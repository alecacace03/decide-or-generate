"""Benchmark: pure LLM vs. decision model vs. hybrid circuit breaker.

    python benchmarks/run_benchmark.py --s1-repeats 10 --s2-repeats 3

Runs every labeled command through System 1 and System 2 several times, then
derives the hybrid pipeline (System 1 first, escalate to System 2 below a
confidence threshold) from the recorded results. Writes data/benchmark.json,
which both the local dashboard and the write-up read.
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dual_engine.dataset import HELDOUT_COMMANDS, LABELED_COMMANDS  # noqa: E402
from dual_engine.engines import DEFAULT_THRESHOLD, S2_MAX_TOKENS, Engines  # noqa: E402

THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.01]


def pct(values, p):
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def lat_stats(values):
    return {"p50": pct(values, 50), "p95": pct(values, 95), "p99": pct(values, 99),
            "mean": statistics.fmean(values)}


def ece(pairs, bins=10):
    """Expected Calibration Error over (confidence, correct) pairs."""
    pairs = [(c, ok) for c, ok in pairs if c is not None]
    if not pairs:
        return None, []
    total, out, table = len(pairs), 0.0, []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inside = [(c, ok) for c, ok in pairs if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not inside:
            continue
        conf = statistics.fmean(c for c, _ in inside)
        acc = statistics.fmean(1.0 if ok else 0.0 for _, ok in inside)
        out += len(inside) / total * abs(acc - conf)
        table.append({"lo": lo, "hi": hi, "n": len(inside), "confidence": conf, "accuracy": acc})
    return out, table


def auroc(pairs):
    """How well confidence separates right from wrong answers: 0.5 = no signal, 1.0 = perfect."""
    right = [c for c, ok in pairs if c is not None and ok]
    wrong = [c for c, ok in pairs if c is not None and not ok]
    if not right or not wrong:
        return None
    wins = sum(1.0 if r > w else 0.5 if r == w else 0.0 for r in right for w in wrong)
    return wins / (len(right) * len(wrong))


def device():
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except Exception:
        chip = ""
    return f"{chip or platform.machine()} · {platform.system()} {platform.release()} · MLX GPU"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-repeats", type=int, default=10)
    ap.add_argument("--s2-repeats", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "data" / "benchmark.json"))
    args = ap.parse_args()

    eng = Engines()
    rows = []
    commands = [(c, e, h, "dev") for c, e, h in LABELED_COMMANDS] + [(c, e, h, "heldout") for c, e, h in HELDOUT_COMMANDS]
    for i, (cmd, expected, hard, split) in enumerate(commands, 1):
        s1_runs = [eng.system1(cmd) for _ in range(args.s1_repeats)]
        s2_runs = [eng.system2(cmd) for _ in range(args.s2_repeats)]
        s1, s2 = s1_runs[0], s2_runs[0]  # both engines are deterministic; later runs only add latency samples
        rows.append({
            "command": cmd, "expected": expected, "hard": hard, "split": split,
            "s1": {"decision": s1["decision"], "confidence": s1["confidence"], "probabilities": s1["probabilities"],
                   "latency_ms": [r["latency_ms"] for r in s1_runs], "input_tokens": s1["input_tokens"]},
            "s2": {"decision": s2["decision"], "confidence": s2["confidence"], "raw": s2["raw"], "valid": s2["valid"],
                   "latency_ms": [r["latency_ms"] for r in s2_runs],
                   "input_tokens": s2["input_tokens"], "output_tokens": s2["output_tokens"],
                   "consistent": len({r["raw"] for r in s2_runs}) == 1},
        })
        print(f"[{i}/{len(commands)}] {cmd[:50]}", flush=True)

    n = len(rows)

    def hybrid_at(t):
        correct, lats, tokens, escalated = 0, [], 0, 0
        for r in rows:
            s1_lat = statistics.median(r["s1"]["latency_ms"])
            if r["s1"]["confidence"] >= t:
                dec, lat = r["s1"]["decision"], s1_lat
            else:
                escalated += 1
                dec = r["s2"]["decision"] if r["s2"]["valid"] else r["s1"]["decision"]
                lat = s1_lat + statistics.median(r["s2"]["latency_ms"])
                tokens += r["s2"]["input_tokens"] + r["s2"]["output_tokens"]
            correct += dec == r["expected"]
            lats.append(lat)
        # a "missed block" is a destructive command that was allowed through: the costly error
        return {"threshold": t, "accuracy": correct / n, "escalation_rate": escalated / n,
                "llm_tokens_per_cmd": tokens / n, "latency": lat_stats(lats),
                "throughput_per_s": 1000 / statistics.fmean(lats)}

    def missed(dec_fn):
        return sum(1 for r in rows if r["expected"] == "destructive" and dec_fn(r) != "destructive")

    s1_lats = [x for r in rows for x in r["s1"]["latency_ms"]]
    s2_lats = [x for r in rows for x in r["s2"]["latency_ms"]]
    s1_ece, s1_bins = ece([(r["s1"]["confidence"], r["s1"]["decision"] == r["expected"]) for r in rows])
    s1_pairs = [(r["s1"]["confidence"], r["s1"]["decision"] == r["expected"]) for r in rows]
    s2_pairs = [(r["s2"]["confidence"], r["s2"]["decision"] == r["expected"]) for r in rows if r["s2"]["valid"]]
    s2_ece, s2_bins = ece([(r["s2"]["confidence"], r["s2"]["decision"] == r["expected"]) for r in rows if r["s2"]["valid"]])
    sweep = [hybrid_at(t) for t in THRESHOLDS]
    chosen = hybrid_at(DEFAULT_THRESHOLD)

    def hybrid_dec(r, t=DEFAULT_THRESHOLD):
        if r["s1"]["confidence"] >= t:
            return r["s1"]["decision"]
        return r["s2"]["decision"] if r["s2"]["valid"] else r["s1"]["decision"]

    def acc_by_split(dec_fn):
        return {sp: statistics.fmean(dec_fn(r) == r["expected"] for r in rows if r["split"] == sp)
                for sp in ("dev", "heldout")}

    summary = {
        "meta": {"s1_model": eng.s1_model_id, "s2_model": eng.s2_model_id, "device": device(),
                 "commands": n, "hard_commands": sum(r["hard"] for r in rows),
                 "heldout_commands": sum(r["split"] == "heldout" for r in rows),
                 "s1_repeats": args.s1_repeats, "s2_repeats": args.s2_repeats,
                 "s2_max_tokens": S2_MAX_TOKENS, "threshold": DEFAULT_THRESHOLD,
                 "created": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "s1": {"accuracy": sum(r["s1"]["decision"] == r["expected"] for r in rows) / n,
               "accuracy_hard": statistics.fmean(r["s1"]["decision"] == r["expected"] for r in rows if r["hard"]),
               "accuracy_split": acc_by_split(lambda r: r["s1"]["decision"]),
               "valid_rate": 1.0, "latency": lat_stats(s1_lats), "llm_tokens_per_cmd": 0,
               "missed_destructive": missed(lambda r: r["s1"]["decision"]),
               "throughput_per_s": 1000 / statistics.fmean(s1_lats), "ece": s1_ece, "confidence_auroc": auroc(s1_pairs), "calibration": s1_bins},
        "s2": {"accuracy": sum(r["s2"]["decision"] == r["expected"] for r in rows) / n,
               "accuracy_hard": statistics.fmean(r["s2"]["decision"] == r["expected"] for r in rows if r["hard"]),
               "accuracy_split": acc_by_split(lambda r: r["s2"]["decision"]),
               "valid_rate": sum(r["s2"]["valid"] for r in rows) / n, "latency": lat_stats(s2_lats),
               "llm_tokens_per_cmd": statistics.fmean(r["s2"]["input_tokens"] + r["s2"]["output_tokens"] for r in rows),
               "output_tokens_per_cmd": statistics.fmean(r["s2"]["output_tokens"] for r in rows),
               "missed_destructive": missed(lambda r: r["s2"]["decision"]),
               "throughput_per_s": 1000 / statistics.fmean(s2_lats), "ece": s2_ece, "confidence_auroc": auroc(s2_pairs),
               "distinct_confidences": sorted({c for c, _ in s2_pairs}), "calibration": s2_bins},
        "hybrid": {**chosen, "missed_destructive": missed(hybrid_dec), "accuracy_split": acc_by_split(hybrid_dec),
                   "accuracy_hard": statistics.fmean(hybrid_dec(r) == r["expected"] for r in rows if r["hard"])},
        "sweep": sweep,
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps({k: {kk: summary[k][kk] for kk in ("accuracy", "latency")} for k in ("s1", "s2", "hybrid")}, indent=1))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
