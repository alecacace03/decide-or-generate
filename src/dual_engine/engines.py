"""The two engines and the hybrid router.

System 1: Laya-MLX decision model. One forward pass, logits read out over a closed
          set of choices, so the answer is always valid and carries a probability.
System 2: a small instruction-tuned LLM (Qwen2.5 via mlx-lm) that writes a JSON
          answer token by token.
Hybrid:   System 1 answers when it is confident; below the threshold the command
          is escalated to System 2. The routing is applied in server.run() and in
          benchmarks/run_benchmark.py, which reuse one System 1 result per command.
"""
import json
import os
import threading
import time

import laya_mlx as laya
from mlx_lm import generate, load

S1_MODEL_ID = os.environ.get("S1_MODEL", "aac6fef/laya-multilingual-mlx")
S2_MODEL_ID = os.environ.get("S2_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
DEFAULT_THRESHOLD = float(os.environ.get("HYBRID_THRESHOLD", "0.90"))
LABELS = ("destructive", "safe")

S1_SCHEMA = {
    "risk_level": {
        "type": "choice",
        "instructions": "Determine if the command is destructive (deletes, drops, wipes data) "
                        "or safe (read-only, inspection, routine build).",
        "criteria": list(LABELS),
    }
}

# Both engines answer the same question. System 1 keeps a short instruction (it scored
# 72% short vs 59% with the long definition below); System 2 gets the full definition
# (78% -> 95% on the dev set). See docs/research-log.md.
S2_DEFINITION = (
    "A command is DESTRUCTIVE if running it can irreversibly delete, overwrite, truncate or discard data, files, "
    "git history, databases, cloud resources, disks or configuration. This includes recursive or wildcard deletes, "
    "DROP/TRUNCATE, DELETE or UPDATE without a WHERE clause, force pushes, hard resets, git clean, formatting disks, "
    "redirecting over files, and destroy/prune/flush commands. A command is SAFE if it only reads, lists, builds, "
    "starts or restarts services, copies, or makes a narrow, recoverable change (for example an UPDATE/DELETE with a "
    "selective WHERE clause)."
)
S2_SYSTEM = (
    "You are a security linter for shell and SQL commands.\n" + S2_DEFINITION + "\n"
    'Return strictly JSON: {"risk": "destructive"|"safe", "confidence": <number between 0 and 1>}'
)
S2_MAX_TOKENS = 40


class Engines:
    """Loads both models once. MLX inference is not thread-safe, so calls are serialized."""

    def __init__(self):
        self.lock = threading.Lock()
        self.s1 = laya.load(S1_MODEL_ID)
        self.s2_model, self.s2_tok = load(S2_MODEL_ID)
        self.s1_model_id, self.s2_model_id = S1_MODEL_ID, S2_MODEL_ID
        # warm both paths so the first real request is not penalized by graph compilation
        self.system1("warmup")
        self.system2("warmup")

    def system1(self, command):
        with self.lock:
            t0 = time.perf_counter()
            result = self.s1.predict(command, S1_SCHEMA)
            ms = (time.perf_counter() - t0) * 1000
        ans = result["answers"]["risk_level"]
        probs = {k: float(v) for k, v in ans["probabilities"].items()}
        decision = ans["choice"]
        return {
            "decision": decision,
            "confidence": probs[decision],
            "probabilities": probs,
            "latency_ms": ms,
            "input_tokens": (result.get("usage") or {}).get("input_tokens"),
            "output_tokens": 0,
            "valid": decision in LABELS,
        }

    def _s2_prompt(self, command):
        return (
            f"<|im_start|>system\n{S2_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\nCommand: {command}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def system2(self, command):
        prompt = self._s2_prompt(command)
        with self.lock:
            t0 = time.perf_counter()
            text = generate(self.s2_model, self.s2_tok, prompt=prompt, max_tokens=S2_MAX_TOKENS, verbose=False)
            ms = (time.perf_counter() - t0) * 1000
        decision, conf, valid = None, None, False
        try:
            obj = json.loads(text.strip())
            decision = obj.get("risk")
            conf = obj.get("confidence")
            conf = float(conf) if conf is not None else None
            valid = decision in LABELS
        except Exception:
            pass
        return {
            "decision": decision if valid else None,
            "confidence": conf if valid else None,
            "raw": text.strip(),
            "latency_ms": ms,
            "input_tokens": len(self.s2_tok.encode(prompt)),
            "output_tokens": len(self.s2_tok.encode(text)),
            "valid": valid,
        }
