"""Local web server for the live-test dashboard (decision model vs LLM vs hybrid).

    python -m dual_engine.server            # http://127.0.0.1:8000

Routes
    GET  /              the dashboard
    GET  /api/info      model names, default threshold, example commands, benchmark summary
    POST /api/run       {"command": str, "threshold": float} -> System 1, System 2 and hybrid results
"""
import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .dataset import LABELED_COMMANDS
from .engines import DEFAULT_THRESHOLD, Engines

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
BENCHMARK = ROOT / "data" / "benchmark.json"
MAX_COMMAND_CHARS = 500

ENGINES = None


def info():
    summary = None
    if BENCHMARK.exists():
        summary = json.loads(BENCHMARK.read_text()).get("summary")
    return {
        "s1_model": ENGINES.s1_model_id,
        "s2_model": ENGINES.s2_model_id,
        "threshold": DEFAULT_THRESHOLD,
        "examples": [{"command": c, "expected": e, "hard": h} for c, e, h in LABELED_COMMANDS],
        "benchmark": summary,
    }


def run(command, threshold):
    # The GPU clocks down after a few idle seconds, so the first pass after a pause takes
    # 40-70 ms instead of ~10 ms. An untimed warm-up pass makes the live numbers match the
    # benchmark, which measures steady traffic.
    ENGINES.system1("warmup")
    s1 = ENGINES.system1(command)
    s2 = ENGINES.system2(command)
    # The hybrid reuses the System 1 result; it only "pays" for System 2 when it escalates.
    escalated = s1["confidence"] < threshold
    if escalated:
        decision = s2["decision"] if s2["valid"] else s1["decision"]
        latency = s1["latency_ms"] + s2["latency_ms"]
        tokens = s2["input_tokens"] + s2["output_tokens"]
    else:
        decision, latency, tokens = s1["decision"], s1["latency_ms"], 0
    expected = next((e for c, e, _ in LABELED_COMMANDS if c == command.strip()), None)
    return {
        "command": command,
        "expected": expected,
        "s1": s1,
        "s2": s2,
        "hybrid": {"decision": decision, "route": "escalated" if escalated else "fast",
                   "latency_ms": latency, "llm_tokens": tokens, "threshold": threshold},
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/info":
            self._send(200, info())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/run":
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            command = str(body.get("command", "")).strip()
            threshold = float(body.get("threshold", DEFAULT_THRESHOLD))
        except (ValueError, TypeError):
            return self._send(400, {"error": "Send JSON like {\"command\": \"git status\"}."})
        if not command:
            return self._send(400, {"error": "Type a command to classify."})
        if len(command) > MAX_COMMAND_CHARS:
            return self._send(400, {"error": f"Keep the command under {MAX_COMMAND_CHARS} characters."})
        self._send(200, run(command, min(max(threshold, 0.5), 1.0)))

    def log_message(self, fmt, *args):
        pass


def main():
    global ENGINES
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    print("Loading System 1 (decision model) and System 2 (LLM)... the first run downloads ~2 GB of weights.")
    ENGINES = Engines()
    url = f"http://{args.host}:{args.port}"
    print(f"Ready: {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
