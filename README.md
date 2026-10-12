# Decide or Generate?

**Decision models vs large language models.** An autonomous-learning research project: how a decision model (which *picks* an answer from a fixed list) differs from an LLM (which *writes* one token at a time), where each one wins, and an experiment that tests both on a real problem.

Read the full write-up on the project site: <https://alecacace03.github.io/decide-or-generate/> (source in [docs/index.html](docs/index.html)).

## The experiment

The prototype checks whether a shell or SQL command is **destructive** (block it) or **safe** (let it run), using two kinds of model:

- **System 1, decision model**: [Laya-MLX](https://huggingface.co/aac6fef/laya-multilingual-mlx). One forward pass reads the logits for a fixed set of answers. It takes about 10 ms, always returns a valid label, and gives a probability.
- **System 2, LLM**: Qwen2.5-3B-Instruct (4-bit, via `mlx-lm`). It writes a JSON answer token by token. It takes about 800 ms and is more accurate on tricky commands.
- **Hybrid**: System 1 answers when it is confident. Below a confidence threshold, the command goes to System 2.

The dashboard lets you type any command and see the three verdicts side by side, with latency, tokens and the route the hybrid took.

## Run the prototype

You need a Mac with Apple silicon (M1 or newer), Python 3.10+, and about 3 GB of free disk space.

```bash
./run.sh
```

The first run creates `.venv`, installs three packages, downloads the two models (~2 GB) and opens <http://127.0.0.1:8000>. Later runs start in a few seconds. Stop the server with `Ctrl+C`.

Options: `./run.sh --port 8080`, `./run.sh --no-browser`. To use a different LLM, set `S2_MODEL`, for example `S2_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit ./run.sh`. To change the default threshold, set `HYBRID_THRESHOLD=0.95`.

## Reproduce the benchmark

```bash
PYTHONPATH=src .venv/bin/python benchmarks/run_benchmark.py --s1-repeats 10 --s2-repeats 3
```

This writes `data/benchmark.json`, which the dashboard's benchmark table reads. Results from an Apple M4:

| 78 labeled commands | Decision model | LLM only | Hybrid (threshold 90%) |
|---|---|---|---|
| Accuracy | 73% | 94% | 85% |
| Accuracy, 20 held-out commands | 75% | 90% | 80% |
| Destructive commands let through | 20 | 5 | 11 |
| Latency p50 / p95 / p99 | 10 / 13 / 15 ms | 775 / 855 / 1084 ms | 11 / 843 / 1043 ms |
| LLM tokens per command | 0 | 198 | 79 |
| Valid output | 100% | 100% | 100% |

See [docs/research-log.md](docs/research-log.md) for the method, the experiments that did not work, and the limits of these numbers.

## Layout

```
run.sh                      one-command launcher
src/dual_engine/engines.py  System 1, System 2 and the hybrid router
src/dual_engine/dataset.py  58 dev commands + 20 held-out commands, labeled
src/dual_engine/server.py   local web server (standard library only)
src/dual_engine/static/     the dashboard
benchmarks/run_benchmark.py benchmark -> data/benchmark.json
```
