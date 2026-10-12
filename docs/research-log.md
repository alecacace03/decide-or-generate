# Research log: Exploring Decision Models

Challenge: compare non-generative decision models with autoregressive LLMs, and build a hybrid "System 1 / System 2" pipeline. Task used throughout: decide whether a shell or SQL command is `destructive` or `safe`.

## 1. Architecture: why a decision model is O(1) in the output

| | Decision model (Laya) | LLM (Qwen2.5-3B) |
|---|---|---|
| Inference | One forward pass over `[instructions + options + input]`, then a readout of one logit per option | Prefill the prompt, then one forward pass **per generated token**, each reading the growing KV cache |
| Cost in output length | O(1): the answer is always "one of K" | O(N) passes for N output tokens, and each pass grows with the context |
| Output guarantee | Always a valid label (argmax over the options) | Free text. Valid JSON only if the model follows the instruction |
| Confidence | Softmax over option logits, with temperature calibration shipped with the checkpoint | Only what the model *writes* as a number |

Measured on an Apple M4 (MLX, GPU): System 1 takes **10 ms p50** with 0 generated tokens. System 2 takes **775 ms p50** for about 14 output tokens on a ~183-token prompt. That is roughly **75x**.

## 2. Timeline

**v1, CPU-only Linux VM.** Qwen2.5-0.5B, 27 commands. S1 took about 1.1 s and S2 about 19 s, because everything ran on CPU. The 0.5B model also broke the JSON format on some commands (for example it answered `ls -la` with a fake terminal listing). The numbers were not representative, and the dashboard tried to show everything at once.

**v2, native on the M4 GPU.** Same models, now on Metal: S1 took 8.5 ms and S2 about 100 ms. But Qwen 0.5B, when asked for a confidence, labeled **every** command destructive (45% accuracy, a constant classifier). A 0.5B LLM is too small to be a useful System 2.

**v3, Qwen2.5-3B (this version).** The dataset grew to 58 dev commands (21 marked "hard", where the risk hides in a flag or target, such as `git push --force` or `UPDATE ... ;` without `WHERE`) plus 20 held-out commands.

## 3. Prompt experiments

System 2 (dev set, 58 commands):

| Prompt | Accuracy | Latency | Output tokens |
|---|---|---|---|
| Original: "deletes or permanently wipes data" | 78% | 330 ms | 6 |
| **With a definition of destructive** (deletes, overwrites, force push, hard reset, unscoped UPDATE/DELETE...) | **95%** | 708 ms | 14 |
| Definition + "give a one-sentence reason first" | 88% | 1305 ms | 43 |

Asking for a reason first made the 3B model *worse*: it talked itself into calling `du -sh *` and `git checkout -b` destructive. The definition was written after looking at the dev-set failures, so the **held-out set** (20 commands written afterwards) is the honest number: **90%**.

System 1:

| Schema | Accuracy |
|---|---|
| **Short instruction, labels `[destructive, safe]`** | **72%** |
| Labels with descriptions | 69% |
| Short instruction, short label descriptions | 67% |
| Yes/no question type (`noul`) | 64% |
| Same long definition as System 2 | 59% |

The decision model does best with a short, focused question. Giving it the LLM's long definition made it worse. **Lesson: prompts don't transfer between the two kinds of model.**

## 4. Calibration

| | ECE | AUROC of confidence (right vs wrong) | Distinct confidence values |
|---|---|---|---|
| Decision model | 0.20 | **0.64** | 78 |
| LLM (verbalized) | 0.06 | 0.50 | **1** (it always says 1.0) |

The LLM's ECE looks better, but only because it is right 94% of the time and always says "1.0". Its confidence carries **no information** (AUROC 0.5 = a coin flip), so it cannot be used to decide anything. The decision model is overconfident (ECE 0.20: when it says 97%, it is right 81% of the time), but its probability *ranks* errors. That ranking is what lets the router work. Its blind spots are commands it is confidently wrong about: `crontab -r` (99% safe), `> /etc/hosts` (99% safe), `wipefs -a`, `echo '' > ~/.ssh/authorized_keys` (100% safe).

Reliability of the decision model (78 commands):

| Confidence bin | n | Mean confidence | Actual accuracy |
|---|---|---|---|
| 0.5–0.6 | 4 | 0.54 | 1.00 |
| 0.6–0.7 | 4 | 0.65 | 0.50 |
| 0.7–0.8 | 13 | 0.75 | 0.46 |
| 0.8–0.9 | 10 | 0.86 | 0.70 |
| 0.9–1.0 | 47 | 0.97 | 0.81 |

## 5. Hybrid: the threshold is the product decision

| Threshold | Escalated to LLM | Accuracy | p50 | Mean latency | LLM tokens/cmd |
|---|---|---|---|---|---|
| (S1 only) | 0% | 73% | 10 ms | 10 ms | 0 |
| 0.80 | 27% | 82% | 10 ms | 226 ms | 53 |
| **0.90 (default)** | **40%** | **85%** | **11 ms** | **328 ms** | **79** |
| 0.95 | 50% | 88% | 362 ms | 404 ms | 99 |
| 0.99 | 83% | 94% | 784 ms | 661 ms | 165 |
| (LLM only) | 100% | 94% | 791 ms | 789 ms | 198 |

At 0.90, most commands get a 10 ms answer. The run uses 60% fewer LLM tokens and has 2.4x lower mean latency, at a cost of 9 accuracy points. At 0.99 it matches the LLM's accuracy while saving 17% of calls. Where to sit on this curve depends on what an error costs.

## 6. When to use which

- **Decision model alone**: closed label set, high volume, latency budget under ~50 ms, and errors are cheap or reviewed later (routing, tagging, triage, first-pass linting).
- **LLM alone**: open-ended output (explanations, rewrites), rare calls, or a task the decision model has no prior for.
- **Hybrid**: closed label set, but some inputs need real reasoning, and the decision model's confidence ranks its errors (AUROC > 0.5). Tune the threshold on labeled data.

## 7. Limits

- 78 commands is small; the accuracy numbers have wide error bars (±5–10 points).
- Labels are mine. Some commands are debatable (`rm -rf node_modules` was dropped for that reason).
- The System 2 prompt was refined on the dev set, so the held-out set (LLM 90%, decision model 75%, hybrid 80%) is the fairer estimate.
- System 1 was used zero-shot, without any training on this task.
- Latency was measured one request at a time on one machine. Batch throughput for System 1 would be much higher.
