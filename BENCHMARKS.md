# Benchmarks — Primus Decision 0.1 (Research Alpha)

Benchmark: `LocalLLaMA/typed-decisions` (revision `ea9306458d6e9563628369a3d1e72e362fb381d2`), official **test** split,
400 cases / 2,000 decisions, one sealed release evaluation after freezing (`SEALED_RESULT.json`, sealed run #1).
Later verification and robustness analyses are documented separately in `EXPERIMENTS.md`; they did not change the released weights or replace the sealed result.
Protocol and metric definitions: `PROTOCOL.md` (bundled). Mode: specialist (same as Laya-typed-decisions).

| Model | accuracy | soft acc | Brier | ECE | score MAE | within-1 | KL |
|---|---|---|---|---|---|---|---|
| **primus-decision-0.1 (raw)** | **0.751** | 0.563 | 0.059 | 0.127 | 0.275 | 0.984 | 0.102 |
| primus-decision-0.1 (temperature-calibrated) | 0.751 | 0.618 | 0.114 | 0.040 | 0.320 | 0.970 | 0.301 |
| Laya-typed-decisions (reproduced by us on the same harness, raw; `EXPERIMENTS.md`) | 0.766 | 0.509 | 0.066 | 0.213 | 0.242 | 0.995 | 0.122 |
| Laya-typed-decisions (published, raw) | 0.766 | 0.471 | 0.062 | 0.213 | 0.242 | - | - |
| TypeSafe Jev 1.13.0 (published, generalist) | 0.727 | 0.580 | 0.148 | 0.144 | 0.391 | 0.952 | 1.442 |
| meraGPT Decider 1 (published, generalist, zero-shot; proprietary) | 0.768 | 0.608 | 0.052 | 0.180 | 0.219 | 0.984 | 0.096 |

Laya's reproduced row comes from running its released checkpoint through this protocol on our machine: accuracy, ECE, score
MAE and every per-workflow and per-primitive accuracy reproduce the model card exactly; the card's soft accuracy and Brier do
not (0.509 and 0.066 with both its author's harness and ours), so the reproduced row is the one to compare with.
Published figures are quoted from their publishers, not re-measured: Laya from
https://huggingface.co/convaiinnovations/laya-typed-decisions (model card, retrieved 2026-09-22); Jev from the dataset card
at the pinned revision; meraGPT Decider 1 from the dataset card's `main` branch on 2026-09-22, which lists it as the
leader (it is not in revision `ea930645…`, the one used for the release rule, and the rule was written against Laya).
It is above this model on accuracy, Brier and score MAE and below it on ECE; as the card itself notes, generalists
answer zero-shot and specialists are fitted to these workflows, so the two kinds are not directly comparable.
Laya's calibrated ECE (≈0.081) is a family-level post-temperature figure from https://huggingface.co/convaiinnovations/laya.

## Release gate (computed by the sealed runner)

| Check | Result |
|---|---|
| accuracy > 0.766 | False |
| Brier < 0.062 | True |
| raw ECE < 0.213 | True |
| score MAE < 0.242 (informational, not part of the rule) | False |
| **broad claim of superiority over Laya allowed** (accuracy AND (Brier OR raw ECE)) | **False** |

## By workflow

| workflow | accuracy (raw) | ECE (raw) | ECE (calibrated) | Laya accuracy | difference |
|---|---|---|---|---|---|
| agent_trace_observability | 0.732 | 0.160 | 0.036 | 0.730 | +0.002 (1 additional matching decision in 500) |
| customer_service | 0.728 | 0.112 | 0.070 | 0.764 | −0.036 |
| invoice_processing | 0.808 | 0.097 | 0.044 | 0.804 | +0.004 (2 additional matching decisions in 500) |
| security_incidents | 0.736 | 0.145 | 0.047 | 0.766 | −0.030 |

Each workflow has 100 cases and 500 related decisions. The table reports observed differences on those cases.
Comparative uncertainty should be estimated from paired predictions while preserving case grouping; the one- and
two-decision differences above are not evidence of a broad performance advantage.

## By primitive

| type | accuracy (raw) | ECE (raw) | ECE (calibrated) | Laya accuracy |
|---|---|---|---|---|
| choice | 0.738 | 0.139 | 0.066 | 0.733 |
| noul | 0.837 | 0.084 | 0.031 | 0.857 |
| score | 0.696 | 0.150 | 0.043 | 0.723 |

## The gap in decisions (raw accuracy × n, rounded)

| slice | n | Primus − Laya (decisions) |
|---|---|---|
| overall | 2000 | -30 |
| agent_trace_observability | 500 | +1 |
| customer_service | 500 | -18 |
| invoice_processing | 500 | +2 |
| security_incidents | 500 | -15 |
| choice | 600 | +3 |
| noul | 600 | -12 |
| score | 800 | -21 |

## Calibration profile (experimental, not the default output)

Raw → temperature-calibrated on the sealed test: ECE 0.127 → 0.040, NLL 0.637 → 0.568,
but Brier 0.059 → 0.114 and score MAE 0.275 → 0.320. The raw probabilities are the
default output; the profile is shipped separately as `model/calibration.json` and applied only on request.

Latency recorded by the sealed runner: 134.04 ms per case (5 decisions), which is the wall time of loading the model
(including the two featurizers) and predicting all 400 cases in batches of 32, divided by 400, on the evaluation CPU,
no GPU. Steady-state figures with the model already loaded: about 140 ms per case one at a time (p50) and 36–65 ms per case
at batch 32; six runs with two methods on the same machine class gave p50 115–162 ms, so expect run-to-run variation of that
size; every run and both methods are in `EXPERIMENTS.md` and `MODEL_SIZE_AND_PARAMETERS.md`.
Environment of the sealed run (same container as training): {"python": "3.11.15", "platform": "Linux x86_64, glibc 2.39", "machine": "x86_64", "cpu_count": 4, "torch": "2.14.0+cpu", "torch_threads": 2, "cuda": false, "numpy": "2.4.6", "cpu_model": "Intel(R) Xeon(R) Processor @ 2.10GHz"}.
