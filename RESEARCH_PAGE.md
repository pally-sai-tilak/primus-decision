# Primus Decision 0.1 — Research Alpha

The first trained model component of Primus: a non-transformer typed-decision research model.

## What it does

Primus Decision 0.1 takes a JSON state and a typed question and returns a probability distribution: is a statement
true (`noul`), which of these options (`choice`), which ordinal level (`score`). It is an ensemble of two small
encoders, a diagonal state-space (S4D) model and a bidirectional GRU, each paired with LSA case features. There is no
transformer anywhere in it. Both members have 3.7 M neural parameters between them and run on a CPU at about 140 ms
per five-decision case (115–162 ms across recorded runs; `EXPERIMENTS.md`).

## Result

One sealed evaluation of the frozen model on the official test split of `LocalLLaMA/typed-decisions`, 400 cases and 2,000 decisions,
raw probabilities:

```
Primus Decision 0.1
Accuracy: 0.751
Brier:    0.059
ECE:      0.127

Laya reference accuracy: 0.766
```

| model | accuracy | soft acc | Brier | ECE | score MAE | within-1 |
|---|---|---|---|---|---|---|
| Primus Decision 0.1 | 0.751 | 0.563 | 0.059 | 0.127 | 0.275 | 0.984 |
| Laya-typed-decisions, published | 0.766 | 0.471 | 0.062 | 0.213 | 0.242 | not published |
| TypeSafe Jev 1.13.0, published generalist | 0.727 | 0.580 | 0.148 | 0.144 | 0.391 | 0.952 |
| meraGPT Decider 1, published generalist (dataset card `main`, 2026-09-22; not in the pinned revision) | 0.768 | 0.608 | 0.052 | 0.180 | 0.219 | 0.984 |

| workflow | Primus accuracy | Laya accuracy |
|---|---|---|
| agent_trace_observability | 0.732 | 0.730 |
| customer_service | 0.728 | 0.764 |
| invoice_processing | 0.808 | 0.804 |
| security_incidents | 0.736 | 0.766 |

Published figures are quoted from their publishers. The full tables are in `BENCHMARKS.md`.

## Reading the result

Primus demonstrates typed decisions with 3.7M neural parameters and local CPU inference. On this benchmark it records
75.1% accuracy, raw ECE 0.127 and Brier 0.059. The reproduced Laya checkpoint records 76.6%, ECE 0.213 and Brier 0.066;
its ordinal MAE is 0.242 versus Primus's 0.275. Published reference figures above retain their original source labels.
The 143 MB of fitted LSA features are additional learned state, separate from the neural parameter count.

## Invoice research

Using its released weights without retraining, Primus scores **26/32 structured reconciliation checks** in a broader synthetic invoice test,
following **32/32** in the initial pilot. The study publishes both input representations, duplicate-ID checks,
all comparison systems and an audit of Qwen answer formats. [Read the case study and evidence](INVOICE_CASE_STUDY.md).

## Why no transformer

The decision layer uses recurrent and state-space encoders by design. The practical consequences are visible in this release: 15 MB of float32 weights, CPU
inference, and a memory footprint set by the featurizers rather than the networks.

## How it was evaluated

Training used the official train split only: 1,005 cases for fitting, 195 held out for validation by a fixed hash
rule. Model, calibration and protocol were frozen before the sealed evaluation. The predeclared comparison rule required accuracy above Laya's published 76.6% plus an improvement in Brier or raw
ECE for a broad superiority claim. Primus recorded 75.1% accuracy, Brier 0.059 and raw ECE 0.127. It is published as
the research alpha provided for in that protocol, with each measured capability and comparison reported explicitly.
Protocol: `PROTOCOL.md`; machine record: `SEALED_RESULT.json`.

## Evaluation conditions

The evidence covers supported schemas, the recorded state formats and one frozen ensemble. Input conventions,
calibration choices, validation sample size and runtime footprint are documented in [Scope and evaluation notes](LIMITATIONS.md).

## Verify and reproduce

`REPRODUCIBILITY.md` covers checksum verification, the environment, the dataset revision, the recorded training setup, the sealed
evaluation and inference. `MODEL_SIZE_AND_PARAMETERS.md` has the exact parameter counts, package sizes and footprint,
and `BEHAVIORAL_EQUIVALENCE.md` shows the public package matching the internal frozen one bit for bit on a 600-decision fixture.

## Context

`ARCHITECTURE.md` describes the released decision component. Our broader research direction and the distinction
between public evidence and protected development are set out in [The Road to Primus](README.md#the-road-to-primus).

## Citation and license

Primus Decision 0.1 (Research Alpha), AAME, 2026-09-22. Model artifacts, runtime and documentation: Apache License
2.0, this release only (`NOTICE`); Primus names and logos are trademarks and are not licensed. Benchmark dataset
`LocalLLaMA/typed-decisions`, Apache-2.0.
