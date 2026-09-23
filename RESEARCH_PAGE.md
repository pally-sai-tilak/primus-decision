# Primus Decision 0.1 — Research Alpha

The first trained model component of Primus: a non-transformer typed-decision research model.

## What it does

Primus Decision 0.1 takes a JSON state and a typed question and returns a probability distribution: is a statement
true (`noul`), which of these options (`choice`), which ordinal level (`score`). It is an ensemble of two small
encoders, a diagonal state-space (S4D) model and a bidirectional GRU, each paired with LSA case features. There is no
transformer anywhere in it. Both members have 3.7 M neural parameters between them and run on a CPU at about 140 ms
per five-decision case (115–162 ms across recorded runs; `EXPERIMENTS.md`).

## Result

Evaluated once on the sealed official test split of `LocalLLaMA/typed-decisions`, 400 cases and 2,000 decisions,
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

The Laya reference wins on accuracy by 30 decisions in 2,000, and it wins clearly on the `score` primitive, on
customer service and on security incidents. Primus Decision 0.1 has the lower Brier and ECE under the same protocol
(the Brier gap, 0.059 against 0.062, is about one standard error; the ECE gap is not); on agent-trace observability
and invoice processing the two are within one or two decisions in 500, which is a tie. It is a different kind of
object: 3.7 M neural parameters plus 143 MB of fitted LSA tables, against a fine-tuned ModernBERT-large that its model
card puts at about 421 M parameters. A model this small gets this close on this benchmark with lower calibration
error; it does not beat anything.

## Why no transformer

The decision layer uses recurrent and state-space encoders by design. The practical consequences are visible in this release: 15 MB of float32 weights, CPU
inference, and a memory footprint set by the featurizers rather than the networks.

## How it was evaluated

Training used the official train split only: 1,005 cases for fitting, 195 held out for validation by a fixed hash
rule. The release rule was committed before the test split was read: primary gate accuracy above the reference,
secondary gate Brier or raw ECE below it. Architecture, hyper-parameters and calibration were frozen with content
hashes; the test split was opened once. Primary failed, secondary passed. Protocol in `PROTOCOL.md`, machine record in
`SEALED_RESULT.json`.

## Limitations

Specialist scope, a small validation split, sensitivity to state formatting, an experimental calibration profile
that trades Brier for ECE, one seed per member, and the memory cost of the featurizers. Details in `LIMITATIONS.md`.

## Verify and reproduce

`REPRODUCIBILITY.md` covers checksum verification, the environment, the dataset revision, retraining, the sealed
evaluation and inference. `MODEL_SIZE_AND_PARAMETERS.md` has the exact parameter counts, package sizes and footprint,
and `BEHAVIORAL_EQUIVALENCE.md` shows the public package matching the internal frozen one bit for bit.

## Context

`ARCHITECTURE.md` places the decision layer inside Primus. It is **not the complete PRGA/Primus system**; other Primus
components are in development and are not part of this release.

## Citation and license

Primus Decision 0.1 (Research Alpha), AAME, 2026-09-22. Model artifacts, runtime and documentation: Apache License
2.0, this release only (`NOTICE`); Primus names and logos are trademarks and are not licensed. Benchmark dataset
`LocalLLaMA/typed-decisions`, Apache-2.0.
