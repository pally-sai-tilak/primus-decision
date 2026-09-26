# Primus Decision 0.1 — Research Alpha

The first trained model component of Primus: a non-transformer typed-decision research model.

We are releasing Primus Decision 0.1 under Apache-2.0. Given a JSON state and a typed question, it returns a
probability distribution: is a statement true, which of these options applies, which ordinal level. It is
an ensemble of a diagonal state-space (S4D) encoder and a bidirectional GRU with LSA case features, 3,715,074 neural
parameters in total, and it runs on a CPU at about 140 ms per five-decision case (115–162 ms across recorded runs; method and every run in `EXPERIMENTS.md`).

We evaluated it once on the sealed official test split of `LocalLLaMA/typed-decisions`, after freezing the
architecture, hyper-parameters and calibration, under a release rule written before the split was read:

```
Primus Decision 0.1
Accuracy: 0.751
Brier:    0.059
ECE:      0.127

Laya reference accuracy: 0.766
```

Primus pairs 75.1% accuracy with raw ECE 0.127 and Brier 0.059, using 3.7M neural parameters plus 143 MB of fitted
LSA features. The published Laya reference records 76.6% accuracy, ECE 0.213 and Brier 0.062 with approximately 421M
neural parameters. Our full tables also distinguish the reproduced reference measurements and ordinal score error.

Our [invoice case study](INVOICE_CASE_STUDY.md) adds a concrete application test: 26/32 structured
reconciliation checks in a broader synthetic experiment, following 32/32 in the initial pilot. We publish both
representations, every comparison system and raw predictions alongside the study.

The release contains the exact frozen bytes with checksums, a minimal inference runtime with a runnable example, the
model card, the full benchmark tables, scope and evaluation notes, the evaluation protocol, reproduction instructions, and a
behavioral-equivalence record showing the public package matching our internal frozen model bit for bit. Exact
parameter counts, package sizes and the runtime footprint are in `MODEL_SIZE_AND_PARAMETERS.md`.

This is our first public component in a longer Primus research programme. We publish evidence for the released
capability while keeping unreleased components and training methods private; see [Public boundary](PUBLIC_BOUNDARY.md).

Tag `primus-decision-0.1`. License: Apache-2.0 for the artifacts and runtime in the repository; see `NOTICE` for
scope and trademarks.
