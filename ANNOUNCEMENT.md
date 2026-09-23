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

The Laya reference is more accurate, by 30 decisions in 2,000. Primus Decision 0.1 has the lower Brier and ECE under
the same protocol (Brier 0.059 vs 0.062, about one standard error apart; ECE 0.127 vs 0.213; Laya figures quoted from
its model card) with 3.7 M neural parameters plus 143 MB of fitted LSA tables, against the reference's fine-tuned
ModernBERT-large of about 421 M parameters. We make no claim of superiority over Laya or over any other system.

The release contains the exact frozen bytes with checksums, a minimal inference runtime with a runnable example, the
model card, the full benchmark tables, the limitations, the evaluation protocol, reproduction instructions, and a
behavioral-equivalence record showing the public package matching our internal frozen model bit for bit. Exact
parameter counts, package sizes and the runtime footprint are in `MODEL_SIZE_AND_PARAMETERS.md`.

It is **not the complete PRGA/Primus system**: other Primus components are in development and are not part of this
release.

Tag `primus-decision-0.1`. License: Apache-2.0 for the artifacts and runtime in the repository; see `NOTICE` for
scope and trademarks.
