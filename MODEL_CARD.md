# Model card — Primus Decision 0.1 (Research Alpha)

**Positioning.** The first trained model component of Primus: a non-transformer typed-decision research model.

**Scope.** This artifact is **Primus System 1**, the typed-decision component. It answers supported questions about
one JSON state with probability distributions. Persistent memory, deliberation and tool orchestration belong to the
broader research roadmap; the released interface evaluates each supplied state independently.

## Result

Sealed official test split of `LocalLLaMA/typed-decisions`, 400 cases and 2,000 decisions, evaluated once after
freezing, raw probabilities:

```
Primus Decision 0.1
Accuracy: 0.751
Brier:    0.059
ECE:      0.127

Laya reference accuracy: 0.766
```

| metric | Primus Decision 0.1 (measured by us, sealed run) | Laya-typed-decisions (reproduced by us, same harness) | Laya-typed-decisions (published model card) |
|---|---|---|---|
| accuracy | 0.751 | 0.766 | 0.766 |
| Brier | 0.059 | 0.066 | 0.062 |
| ECE, 15 bins on the top probability | 0.127 | 0.213 | 0.213 |
| score MAE | 0.275 | 0.242 | 0.242 |
| soft accuracy | 0.563 | 0.509 | 0.471 |
| NLL | 0.637 | 0.707 | not published |

Primus pairs 75.1% accuracy with Brier 0.059 and ECE 0.127. Reproduced Laya scores 76.6% accuracy, Brier 0.066,
ECE 0.213 and ordinal MAE 0.242, compared with Primus MAE 0.275. The accuracy difference is 30 decisions in 2,000.
The published and reproduced columns identify their evidence sources: accuracy, ECE and score MAE reproduce Laya's
card exactly; soft accuracy and Brier differ under the recorded harness. See `EXPERIMENTS.md` and `BENCHMARKS.md`
for the full comparisons, including the externally reported meraGPT result (accuracy 0.768, Brier 0.052,
2026-09-22 snapshot). The machine record remains `SEALED_RESULT.json`.

## Invoice case study

The unchanged model scored **26/32 on structured reconciliation** in the broader synthetic invoice experiment and
**32/32 in the initial pilot**. The broader study also records 17/32 on the narrative presentation of the same facts,
alongside classifier, rule and standalone Qwen3-1.7B comparisons. Read the
[complete case study and evidence](INVOICE_CASE_STUDY.md). These are separate task-specific experiments.

## Model

Two members, averaged with equal weight, each a token encoder over the flattened state text plus a 256-d LSA case
vector, additive attention pooling conditioned on the question, and an option scorer:

| member | encoder | neural parameters |
|---|---|---|
| `model/member0` | 2 diagonal state-space (S4D) blocks, d = 192, 16 states per channel | 1,932,609 |
| `model/member1` | 2 bidirectional GRU layers, d = 192 | 1,782,465 |

That is 3,715,074 neural parameters in total, all of them used on every prediction because both members run each
time. Each member also carries an LSA featurizer (word and character TF-IDF plus a 256-d truncated SVD fitted on the
training split), which is a learned statistical table rather than trained weights: 71 MB each on disk and the reason
the package is 158 MB rather than 15 MB. Exact counts, sizes and measurements: `MODEL_SIZE_AND_PARAMETERS.md`.

- **Question types.** `noul` gives the probability that a statement is true; `choice` a distribution over the
  supplied options; `score` a distribution over ordinal levels plus its expected value. The number of options is not fixed.
- **Inputs.** The state flattened to text, deterministic derived-relation sentences (numeric comparisons within a
  key family, list membership, signs, nulls), the question instructions and the option criteria.
- **Training data.** The official train split only, 1,005 cases for fitting and 195 held out as validation by a
  fixed hash rule. This public dataset is synthetic and teacher-labelled; no additional external data were used.
  Primus itself uses trained-from-scratch, non-transformer encoders and no LLM call at inference.
- **Interface.** A request is one JSON state plus typed questions; a reply is one distribution per question.
  `INTERFACE.md` and `schemas/` specify both, and the Decision History record for logging answers and outcomes.
- **Output.** Raw probabilities. `model/calibration.json` is an experimental temperature profile fitted on the
  validation split; on the sealed test it improves ECE (0.127 → 0.040) and NLL but worsens Brier (0.059 → 0.114) and
  score MAE, so it is off unless asked for.
- **Speed and memory.** About 140 ms for a five-decision case (p50) with two CPU threads and the model already loaded:
  six runs with two methods on the same machine class gave p50 115–162 ms, 36–65 ms per case in batches of 32 and 20–34
  cases per second over the whole test split (every run in `EXPERIMENTS.md`; the probe method in `MODEL_SIZE_AND_PARAMETERS.md`);
  roughly 0.5–0.6 GB resident after loading and 1.4–1.6 GB at peak, almost all of it the featurizers.

## Intended use

A research artifact and specialist for agent-trace observability, customer service, invoice processing and security
incidents within the 20 released schemas. Application development should evaluate the intended records and adapt the
model or schema coverage where needed. Use human review for consequential decisions. See
[Scope and evaluation notes](LIMITATIONS.md).

## Evaluation protocol

`PROTOCOL.md` describes the fixed train/validation hash split, the once-only sealed read of the test split, the
metric definitions (the same conventions as the published Laya evaluation) and the release rule, which was written
down before the test split was opened. The rule's primary gate, accuracy above the reference, failed; the secondary
gate, Brier or ECE below the reference, passed. That is why this is a research alpha.

## Reproduction and provenance

`REPRODUCIBILITY.md` has the environment, dataset revision, training configuration, the sealed-evaluation procedure,
an inference example and the expected numbers. `PROVENANCE.json` has the frozen hash of every model file. Frozen
2026-09-22T06:22:58Z, sealed evaluation 2026-09-22T06:32:28Z, dataset manifest sha256 `4752590be05d1f40be378eb21cd0b42b75877b2b5262eddf59892f93c2cc02d1`, sealed test file sha256
`4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c`. The model bytes are identical to the internal
frozen release; `BEHAVIORAL_EQUIVALENCE.md` shows the public package producing bit-identical outputs on a
600-decision fixture.

## License

Model artifacts, runtime and documentation: **Apache License 2.0** (`LICENSE`, `NOTICE`). The license covers this
release only; other Primus components, past or future, are licensed separately. "Primus", "Primus AI", "AAME" and the
associated logos are trademarks of Pally Sai Tilak / AAME and are not licensed. Dataset: Apache-2.0
(`LocalLLaMA/typed-decisions`); no dataset rows are included.

## Broader Primus programme

The released decision component is the first public research alpha. The longer-term architecture and the boundary
between auditable public evidence and protected research are described in [Public boundary](PUBLIC_BOUNDARY.md).
