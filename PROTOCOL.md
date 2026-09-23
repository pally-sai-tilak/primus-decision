# Evaluation protocol — Primus Decision 0.1

Benchmark: `LocalLLaMA/typed-decisions`, revision `ea9306458d6e9563628369a3d1e72e362fb381d2` (Apache-2.0). The dataset
manifest and the two split files used are identified by hash in `PROVENANCE.json`.

## Splits and sealing

| split | cases | decisions | use |
|---|---|---|---|
| official `train` | 1,200 | 6,000 | everything that happened before the sealed run |
| ↳ training part (hash bucket ≥ 20) | 1,005 | 5,025 | model fitting |
| ↳ validation part (hash bucket < 20) | 195 | 975 | early stopping, model selection, ensemble weights, calibration, hyper-parameters |
| official `test` | 400 | 2,000 | read exactly once, after freezing |

The training / validation split is `int(sha256(case_id.encode()).hexdigest(), 16) % 100` (the whole hexadecimal digest
of the UTF-8 case id read as one integer): a bucket below 20 is validation, which selects 195 of the 1,200 cases. No
seed, no shuffling, the same on every machine. The upstream packaging guarantees that no case id or state occurs in both official splits;
the sealed runner checked that again when it loaded the test split.

Rules, written down before the test split was opened: never train on the test split; never use test outputs for any
architectural, hyper-parameter or calibration choice; run the official test only after the architecture, the
hyper-parameters and the calibration are frozen with content hashes; append every sealed invocation to a ledger and
refuse a second run of the same frozen candidate. `SEALED_RESULT.json` is the one run for this model.

## Metrics (the conventions of the published Laya evaluation)

Per decision, with prediction `p` and gold distribution `g` aligned to the canonical option order (`noul`:
[false, true]; `choice`: criteria order; `score`: levels 0..n−1):

| metric | definition | over |
|---|---|---|
| accuracy | argmax(p) == gold label (`noul`: p_true ≥ 0.5 ⇒ "true") | all 2,000 decisions |
| soft accuracy | Σ_k p_k g_k | noul + choice (1,200) |
| Brier | Σ_k (p_k − g_k)² | noul + choice |
| KL / TV | Σ_k g_k log(g_k / p_k) and ½ Σ_k \|p_k − g_k\| | noul + choice |
| ECE | 15 equal-width bins; confidence = max_k p_k, correctness = argmax correctness | all 2,000 |
| score MAE / within-1 | \|Σ_i i·p_i − gold expected score\|, and whether that is ≤ 1 | score (800) |
| NLL | −log p[gold label]; ours, in addition to the published set | all |

Implementation: `primus_decision/metrics.py`. The conventions follow the Luni `laya-jev-benchmark` harness
(`bench/eval.py`) and the Laya notebook that produced the published 0.766; sources and retrieval dates are in
`THIRD_PARTY.md`.

## Published comparison targets (quoted, not measured here)

The targets used by the release rule are the `laya-typed-decisions` model-card figures on the official test split:
accuracy 0.766, soft accuracy 0.471, Brier 0.062, ECE 0.213, score MAE 0.242; by workflow 0.804 / 0.766 / 0.764 / 0.730;
by primitive noul 0.857, choice 0.733, score 0.723. The "calibrated ECE ≈ 0.081" figure comes from the base `laya`
model card (post-temperature, family-level) and is quoted for context only.

Laya-typed-decisions is a specialist fine-tuned on the official train split. Primus Decision 0.1 is likewise a
specialist, so the comparison is like for like; the dataset card notes that specialists and generalists are not
comparable.

## Release rule

Written before the test split was read, hash-pinned in the freeze record and checked by the sealed runner before it
starts. Exactly one sealed run per frozen candidate.

A broad claim of superiority over Laya requires **raw** accuracy above 0.766 **and** an improvement on at least one
probability-quality metric: raw Brier below 0.062 or raw ECE below 0.213. Score MAE is reported but is an expected-value
error, not a probability-quality metric, so it never unlocks the broad claim. Calibrated numbers are reported separately
and never feed the rule (Laya's post-temperature ECE is a family-level figure from a different evaluation). A win on
one workflow may only be released as an explicitly named specialist. Otherwise the strongest candidate ships as a
research alpha with the precise gap stated.

That is what happened: the primary gate failed (0.751 against 0.766), the secondary gate passed (Brier 0.059, ECE
0.127), and this release states the gap instead of making a claim.
