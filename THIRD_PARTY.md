# Third-party material, licenses and provenance — Primus Decision 0.1 (Research Alpha)

## Benchmark dataset (not redistributed here)

| item | value |
|---|---|
| dataset | `LocalLLaMA/typed-decisions` — https://huggingface.co/datasets/LocalLLaMA/typed-decisions |
| revision used | `ea9306458d6e9563628369a3d1e72e362fb381d2` (last modified 2026-09-18, downloaded 2026-09-22) |
| license | Apache-2.0 (dataset card) |
| files used | `all/train-00000-of-00001.parquet` sha256 `46a58d63edfd86e23229c78afe8b72307bb4ca9fb0e8df180cabb3c67ec9dcd5` (training / validation); `all/test-00000-of-00001.parquet` sha256 `4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c` (sealed test, read once) |
| gold | the dataset's own teacher-derived soft labels; nothing was relabelled |

No dataset rows are included in this repository.

## Published reference figures (quoted, not re-measured)

| reference | source | retrieved |
|---|---|---|
| Laya-typed-decisions: accuracy 0.766, soft accuracy 0.471, Brier 0.062, ECE 0.213, score MAE 0.242; per-workflow and per-primitive figures | model card "Benchmark" table, https://huggingface.co/convaiinnovations/laya-typed-decisions | 2026-09-22 |
| Laya calibrated ECE ≈ 0.081 (family-level, post-temperature; context only) | https://huggingface.co/convaiinnovations/laya | 2026-09-22 |
| TypeSafe Jev 1.13.0 (generalist), ModernBERT-base and MiniLM-L6 specialists, and the dataset card's teacher-agreement and factor-predictor reference measurements | dataset card baseline table, https://huggingface.co/datasets/LocalLLaMA/typed-decisions, revision `ea930645…` | 2026-09-22 |
| meraGPT Decider 1 (generalist, zero-shot, proprietary): accuracy 0.768, soft accuracy 0.608, Brier 0.052, KL 0.096, ECE 0.180, score MAE 0.219, within-1 0.984 | dataset card leaderboard on `main` (added after revision `ea930645…`) | 2026-09-22 |

## Metric conventions

Accuracy, soft accuracy, Brier, KL / TV, ECE (15 bins) and score MAE follow the conventions of the published Laya
evaluation (`laya/common.py::ece_score`, https://github.com/NandhaKishorM/laya, Apache-2.0) and of the Luni
`laya-jev-benchmark` `bench/eval.py` (https://huggingface.co/datasets/Luni/laya-jev-benchmark, code Apache-2.0). We
read those implementations to match the conventions; no code from them is in this repository. `PROTOCOL.md` restates the definitions.

## Runtime dependencies (`requirements.txt`; versions of the recorded training and sealed runs)

| package | version | license (installed package metadata) |
|---|---|---|
| torch | 2.14.0 (+cpu build) | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| scipy | 1.17.1 | BSD-3-Clause |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| safetensors | 0.8.0 | Apache-2.0 |

The two LSA featurizers (`model/member*/lsa.pkl`) are pickled scikit-learn objects (TF-IDF + truncated SVD) fitted
on the training split; loading them requires the scikit-learn version above.

## Redistribution rights (provenance audit)

| item in this repository | derived from | right to distribute |
|---|---|---|
| model weights (`model/member*/model.safetensors`), configs, calibration profiles | training on the official train split of `LocalLLaMA/typed-decisions` (Apache-2.0) | Apache-2.0 permits preparing and distributing derivative works; no dataset row, state text, question or gold label is included |
| vocabularies (`model/member*/vocab.json`) and LSA featurizers (`model/member*/lsa.pkl`) | token statistics (TF-IDF vocabulary, SVD components) of the training split text | derived statistics, not dataset rows; the same Apache-2.0 grant applies |
| benchmark figures for Laya, TypeSafe Jev and the dataset-card baselines | published model / dataset cards | quoted facts with source and retrieval date; nothing copied beyond the numbers |
| metric conventions | Laya `common.py`, Luni `bench/eval.py` | consulted for parity; no code from them is included; definitions restated in `PROTOCOL.md` |
| inference runtime (`primus_decision/`) | AAME's own code | Apache-2.0 (this release) |
| software dependencies | PyPI packages | not bundled; installed by the user under their own licenses (table above) |

The model release includes AAME's runtime and model artifacts, learned statistics from the cited public dataset,
and attributed benchmark facts. The invoice evidence additionally includes generated cases, model outputs and
evaluation records. Third-party model checkpoints used as comparators are obtained from their publishers and are
not bundled in the evidence archive.

## This repository

Model artifacts, featurizers, configuration, runtime and documentation: Apache License 2.0 (`LICENSE`, `NOTICE`),
applying to this release only. The inference runtime in `primus_decision/` is AAME's own code.
