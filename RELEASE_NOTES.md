# Release notes — Primus Decision 0.1, Research Alpha (2026-09-22)

The first trained model component of Primus: a non-transformer typed-decision research model.

- Two-member ensemble, a diagonal state-space (S4D) encoder and a bidirectional GRU, each with 256-d LSA case
  features. 3,715,074 neural parameters, both members evaluated on every prediction. CPU inference.
- The sealed benchmark records 75.1% accuracy, Brier 0.059 and raw ECE 0.127. Reproduced Laya records 76.6%, 0.066
  and 0.213, respectively; its published reference Brier is 0.062. Primus demonstrates the reported combination of
  task accuracy, probability estimates and local CPU execution. The original protocol and comparison rule remain
  available in `PROTOCOL.md`.
- Output is raw probabilities. `model/calibration.json` is an experimental temperature profile, applied only on request.
- Package: 157.9 MB installed for the original release, 114.5 MB as `.tar.gz`; the two 71 MB LSA featurizers are Git LFS objects. Exact sizes,
  parameter counts and the runtime footprint are in `MODEL_SIZE_AND_PARAMETERS.md`.
- The model bytes are byte-identical to the internal frozen release. `BEHAVIORAL_EQUIVALENCE.md` records the public
  package reproducing the internal package bit for bit on 600 decisions covering all three question types.
- License: Apache License 2.0 for the artifacts, runtime and documentation in this repository (`LICENSE`, `NOTICE`).
  Primus names and logos are trademarks and are not licensed; other Primus components are licensed separately.
- Interface: `INTERFACE.md` and `schemas/` specify the request, the response and the Decision History record.
- Contents: `model/`, `primus_decision/`, `examples/`, `schemas/`, the documents listed in `README.md`, `manifest.toml`,
  `SHA256SUMS`, `PROVENANCE.json`, `SEALED_RESULT.json`.
- The sealed result remains the release record. Later verification and diagnostic measurements are reported separately; no replacement sealed result or test-driven retuning is presented for this version.

## Documentation correction of 2026-09-23 (model bytes unchanged)

- The wording of the sealed run's 134.04 ms latency (load plus prediction of the 400 cases in batches of 32, divided by 400)
  is now the same in every document; the runtime footprint was re-measured on the final release bytes; the latency and
  memory figures give the spread over all recorded runs instead of one run.
- `EXPERIMENTS.md` added: every measurement made on the frozen model after the sealed run (a separate metric calculation checking
  the sealed result, bootstrap intervals, calibration and abstention, latency and memory, robustness, and Laya's released
  checkpoint reproduced on the same harness), each with its evidence class and method; `examples/measure_latency.py`
  reproduces the latency method; `CITATION.cff` and a verification workflow added; `README.md` rewritten around the same numbers.
- No model file, no runtime file and no answer changed: the behavioral-equivalence record is unchanged.

## Invoice case study and documentation update — 2026-09-26

- Published three synthetic invoice experiments with their recorded cases, predictions, protocols, baselines and
  checksums under `INVOICE_CASE_STUDY.md`.
- Added a public-model pilot runner and a standard-library evidence verifier. Private candidate implementation and
  checkpoints remain outside the publication; the evidence guide records the export boundary.
- Revised the repository introduction, model card and public summary pages to lead with demonstrated capabilities,
  retaining benchmark results, comparison figures and evaluation conditions. Clarified teacher-labelled training
  data, the broader research direction and the distinction between public and protected work.
- Updated documentation checksums. Frozen model files, inference runtime, original sealed result, release tag and
  original protocol remain unchanged. This is an evidence and documentation publication, not a new model version.

## Public explanation and research FAQ — 2026-09-26

- Added a cross-linked research FAQ and aligned current documentation around demonstrated capabilities, the released
  interface and the broader research programme.
- Clarified the teacher-label reference, neural parameter count, original package size and invoice preprocessing
  settings. Replaced statistical equivalence wording with descriptions supported by the recorded comparisons.
- Frozen model artifacts, numerical records and the original evaluation protocol remain unchanged. The invoice evidence records
  and evaluation scripts are preserved; the archive article and guide are synchronized with the current explanation.
