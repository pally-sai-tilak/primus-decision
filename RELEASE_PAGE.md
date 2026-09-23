# Primus Decision 0.1 — Research Alpha

The first trained model component of Primus: a non-transformer typed-decision research model.

```
Primus Decision 0.1
Accuracy: 0.751
Brier:    0.059
ECE:      0.127

Laya reference accuracy: 0.766
```

Sealed official test split of `LocalLLaMA/typed-decisions`, 400 cases and 2,000 decisions, evaluated once after
freezing, raw probabilities. The model does not outperform the Laya reference on accuracy; it has the lower Brier
and ECE under the same protocol (Laya figures quoted from its model card). No claim of superiority is made. Full tables: `BENCHMARKS.md`.

3,715,074 neural parameters (S4D member 1,932,609, GRU member 1,782,465, both run per prediction); 157.9 MB installed,
114.5 MB `.tar.gz`; about 140 ms per five-decision case on a CPU (115–162 ms across recorded runs). Details and method: `EXPERIMENTS.md`, `MODEL_SIZE_AND_PARAMETERS.md`.

## In this release

- `model/`: the frozen bytes, checksummed; the two 71 MB LSA featurizers are Git LFS objects.
- `primus_decision/`: minimal inference runtime; `examples/predict_example.py`; `python -m primus_decision.serve model`.
- `INTERFACE.md`, `schemas/`: the request, response and Decision History shapes.
- `MODEL_CARD.md`, `BENCHMARKS.md`, `LIMITATIONS.md`, `PROTOCOL.md`, `REPRODUCIBILITY.md`, `ARCHITECTURE.md`,
  `RESEARCH_PAGE.md`, `RELEASE_NOTES.md`, `PUBLIC_BOUNDARY.md`, `MODEL_SIZE_AND_PARAMETERS.md`,
  `BEHAVIORAL_EQUIVALENCE.md`, `THIRD_PARTY.md`, `LICENSE` (Apache-2.0), `NOTICE`.
- `manifest.toml`, `SHA256SUMS`, `PROVENANCE.json`, `SEALED_RESULT.json`.

## Verify

```bash
git lfs install && git lfs pull        # install git-lfs before cloning
sha256sum -c SHA256SUMS                # macOS: shasum -a 256 -c SHA256SUMS
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
python examples/predict_example.py
```

Python 3.10 or newer; the `--index-url` option selects the CPU build of torch. This release page carries
`primus-decision-0.1.tar.gz`, `primus-decision-0.1.zip` and `DOWNLOAD_CHECKSUMS.txt`; use one of those archives or a
Git LFS clone, not the archive GitHub generates for the tag, which holds LFS pointer files instead of the featurizers.

## Not in this release

It is **not the complete PRGA/Primus system**. Not in this release: a general decision engine, other workflows, a
replacement for the reference model, or any other Primus component. See `LIMITATIONS.md` and `PUBLIC_BOUNDARY.md`.
