# Primus Decision 0.1 — Research Alpha

The first trained model component of Primus: a non-transformer typed-decision research model.

```
Primus Decision 0.1
Accuracy: 0.751
Brier:    0.059
ECE:      0.127

Laya reference accuracy: 0.766
```

One sealed evaluation after freezing, on the official test split of `LocalLLaMA/typed-decisions`,
400 cases and 2,000 decisions, raw probabilities. Primus pairs 75.1% accuracy with Brier 0.059 and ECE 0.127; the published Laya reference
records 76.6% accuracy, Brier 0.062 and ECE 0.213. Full published and reproduced tables: `BENCHMARKS.md`.

3,715,074 neural parameters (S4D member 1,932,609, GRU member 1,782,465, both run per prediction); 157.9 MB installed for the original release,
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

Python 3.11 or newer; the `--index-url` option selects the CPU build of torch. This release page carries
`primus-decision-0.1.tar.gz`, `primus-decision-0.1.zip` and `DOWNLOAD_CHECKSUMS.txt`; use one of those archives or a
Git LFS clone, not the archive GitHub generates for the tag, which holds LFS pointer files instead of the featurizers.

## Further research and evidence

The [invoice case study](INVOICE_CASE_STUDY.md) evaluates the released model weights without retraining against a separate LLM,
a classifier and explicit rules, with per-case evidence. The model release remains version 0.1.

This is the first public decision component of the broader Primus programme. [Public boundary](PUBLIC_BOUNDARY.md)
describes the research direction and protected work; [Scope and evaluation notes](LIMITATIONS.md) document the
released component's supported conditions.
