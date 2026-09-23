# Public boundary

## In this release

| what | where |
|---|---|
| the frozen Primus Decision 0.1 bytes: ensemble descriptor, two members' weights, configurations, vocabularies, LSA featurizers, calibration profiles | `model/` |
| the inference runtime, a runnable example and the latency measurement script | `primus_decision/`, `examples/` |
| the typed request, response and Decision History interface | `INTERFACE.md`, `schemas/` |
| model card, benchmark tables and the sealed record, limitations, evaluation protocol, reproducibility, architecture, research page, release notes | the `.md` files and `SEALED_RESULT.json` |
| exact parameter counts, sizes and footprint; behavioral equivalence with the internal frozen package; every measurement made on the frozen model after the sealed run | `MODEL_SIZE_AND_PARAMETERS.md`, `BEHAVIORAL_EQUIVALENCE.md`, `EXPERIMENTS.md` |
| provenance: frozen hashes, dataset revision, training configuration, validation metrics, environment | `PROVENANCE.json`, `manifest.toml`, `SHA256SUMS` |
| license (Apache-2.0, this release only), trademark notice, third-party provenance and redistribution rights, citation | `LICENSE`, `NOTICE`, `THIRD_PARTY.md`, `CITATION.cff` |
| a verification workflow for the hosted repository (checksums, dependencies, one inference, the bridge) | `.github/workflows/verify.yml` |

## Deliberately not in this release

The system around the model and the other Primus components in development: their architectures, data, objectives,
evaluation sets, checkpoints and deployment. The training, model-selection and hyper-parameter-search code, and every
training run other than the two members shipped here. Internal research records and evaluation sets other than the
public benchmark. The dataset rows themselves, which are public under the dataset's own license.

## How the boundary is enforced

Before export, an automated gate scans every file of this tree, and the single commit that publishes it, for private
paths, internal identifiers, names of other components, credentials and unsupported claims; verifies every model file
against the frozen hashes; recomputes the generated records from the frozen ones; re-runs the behavioral-equivalence
suite; and checks that the tree contains only the entries listed above. The runtime is not an obfuscated or minified
copy of anything: it is the model definition and the inference path in plain Python.
