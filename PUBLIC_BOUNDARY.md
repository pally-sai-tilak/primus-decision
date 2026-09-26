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

## Protected research

The system around the model and the other Primus components in development: their architectures, data, objectives,
evaluation sets, checkpoints and deployment. The training, model-selection and hyper-parameter-search code, and every
training run other than the two members shipped here. Internal research records and evaluation sets beyond those explicitly published. Original benchmark dataset rows
are available from their public source under its own license; the invoice study adds its generated evaluation records.

## The Road to Primus

Primus Decision 0.1 is the first public research alpha and one trained component. AAME's longer-term goal is a broader
architecture connecting persistent memory, recurrent graph reasoning, temporal state and history, salience and
priority mechanisms, imagination and simulation, consolidation and learning cycles, language and grounding,
planning and decision layers, and agent and tool interfaces. These are intended research directions, not claims that
the released decision model already implements those capabilities or that AGI has been achieved.

## Public vs Protected

Public releases will include enough architecture, benchmarks, interfaces and reproducibility evidence to make their
stated claims auditable. The extent of reproduction is stated for each artifact: public Decision 0.1 inference can
be reproduced from the shipped model; the original training pipeline remains private.

Unreleased components, training methods, system integration details, internal datasets and implementation specifics
remain protected and private until AAME chooses to publish them. This boundary protects the broader roadmap while
allowing readers to examine each demonstrated public capability.

## Invoice study export

The [invoice case study](INVOICE_CASE_STUDY.md) adds generated evaluation cases, protocols, recorded
predictions, model identifiers, checksums and public-model evaluation scripts. Original numerical records are
preserved. Its evidence guide identifies publication adaptations and omitted private candidate code; no private
training code or checkpoints are included. The model files, runtime and sealed release results remain unchanged.
