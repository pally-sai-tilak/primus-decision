# Primus Decision 0.1

**Decisions, not generations.**

An open non-transformer model for typed probabilistic decisions, with 3.7M neural parameters, fitted LSA features and local CPU inference.

[![verify](https://github.com/pally-sai-tilak/primus-decision/actions/workflows/verify.yml/badge.svg)](https://github.com/pally-sai-tilak/primus-decision/actions/workflows/verify.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![release](https://img.shields.io/github/v/tag/pally-sai-tilak/primus-decision?label=release)](https://github.com/pally-sai-tilak/primus-decision/releases)
[![python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](requirements.txt)

Primus Decision 0.1 is [AAME](https://theaame.com)'s first public research alpha: a non-transformer model that returns
probabilities for typed questions over a supplied state. Its released inference path runs locally on CPU without an
LLM call. It is the first published decision component in the broader Primus research programme.

[Research FAQ](RESEARCH_FAQ.md) · [Model on Hugging Face](https://huggingface.co/The-Aame/primus-decision-0.1) ·
[AAME research](https://theaame.com/primus)

Give it a JSON state (an invoice, a support thread, a security alert, an agent trace) and a typed question, and it
returns a probability distribution over the answers instead of generated text:

| question type | asks | returns |
|---|---|---|
| `noul` | is this statement true? | P(false), P(true) |
| `choice` | which of these options applies? | one probability per option |
| `score` | which level on an ordinal scale? | one probability per level, plus the expected level |

**Key result.** One sealed evaluation of the frozen model on the official test split of
`LocalLLaMA/typed-decisions` (400 cases, 2,000 decisions), raw probabilities:

```
Primus Decision 0.1
Accuracy: 0.751
Brier:    0.059
ECE:      0.127

Laya reference accuracy: 0.766
```

Primus combines **75.1% accuracy**, NLL **0.637** and raw ECE **0.127** with 3,715,074 neural parameters.
On the same harness, the reproduced [421M-parameter Laya checkpoint](https://huggingface.co/convaiinnovations/laya-typed-decisions) scores 76.6% accuracy, NLL 0.707 and ECE 0.213.
That is about 113× fewer neural parameters for Primus; its separate fitted LSA tables and runtime memory are listed
below. Laya records the lower ordinal score error (MAE 0.242 versus 0.275). These figures describe different aspects
of the two models on this benchmark; full results and measurement conditions follow.

## Invoice case study

Using its released weights without retraining, Primus answered **26/32 structured reconciliation questions correctly (81.25%)** in a broader
synthetic invoice experiment, following **32/32** in the initial pilot. Qwen3-1.7B was evaluated as a standalone
competitor, alongside a TF-IDF classifier and explicit rules. The broader classifier result is 27/32; Primus's
narrative result on the same invoice facts is 17/32. All tasks, formats and comparison results are published together.

**[Read the case study and download its evidence](INVOICE_CASE_STUDY.md)** — generated cases, raw predictions,
protocols, all baselines, public evaluation scripts and checksums. This task-specific research is separate from the
original 75.1% Typed Decisions benchmark.

## At a glance

| item | value |
|---|---|
| neural parameters | 3,715,074: 1,932,609 in the S4D member and 1,782,465 in the GRU member; both run on every prediction |
| other learned state | two LSA featurizers (word and character TF-IDF with a 256-d truncated SVD), 71.4 MB each |
| original release download / installed | 114.5 MB `.tar.gz` / 157.9 MB, of which 142.8 MB are the featurizers |
| CPU latency (one five-decision case, model loaded, 2 threads) | p50 about 140 ms; 115–162 ms across six runs with two methods on the same machine class ([Latency and memory](#latency-and-memory)) |
| throughput | 20–34 cases (100–171 decisions) per second over the whole test split with 2 threads |
| memory | 0.5–0.6 GB resident after loading; 1.4–1.6 GB peak during inference, almost all of it the featurizers |
| training data | the official train split only: 1,005 cases for fitting, 195 for validation; no pretrained encoder, no transformer, no external data |
| license | Apache-2.0 for everything here (`LICENSE`, `NOTICE`); Primus and AAME names and logos are trademarks and are not licensed |

## Quickstart

```bash
git clone https://github.com/pally-sai-tilak/primus-decision.git && cd primus-decision
git lfs pull                          # the two featurizers are Git LFS objects; install git-lfs before cloning
sha256sum -c SHA256SUMS               # macOS: shasum -a 256 -c SHA256SUMS
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
python examples/predict_example.py    # one invoice, three questions, raw probabilities
```

Python 3.11 or newer; CPU only. The `--index-url` option installs the CPU build of torch (about 200 MB); without it,
PyPI serves the CUDA build with several GB of dependencies that this model does not use. Take the model from a clone
with Git LFS or from `primus-decision-0.1.tar.gz` on the release page of tag `primus-decision-0.1`; the archive GitHub
generates for the tag ("Source code") holds LFS pointer files instead of the featurizers, and loading one of those
fails with a message that says so.

The example asks three real benchmark questions about one invoice whose amount exceeds its purchase order by 20 %:

```json
{"workflow": "invoice_processing",
 "state": {"invoice": {"invoice_number": "INV-2041", "amount_usd": 1200.0, "po_amount_usd": 1000.0, "status": "received", "vendor": "Acme Supplies"},
           "purchase_order": {"po_number": "PO-7781", "amount_usd": 1000.0}},
 "questions": {"duplicate": {"type": "noul", "instructions": "Is this invoice a duplicate of one already paid?"},
               "disposition": {"type": "choice", "instructions": "What should happen to this invoice?",
                               "criteria": {"approve": "Approve and pay.", "hold": "Hold until the discrepancy is resolved.",
                                            "manual_review": "Send to a person for review.", "reject": "Reject the invoice."}},
               "discrepancy_severity": {"type": "score", "instructions": "How severe is the discrepancy between invoice and purchase order?",
                                        "criteria": ["No discrepancy.", "Minor.", "Material.", "Severe."]}}}
```

The model's actual answer (probabilities rounded to three decimals; the script prints them in full):

```json
{"duplicate":            {"type": "noul",   "probabilities": {"false": 0.986, "true": 0.014}, "confidence": 0.986, "noul": 0.014},
 "disposition":          {"type": "choice", "probabilities": {"approve": 0.002, "hold": 0.206, "manual_review": 0.708, "reject": 0.084}, "confidence": 0.708, "choice": "manual_review"},
 "discrepancy_severity": {"type": "score",  "probabilities": {"0": 0.043, "1": 0.041, "2": 0.118, "3": 0.798}, "confidence": 0.798, "score": 2.672}}
```

From Python, the same three lines:

```python
from primus_decision.ensemble import Ensemble
from primus_decision.predict import request_to_case, answers_from_predictions

model = Ensemble.load("model")
case = request_to_case(state, questions, workflow="invoice_processing")
decisions, predictions = model.predict_cases([case])            # raw probabilities
answers = answers_from_predictions(decisions, predictions)      # the response shape of INTERFACE.md
```

`python -m primus_decision.serve model` starts a JSON-lines bridge: one request per line in, one reply per line out,
the same shapes, `{"calibrated": true}` to apply the optional temperature profile. `INTERFACE.md` specifies both.

## What it answers

The released interface supports 20 question schemas, identified by workflow and question ID and listed under
`schemas` in `model/member0/config.json`: five questions in each of agent-trace observability, customer service,
invoice processing and security incidents. For a supported `choice` question, the request can select and reorder
trained option keys and supply their descriptions; K supplied supported keys produce a K-way distribution. The keys
available for each schema are listed under `schema_options` in the model configuration. The runtime validates the
question schema and looks up its trained option keys before inference.

## Architecture

```
JSON state ──▶ flattened text (one path: value line per leaf) + derived-relation sentences
           ├─▶ S4D member: token embeddings ▶ 2 diagonal state-space blocks per direction ▶ question-conditioned attention pooling ─┐
           ├─▶ GRU member: token embeddings ▶ 2 bidirectional GRU layers ▶ question-conditioned attention pooling ──────────────────┤
           └─▶ LSA case vector (TF-IDF ▶ 256-d SVD), one per member, merged into each member's context ─────────────────────────────┤
                                                                                                                                     ▼
                                     option scorer per member ▶ softmax over the offered options ▶ average of the two members (0.5 / 0.5)
                                     ▶ raw probability distribution (optional temperature profile on request)
```

No transformer and no pretrained encoder anywhere: both encoders are trained from scratch on the benchmark's training
part, and the only attention is the pooling of one question's token states, never token-to-token. Per-module parameter
counts, the featurizer tables and the design commitments are in `ARCHITECTURE.md` and `MODEL_SIZE_AND_PARAMETERS.md`.

## Benchmarks

Official test split, 2,000 decisions, raw probabilities. One column per evidence class: what we measured on our own
model in the sealed run, what we reproduced by running Laya's released checkpoint through the same protocol on the same
machine, and what Laya's model card reports.

| metric | Primus Decision 0.1 (measured by us, sealed run) | Laya-typed-decisions (reproduced by us, same harness) | Laya-typed-decisions (published model card) |
|---|---|---|---|
| accuracy | **0.751** | 0.766 | 0.766 |
| soft accuracy | 0.563 | 0.509 | 0.471 |
| Brier | 0.059 | 0.066 | 0.062 |
| ECE (15 bins) | 0.127 | 0.213 | 0.213 |
| NLL | 0.637 | 0.707 | not published |
| score MAE | 0.275 | 0.242 | 0.242 |
| within-1 (score) | 0.984 | 0.995 | not published |

Laya's accuracy, ECE, score MAE and every per-workflow and per-primitive accuracy reproduce its model card exactly; its
published soft accuracy and Brier do not reproduce under either its author's harness or ours (0.509 and 0.066 instead of
0.471 and 0.062), so those two published cells are not comparable and the reproduced column is the one to read.

Case-level bootstrap intervals for Primus (95%, 2,000 resamples of the 400 cases): accuracy 0.729–0.774, NLL
0.612–0.662, Brier 0.052–0.066, ECE 0.108–0.146, score MAE 0.256–0.293. These describe case-sampling uncertainty
for the recorded Primus run; establishing model equivalence or a difference requires a suitable comparative analysis.

By workflow (accuracy, Primus / reproduced Laya): agent-trace observability **0.732 / 0.730**, invoice processing
**0.808 / 0.804**, customer service **0.728 / 0.764**, security incidents **0.736 / 0.766**. By question type:
`choice` **0.738 / 0.733**, `noul` **0.837 / 0.857**, `score` **0.696 / 0.723**. These identify where the released
component is most effective and where further work is useful. Full tables: `BENCHMARKS.md`; recorded result:
`SEALED_RESULT.json`.

The dataset card also reported TypeSafe Jev 1.13.0 at accuracy 0.727, Brier 0.148 and ECE 0.144, and meraGPT Decider 1
at accuracy 0.768, Brier 0.052 and score MAE 0.219 (2026-09-22 snapshot). Those published generalist results provide
context; their systems and supervision differ from this task-trained specialist.

## Calibration and abstention

The raw model is under-confident: mean confidence 0.625 against accuracy 0.751, ECE 0.127. The optional temperature
profile in `model/calibration.json` (fitted on the validation part, never on the test split) brings ECE to 0.040 and
NLL to 0.568 on the test split but worsens Brier (0.059 → 0.114) and score MAE (0.275 → 0.320), so it is off unless
asked for and its ECE is not the headline calibration.

Confidence is usable for abstention. Answering only the most confident half of the decisions gives 0.909 accuracy
(area under the risk–coverage curve 0.102); at a confidence threshold of 0.7 the model answers 28 % of decisions at
0.960 accuracy, at 0.6 it answers 47 % at 0.915, at 0.5 it answers 73 % at 0.837. Reliability diagrams, the interval
method and the full coverage table are in `EXPERIMENTS.md`.

## Latency and memory

All runs use the public runtime in this repository on the release bytes, a 4-vCPU Intel Xeon at 2.10 GHz, 2 threads and
the model already loaded; the case is one five-decision test case at a time. Two runs of the method in
`examples/measure_latency.py` (100 distinct test cases after a warm-up call, page cache dropped before loading):

| measurement | fastest run (idle machine) | run of the packaged script |
|---|---|---|
| load (both members, both featurizers) | 0.28 s | 1.53 s |
| first inference after loading | 370 ms | 663 ms |
| one five-decision case, warm, p50 / p95 | 115 ms / 134 ms | 139 ms / 182 ms |
| batch of 32 cases, per case (mean) | 36 ms | 60 ms |
| whole test split, 400 cases | 11.7 s (34 cases, 171 decisions per second) | 20.1 s (20 cases, 99 decisions per second) |
| resident memory after loading / peak | 602 MB / 1,563 MB | 578 MB / 1,499 MB |

Four cold-start probe runs with a second method on the same machine class (`MODEL_SIZE_AND_PARAMETERS.md`) measured p50
128–162 ms, 39–65 ms per case at batch 32 and peak 1,403 MB. Over all six runs the p50 is 115–162 ms, about 140 ms in the
middle, so the recorded spread is approximately ±25 %. Reproduced Laya used 4 CPU threads on the same machine and
recorded 3,291 ms per case, compared with Primus's observed 115–162 ms using 2 threads. That is a 20–29× difference
in these per-case latency measurements. Peak memory was 3,029 MB for Laya and 1,403–1,563 MB across the Primus runs.
Reproduce the method on your hardware with
`python examples/measure_latency.py --parquet <official test split parquet>`; every recorded run is listed in
`EXPERIMENTS.md`.

## Robustness

Measured on the test split with synthetic rewrites of the inputs (2,000 decisions each unless stated):

- **Option order:** exactly invariant. Shuffling the options of every `choice` question changes no decision and moves no
  probability by more than 4e-8.
- **Wording:** paraphrased instructions −0.1 point (96.7 % of decisions keep their answer); synonyms in the state −0.1;
  long or US date formats −0.4 and −0.5; three or five added noise fields −0.8 and −1.1; a different nesting layout −1.3;
  kebab-case keys −1.4; rewritten relation statements −1.6.
- **Input conventions:** key synonyms change accuracy by −4.1 points and camelCase keys by −5.4 points (10%
  of decisions change). The lexical features learn field-name conventions; adopting another naming scheme calls for
  adaptation and evaluation on that scheme.
- **Deterministic:** two identical runs agree bit for bit; batched and one-at-a-time inference differ by at most
  1.1e-7 and never change a decision.

## Supported scope and evaluation conditions

The released component supports four synthetic workflows and 20 typed question schemas, with flattened JSON states
and derived-relation features. Its original benchmark measures agreement with teacher-generated labels. The invoice
case study adds controlled synthetic records with arithmetic and membership labels.

Use the per-workflow and per-question results above to assess a proposed application. The evidence covers one frozen
ensemble, with one seed per member; reported intervals describe case sampling. Deployment work should validate the
application's records, field conventions and decision criteria, with human review for consequential decisions.

The installed release is about 158 MB and peaks at 1.4–1.6 GB during recorded inference runs, mainly from its LSA
features. Calibration choices, input conventions and operational details are collected in
[Scope and evaluation notes](LIMITATIONS.md).

## Reproducibility

- **Verify the bytes.** `sha256sum -c SHA256SUMS` checks every file; `PROVENANCE.json` holds the hash of every model
  file as frozen before the sealed run, the dataset revision and file hashes, the training configuration of each member,
  the validation metrics and the environment.
- **The sealed protocol.** Model, calibration and protocol were frozen before the release evaluation. The historical
  comparison rule and its outcome are recorded in [the evaluation protocol](PROTOCOL.md) and
  [benchmark record](BENCHMARKS.md#release-gate-computed-by-the-sealed-runner). Later verification and diagnostic
  measurements remain separately identified.
- **Expected numbers.** Raw: accuracy 0.751, soft accuracy 0.563, Brier 0.059, ECE 0.127, NLL 0.637, score MAE 0.275,
  within-1 0.984, KL 0.102. A separately implemented metric calculation on a fresh run of the hash-checked bytes
  reproduces every cell of the recorded result to 1e-16 (`EXPERIMENTS.md`).
- **Retraining.** The training code is not part of this release; `REPRODUCIBILITY.md` gives the full configuration of
  both members (seeds, widths, losses, optimiser, schedule, early stopping, calibration) against the runtime's model
  definition, and the ensemble weights and how they were chosen.
- **Equivalence.** `BEHAVIORAL_EQUIVALENCE.md` records this package producing bit-identical outputs to the internal
  frozen package on 600 decisions covering all three question types.
- **Measurements.** `examples/measure_latency.py` reproduces the latency and memory method above on any machine.

## Provenance

Frozen 2026-09-22T06:22:58Z; sealed evaluation 2026-09-22T06:32:28Z; dataset `LocalLLaMA/typed-decisions` at revision
`ea9306458d6e9563628369a3d1e72e362fb381d2`; sealed test file sha256
`4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c`. The public form of the ensemble descriptor
(`model/ensemble.json`) omits the training-run identifiers of the internal record and nothing else; its hash is named
in `manifest.toml` and `PROVENANCE.json`. The documentation of this release was corrected on 2026-09-23 with the model
bytes unchanged (`RELEASE_NOTES.md`).

## Interface, schemas and Decision History

`INTERFACE.md` and `schemas/` specify the three JSON shapes the model reads and writes: the request (one state, typed
questions), the response (one distribution per question with its confidence and type-specific summary) and the
**Decision History** schema, a record format for what was asked, what was answered and a later observed outcome.
It is designed for applications to audit decisions, replay them against a later model version and measure calibration;
the integrating application supplies the log storage.

## Repository contents

| file | purpose |
|---|---|
| `RESEARCH_FAQ.md` | release capabilities, benchmark interpretation, adaptation and broader research direction |
| `INVOICE_CASE_STUDY.md`, `INVOICE_EVIDENCE.md` | invoice case study, evidence download and reproduction guide |
| `model/` | the frozen bytes: `ensemble.json`, `member0/` (S4D), `member1/` (GRU), calibration profiles |
| `primus_decision/` | the inference runtime: text pipeline, featurizer, networks, ensemble, calibration, metrics, request handling, JSON-lines bridge |
| `examples/` | `predict_example.py` (one request, real answers) and `measure_latency.py` (the latency and memory method) |
| `INTERFACE.md`, `schemas/` | request, response and Decision History shapes as JSON Schemas |
| `MODEL_CARD.md` | scope, results, model, intended use, provenance |
| `BENCHMARKS.md`, `SEALED_RESULT.json` | the sealed run, per workflow and per question type, with the published references |
| `EXPERIMENTS.md` | recorded post-release benchmark analyses, with their evidence class and method |
| `MODEL_SIZE_AND_PARAMETERS.md` | exact parameter counts, sizes and the probe footprint |
| `LIMITATIONS.md` | supported scope, input conventions and evaluation conditions |
| `PROTOCOL.md` | evaluation protocol and the release rule, written before the test split was read |
| `REPRODUCIBILITY.md` | verification, environment, dataset, training configuration, sealed evaluation, inference |
| `ARCHITECTURE.md` | the model, end to end, and the design commitments |
| `BEHAVIORAL_EQUIVALENCE.md` | the 600-decision equivalence record between this package and the internal frozen one |
| `RESEARCH_PAGE.md`, `RELEASE_PAGE.md`, `ANNOUNCEMENT.md`, `RELEASE_NOTES.md` | the research summary, the release page text, the announcement and the notes |
| `THIRD_PARTY.md`, `LICENSE`, `NOTICE`, `CITATION.cff` | provenance and redistribution rights, license, trademarks, citation |
| `manifest.toml`, `SHA256SUMS`, `PROVENANCE.json`, `PUBLIC_BOUNDARY.md` | machine-readable manifest, checksums, frozen hashes and training provenance, what is and is not in this release |

## Citation

`CITATION.cff` is in the repository root. BibTeX:

```bibtex
@software{primus_decision_0_1,
  author  = {Pally, Sai Tilak and {AAME}},
  title   = {Primus Decision 0.1: an open 3.7M-parameter non-transformer model for typed probabilistic decisions},
  year    = {2026},
  version = {0.1},
  license = {Apache-2.0},
  url     = {https://github.com/pally-sai-tilak/primus-decision}
}
```

## License

Apache License 2.0 for the model artifacts, the runtime and the documentation in this repository (`LICENSE`,
`NOTICE`). The license covers this release only. "Primus", "AAME" and the associated logos are trademarks and are not
licensed. The dataset is Apache-2.0 (`LocalLLaMA/typed-decisions`); no dataset rows are included.

## The Road to Primus

Primus Decision 0.1 is our first public research alpha: one trained decision component in a longer research programme.
Our goal is a broader architecture connecting persistent memory, recurrent graph reasoning, temporal state and history,
salience and priority, imagination and simulation, consolidation and learning cycles, language and grounding, planning
and decision layers, and agent and tool interfaces. These are research directions; this release demonstrates the
typed-decision component described here.

### Public vs Protected

We publish architecture descriptions, interfaces, benchmarks and reproducibility evidence for released components so
their claims can be audited. Unreleased components, training methods, system integration details, internal datasets
and implementation specifics remain private until AAME chooses to publish them. Public evidence documents what each
release demonstrates while protecting the broader research IP. See [Public boundary](PUBLIC_BOUNDARY.md).
