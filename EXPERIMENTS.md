# Experiments — Primus Decision 0.1 (Research Alpha)

Every measurement made on the frozen model after its one sealed run, with its evidence class and method. Three classes
are kept apart throughout this repository:

- **measured by us** — our model, our harness, our machine; the record of each run is described with its method below;
- **reproduced by us** — another author's released checkpoint run through the same protocol on the same machine;
- **reported externally** — figures quoted from a model or dataset card, with the retrieval date, never re-measured.

Nothing in this document changed the model. The sealed run of 2026-09-22 (`SEALED_RESULT.json`, `PROTOCOL.md`) is
the claim; the measurements below were taken afterwards on the hash-checked release bytes, the official test split was
not used for any architectural, hyper-parameter or calibration choice, and no second sealed run exists. The test split
is therefore open for diagnosis of this version and closed for claims. All runs use 2 CPU threads on the same machine
class (4-vCPU Intel Xeon at 2.10 GHz, 15.7 GiB, Linux x86_64, Python 3.11, PyTorch 2.14 CPU) unless stated.

## 1. The sealed result, verified independently (measured by us)

An implementation of the protocol's metrics written independently of the runtime's `metrics.py`, applied to a fresh
run of the release bytes (every model file hash-checked against `SHA256SUMS` first), reproduces every recorded cell of
`SEALED_RESULT.json` exactly: the largest difference over 188 cells (overall, per workflow, per question type, raw and
calibrated) is 1.1e-16. The fresh run's probabilities differ from the recorded 5-decimal diagnostic dump by at most
5.0e-6, which is the rounding of that dump.

Raw: accuracy 0.7510, soft accuracy 0.5635, Brier 0.0591, KL 0.1024, TV 0.1434, NLL 0.6368, ECE 0.1272, score MAE
0.2747, within-1 0.9838. With the optional temperature profile: Brier 0.1136, NLL 0.5680, ECE 0.0404, score MAE 0.3199.

| workflow | accuracy | soft accuracy | Brier | NLL | ECE | score MAE |
|---|---|---|---|---|---|---|
| agent_trace_observability | 0.732 | 0.472 | 0.056 | 0.701 | 0.160 | 0.210 |
| customer_service | 0.728 | 0.547 | 0.092 | 0.692 | 0.112 | 0.297 |
| invoice_processing | 0.808 | 0.720 | 0.050 | 0.474 | 0.097 | 0.280 |
| security_incidents | 0.736 | 0.515 | 0.038 | 0.680 | 0.145 | 0.311 |

| question type | n | accuracy | soft accuracy | Brier | NLL | ECE | score MAE | within-1 |
|---|---|---|---|---|---|---|---|---|
| choice | 600 | 0.738 | 0.458 | 0.079 | 0.713 | 0.139 | – | – |
| noul | 600 | 0.837 | 0.669 | 0.039 | 0.381 | 0.084 | – | – |
| score | 800 | 0.696 | – | – | 0.771 | 0.150 | 0.275 | 0.984 |

## 2. Uncertainty: case-level bootstrap intervals (measured by us)

2,000 resamples of the 400 test cases (whole cases, so that the five decisions of a case stay together), seed 0,
95 % percentile intervals:

| metric | point | 95 % interval |
|---|---|---|
| accuracy | 0.751 | 0.729 – 0.774 |
| NLL | 0.637 | 0.612 – 0.662 |
| Brier (noul + choice) | 0.059 | 0.052 – 0.066 |
| ECE | 0.127 | 0.108 – 0.146 |
| score MAE | 0.275 | 0.256 – 0.293 |

Read against Laya's reproduced figures of section 6: its accuracy 0.766 is inside the accuracy interval (the gap is
not statistically meaningful at this test size); its NLL 0.707 and ECE 0.213 are outside (Primus is better on both);
its Brier 0.066 sits at the edge (marginal); its score MAE 0.242 is outside (Laya is better). The intervals cover case
sampling only: with one seed per member, training variance is not measured.

## 3. Calibration and abstention (measured by us)

Reliability on the 2,000 decisions, confidence = largest probability, 15 equal-width bins:

| output | mean confidence | accuracy | ECE | maximum calibration error |
|---|---|---|---|---|
| raw (the default) | 0.625 | 0.751 | 0.127 | 0.328 |
| temperature profile applied | 0.780 | 0.751 | 0.040 | 0.371 |

The raw model is under-confident by 0.126 on average; its low raw ECE is caution as much as calibration. The
temperature profile (`model/calibration.json`, fitted on the validation part only) removes most of that gap and lowers
NLL to 0.568, at the price of Brier against the soft gold (0.059 → 0.114) and score MAE (0.275 → 0.320). Both are
reported; the raw output is the default.

Selective prediction with the raw confidence (decisions sorted by confidence, most confident first):

| coverage | decisions answered | accuracy on them | confidence threshold |
|---|---|---|---|
| 10 % | 200 | 1.000 | 0.917 |
| 20 % | 400 | 0.988 | 0.770 |
| 30 % | 600 | 0.955 | 0.689 |
| 40 % | 800 | 0.935 | 0.634 |
| 50 % | 1,000 | 0.909 | 0.587 |
| 60 % | 1,200 | 0.878 | 0.546 |
| 70 % | 1,400 | 0.841 | 0.510 |
| 80 % | 1,600 | 0.816 | 0.474 |
| 90 % | 1,800 | 0.783 | 0.438 |
| 100 % | 2,000 | 0.751 | – |

Area under the risk–coverage curve: 0.102 (raw), 0.101 (temperature profile). By fixed threshold on the raw
confidence: 0.5 answers 73 % of decisions at 0.837 accuracy, 0.6 answers 47 % at 0.915, 0.7 answers 28 % at 0.960,
0.8 answers 18 % at 0.992, 0.9 answers 11 % at 0.996. These figures describe this test split; a deployment would set
its own threshold on its own Decision History.

## 4. Latency, throughput and memory (measured by us; six runs)

Two methods, both with the public runtime of this repository on the release bytes, a fresh process, the page cache
dropped before loading, 2 threads and the model already loaded when the per-case times are taken:

- **script method** (`examples/measure_latency.py`): one warm-up call, then 100 distinct test cases one at a time
  (p50 / p95 over those 100), ten batches of 32 test cases, then the whole 400-case split in one call;
- **probe method** (`MODEL_SIZE_AND_PARAMETERS.md`): 60 timings over a 120-case fixture drawn from the train split
  (cases repeat), three batches of 32; run inside the release build each time the tree was rebuilt.

| run | method | date (UTC) | load | first inference | one case, p50 | one case, p95 | batch of 32, per case | whole split | RSS after load | peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| A | script, idle machine | 2026-09-23 08:37 | 0.28 s | 370 ms | **115 ms** | 134 ms | 36 ms | 11.7 s (34 cases/s) | 602 MB | 1,563 MB |
| B | script, packaged copy | 2026-09-23 12:29 | 1.53 s | 663 ms | **139 ms** | 182 ms | 60 ms | 20.1 s (20 cases/s) | 578 MB | 1,499 MB |
| C | probe | 2026-09-22 19:39 | 2.07 s | 836 ms | **130 ms** | 287 ms | 49 ms | – | 539 MB | 1,403 MB |
| D | probe | 2026-09-22 20:20 | 0.18 s | 246 ms | **128 ms** | 280 ms | 46 ms | – | 539 MB | 1,403 MB |
| E | probe | 2026-09-22 22:54 | 0.17 s | 314 ms | **137 ms** | 285 ms | 39 ms | – | 539 MB | 1,404 MB |
| F | probe, final tree | 2026-09-23 11:28 | 0.84 s | 1,005 ms | **162 ms** | 316 ms | 65 ms | – | 539 MB | 1,403 MB |

Across the six runs the one-case p50 is 115–162 ms (median 138 ms), the batch-of-32 time 36–65 ms per case, the
whole-split throughput 20–34 cases (100–171 decisions) per second, resident memory 539–602 MB after loading and
1,403–1,563 MB at peak. The spread comes from the shared cloud machine, not from the model (section 5: the answers are
deterministic); any single figure should be read as ±25 %. The sealed runner's own 134.04 ms per case
(`BENCHMARKS.md`) is a different quantity: loading plus the whole split in batches of 32, divided by 400.

Laya's released checkpoint on the same machine (section 6, 4 threads because that is what its runtime uses on CPU):
p50 3,291 ms and p95 4,656 ms per case, peak RSS 3,029 MB, load 2.6 s; 20–29x the Primus per-case figures depending on
the Primus run. Its author reports GPU figures; none were measured here.

## 5. Robustness and consistency (measured by us)

Synthetic, seed-deterministic rewrites of the test inputs; 2,000 decisions per row unless the transform applies to a
subset. Accuracy is measured on the same subset before and after; "same answer" is the share of decisions whose argmax
does not change; TV is the mean total-variation distance between the two distributions.

| transform | n | accuracy before → after | change (points) | same answer | mean TV |
|---|---|---|---|---|---|
| option order of every `choice` question shuffled | 600 | 0.738 → 0.738 | 0.0 | 100.0 % | 6e-9 |
| instructions paraphrased | 2,000 | 0.751 → 0.750 | −0.1 | 96.7 % | 0.026 |
| synonyms in state values | 1,965 | 0.750 → 0.749 | −0.1 | 99.6 % | 0.003 |
| dates in long format | 980 | 0.767 → 0.762 | −0.5 | 96.9 % | 0.018 |
| dates in US format | 980 | 0.767 → 0.763 | −0.4 | 97.0 % | 0.018 |
| three noise fields added | 2,000 | 0.751 → 0.744 | −0.8 | 96.6 % | 0.018 |
| five noise fields added | 2,000 | 0.751 → 0.741 | −1.1 | 95.8 % | 0.024 |
| nesting layout changed | 2,000 | 0.751 → 0.738 | −1.3 | 95.6 % | 0.027 |
| keys renamed to kebab-case | 2,000 | 0.751 → 0.738 | −1.4 | 96.1 % | 0.025 |
| relation statements rewritten | 1,820 | 0.747 → 0.731 | −1.6 | 94.8 % | 0.028 |
| keys replaced by synonyms | 2,000 | 0.751 → 0.711 | −4.1 | 90.0 % | 0.062 |
| keys renamed to camelCase | 2,000 | 0.751 → 0.698 | −5.4 | 89.8 % | 0.061 |

Option order is an exact invariance (the largest probability shift is 3.7e-8). Wording, dates, noise fields and layout
cost at most 1.3 points; kebab-case keys and rewritten relation statements 1.4 and 1.6; the spelling of the JSON keys is
the weak spot, because the lexical featurizer ties the model to the field names of the benchmark. Consistency: two
identical runs differ by 0.0 in every probability; batched and one-at-a-time inference differ by at most 1.1e-7 and
never change a decision.

## 6. Laya's released checkpoint reproduced on the same harness (reproduced by us)

`convaiinnovations/laya-typed-decisions` (Hugging Face Hub revision `dd079950`, `model.safetensors` 842,609,220 bytes,
hashed in our record; laya 0.3.7, transformers 5.17.0, PyTorch 2.14 CPU, 4 threads) run on the official test split of
the same dataset revision through its author's own `Agent.predict`, scored two ways: with a port of the author's
published evaluation and with our implementation of the protocol.

| metric | published model card | reproduced, author's harness ported | reproduced, our protocol |
|---|---|---|---|
| accuracy | 0.766 | 0.7660 | 0.7660 |
| soft accuracy | 0.471 | 0.5087 | 0.5087 |
| Brier | 0.062 | 0.0658 | 0.0658 |
| ECE | 0.213 | 0.2132 | 0.2132 |
| score MAE | 0.242 | 0.2424 | 0.2424 |
| NLL | – | – | 0.7069 |
| within-1 | – | 0.9950 | 0.9950 |

Accuracy, ECE, score MAE and every per-workflow accuracy (agent-trace 0.730, customer service 0.764, invoice 0.804,
security 0.766) and per-question-type accuracy (choice 0.733, noul 0.857, score 0.723) reproduce the model card
exactly. The card's soft accuracy 0.471 and Brier 0.062 do not reproduce under either scoring (0.509 and 0.066); the
card does not document the convention behind those two cells, so they are not comparable and the reproduced values are
the ones used in this repository. Deviations from the author's setup: CPU instead of GPU; the parquet files of the
pinned dataset revision instead of the datasets library; our protocol uses the gold distribution without an epsilon and
clips probabilities at 1e-12 in the KL term (the ported harness is reported alongside and agrees).

Like for like on this harness, Primus Decision 0.1 is 1.5 accuracy points behind (inside its interval), ahead on NLL
(0.637 vs 0.707), ECE (0.127 vs 0.213), Brier (0.059 vs 0.066, marginal), KL (0.102 vs 0.122), TV (0.143 vs 0.179) and
soft accuracy (0.563 vs 0.509), and behind on score MAE (0.275 vs 0.242, meaningful) and within-1 (0.984 vs 0.995).
Parameters: 3,715,074 against about 421 million according to Laya's model card (its weights file is 843 MB), about
1/113; installed size 158 MB against 843 MB, about 1/5, because of the two 71 MB LSA tables.

## 7. What this evidence supports, and what it does not

Supported on this benchmark: a compact open specialist with typed-decision accuracy within sampling noise of the
strongest open specialist we could reproduce, better probability quality (NLL, ECE; Brier marginally), 20–29x lower
per-case CPU latency on the same machine and 113x fewer neural parameters; usable confidence for abstention; exact
invariance to option order.

Not supported: any "best" wording (Laya is more accurate; the dataset card's meraGPT Decider 1 reports higher accuracy,
lower Brier and lower score MAE, reported externally and not reproduced here); any claim about TypeSafe Jev (reported
externally only, not reproduced under an identical protocol); any generalist claim (the model answers only the 20
benchmark schemas); any robustness claim beyond the benchmark's field names; any statement about training variance
(one seed per member).

## 8. Records

The scripts and raw JSON records of sections 1–3, 5 and 6 are kept with the research code and are not part of this
package. Section 4's script method is `examples/measure_latency.py`, and its two runs above can be repeated with it on
any machine; the probe method is described in `MODEL_SIZE_AND_PARAMETERS.md`. The dataset revision and the frozen
model hashes that every measurement checked first are in `PROVENANCE.json`.
