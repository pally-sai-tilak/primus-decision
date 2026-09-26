# Scope and evaluation notes — Primus Decision 0.1

Primus Decision 0.1 is a locally runnable specialist for 20 typed question schemas across invoice processing,
customer service, security incidents and agent-trace observability. These notes describe the conditions under which
its results apply. The filename is retained for existing links.

## Supported interface

Requests contain a JSON state and supported question IDs; replies contain yes/no, choice or ordinal distributions.
The runtime validates workflow and question IDs, returning an error for schemas outside the released set. State
features use flattened JSON and derived relations; instruction and option features use a bag-of-words representation.
Changes to field conventions or free-text layouts need application-specific evaluation.

## Measured performance

The sealed benchmark records accuracy 0.751 against reproduced Laya accuracy 0.766, a difference of 30 decisions in
2,000. Primus records lower reproduced Brier and ECE; Laya records lower ordinal MAE (0.242 versus 0.275).
Per-question and per-workflow results are in [BENCHMARKS.md](BENCHMARKS.md), and the
[invoice case study](INVOICE_CASE_STUDY.md) adds task-specific synthetic experiments.

The original benchmark's labels average three samples from a 4B-class teacher, so accuracy measures agreement with
that reference. The [dataset card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions#what-a-score-here-means)
reports 73.5% teacher self-agreement and cautions that scores around 75% approach saturation for its labels. This
context is separate from real-world correctness. The 195-case validation split supported model selection and calibration; quote the sealed test
metrics for release performance. One seed per member was evaluated, and case-bootstrap intervals describe sampling
uncertainty rather than variation across training runs.

## Calibration and operation

Raw probabilities are the default. The optional validation-fitted temperature profile improves ECE and NLL while
increasing Brier and ordinal MAE on the sealed test. Choose the output mode against the application's decision cost
and evaluate its confidence thresholds on appropriate held-out records.

CPU inference uses PyTorch and pinned scikit-learn dependencies. The two float64 LSA artifacts account for most of
the approximately 158 MB installed package and 1.4–1.6 GB measured peak memory. Load checksum-verified artifacts:
the serialized scikit-learn objects use pickle, which executes code when loading.

For consequential applications, use human review and validate records, output criteria and operating thresholds in
the intended setting. Primus Decision 0.1 is a research alpha; its evidence supports the tasks and conditions reported
here.
