# Primus Decision 0.1: research FAQ

Primus Decision 0.1 is AAME's first public research alpha for typed probabilistic decisions. It runs locally on CPU
with non-transformer encoders and returns probabilities for questions over a supplied state.

## What can I use the public release to test?

Version 0.1 exposes 20 question schemas across invoice processing, customer service, security incidents and
agent-trace observability. You supply a state, supported question IDs, instructions and option descriptions; the
model returns yes/no, choice or ordinal probabilities. For `choice` questions, requests may select and reorder the
schema's trained option keys, which are listed under `schema_options` in the model configuration. The [interface](INTERFACE.md) documents the request contract,
and the [quickstart](README.md#quickstart) runs the released weights locally.

## Does the current result define how far Primus can go?

The results describe a particular released checkpoint, interface and evaluation. They are measurements of demonstrated
capabilities, not a measured ceiling for the Primus research programme. New schemas, representations and applications
are open directions for further work, with new claims supported by their own evaluations.

## What does the 75.1% benchmark score mean?

Primus matched the benchmark's labels on 1,502 of 2,000 decisions across 400 test cases in its sealed release evaluation.
Each case asks five related questions. The benchmark uses synthetic teacher-derived labels. Its teacher-agreement
reference measures that labelling process; it is not a mathematical upper bound on Primus or future application
accuracy. [Benchmark tables](BENCHMARKS.md) and [evaluation notes](LIMITATIONS.md) explain the reference and metrics.

Training used 1,005 cases for fitting and 195 for validation from the official 1,200-case training split. The original
[protocol](PROTOCOL.md) and [sealed result](SEALED_RESULT.json) remain the historical release records.

## What did the invoice study demonstrate?

Using the released weights without retraining, Primus answered 26 of 32 structured reconciliation questions correctly
(81.25%) in the broader synthetic invoice test, following 32 of 32 in the initial pilot. The broader study also tests
narrative inputs and duplicate IDs and includes TF-IDF, Qwen3-1.7B and rule comparisons. Its exact settings and every
reported condition are available in the [case study](INVOICE_CASE_STUDY.md) and [evidence guide](INVOICE_EVIDENCE.md).
These task-specific results are separate from the original 75.1% benchmark.

## Does Primus call an LLM?

The released Primus inference path runs locally without an LLM call or pretrained language encoder. Its encoders were
trained from scratch; the original public training dataset uses teacher-derived labels. Qwen3-1.7B is a standalone
competitor in the invoice study, and its outputs do not enter Primus inference.

## What does 3.7 million parameters include?

The two neural encoders and their scoring layers have 3,715,074 neural parameters in total. Fitted TF-IDF/LSA features
are separate learned state, occupying approximately 143 MB of the original 158 MB installed release. The
[footprint report](MODEL_SIZE_AND_PARAMETERS.md) gives neural counts, artifact sizes and measured runtime memory
separately. Parameter ratios should not be read as package-size or memory ratios.

## Can researchers adapt it and publish better results?

The released artifacts and inference runtime are available under Apache-2.0, subject to the license and notice.
Researchers can investigate new input representations, schemas, training approaches and integration designs. Report
the model version, data, configuration and evaluation protocol with new results so improvements can be examined and
reproduced. The original frozen model and its recorded result remain identifiable alongside later work.

## What is the longer-term Primus plan?

AAME's research direction connects persistent memory, recurrent graph reasoning, temporal state and history, salience
and priority mechanisms, imagination and simulation, consolidation and learning cycles, language and grounding,
planning and decision layers, and agent and tool interfaces. Decision 0.1 is the first public component. Each later
release will identify the capabilities it implements and the evidence supporting them.

## What is public and what is protected?

Public releases provide architecture descriptions, interfaces, benchmarks and reproduction evidence for their stated
capabilities. Unreleased components, training methods, integration details, internal datasets and implementation
specifics remain private until AAME chooses to publish them. Public Decision 0.1 inference is reproducible from the
shipped model; its original training pipeline remains private. [Public boundary](PUBLIC_BOUNDARY.md) describes the
programme and publication boundary.

## Where are the official project sources?

[AAME research](https://theaame.com/primus) ·
[GitHub repository](https://github.com/pally-sai-tilak/primus-decision) ·
[Hugging Face model](https://huggingface.co/The-Aame/primus-decision-0.1)
