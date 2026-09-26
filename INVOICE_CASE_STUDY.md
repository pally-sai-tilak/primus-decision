# Primus for invoice decisions

**AAME · 26 September 2026 · Three controlled synthetic experiments**

Primus Decision 0.1 performs structured invoice checks locally with a 3.7-million-parameter model and no LLM call at inference. In our broader invoice experiment, the unchanged public model answered **26 of 32 reconciliation questions correctly (81.25%)**. In the initial, simpler pilot it answered **32 of 32 correctly**, including after the question was reworded.

We built this study to make that capability inspectable: the records, questions, model answers, comparison systems and evaluation conditions accompany the result. Qwen3-1.7B runs as an independent competitor. Its outputs never enter Primus.

[Download the evidence](primus-invoice-evidence-2026-09-26.zip) · [Verify or reproduce](INVOICE_EVIDENCE.md) · [Public model card](MODEL_CARD.md)

## The decision task

Given an invoice, a purchase order and a delivery record, Primus returns a probability that the invoiced items, quantities and amounts reconcile. A second question checks whether the invoice ID already appears in the supplied history. The study uses synthetic records with exact arithmetic and membership labels, checked by a separate calculation from the records before inference.

Primus and the TF-IDF/logistic-regression baseline use their original task supervision. Neither was fitted on these evaluation invoices. Qwen uses its official non-thinking chat template with no demonstrations, additional training or external tools. These are comparisons of the specified deployment configurations, with their different training histories.

## Broader test: 32 new invoices

The broader experiment covers delivery shortages, price discrepancies, offsetting price discrepancies across two items with an unchanged grand total, and distracting historical amounts. Every invoice has a structured and a narrative presentation of the same facts.

All entries below are correct answers out of 32; each row contains 16 true and 16 false cases.

| Presentation | Question | Primus 0.1 | TF-IDF classifier | Qwen token scoring | Qwen free answer | Explicit rules |
|---|---|---:|---:|---:|---:|---:|
| Structured | Reconciliation | **26/32** | 27/32 | 16/32 | 16/32 | 32/32 |
| Structured | Duplicate ID | 16/32 | 16/32 | 18/32 | 18/32 | 32/32 |
| Narrative | Reconciliation | 17/32 | 24/32 | 18/32 | 18/32 | 32/32 |
| Narrative | Duplicate ID | 16/32 | 16/32 | 16/32 | 14/32 | 32/32 |

Structured reconciliation is Primus's strongest result in this experiment. It scores 8/8 on delivery-quantity cases and 6/8 in each of the other three families. The narrative result is 17/32, making input representation a concrete direction for the next iteration. The classifier reaches 27/32 structured and 24/32 narrative; explicit rules provide the exact reference solution for these arithmetic and membership checks.

The 32 invoices form 16 matching/mismatching pairs. Counting a pair as solved only when both answers are correct gives Primus **10/16 structured pairs** and **1/16 narrative pairs**. The classifier reaches 11/16 and 8/16; Qwen reaches 0/16 and 2/16 under each reported answer method. This check measures whether a model distinguishes the changed facts within a pair.

[Per-family scores, paired results and raw predictions](primus-invoice-evidence-2026-09-26.zip)

## Initial pilot: supported questions and rewording

The first experiment used 32 synthetic invoices varying duplicate ID, price discrepancy, delivery shortage, payment timing and vendor. Six questions per invoice produced 192 related answer instances across supported, reworded and new-question tracks.

| Track | Question | Primus 0.1 | Qwen3-1.7B | TF-IDF classifier | Explicit rules |
|---|---|---:|---:|---:|---:|
| Supported | Reconciliation | **32/32** | 8/32 | 32/32 | 32/32 |
| Reworded | Reconciliation | **32/32** | 8/32 | 32/32 | 32/32 |
| Supported | Duplicate ID | 18/32 | 19/32 | 16/32 | 32/32 |
| Reworded | Duplicate ID | 18/32 | 16/32 | 16/32 | 32/32 |

Reconciliation has 8 true and 24 false labels in this pilot. Its 32/32 result describes this specific template family; the balanced broader experiment above extends the evidence. Rewording retains the supported question IDs and answer meanings, so it tests stability within the released interface.

The pilot also records a separate development candidate on new payment-timing and delivery-shortage questions: 17/32 and 14/32 respectively. Primus 0.1 returns an unsupported-schema response for those two question IDs. All candidate scores and recorded predictions remain visible in the [pilot evidence](primus-invoice-evidence-2026-09-26.zip), identified separately from the public model. Its implementation and checkpoints remain private.

## Checking the competitor's answer format

We followed up on Qwen's pilot answers with six answer-format conditions on the same 64 supported-question instances. All six are reported:

| Qwen answer method | Duplicate ID | Reconciliation |
|---|---:|---:|
| Original lowercase-token replay | 19/32 | 8/32 |
| Lowercase and capitalized token aliases | 17/32 | 8/32 |
| Free greedy answer | 14/32 | 8/32 |
| Yes/no token aliases | 20/32 | 8/32 |
| A=false, B=true | 16/32 | 8/32 |
| A=true, B=false | 16/32 | 8/32 |

The original scoring replay reproduced every probability and label exactly. The audit is a retrospective diagnostic on already inspected cases. The subsequent broader experiment reports both capitalization-aware token scoring and free answers. These results describe Qwen3-1.7B in this non-thinking setup, rather than its strongest possible reasoning configuration or LLMs as a whole.

[Answer-format protocols and all 384 predictions](primus-invoice-evidence-2026-09-26.zip)

## Model and evaluation details

- **Primus:** the unchanged public Decision 0.1 ensemble, 3,715,074 neural parameters, S4D and bidirectional GRU encoders with fitted TF-IDF/LSA features; local CPU float32 inference, two threads. No transformer or pretrained language encoder is part of Primus. Its original training uses teacher-labelled data; the teacher is absent at inference.
- **Qwen:** the official Qwen/Qwen3-1.7B checkpoint at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, Apple GPU float16, official non-thinking template. Token scoring normalizes over the specified answer tokens. Free generation is greedy with a 12-token cap; initial true/false answers are parsed ignoring case and leading punctuation, and invalid answers count as incorrect.
- **Inputs:** both systems receive the same supplied facts and question text for each presentation. The broader experiment fits complete states within Primus's 640-token limit and uses the same six-line cap on generic derived relations for both presentations. The pilot used the original default cap of 40, so changes between experiments reflect both cases and preprocessing.
- **Evidence units:** the pilot has 32 base invoices, the answer-format audit reuses them, and the broader experiment has 32 new base invoices. Multiple questions, paired variants and two presentations are related observations. Scores are reported by task and condition rather than pooled into a single accuracy.
- **Timing:** recorded pilot timings use different devices and timing boundaries. They are descriptive measurements; this study's comparison is the task-level prediction result.

These are AAME's controlled, assistant-generated synthetic experiments. The procedures and cases were frozen locally before their respective predictions; the records document that sequence rather than external preregistration. The original rule-baseline implementation correction is disclosed in the pilot evidence and left all predictions unchanged.

## What this gives us

This study establishes a concrete starting capability: a compact, locally runnable Primus component can make supported structured invoice decisions, with results that can be checked per case. The next research step is to extend that performance across representations and richer records, then evaluate externally sourced invoices with human-checked labels.

The public model's original **75.1% Typed Decisions benchmark** remains a separate result on 400 cases and 2,000 decisions. The invoice experiments add task-specific evidence; they do not replace that benchmark or imply a completed general Primus system.

## Public evidence and protected work

We publish the generated evaluation cases, recorded probabilities and answers, every comparison condition, metrics, model revisions, checksums and public-model evaluation code. Readers can audit the tables directly from these files.

Unreleased candidate implementation, private training code, checkpoints, internal datasets and broader Primus integration details stay private. The [evidence guide](INVOICE_EVIDENCE.md) identifies the public export changes and the exact reproduction boundary. Public inference and evidence remain inspectable; the broader research roadmap is described in [Public vs Protected](PUBLIC_BOUNDARY.md).
