# Interface — Primus Decision 0.1

Three JSON shapes cover everything this model reads and writes. Their JSON Schemas are in `schemas/`; this page is the
readable version with one example each. The Python API in `primus_decision/predict.py` and the JSON-lines bridge in
`primus_decision/serve.py` use exactly these shapes.

## Request (`schemas/request.schema.json`)

A request is one state and one or more typed questions about it. The model answers the 20 benchmark schemas, identified
by workflow and question id and listed under `schemas` in `model/member0/config.json`; any other pair is rejected with an
error rather than answered. Each schema also has trained option keys listed under `schema_options` in the model configuration. Inference looks up those keys; callers can select and reorder supported `choice` keys and supply their descriptions.

```json
{
  "workflow": "invoice_processing",
  "state": {
    "invoice": {"invoice_number": "INV-2041", "amount_usd": 1200.0, "po_amount_usd": 1000.0, "status": "received", "vendor": "Acme Supplies"},
    "purchase_order": {"po_number": "PO-7781", "amount_usd": 1000.0}
  },
  "questions": {
    "duplicate": {"type": "noul", "instructions": "Is this invoice a duplicate of one already paid?"},
    "disposition": {"type": "choice", "instructions": "What should happen to this invoice?",
                    "criteria": {"approve": "Approve and pay.", "hold": "Hold until the discrepancy is resolved.",
                                 "manual_review": "Send to a person for review.", "reject": "Reject the invoice."}},
    "discrepancy_severity": {"type": "score", "instructions": "How severe is the discrepancy between invoice and purchase order?",
                             "criteria": ["No discrepancy.", "Minor.", "Material.", "Severe."]}
  },
  "calibrated": false
}
```

| field | meaning |
|---|---|
| `workflow` | workflow name; `null` or absent means the empty workflow |
| `state` | the JSON object the questions are about; a string that parses as JSON is treated as that JSON, any other string is used as raw text |
| `questions` | question id → `{type, instructions, criteria}` |
| `type` | `noul` (is the statement true), `choice` (which option), `score` (which ordinal level) |
| `criteria` | `noul`: `null`, or `{"false": text, "true": text}`; `choice`: trained option key → option text, in scoring order; a subset or reordering of the supported keys is accepted; `score`: one text per supported level, level 0 first |
| `calibrated` | apply the temperature profile in `model/calibration.json`; off by default |

The state is flattened to one `path: value` line per leaf and extended with derived-relation sentences (numeric
comparisons within a key family, list membership, signs, nulls); `primus_decision/data.py` is the exact rule. Only the
first 640 tokens of that text reach the encoders, and at most 40 derived-relation lines are added. Question instructions
and option texts enter as bags of words, truncated to 64 and 48 tokens. Fields other than the four above are ignored by
the runtime; `schemas/request.schema.json` rejects them.

## Response (`schemas/response.schema.json`)

```json
{
  "answers": {
    "duplicate": {"type": "noul", "probabilities": {"false": 0.93, "true": 0.07}, "confidence": 0.93, "noul": 0.07},
    "disposition": {"type": "choice", "probabilities": {"approve": 0.09, "hold": 0.64, "manual_review": 0.22, "reject": 0.05}, "confidence": 0.64, "choice": "hold"},
    "discrepancy_severity": {"type": "score", "probabilities": {"0": 0.04, "1": 0.31, "2": 0.55, "3": 0.10}, "confidence": 0.55, "score": 1.71}
  }
}
```

Every answer carries the full distribution over the request's options, `confidence` (the largest probability) and one
type-specific summary: `noul` (probability of true), `choice` (the option key with the largest probability) or `score`
(the expected level, Σ level × probability). Probabilities sum to 1 within floating-point rounding. A request that
cannot be answered yields `{"error": "<reason>"}` instead of `answers`; the bridge never exits on a bad request.
The numbers above are illustrative; run `examples/predict_example.py` for the model's actual answers.

## Decision History record (`schemas/decision_history.schema.json`)

Decision History defines a record format for logging each request, its probabilities and a later observed outcome.
Integrators can use it to audit decisions, replay requests against a later model and measure calibration in their
application. The schema is storage-agnostic; a JSON-lines file, database or message queue can store the records.
Each record is self-contained. Logging is provided by the integrating application.

```json
{
  "schema_version": "primus-decision-history/1",
  "record_id": "018f6c2e-4b7a-7c1d-9e21-3f0d5a8b2c44",
  "recorded_at": "2026-09-22T10:15:30Z",
  "model": {"model_id": "primus-decision-0.1", "version": "0.1.0", "weights_sha256": "368f8eca1186cc8c68cc3bb320cf4fb22b26205b800f22fb8d2e1952b81e120f"},
  "request": {
    "workflow": "invoice_processing",
    "state_sha256": "41764d85fb31541fcbe1e1f2fcc390e563bc88d238813175a277711060a91477",
    "questions": {"duplicate": {"type": "noul", "instructions": "Is this invoice a duplicate of one already paid?"}}
  },
  "decisions": [
    {"qid": "duplicate", "type": "noul", "option_keys": ["false", "true"], "probabilities": [0.93, 0.07], "confidence": 0.93, "chosen": "false", "expected_score": null}
  ],
  "calibrated": false,
  "latency_ms": 41.2,
  "outcomes": [
    {"qid": "duplicate", "gold_label": "false", "source": "human", "noted_at": "2026-09-23T08:00:00Z"}
  ]
}
```

| field | meaning |
|---|---|
| `schema_version` | always `primus-decision-history/1` for this schema |
| `record_id` | unique within the history; the producer chooses the scheme |
| `recorded_at` | when the answer was produced, RFC 3339 |
| `model` | which frozen artifact answered: id, version and the sha256 of `model/ensemble.json` (or of `model.safetensors` for a single member) |
| `request` | the workflow and questions as asked; the state itself, or its canonical-JSON sha256 when storing states is not acceptable |
| `decisions` | one entry per question: the option keys, the aligned probabilities, the confidence, the chosen key, and the expected level for `score` |
| `calibrated` | whether the temperature profile was applied to the recorded probabilities |
| `outcomes` | what turned out to be right, per question, with its source (`dataset`, `human`, `downstream`, `other`) |
| `extensions` | producer-specific data readers ignore |

Canonical JSON text for `state_sha256`: keys sorted, separators `,` and `:`, non-ASCII characters escaped as `\uXXXX`
(the defaults of Python's `json.dumps`), no trailing newline, hashed as its UTF-8 bytes (Python:
`json.dumps(state, sort_keys=True, separators=(",", ":"))`). The example above hashes the state of the request example.

## Python API

```python
from primus_decision.ensemble import Ensemble
from primus_decision.predict import request_to_case, answers_from_predictions

model = Ensemble.load("model")
case = request_to_case(state, questions, workflow="invoice_processing")
decisions, predictions = model.predict_cases([case])            # raw probabilities
answers = answers_from_predictions(decisions, predictions)      # the response shape above
```

`Ensemble.predict_cases` takes a list of cases and returns the decisions in request order with their predictions;
`PrimusDecision.load("model/member0")` loads one member on its own with the same API plus `calibrated=True`.
