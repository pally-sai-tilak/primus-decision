"""Run Primus Decision 0.1 on one invoice and print the three answers.

    python examples/predict_example.py [model_dir]        (from the repository root)

The three questions are real benchmark schemas (invoice_processing/duplicate, disposition, discrepancy_severity).
The model only answers the 20 schemas listed under "schemas" in model/member0/config.json; any other workflow or
question id raises a KeyError on purpose rather than returning a made-up distribution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from primus_decision.ensemble import Ensemble  # noqa: E402
from primus_decision.predict import answers_from_predictions, request_to_case  # noqa: E402

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "model")
STATE = {"invoice": {"invoice_number": "INV-2041", "amount_usd": 1200.0, "po_amount_usd": 1000.0, "status": "received", "vendor": "Acme Supplies"},
         "purchase_order": {"po_number": "PO-7781", "amount_usd": 1000.0}}
QUESTIONS = {
    "duplicate": {"type": "noul", "instructions": "Is this invoice a duplicate of one already paid?"},
    "disposition": {"type": "choice", "instructions": "What should happen to this invoice?",
                    "criteria": {"approve": "Approve and pay.", "hold": "Hold until the discrepancy is resolved.",
                                 "manual_review": "Send to a person for review.", "reject": "Reject the invoice."}},
    "discrepancy_severity": {"type": "score", "instructions": "How severe is the discrepancy between invoice and purchase order?",
                             "criteria": ["No discrepancy.", "Minor.", "Material.", "Severe."]},
}

if __name__ == "__main__":
    model = Ensemble.load(MODEL_DIR)
    case = request_to_case(STATE, QUESTIONS, workflow="invoice_processing")
    decisions, preds = model.predict_cases([case])  # raw probabilities
    print(json.dumps(answers_from_predictions(decisions, preds), indent=1))
