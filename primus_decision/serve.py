"""JSON-lines serving bridge for Primus Decision 0.1.

    python -m primus_decision.serve <model_dir>

On start it prints one line, ``{"ready": true, "model_dir": ..., "kind": "ensemble" | "member", "members": n}``.
Then, for every request line (the request object of ``INTERFACE.md``: ``{"workflow": str | null, "state": object | str,
"questions": {qid: {...}}, "calibrated": bool}``), it prints one reply line, ``{"answers": {qid: {...}}}`` or
``{"error": str}``. Answers are raw probabilities unless ``calibrated`` is true, which applies the temperature profile in
``<model_dir>/calibration.json``. A malformed request produces an error line, never a crash, so a supervisor can keep the
process alive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .calibration import TemperatureCalibrator
from .ensemble import Ensemble
from .predict import PrimusDecision, answers_from_predictions, request_to_case


def main() -> int:
    model_dir = Path(sys.argv[1])
    if (model_dir / "ensemble.json").exists():
        model = Ensemble.load(model_dir)
        kind, members = "ensemble", len(model.members)
        derived = all(m.cfg.derived_relations for _, m, _ in model.members)
        calibrator = TemperatureCalibrator.from_json((model_dir / "calibration.json").read_text()) if (model_dir / "calibration.json").exists() else None
    else:
        model = PrimusDecision.load(model_dir)
        kind, members = "member", 1
        derived, calibrator = model.cfg.derived_relations, model.calibrator
    sys.stdout.write(json.dumps({"ready": True, "model_dir": str(model_dir), "kind": kind, "members": members}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            case = request_to_case(req.get("state"), req.get("questions") or {}, req.get("workflow"), derived=derived)
            decisions, preds = model.predict_cases([case])
            if req.get("calibrated") and calibrator is not None:
                for d, p in zip(decisions, preds):
                    p.probs = calibrator.apply(d, np.asarray(p.probs, float))
                    if d.qtype == "score":
                        p.expected_score = float((np.arange(len(p.probs)) * p.probs).sum())
            sys.stdout.write(json.dumps({"answers": answers_from_predictions(decisions, preds)}) + "\n")
        except Exception as e:  # noqa: BLE001 — report on the wire; the caller decides what to do
            sys.stdout.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
