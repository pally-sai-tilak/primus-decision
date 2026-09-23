"""Probability-averaging ensemble of Primus Decision members. The weights were chosen on the
validation split only. Members are non-transformer by construction, so the ensemble is too. Stored
as a directory with ``ensemble.json`` and one sub-directory per member.
"""

from __future__ import annotations

import json

from pathlib import Path

import numpy as np

from .metrics import Prediction
from .predict import PrimusDecision


class Ensemble:
    def __init__(self, members: list[tuple[str, object, float]]):
        # (kind, model, weight); every member of the released ensemble has kind "nn"
        self.members = members

    def predict_cases(self, cases):
        acc = None
        decisions = None
        for _kind, model, w in self.members:
            dec, preds = model.predict_cases(cases, calibrated=False)
            if acc is None:
                decisions = dec
                acc = [w * np.asarray(p.probs, float) for p in preds]
            else:
                for i, p in enumerate(preds):
                    acc[i] = acc[i] + w * np.asarray(p.probs, float)
        total_w = sum(w for _, _, w in self.members)
        out = []
        for d, p in zip(decisions, acc):
            p = p / total_w
            p = p / p.sum()
            es = float((np.arange(len(p)) * p).sum()) if d.qtype == "score" else None
            out.append(Prediction(probs=p, expected_score=es))
        return decisions, out

    @classmethod
    def load(cls, directory: Path) -> "Ensemble":
        spec = json.loads((Path(directory) / "ensemble.json").read_text())
        members = []
        for m in spec["members"]:
            if m["kind"] != "nn":
                raise ValueError(f"unsupported member kind {m['kind']!r}")
            members.append(("nn", PrimusDecision.load(Path(directory) / m["dir"]), float(m["weight"])))
        return cls(members)
