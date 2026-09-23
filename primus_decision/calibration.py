"""Post-hoc temperature calibration, fitted on validation data only and shipped as part of the model.

Temperature scaling per (question type, option count): p_cal = softmax(log p / T).
Fitted by minimising negative log-likelihood of the gold *discrete label*, which is what the
argmax-based ECE measures. Raw (uncalibrated) and calibrated metrics are always reported side by
side; fitting never touches the test split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


def _group_key(qtype: str, n_options: int) -> str:
    return f"{qtype}/{n_options}"


@dataclass
class TemperatureCalibrator:
    """One temperature per (question type, option count); fitted by minimising the NLL of the gold discrete label."""

    temperatures: dict[str, float] = field(default_factory=dict)
    fallback: float = 1.0
    objective: str = "nll_vs_hard_label"

    def apply(self, decision, probs: np.ndarray) -> np.ndarray:
        t = self.temperatures.get(_group_key(decision.qtype, decision.n_options), self.fallback)
        z = np.log(np.clip(np.asarray(probs, float), 1e-12, None)) / t
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    @classmethod
    def from_json(cls, text: str) -> "TemperatureCalibrator":
        obj = json.loads(text)
        objective = obj.get("objective", "nll_vs_hard_label")
        if objective != "nll_vs_hard_label":
            raise ValueError(f"unsupported calibration objective {objective!r}")
        return cls(temperatures={k: float(v) for k, v in obj["temperatures"].items()}, fallback=float(obj.get("fallback", 1.0)), objective=objective)
