"""Benchmark metrics, matching the published typed-decisions harness conventions.

Reference: the Luni/laya-jev-benchmark harness (bench/eval.py), whose docstring states that its
metrics match the Laya notebook that produced the published 0.766.

Conventions (per decision):
- accuracy: argmax of the predicted distribution equals the gold discrete label. For noul the
  prediction is "true" iff p_true >= 0.5. Averaged over ALL decisions (noul + choice + score).
- soft accuracy, Brier, KL, TV: computed against the gold *distribution*, over noul and choice
  decisions only (the reference harness skips them for score questions).
    soft  = sum_k p_k g_k          Brier = sum_k (p_k - g_k)^2
    TV    = 0.5 sum_k |p_k - g_k|  KL    = sum_k g_k log(g_k / p_k)
- score MAE / within-1: |expected_score - gold_score| over score decisions only.
- ECE: confidence = max predicted probability, correctness = argmax correctness, over all
  decisions, 15 equal-width bins as in laya/common.py::ece_score (the reference harness imports it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-12


def ece_score(confidences: np.ndarray, corrects: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error, 15 equal-width bins on (0, 1] (laya/common.py::ece_score)."""
    confidences = np.asarray(confidences, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    if len(confidences) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += mask.sum() / n * abs(corrects[mask].mean() - confidences[mask].mean())
    return float(ece)


@dataclass
class Prediction:
    """One predicted decision: distribution aligned with Decision.option_keys."""
    probs: np.ndarray  # shape (K,), sums to 1
    expected_score: float | None = None  # for score questions


@dataclass
class MetricAccumulator:
    acc: list[float] = field(default_factory=list)
    conf: list[float] = field(default_factory=list)
    corr: list[float] = field(default_factory=list)
    soft: list[float] = field(default_factory=list)
    brier: list[float] = field(default_factory=list)
    kl: list[float] = field(default_factory=list)
    tv: list[float] = field(default_factory=list)
    nll: list[float] = field(default_factory=list)
    mae: list[float] = field(default_factory=list)
    within1: list[float] = field(default_factory=list)

    def add(self, decision, pred: Prediction) -> None:
        p = np.clip(np.asarray(pred.probs, dtype=float), EPS, None)
        p = p / p.sum()
        g = np.asarray(decision.gold_probs, dtype=float)
        g = g / g.sum()
        gold_idx = decision.gold_index
        if decision.qtype == "noul":
            pred_idx = 1 if p[1] >= 0.5 else 0
        else:
            pred_idx = int(np.argmax(p))
        is_corr = float(pred_idx == gold_idx)
        self.acc.append(is_corr)
        self.corr.append(is_corr)
        self.conf.append(float(p.max()))
        self.nll.append(float(-np.log(p[gold_idx])))
        if decision.qtype == "score":
            es = pred.expected_score
            if es is None:
                es = float((np.arange(len(p)) * p).sum())
            gs = decision.gold_score if decision.gold_score is not None else float((np.arange(len(g)) * g).sum())
            self.mae.append(abs(es - gs))
            self.within1.append(float(abs(es - gs) <= 1.0))
        else:
            self.soft.append(float((p * g).sum()))
            self.brier.append(float(((p - g) ** 2).sum()))
            self.tv.append(float(0.5 * np.abs(p - g).sum()))
            gc = np.clip(g, EPS, None)
            self.kl.append(float((g * np.log(gc / p)).sum()))

    def summary(self) -> dict:
        def m(x):
            return float(np.mean(x)) if x else None

        return {
            "n": len(self.acc),
            "accuracy": m(self.acc),
            "soft_accuracy": m(self.soft),
            "brier": m(self.brier),
            "kl": m(self.kl),
            "tv": m(self.tv),
            "nll": m(self.nll),
            "ece": ece_score(np.array(self.conf), np.array(self.corr)) if self.acc else None,
            "score_mae": m(self.mae),
            "within_1": m(self.within1),
        }


def evaluate(decisions, predictions, group_keys=("workflow", "qtype", "schema_id")) -> dict:
    """Overall + grouped metrics. ``predictions`` aligns with ``decisions``."""
    overall = MetricAccumulator()
    groups: dict[str, dict[str, MetricAccumulator]] = {k: {} for k in group_keys}
    for d, p in zip(decisions, predictions):
        overall.add(d, p)
        for k in group_keys:
            key = getattr(d, k)
            groups[k].setdefault(key, MetricAccumulator()).add(d, p)
    out = {"overall": overall.summary()}
    for k in group_keys:
        out[f"by_{k}"] = {name: acc.summary() for name, acc in sorted(groups[k].items())}
    return out
