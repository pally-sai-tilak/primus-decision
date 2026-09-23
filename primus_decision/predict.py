"""Inference API for one Primus Decision member directory (``model/member0`` or ``model/member1``).

    member = PrimusDecision.load("model/member0")
    answers = member.predict(state, questions)      # state: dict or JSON text; questions: {qid: {...}}

The released model is the two-member ensemble: ``Ensemble.load("model")`` in ``ensemble.py`` wraps two of these
and averages their probabilities. ``answers[qid]`` follows the reply shape of the benchmark harness:
    noul   → {"noul": p_true, "probabilities": {"false": .., "true": ..}, "confidence": ..}
    choice → {"choice": key, "probabilities": {key: p}, "confidence": ..}
    score  → {"score": expected_level, "probabilities": {"0": p0, ...}, "confidence": ..}
``confidence`` is the largest probability. The temperature profile in ``calibration.json`` (fitted on the
validation split only) is applied when ``calibrated=True``; the default output is the raw distribution. The
request and reply shapes are specified in ``INTERFACE.md`` and ``schemas/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .calibration import TemperatureCalibrator
from .data import Case, Decision, parse_options, state_to_text
from .features import LsaFeaturizer
from .metrics import Prediction
from .nn import Batcher, NetConfig, TypedDecisionNet, Vocab


def request_to_case(state, questions: dict, workflow: str | None, derived: bool = True) -> Case:
    """Build a ``Case`` from a request. The gold fields get a uniform placeholder; nothing reads them at inference."""
    raw = state if isinstance(state, str) else json.dumps(state)
    case = Case(case_id="live", workflow=workflow or "", state_text=state_to_text(raw, derived=derived), raw_state=raw)
    for qid, spec in questions.items():
        qtype = spec["type"]
        keys, texts = parse_options(qtype, spec.get("criteria"))
        case.decisions.append(Decision(case_id="live", workflow=workflow or "", qid=qid, qtype=qtype,
                                       instructions=str(spec.get("instructions", "")), option_keys=keys, option_texts=texts,
                                       gold_probs=[1.0 / len(keys)] * len(keys), gold_label=keys[0], gold_score=None))
    return case


def answers_from_predictions(decisions, preds) -> dict:
    out = {}
    for d, p in zip(decisions, preds):
        probs = {k: float(v) for k, v in zip(d.option_keys, p.probs)}
        ans = {"type": d.qtype, "probabilities": probs, "confidence": float(max(p.probs))}
        if d.qtype == "noul":
            ans["noul"] = probs["true"]
        elif d.qtype == "choice":
            ans["choice"] = d.option_keys[int(np.argmax(p.probs))]
        else:
            ans["score"] = p.expected_score
        out[d.qid] = ans
    return out


class PrimusDecision:
    def __init__(self, model: TypedDecisionNet, vocab: Vocab, cfg: NetConfig, calibrator: TemperatureCalibrator | None, featurizer=None):
        self.model, self.vocab, self.cfg, self.calibrator = model, vocab, cfg, calibrator
        self.model.eval()
        self.batcher = Batcher(vocab, cfg, featurizer=featurizer)

    @classmethod
    def load(cls, directory: str | Path) -> "PrimusDecision":
        d = Path(directory)
        cfg = NetConfig.from_json((d / "config.json").read_text())
        vocab = Vocab.from_json((d / "vocab.json").read_text())
        model = TypedDecisionNet(cfg)
        model.load_state_dict(load_file(d / "model.safetensors"))
        cal = TemperatureCalibrator.from_json((d / "calibration.json").read_text()) if (d / "calibration.json").exists() else None
        featurizer = LsaFeaturizer.load(d / "lsa.pkl") if cfg.lsa_dims > 0 else None
        return cls(model, vocab, cfg, cal, featurizer)

    def _case(self, state, questions: dict, workflow: str | None) -> Case:
        return request_to_case(state, questions, workflow, derived=self.cfg.derived_relations)

    def predict_cases(self, cases: list[Case], calibrated: bool = False) -> tuple[list[Decision], list[Prediction]]:
        decisions, preds = [], []
        with torch.no_grad():
            for i in range(0, len(cases), 32):
                dec, batch = self.batcher.make(cases[i : i + 32])
                probs = self.model(batch).exp().cpu().numpy()
                for j, d in enumerate(dec):
                    p = probs[j, : d.n_options]
                    p = p / p.sum()
                    if calibrated and self.calibrator is not None:
                        p = self.calibrator.apply(d, p)
                    es = float((np.arange(d.n_options) * p).sum()) if d.qtype == "score" else None
                    decisions.append(d)
                    preds.append(Prediction(probs=p, expected_score=es))
        return decisions, preds

    def predict(self, state, questions: dict, workflow: str | None = None, calibrated: bool = False) -> dict:
        case = self._case(state, questions, workflow)
        dec, preds = self.predict_cases([case], calibrated=calibrated)
        return answers_from_predictions(dec, preds)
