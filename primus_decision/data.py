"""Data model for typed decisions (LocalLLaMA/typed-decisions schema).

A *case* is one shared ``state`` plus several typed questions. A *decision* is one question of
one case, with its gold distribution aligned to a fixed option order:

- ``noul``   options ``["false", "true"]`` (index 1 is "true")
- ``choice`` options in the order of the ``criteria`` dict keys (as the reference harness does)
- ``score``  levels ``"0" .. "n-1"`` in rubric order; expected score = sum(i * p_i)

At inference the runtime builds a ``Case`` from a live request (``predict.request_to_case``); the gold
fields then hold a uniform placeholder, so the same types serve evaluation code and serving.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from typing import Any

NOUL_OPTIONS = ["false", "true"]


@dataclass
class Decision:
    case_id: str
    workflow: str
    qid: str
    qtype: str  # noul | choice | score
    instructions: str
    option_keys: list[str]
    option_texts: list[str]
    gold_probs: list[float]  # aligned with option_keys, sums to 1
    gold_label: str  # discrete gold (key for choice, "true"/"false" for noul, level index for score)
    gold_score: float | None  # expected score for score questions

    @property
    def schema_id(self) -> str:
        return f"{self.workflow}/{self.qid}"

    @property
    def n_options(self) -> int:
        return len(self.option_keys)

    @property
    def gold_index(self) -> int:
        return self.option_keys.index(self.gold_label)


@dataclass
class Case:
    case_id: str
    workflow: str
    state_text: str
    decisions: list[Decision] = field(default_factory=list)
    raw_state: str = ""  # original JSON text


def flatten_state(state: Any, prefix: str = "") -> list[str]:
    """Recursively render a JSON state as ``path: value`` lines. Text states pass through."""
    lines: list[str] = []
    if isinstance(state, dict):
        for k, v in state.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                lines.extend(flatten_state(v, p))
            else:
                lines.append(f"{p}: {v}")
    elif isinstance(state, list):
        for i, v in enumerate(state):
            p = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                lines.extend(flatten_state(v, p))
            else:
                lines.append(f"{p}: {v}")
    else:
        lines.append(str(state))
    return lines


def _leaves(state: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(state, dict):
        for k, v in state.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_leaves(v, p))
    elif isinstance(state, list):
        for i, v in enumerate(state):
            out.extend(_leaves(v, f"{prefix}[{i}]"))
    else:
        out.append((prefix, state))
    return out


_INDEX_RE = re.compile(r"\[\d+\]")


def _words(path: str) -> str:
    return " ".join(_INDEX_RE.sub("", path).replace("_", " ").replace(".", " ").split())


def _base_key(path: str) -> str:
    return _INDEX_RE.sub("", path).rsplit(".", 1)[-1]


def derived_relations(obj: Any, max_lines: int = 40) -> list[str]:
    """Deterministic, domain-agnostic relation extraction over a JSON state. Emits plain-English lines
    so any tokenizer can use them:

    - numeric leaves sharing a base key (or one key ending in the other) are compared pairwise:
      "invoice total usd is greater than purchase order total usd by about 40 percent"
    - a string value that also occurs inside a list elsewhere: "invoice id appears in vendor
      history prior invoice ids"; a string with no such occurrence in a list of the same base key
      family: "... does not appear in ..."
    - numeric sign and null markers: "payment days until due is negative", "... is null"
    """
    leaves = _leaves(obj)
    lines: list[str] = []
    numeric = [(p, float(v)) for p, v in leaves if isinstance(v, (int, float)) and not isinstance(v, bool)]
    strings = [(p, v) for p, v in leaves if isinstance(v, str)]
    nulls = [p for p, v in leaves if v is None]
    for p in nulls:
        lines.append(f"{_words(p)} is null")
    for p, v in numeric:
        if v < 0:
            lines.append(f"{_words(p)} is negative")
        elif v == 0:
            lines.append(f"{_words(p)} is zero")
    # pairwise numeric comparisons within a base-key family, across different parents
    for i in range(len(numeric)):
        for j in range(i + 1, len(numeric)):
            (pa, va), (pb, vb) = numeric[i], numeric[j]
            ka, kb = _base_key(pa), _base_key(pb)
            if not (ka == kb or ka.endswith(kb) or kb.endswith(ka)):
                continue
            if _INDEX_RE.sub("", pa).rsplit(".", 1)[0] == _INDEX_RE.sub("", pb).rsplit(".", 1)[0]:
                continue  # same parent (e.g. two list items) — skip
            if va == vb:
                rel = "is equal to"
            else:
                rel = "is greater than" if va > vb else "is less than"
            denom = max(abs(va), abs(vb), 1e-9)
            pct = abs(va - vb) / denom * 100
            if va == vb:
                lines.append(f"{_words(pa)} {rel} {_words(pb)}")
            else:
                bucket = "under 1" if pct < 1 else "about 5" if pct < 7.5 else "about 10" if pct < 17.5 else "about 25" if pct < 37.5 else "about 50" if pct < 75 else "over 75"
                lines.append(f"{_words(pa)} {rel} {_words(pb)} by {bucket} percent")
            if len(lines) >= max_lines:
                return lines
    # string membership in lists elsewhere
    list_values: dict[str, set[str]] = {}
    for p, v in strings:
        if _INDEX_RE.search(p):
            list_values.setdefault(_INDEX_RE.sub("", p), set()).add(v)
    for p, v in strings:
        if _INDEX_RE.search(p) or len(v) > 40:
            continue
        for lp, vals in list_values.items():
            if lp == p:
                continue
            if _base_key(lp).rstrip("s").endswith(_base_key(p).rstrip("s")) or _base_key(p) in _base_key(lp):
                verb = "appears in" if v in vals else "does not appear in"
                lines.append(f"{_words(p)} {verb} {_words(lp)}")
                if len(lines) >= max_lines:
                    return lines
    return lines


DERIVED_RELATIONS = True  # module default; models record their own setting in NetConfig.derived_relations


def state_to_text(raw_state: str, derived: bool | None = None) -> str:
    """One ``path: value`` line per leaf of the JSON state, then the derived-relation lines when ``derived``."""
    if derived is None:
        derived = DERIVED_RELATIONS
    try:
        obj = json.loads(raw_state)
    except (json.JSONDecodeError, TypeError):
        return str(raw_state)
    if isinstance(obj, str):
        return obj
    lines = flatten_state(obj)
    if derived:
        rel = derived_relations(obj)
        if rel:
            lines.append("derived relations:")
            lines.extend(rel)
    return "\n".join(lines)


def parse_options(qtype: str, criteria: Any) -> tuple[list[str], list[str]]:
    """Returns (option_keys, option_texts) in the canonical order."""
    if qtype == "noul":
        texts = ["No.", "Yes."]
        if isinstance(criteria, dict):
            texts = [str(criteria.get("false", "No.")), str(criteria.get("true", "Yes."))]
        return list(NOUL_OPTIONS), texts
    if qtype == "choice":
        if not isinstance(criteria, dict):
            raise ValueError("choice question without a criteria dict")
        keys = list(criteria.keys())
        return keys, [str(criteria[k]) for k in keys]
    if qtype == "score":
        if not isinstance(criteria, list):
            raise ValueError("score question without a criteria list")
        return [str(i) for i in range(len(criteria))], [str(c) for c in criteria]
    raise ValueError(f"unknown question type {qtype!r}")


_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?|[^\sa-z\d]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
