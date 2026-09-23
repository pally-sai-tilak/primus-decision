# Behavioral equivalence — internal frozen 0.1 vs this public package

**PASS** · fixture: 120 cases / 600 decisions (noul 180, choice 180, score 240) from the official train split, sha256 `c4e143fb88ed83834d804ebb89c19e8942c7f5fc3683ed11304a3ab466ed6113` (not redistributed: dataset rows).

| quantity | value |
|---|---|
| predicted-class mismatches | 0 (exact equality required) |
| max abs probability difference | 0.000e+00 (tolerance 1e-09) |
| max abs expected-score difference | 0.000e+00 (tolerance 1e-09) |
| decisions bit-identical | 600 of 600 |
| internal `ensemble.json` sha256 | `df9a602b968ce8ac32d519c6a41a3ffeb25dfb5e019708219d6c87cbaec74d7f` |
| public `ensemble.json` sha256 | `368f8eca1186cc8c68cc3bb320cf4fb22b26205b800f22fb8d2e1952b81e120f` |
| reason for the difference | the public descriptor omits the two informational fields of the frozen one (training run identifiers and their directories); `Ensemble.load` reads only kind, dir and weight |
| every other model file | byte-identical to the internal frozen release; every model file is listed with its hash in PROVENANCE.json and SHA256SUMS |

The public side ran with the exported runtime as its only importable package; the internal side ran the internal frozen package with its private inference code.
