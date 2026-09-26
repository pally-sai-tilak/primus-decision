# Reproducibility — Primus Decision 0.1 (Research Alpha)

How to verify this package, run it, and see how it was made. Exact sizes, parameter counts and the runtime footprint
are in `MODEL_SIZE_AND_PARAMETERS.md`; the record that this public package behaves identically to the internal frozen
one is `BEHAVIORAL_EQUIVALENCE.md`.

## 1. Verify the package

```bash
git lfs pull                 # fetches the two 71 MB LSA featurizers (pointer oid = sha256 of the bytes)
sha256sum -c SHA256SUMS      # every file in this folder
```

`PROVENANCE.json` records the hash of every model file as frozen before the sealed run, the dataset revision and file
hashes, the training configuration of each member, the validation metrics and the environment. `manifest.toml` is the
short machine-readable summary; `SEALED_RESULT.json` is the single sealed evaluation.

## 2. Environment of the recorded training and sealed runs (same container)

Python 3.11.15 · PyTorch 2.14.0+cpu (2 threads) · NumPy 2.4.6 · scikit-learn 1.9.1 · safetensors 0.8.0 · Linux x86_64,
4 CPUs (Intel Xeon @ 2.10 GHz), no GPU. Retraining on other hardware will not reproduce the weights bit for bit; compare
the validation metrics in `PROVENANCE.json` and the sealed numbers in section 6 instead.

## 3. Dataset

`LocalLLaMA/typed-decisions`, revision `ea9306458d6e9563628369a3d1e72e362fb381d2` (Apache-2.0); dataset manifest
sha256 `4752590be05d1f40be378eb21cd0b42b75877b2b5262eddf59892f93c2cc02d1`. Official `train` split (1,200 cases /
6,000 decisions; file sha256 `46a58d63edfd86e23229c78afe8b72307bb4ca9fb0e8df180cabb3c67ec9dcd5`) for fitting and
validation; official `test` split (400 cases / 2,000 decisions; file sha256
`4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c`) used for one sealed release evaluation. Later verification and diagnostic uses are recorded in `EXPERIMENTS.md`. Validation rule:
`int(sha256(case_id.encode()).hexdigest(), 16) % 100 < 20` (195 cases), the rest is the training part (1,005 cases). No seed, no shuffling.
Full protocol: `PROTOCOL.md`.

## 4. How the members were trained

The public runtime defines the networks (`primus_decision/nn.py`) and text pipeline (`primus_decision/data.py`);
the configuration below documents the recorded training setup for researchers implementing their own training run.
The original training pipeline remains private. Public inference can be reproduced from the shipped artifacts.

| setting | `member0` (S4D) | `member1` (GRU) |
|---|---|---|
| encoder | 2 diagonal state-space blocks per direction, 16 states per channel | 2 bidirectional GRU layers |
| seed | 2 | 1 |
| embedding width / model width | 128 / 192 | 128 / 192 |
| attention-pooling heads | 4 | 4 |
| dropout | 0.3 | 0.3 |
| state tokens kept | first 640 | first 640 |
| LSA case features | 256-d: word 1–2-gram and char 3–5-gram TF-IDF, truncated SVD, fitted on the training part | same |
| loss | soft cross-entropy against the gold distribution + 0.5 × Brier | same |
| optimiser | AdamW, lr 0.002, weight decay 1e-4, gradient norm clipped at 1.0; lr halved after 3 epochs without validation improvement | same |
| batches | 16 cases, every decision of each case | same |
| epochs | up to 40; early stopping with patience 8 on validation soft cross-entropy; the best epoch's weights are kept | same |
| calibration | temperature per (question type, option count), fitted on the validation part | same |

The two members are combined by a weighted probability average. The weights, 0.5 and 0.5, were chosen on the
validation part by lowest NLL on raw probabilities over a 0.1-step grid. Ensemble validation metrics (raw): accuracy
0.779, Brier 0.057, ECE 0.156, NLL 0.615; the full set, raw and calibrated, is in `PROVENANCE.json`. Validation numbers
guided selection and calibration, so they are optimistic; quote the sealed numbers.

## 5. Sealed evaluation

After the freeze record was written (content hashes of every model file, of the metric code, of the release rule and
of the reference figures), the official test split was read once and `SEALED_RESULT.json` was written. The runner
refuses to start if any frozen file changed since the freeze, appends every invocation to a ledger and refuses a second
run of the same candidate, which is why there is exactly one sealed result for this model. This package does not
include the dataset loader or the evaluation driver: re-evaluating it on the public test split needs the parquet files
of the revision above, the gold distribution of every decision in the canonical option order, and the metric
definitions in `primus_decision/metrics.py` (`PROTOCOL.md`); the numbers to expect are in section 6.

## 6. Expected numbers (raw probabilities, sealed test)

accuracy 0.751 · soft accuracy 0.563 · Brier 0.059 · ECE 0.127 · NLL 0.637 · score MAE 0.275 · within-1 0.984 ·
KL 0.102 · latency 134.04 ms per case (5 decisions): model loading plus prediction of the 400 cases in batches of 32, divided by 400, on the CPU above
(steady-state latency with the model loaded: `EXPERIMENTS.md`; `examples/measure_latency.py` reproduces that method).
With the experimental calibration profile applied: ECE 0.040, NLL 0.568, Brier 0.114, score MAE 0.320.
Laya reference (published, quoted): accuracy 0.766, Brier 0.062, ECE 0.213, score MAE 0.242.

## 7. Inference on the frozen bytes

The `primus_decision/` package in this repository is the runtime: `data.py` (state flattening and derived relations),
`features.py` (the LSA featurizer), `nn.py` (the S4D and GRU networks), `ensemble.py`, `calibration.py`, `metrics.py`,
`predict.py` and `serve.py`. It is the model definition and the inference path, nothing else. Request and reply shapes:
`INTERFACE.md` and `schemas/`.

```python
from primus_decision.ensemble import Ensemble
from primus_decision.predict import request_to_case, answers_from_predictions

model = Ensemble.load("model")      # ensemble.json + member0/ + member1/
# questions are identified by workflow + question id and must be one of the 20 benchmark schemas
# listed under "schemas" in model/member0/config.json (e.g. invoice_processing/duplicate | disposition | discrepancy_severity)
state = {"invoice": {"invoice_number": "INV-2041", "amount_usd": 1200.0, "po_amount_usd": 1000.0, "status": "received", "vendor": "Acme Supplies"},
         "purchase_order": {"po_number": "PO-7781", "amount_usd": 1000.0}}
questions = {
    "duplicate": {"type": "noul", "instructions": "Is this invoice a duplicate of one already paid?"},
    "disposition": {"type": "choice", "instructions": "What should happen to this invoice?",
                    "criteria": {"approve": "Approve and pay.", "hold": "Hold until the discrepancy is resolved.",
                                 "manual_review": "Send to a person for review.", "reject": "Reject the invoice."}},
    "discrepancy_severity": {"type": "score", "instructions": "How severe is the discrepancy between invoice and purchase order?",
                             "criteria": ["No discrepancy.", "Minor.", "Material.", "Severe."]},
}
case = request_to_case(state, questions, workflow="invoice_processing")
decisions, preds = model.predict_cases([case])          # raw probabilities
print(answers_from_predictions(decisions, preds))       # {"duplicate": {"noul": p_true, ...}, "disposition": {"choice": ..., ...}, "discrepancy_severity": {"score": expected level, ...}}
```

Serving bridge: `python -m primus_decision.serve model` reads one request per line on stdin and writes one reply per
line; `calibrated: true` in a request applies `model/calibration.json`.
