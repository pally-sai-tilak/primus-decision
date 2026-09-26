# Invoice study evidence

[Download the complete evidence archive](primus-invoice-evidence-2026-09-26.zip). It contains the case study, three experiment folders,
original numerical records, public evaluation scripts and a checksum verifier. Extract it, then open `evidence/`.
The folder names in the table below refer to the extracted archive.

The three folders contain the complete recorded results, including all baselines and the separate development-candidate scores. Open `cases.json` for inputs, the prediction files for individual answers, and `metrics.json` for scores.

| Experiment | Records | Key files |
|---|---|---|
| [Invoice pilot](primus-invoice-evidence-2026-09-26.zip) | 32 invoices, six questions each | `cases.json`, `primus-predictions.json`, `llm-predictions.json`, `metrics.json`, `protocol.json` |
| [Answer-format audit](primus-invoice-evidence-2026-09-26.zip) | Same 64 supported instances, six answer formats | `predictions.json`, `metrics.json`, `replay-check.json`, `protocol.json` |
| [Broader invoice test](primus-invoice-evidence-2026-09-26.zip) | 32 new invoices, two presentations, two questions | `cases.json`, `primus-predictions.json`, `llm-predictions.json`, `metrics.json`, `matched-pairs.json` |

## Verify the published result

From the extracted `evidence/` directory, using Python 3:

```bash
python verify_evidence.py
```

The verifier checks the publication checksums, the preservation of original JSON records, all reported accuracy counts, pilot balanced accuracy and Brier scores, class balance, paired reconciliation counts and the answer-format replay. It requires only the Python standard library. Its output is an audit of the recorded predictions; running model inference is a separate step.

`SHA256SUMS` covers the published evidence files. Each experiment's `ORIGINAL_SHA256.json` preserves the earlier local manifest, including hashes of files withheld or replaced for this public export. `EXPORT.json` states what is retained and omitted. Every original JSON result and protocol is preserved byte for byte; numerical results have not been edited. Historical statements such as “Nothing published” describe the run's original status.

## Re-run the public configurations

Use a fresh working directory so the scripts' overwrite guards protect the recorded evidence. The runners retain the original workspace layout:

```text
work/public-release/primus-decision-0.1/  # public model, including real Git LFS files
work/primus-semantic-v1/train.parquet    # public Typed Decisions training file; historical path name
work/primus-llm-v1/                     # run_public.py, download_pinned.py, qwen/, qwen-source.json
work/primus-llm-robustness-v1/           # run.py
work/primus-invoice-transfer-v1/        # run.py
outputs/primus-llm-v1/                  # published cases.json and protocol.json
outputs/primus-llm-robustness-v1/        # empty before the diagnostic rerun
outputs/primus-invoice-transfer-v1/     # published cases.json and protocol.json
```

1. Obtain the public Primus release at tag `primus-decision-0.1` with Git LFS files, and verify its checksums. Its model hashes are also recorded here. Use the saved cases and protocols to reproduce the evaluated inputs exactly.
2. Obtain the public `LocalLLaMA/typed-decisions` training file from revision `ea9306458d6e9563628369a3d1e72e362fb381d2`. Its expected SHA-256 is `46a58d63edfd86e23229c78afe8b72307bb4ca9fb0e8df180cabb3c67ec9dcd5`. The historical folder name above is used only for this public data; the public runners need no private checkpoint.
3. Install the public model dependencies, plus `transformers==4.57.6`. Recorded runs used Python 3.13, PyTorch 2.14.0, NumPy 2.4.6 and Apple MPS for Qwen. Runtime JSON files record execution details; device and library changes may change floating-point outputs.
4. In `work/primus-llm-v1/`, run `python download_pinned.py`, then `python run_public.py primus` and `python run_public.py llm`. The download uses Qwen revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Compare weight hashes with `llm-runtime.json`. The public export runner omits only the private candidate loading/execution branch; it retains public Primus, TF-IDF, rule and Qwen prediction logic.
5. Run `python work/primus-llm-robustness-v1/run.py` and both `python work/primus-invoice-transfer-v1/run.py primus` and `python work/primus-invoice-transfer-v1/run.py llm`. Copy each experiment's runner from its evidence folder to the layout above first. The audit writes a new dated protocol; compare predictions, not timestamps. Its source-case and prediction hashes, and the Qwen weight checks, protect the comparison inputs.

The original private candidate's recorded scores can be recomputed from its published answers. Re-running that candidate's inference requires its withheld checkpoints and implementation. Its scores are separate from Primus 0.1's public-model claims. No private training or candidate implementation is included.

## Public export history

The original pilot runner included private candidate-loading code. That code is excluded from publication. The new `run_public.py` is explicitly a publication adaptation, with its own hash; the historical runtime record continues to identify the original runner hash. The original download helper resolved a moving revision; the publication helper pins the recorded revision. Inference scripts for the answer-format audit and broader test are preserved unchanged.

The current article presents the experiments in capability-first language. Original inputs, protocols, probabilities, scores and correction notes remain auditable beneath that presentation. No model files, original sealed benchmark results or private Primus training methods are changed by this publication.
