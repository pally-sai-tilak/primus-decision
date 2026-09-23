# Limitations — Primus Decision 0.1 (Research Alpha)

- **Specialist, and a research alpha.** Fitted to four synthetic workflows whose gold labels are the mean of three
  samples from a 4B-class teacher. The scores measure agreement with that teacher, not ground truth.
- **It loses to the reference on accuracy**: 0.751 against 0.766, 30 decisions in 2,000. The sealed run's weak
  spots: the `score` primitive (0.696 vs 0.723; score MAE 0.275 vs 0.242), `noul` (0.837 vs 0.857),
  customer service (0.728 vs 0.764) and security incidents (0.736 vs 0.766).
- **Small validation split.** 195 cases were used for model selection and for fitting the calibration profile, so
  calibrated validation numbers are optimistic. The sealed test numbers in `BENCHMARKS.md` are the only ones to quote.
- **Only the 20 benchmark schemas.** A question is identified by workflow and question id (for example
  `invoice_processing/disposition`); anything else is rejected. The model does not read arbitrary questions; its view
  of the instructions and options is a bag of words.
- **State format matters.** It was trained on flattened JSON with derived-relation sentences. Free-text states of
  other shapes are out of distribution.
- **Memory is set by the featurizers, not the networks.** Two float64 LSA tables of 70 MB each make the package
  158 MB and the peak resident memory about 1.4 GB during inference. They are the frozen bytes and were not shrunk.
- **CPU only, PyTorch and scikit-learn required.** The featurizers are pickled scikit-learn objects and need the
  scikit-learn version pinned in `requirements.txt`. Verify the checksums before loading a copy of unknown origin:
  `pickle` executes what it loads.
- **The calibration profile is a trade-off**, better ECE and NLL for worse Brier and score MAE. It is off by default
  and its ECE is not the model's headline calibration.
- **One sealed run, one seed per member.** There is no confidence interval over seeds for the headline number.
