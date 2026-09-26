# Architecture — Primus Decision 0.1

Primus Decision 0.1 is the first public typed-decision component of Primus: given a JSON state and supported typed
questions, it returns probability distributions. This page describes the released inference architecture. Applications
can use the distributions within their own decision and orchestration layers; the broader research direction is
introduced in the [Research FAQ](RESEARCH_FAQ.md).

## One decision, end to end

1. **State to text.** The JSON state is flattened to one `path: value` line per leaf. Deterministic derived-relation
   sentences are appended: pairwise comparisons of numeric leaves that share a key family ("invoice amount usd is
   greater than purchase order amount usd by about 25 percent"), membership of a string in a list elsewhere in the
   state, signs and nulls. `primus_decision/data.py` is the exact rule. The first 640 tokens reach the encoders. (`member0/config.json` sets
   `derived_relations: true` explicitly; `member1/config.json` omits the key and the runtime default, `true`, applies:
   both members see the same text.)
2. **Two views of the text.** Each member tokenises the text with a fixed lower-case regular expression and maps the
   tokens to a 4,819-row embedding table: 2,770 vocabulary words, 2,048 hash buckets for unknown tokens and padding.
   The same text also goes through the member's LSA featurizer: word 1–2-gram and character 3–5-gram TF-IDF, a
   256-dimensional truncated SVD, standardised per dimension.
3. **Encoding.** `member0` runs the token embeddings through two diagonal state-space (S4D) blocks in each direction,
   computed as FFT convolutions; `member1` through two bidirectional GRU layers. Both produce one 192-dimensional
   state per token. Neither has token-to-token attention.
4. **Pooling by question.** A query is formed from a learned embedding of the question schema (workflow and question
   id) and the bag of words of the instructions. Four heads of additive attention score every token state against
   that query and pool them into one context vector; the projected LSA vector is concatenated and merged in.
5. **Scoring the options.** Each option gets a vector from a learned per-option embedding plus the bag of words of its
   text. A small MLP scores the context together with each option, and a softmax over exactly the options offered
   gives the distribution. `noul` is a two-option question (false, true); `score` questions also report the expected
   level, Σ level × probability.
6. **Ensemble.** The two members' distributions are averaged with weights 0.5 and 0.5, chosen on the validation part.
   The raw average is the released output. `model/calibration.json` holds an optional temperature per question type
   and option count, fitted on the validation part and applied only on request.

Sizes: 1,932,609 parameters in the S4D member and 1,782,465 in the GRU member, 3,715,074 in total, all float32; 15 MB of
weights and 143 MB of LSA tables. `MODEL_SIZE_AND_PARAMETERS.md` has the per-module counts and the runtime footprint.

## Design commitments visible in this release

- No transformer and no pretrained encoder: recurrent and state-space encoders trained from scratch on the benchmark's
  training part. The additive pooling has query and key projections, but it attends over the encoder states of one
  question only, never between tokens.
- For a supported question schema, K supplied trained option keys yield a K-way softmax. A `choice` request can select and reorder those keys and supply their descriptions; the public configuration lists them under `schema_options`.
- Sealed evaluation: the release result comes from one sealed evaluation after the architecture, hyper-parameters
  and calibration were frozen with content hashes, under a rule written before the split was opened (`PROTOCOL.md`).
  Later verification and diagnostic measurements are identified separately in `EXPERIMENTS.md`.
- Raw probabilities by default; calibration is a separate, labelled, optional profile.
- Everything content-addressed: weights, featurizers, configurations, dataset files and the sealed result
  (`PROVENANCE.json`, `SHA256SUMS`).

## Broader research and public evidence

This release makes the decision architecture and inference artifacts inspectable. AAME's broader Primus research
connects this direction with additional components and integration work. [Public boundary](PUBLIC_BOUNDARY.md)
describes the research programme and identifies the work AAME protects until publication.
