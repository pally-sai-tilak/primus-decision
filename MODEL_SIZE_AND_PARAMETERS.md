# Primus Decision 0.1 — exact size, parameter count and runtime footprint

Every number below is measured on the frozen public artifacts in this repository: parameters by instantiating each member network from its `config.json` with the shipped runtime and loading its `model.safetensors`; featurizer statistics by unpickling `lsa.pkl`; sizes by walking the release tree; the footprint by a separate process that can import only the shipped runtime. Nothing was retrained or re-saved.

## 1. Neural parameters

| component | architecture | trainable | non-trainable | total | dtype | checkpoint |
|---|---|---|---|---|---|---|
| `model/member0` (S4D member) | token embedding 4819×128 → 2× diagonal state-space (S4D, n_state 16) d=192 + 256-d LSA case features → additive attention pooling → option scorer | 1,932,609 | 0 | 1,932,609 | torch.float32 | 7,735,780 B = 7.74 MB = 7.38 MiB |
| `model/member1` (GRU member) | token embedding 4819×128 → 2× bidirectional GRU d=192 + 256-d LSA case features → additive attention pooling → option scorer | 1,782,465 | 0 | 1,782,465 | torch.float32 | 7,133,084 B = 7.13 MB = 6.80 MiB |

Checkpoint tensors match the instantiated networks exactly: 61 tensors / 1,932,609 values for member0 and 37 tensors / 1,782,465 values for member1; registered buffers: 0 values.

| quantity | value |
|---|---|
| total unique neural parameters stored | **3,715,074** (1,932,609 S4D member + 1,782,465 GRU member; no weights are shared) |
| active neural parameters for one ensemble inference | **3,715,074** — the ensemble evaluates both members for every prediction and averages their probabilities (weights 0.5 / 0.5) |
| per-module split, S4D member | emb 616,832, encoder 447,744, query 295,680, key 148,224, pool_out 147,648, head 110,977, ctx_merge 73,920, lsa_proj 49,344, text_proj 24,768, option_emb 13,632, schema_emb 3,840 |
| per-module split, GRU member | emb 616,832, encoder 297,600, query 295,680, key 148,224, pool_out 147,648, head 110,977, ctx_merge 73,920, lsa_proj 49,344, text_proj 24,768, option_emb 13,632, schema_emb 3,840 |

`query`, `key` and `pool_out` are the additive attention-pooling projections that summarise the encoder states for each question; there is no token-to-token attention and no transformer block anywhere in either member. `emb` is the token embedding table, `encoder` the S4D or GRU stack, `head` the option scorer, `lsa_proj` the projection of the 256-d LSA case vector.

## 2. Learned statistical artifacts (not neural parameters)

Each member carries an LSA featurizer: a word 1–2-gram TF-IDF vectorizer, a character 3–5-gram TF-IDF vectorizer and a truncated SVD fitted on the training split, plus per-dimension mean and standard deviation. These are fitted tables, not trained weights, so they are reported here and excluded from the parameter count.

| member | word n-gram vocabulary | char n-gram vocabulary | SVD dims | component matrix | matrix dtype | matrix bytes | file |
|---|---|---|---|---|---|---|---|
| `member0` | 9,166 | 25,107 | 256 | 256×34,273 | float64 | 70,191,104 B = 70.19 MB = 66.94 MiB | 71,413,743 B = 71.41 MB = 68.11 MiB |
| `member1` | 9,166 | 25,107 | 256 | 256×34,273 | float64 | 70,191,104 B = 70.19 MB = 66.94 MiB | 71,413,743 B = 71.41 MB = 68.11 MiB |

The embedding table of each member has 4,819 rows: the words listed in `model/member*/vocab.json` plus the hash buckets for unknown tokens and padding.

## 3. Package size (this release tree, real bytes; the two `lsa.pkl` files are stored as Git LFS objects of exactly these sizes, not pointer files)

| part | bytes | decimal | binary |
|---|---|---|---|
| model checkpoints, configs, vocabularies, calibration | 14,930,146 | 14.93 MB | 14.24 MiB |
| LSA featurizers (2 × `lsa.pkl`) | 142,827,486 | 142.83 MB | 136.21 MiB |
| runtime code, example, requirements, publish script | 53,209 | 0.05 MB | 0.05 MiB |
| documentation, manifests, checksums, license | 158,471 | 0.16 MB | 0.15 MiB |
| **total raw package size (installed)** | **157,969,312** | **157.97 MB** | **150.65 MiB** |

Download: `primus-decision-0.1.tar.gz` is 114.5 MB (gzip; the zip is within 0.1 MB of it). Both archives are built from exactly this tree with fixed timestamps. The archives, their exact byte counts and SHA-256 (`DOWNLOAD_CHECKSUMS.txt`) are on the GitHub release page for tag `primus-decision-0.1`, not in the repository; an archive cannot carry its own checksum. A git clone transfers the same 142.8 MB of LFS objects plus the compressed git objects of the remaining 15.1 MB.

## 4. Runtime footprint (release test machine, CPU only)

Hardware: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 vCPU, 15.7 GiB RAM, no GPU; Linux x86_64, glibc 2.39; Python 3.11.15, torch 2.14.0+cpu (2 threads), numpy 2.4.6. Page cache dropped before the load measurement.

| measurement | value |
|---|---|
| RSS after importing the runtime (torch, numpy, scikit-learn) | 364 MB |
| cold-load time, `Ensemble.load` (both members, both featurizers) | 0.84 s |
| steady RSS after load | 539 MB (peak so far 538 MB) |
| first inference (one case, 5 decisions) | 1005 ms |
| peak RSS during inference | 1403 MB |
| warm inference, one case of 5 decisions, p50 | 161.9 ms |
| warm inference, one case, p95 (n = 60) | 315.7 ms |
| warm inference, batch of 32 cases, per case | 65.2 ms |

Memory is dominated by the two LSA featurizers, not the networks: each holds a 256×34,273 float64 SVD matrix (70 MB) and two TF-IDF vocabularies, and the TF-IDF → SVD transform of a case allocates dense intermediates, which is where the peak during inference comes from. The two networks together are 15 MB of float32 weights.

## 5. Comparison-ready summary

```
Primus Decision 0.1
Neural parameters: 3,715,074 (both members evaluated per prediction)
Download size: 114.5 MB (tar.gz; exact bytes and SHA-256 in DOWNLOAD_CHECKSUMS.txt)
Installed size: 158.0 MB (150.7 MiB)
CPU latency p50: 162 ms
CPU latency p95: 316 ms
Accuracy: 75.1%
Brier: 0.059
ECE: 0.127
```
