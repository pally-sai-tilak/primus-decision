"""Dense per-case lexical features: TF-IDF (word 1-2 grams + char 3-5 grams) → truncated SVD (LSA).

Statistical, non-transformer. Each ensemble member feeds this vector to its decision head next to
the pooled encoder state. Fitted on the training split only; the fitted object is pickled beside
the weights (``lsa.pkl``) so inference reproduces it exactly.
"""

from __future__ import annotations

import pickle

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class LsaFeaturizer:
    def __init__(self, dims: int = 256, seed: int = 0):
        self.dims = dims
        self.seed = seed
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=40000,
                                        token_pattern=r"[a-z]+|\d+(?:\.\d+)?|[^\sa-z\d]", lowercase=True)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=60000)
        self.svd = TruncatedSVD(n_components=dims, random_state=seed)
        self.mean = None
        self.std = None
        self._cache: dict[str, np.ndarray] = {}

    def transform(self, cases) -> np.ndarray:
        out = np.zeros((len(cases), self.dims), dtype=np.float32)
        todo = [i for i, c in enumerate(cases) if c.case_id == "live" or c.case_id not in self._cache]
        if todo:
            texts = [cases[i].state_text for i in todo]
            x = hstack([self.word_vec.transform(texts), self.char_vec.transform(texts)]).tocsr()
            z = ((self.svd.transform(x) - self.mean) / self.std).astype(np.float32)
            for j, i in enumerate(todo):
                if cases[i].case_id != "live":
                    self._cache[cases[i].case_id] = z[j]
                out[i] = z[j]
        for i, c in enumerate(cases):
            if i not in todo:
                out[i] = self._cache[c.case_id]
        return out

    @classmethod
    def load(cls, path) -> "LsaFeaturizer":
        with open(path, "rb") as f:
            head = f.read(64)
            if head.startswith(b"version https://git-lfs.github.com/spec"):
                raise RuntimeError(f"{path} is a Git LFS pointer file, not the featurizer: run `git lfs pull` in the clone, or use the release archive")
            f.seek(0)
            obj = pickle.load(f)
        obj._cache = {}
        return obj
