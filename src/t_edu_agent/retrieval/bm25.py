"""Zero-dependency Okapi BM25 lexical index over contextualized chunks."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from t_edu_agent.retrieval.text_processing import tokenize


class BM25Index:
    """In-memory Okapi BM25 scorer tuned for Russian/English educational texts."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_count: int = 0
        self._avgdl: float = 0.0
        self._doc_lens: list[int] = []
        self._doc_tfs: list[Counter[str]] = []
        self._idf: dict[str, float] = {}

    def fit(self, corpus: list[str]) -> BM25Index:
        """Tokenize corpus documents and compute IDF and term-frequency statistics."""
        self._doc_count = len(corpus)
        self._doc_tfs = []
        self._doc_lens = []
        df: Counter[str] = Counter()

        for text in corpus:
            tokens = tokenize(text)
            tf = Counter(tokens)
            self._doc_tfs.append(tf)
            self._doc_lens.append(len(tokens))
            for term in tf:
                df[term] += 1

        self._avgdl = (sum(self._doc_lens) / self._doc_count) if self._doc_count else 0.0
        self._idf = {
            term: math.log(1.0 + (self._doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        return self

    def score_all(self, query: str) -> np.ndarray:
        """Compute BM25 scores for every document in the fitted corpus."""
        scores = np.zeros(self._doc_count, dtype=np.float32)
        if self._doc_count == 0 or self._avgdl == 0.0:
            return scores

        query_tokens = tokenize(query)
        if not query_tokens:
            return scores

        unique_q = list(dict.fromkeys(query_tokens))
        for term in unique_q:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for idx, tf_map in enumerate(self._doc_tfs):
                freq = tf_map.get(term, 0)
                if freq == 0:
                    continue
                dl = self._doc_lens[idx]
                denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl))
                scores[idx] += idf * ((freq * (self.k1 + 1.0)) / denom)

        return scores

    def search(self, query: str, top_n: int = 20) -> list[tuple[int, float]]:
        """Return top_n (index, score) pairs sorted by descending BM25 score."""
        scores = self.score_all(query)
        order = np.argsort(-scores)
        results: list[tuple[int, float]] = []
        for idx in order[:top_n]:
            val = float(scores[idx])
            if val > 0.0:
                results.append((int(idx), val))
        return results
