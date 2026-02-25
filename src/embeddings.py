"""Embedding methods: tfidf-custom, tfidf-scikit-learn, word2vec."""

from typing import List
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec


class TfidfCustom:
    """Custom TF-IDF: O(D*N) IDF, O(N) per-doc TF via Counter."""

    def __init__(self):
        self.idf_dict = None
        self.vocab = None

    def fit(self, docs: List[List[str]]) -> "TfidfCustom":
        n = len(docs)
        df = defaultdict(int)
        for doc in docs:
            for w in set(doc):
                df[w] += 1
        self.idf_dict = {}
        for w, c in df.items():
            self.idf_dict[w] = np.log(n / c)
        self.vocab = list(self.idf_dict.keys())
        return self

    def transform(self, docs: List[List[str]]) -> np.ndarray:
        idf_list = []
        for w in self.vocab:
            idf_list.append(self.idf_dict[w])
        idf = np.array(idf_list, dtype=np.float64)
        rows = []
        for doc in docs:
            d = len(doc) or 1
            cnt = Counter(doc)
            tf_list = []
            for w in self.vocab:
                tf_list.append(cnt.get(w, 0) / d)
            tf = np.array(tf_list, dtype=np.float64)
            rows.append(tf * idf)
        return np.array(rows, dtype=np.float64)

    def fit_transform(self, docs: List[List[str]]) -> np.ndarray:
        self.fit(docs)
        return self.transform(docs)


class TfidfSklearn:
    """TF-IDF via sklearn TfidfVectorizer."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def _join(self, docs: List[List[str]]) -> List[str]:
        out = []
        for d in docs:
            out.append(' '.join(d))
        return out

    def fit(self, docs: List[List[str]]) -> "TfidfSklearn":
        self.vectorizer.fit(self._join(docs))
        return self

    def transform(self, docs: List[List[str]]) -> np.ndarray:
        return self.vectorizer.transform(self._join(docs)).toarray().astype(np.float64)

    def fit_transform(self, docs: List[List[str]]) -> np.ndarray:
        return self.vectorizer.fit_transform(self._join(docs)).toarray().astype(np.float64)


class Word2VecEmbedding:
    """Word2Vec (gensim): document = L2-normalized mean of word vectors."""

    def __init__(self, vector_size=100, window=5, min_count=2, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, docs: List[List[str]]) -> "Word2VecEmbedding":
        self.model = Word2Vec(
            docs, vector_size=self.vector_size, window=self.window,
            min_count=self.min_count, workers=self.workers,
        )
        return self

    def transform(self, docs: List[List[str]]) -> np.ndarray:
        vecs = []
        for tokens in docs:
            wv = []
            for w in tokens:
                if w in self.model.wv:
                    wv.append(self.model.wv[w])
            if wv:
                vec = np.mean(wv, axis=0)
            else:
                vec = np.zeros(self.vector_size)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vecs.append(vec)
        return np.array(vecs, dtype=np.float64)

    def fit_transform(self, docs: List[List[str]]) -> np.ndarray:
        self.fit(docs)
        return self.transform(docs)


EMBEDDINGS = {
    "tfidf-custom": TfidfCustom,
    "tfidf-scikit-learn": TfidfSklearn,
    "word2vec": Word2VecEmbedding,
}
