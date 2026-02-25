"""KMeans clustering via NLTK KMeansClusterer."""

import random
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.util import cosine_distance, euclidean_distance


class KMeansModel:
    """KMeans using NLTK KMeansClusterer."""

    def __init__(
        self,
        n_clusters: int = 8,
        distance: str = "cosine",
        repeats: int = 1,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None

        dist_fn = cosine_distance if distance == "cosine" else euclidean_distance
        rng = random.Random(random_state) if random_state is not None else None
        self._model = KMeansClusterer(
            num_means=n_clusters, distance=dist_fn,
            repeats=repeats, normalise=True,
            avoid_empty_clusters=True, rng=rng,
        )

    def fit(self, X: np.ndarray) -> "KMeansModel":
        X = np.asarray(X, dtype=np.float64)
        if len(X) < self.n_clusters:
            raise ValueError("Need >= " + str(self.n_clusters) + " samples, got " + str(len(X)))
        vecs = []
        for row in X:
            vecs.append(np.asarray(row, dtype=np.float64))
        self._model.cluster_vectorspace(vecs, trace=False)
        labels_list = []
        for v in vecs:
            labels_list.append(self._model.classify_vectorspace(v))
        self.labels_ = np.array(labels_list)
        self.cluster_centers_ = np.array(self._model.means())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        vecs = []
        for row in X:
            vecs.append(np.asarray(row, dtype=np.float64))
        out = []
        for v in vecs:
            out.append(self._model.classify_vectorspace(v))
        return np.array(out)

    def wcss(self, X: np.ndarray) -> float:
        """Within-cluster sum of squares."""
        X = np.asarray(X, dtype=np.float64)
        labels = self.predict(X)
        total = 0.0
        for i, x in enumerate(X):
            c = self.cluster_centers_[labels[i]]
            total += float(np.sum((x - c) ** 2))
        return total

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
