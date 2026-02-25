#!/usr/bin/env python3
"""Load saved model + test vectors -> evaluate clustering performance."""

import sys
import os
import argparse

import joblib
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..', 'src'))

from kmeans import KMeansModel


def main():
    parser = argparse.ArgumentParser(description="Evaluate clustering on test vectors")
    parser.add_argument("--model", required=True, help="Path to saved .joblib model")
    parser.add_argument("--input", required=True, help="Path to test vectors CSV")
    args = parser.parse_args()

    km: KMeansModel = joblib.load(args.model)
    X = np.loadtxt(args.input, delimiter=",", ndmin=2, dtype=np.float64)
    print("\t- Model:", args.model, "(k=" + str(km.n_clusters) + ")")
    print("\t- Test vectors:", X.shape)

    labels = km.predict(X)
    n_found = len(set(labels))
    print("\t- Clusters used:", n_found, "/", km.n_clusters, sep="")

    if n_found < 2:
        print("Error: need >= 2 clusters for evaluation metrics.")
        return 1

    wcss = km.wcss(X)
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)

    print("\n\tWCSS             ", round(wcss, 2), sep="")
    print("\tSilhouette       ", round(sil, 4), sep="")
    print("\tDavies-Bouldin   ", round(db, 4), sep="")


if __name__ == "__main__":
    raise SystemExit(main())
