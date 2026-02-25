#!/usr/bin/env python3
"""Load vectors from CSV -> KMeans -> save model."""

import sys
import os
import re
import argparse

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..', 'src'))

from kmeans import KMeansModel


def _next_version(d):
    os.makedirs(d, exist_ok=True)
    pat = re.compile(r"^kmeans_v(\d+)\.joblib$")
    vs = []
    for f in os.listdir(d):
        m = pat.match(f)
        if m:
            vs.append(int(m.group(1)))
    return max(vs, default=0) + 1


def main():
    parser = argparse.ArgumentParser(description="Run KMeans on precomputed vectors")
    parser.add_argument("--input", required=True, help="Path to vectors CSV")
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--distance", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--models_dir", type=str,
                        default=os.path.join(_SCRIPT_DIR, '..', 'data', 'models'))
    args = parser.parse_args()

    X = np.loadtxt(args.input, delimiter=",", ndmin=2, dtype=np.float64)
    print("\t- Loaded vectors:", X.shape)

    km = KMeansModel(
        n_clusters=args.n_clusters, distance=args.distance,
        repeats=args.repeats, random_state=args.random_state,
    )
    km.fit(X)
    print("\t- Clusters:", args.n_clusters)
    print("\t- WCSS:", round(km.wcss(X), 2))

    v = _next_version(args.models_dir)
    path = os.path.join(args.models_dir, "kmeans_v" + str(v) + ".joblib")
    km.save(path)
    print("\t- Saved:", path)


if __name__ == "__main__":
    raise SystemExit(main())
