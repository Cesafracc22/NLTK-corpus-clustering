#!/usr/bin/env python3
"""Load vectors from CSV -> tune KMeans k -> save models + WCSS results."""

import sys
import os
import argparse
import random

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..', 'src'))

from kmeans import KMeansModel


def main():
    parser = argparse.ArgumentParser(description="Tune KMeans k on precomputed vectors")
    parser.add_argument("--input", required=True, help="Path to vectors CSV")
    parser.add_argument("--k_min", type=int, default=3)
    parser.add_argument("--k_max", type=int, default=10)
    parser.add_argument("--distance", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(_SCRIPT_DIR, '..', 'data', 'models', 'tune'))
    args = parser.parse_args()

    if args.k_min > args.k_max:
        parser.error("k_min must be <= k_max")

    X = np.loadtxt(args.input, delimiter=",", ndmin=2, dtype=np.float64)
    print("\t- Loaded vectors:", X.shape)

    if args.sample_ratio < 1.0:
        rng = random.Random(args.random_state)
        n = max(1, min(int(len(X) * args.sample_ratio), len(X)))
        X = X[rng.sample(range(len(X)), n)]
        print("\t- Sampled:", n, "(" + str(int(100 * args.sample_ratio)) + "%)")

    if len(X) < args.k_max:
        print("Error: need >= k_max=", args.k_max, " samples, got ", len(X), sep="")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for k in range(args.k_min, args.k_max + 1):
        try:
            print("\tk=", k, " ...", sep="", end=" ", flush=True)
            km = KMeansModel(
                n_clusters=k, distance=args.distance,
                repeats=args.repeats, random_state=args.random_state,
            )
            km.fit(X)
            w = float(km.wcss(X))
            results.append((k, w))
            print("WCSS=", round(w, 2), sep="")
        except Exception as e:
            print("FAILED:", e)

    results.sort(key=lambda x: x[0])
    csv_path = os.path.join(args.output_dir, "wcss_results.csv")
    with open(csv_path, "w") as f:
        f.write("k,wcss\n")
        for k, w in results:
            f.write(str(k) + "," + str(round(w, 2)) + "\n")
    print("\n\t- Results:", csv_path, "(", len(results), " runs)")


if __name__ == "__main__":
    raise SystemExit(main())
