#!/usr/bin/env python3
"""Preprocess Reuters corpus -> save vectors to CSV."""

import sys
import os
import argparse
import random

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..', 'src'))

from load_data import load_data
from text_processing import preprocess
from embeddings import EMBEDDINGS


def main():
    parser = argparse.ArgumentParser(description="Preprocess corpus and save vectors to CSV")
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument("--stopwords", action="store_true")
    parser.add_argument("--stem", action="store_true")
    parser.add_argument("--embedding", required=True,
                        choices=["tfidf-custom", "tfidf-scikit-learn", "word2vec"])
    parser.add_argument("--svd", type=int, default=None, help="Apply TruncatedSVD to reduce dimensions")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(_SCRIPT_DIR, '..', 'data', 'vectors'))
    args = parser.parse_args()

    if not args.tokenize:
        parser.error("--tokenize is required when using --embedding")

    print("Pipeline:")

    texts = load_data()['text'].tolist()
    print("\t- Loaded", len(texts), "documents")

    if args.split:
        rng = random.Random(args.random_state)
        shuffled = texts.copy()
        rng.shuffle(shuffled)
        n_test = int(len(shuffled) * args.test_size)
        test_texts, train_texts = shuffled[:n_test], shuffled[n_test:]
        print("\t- Split: train=", len(train_texts), ", test=", len(test_texts), sep="")
    else:
        train_texts = texts
        test_texts = None

    train_docs = preprocess(train_texts, args.tokenize, args.stopwords, args.stem)

    embedder = EMBEDDINGS[args.embedding]()
    X_train = embedder.fit_transform(train_docs)
    print("\t- Embedded train:", X_train.shape)

    svd = None
    if args.svd is not None:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=args.svd, random_state=args.random_state)
        X_train = svd.fit_transform(X_train)
        pct = svd.explained_variance_ratio_.sum() * 100
        print("\t- SVD:", X_train.shape, "(variance explained: " + str(round(pct, 2)) + "%)")

    if args.svd is None:
        out_name = args.embedding
    else:
        out_name = args.embedding + "-svd" + str(args.svd)
    out_dir = os.path.join(args.output_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)

    np.savetxt(os.path.join(out_dir, "train.csv"), X_train, delimiter=",", fmt="%.6g")
    print("\t- Saved:", out_dir + "/train.csv")

    if test_texts is not None:
        test_docs = preprocess(test_texts, args.tokenize, args.stopwords, args.stem)
        X_test = embedder.transform(test_docs)
        if svd is not None:
            X_test = svd.transform(X_test)
        np.savetxt(os.path.join(out_dir, "test.csv"), X_test, delimiter=",", fmt="%.6g")
        print("\t- Saved:", out_dir + "/test.csv", "(", X_test.shape, ")")


if __name__ == "__main__":
    raise SystemExit(main())
