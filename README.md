# NLTK Corpus Clustering

Clusters the **Reuters** corpus into a specified number of classes using K-Means, with document vectors from TF-IDF or Word2Vec and optional cosine or Euclidean distance.

## Overview

1. **Load** the Reuters corpus (NLTK).
2. **Preprocess** text (tokenize, optional stopword removal, optional stemming).
3. **Embed** documents with TF-IDF (custom or scikit-learn) or Word2Vec, optionally reduce dimensions with TruncatedSVD.
4. **Cluster** with K-Means; optionally **tune** the number of clusters \(k\).
5. **Evaluate** a saved model on test vectors (WCSS, Silhouette, Davies–Bouldin).

Outputs are CSV vectors and serialized models under `data/`.

## Project layout

```
├── src/                 # Library code (run from repo root or scripts/)
│   ├── load_data.py     # Reuters loading, NLTK data under data/nltk_data
│   ├── text_processing.py
│   ├── embeddings.py    # tfidf-custom, tfidf-scikit-learn, word2vec
│   └── kmeans.py        # KMeansModel (NLTK KMeansClusterer, cosine, L2 norm)
├── scripts/
│   ├── preprocess.py    # Corpus → vectors CSV
│   ├── cluster.py       # Vectors → K-Means → save model
│   ├── tune.py          # Vectors → sweep k → WCSS CSV
│   └── evaluate.py      # Model + test vectors → metrics
├── data/
│   ├── nltk_data/       # NLTK downloads (stopwords, reuters, punkt)
│   ├── vectors/         # Per-embedding subdirs with train.csv, test.csv
│   └── models/          # kmeans_v*.joblib, tune/wcss_results.csv
└── docs/                # Notebooks, report
```

## Scripts and flags

### `scripts/preprocess.py`

Preprocesses the Reuters corpus and writes document vectors to CSV (and optionally a train/test split).

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--tokenize` | Yes* | — | Use NLTK tokenization (required when using `--embedding`) |
| `--stopwords` | No | off | Remove NLTK English stopwords |
| `--stem` | No | off | Apply NLTK Porter stemmer |
| `--embedding` | Yes | — | `tfidf-custom`, `tfidf-scikit-learn`, or `word2vec` |
| `--svd` | No | — | TruncatedSVD dimension (e.g. `300`) |
| `--split` | No | off | Create train/test split |
| `--test_size` | No | 0.2 | Fraction of data for test set (if `--split`) |
| `--random_state` | No | 42 | Seed for split and SVD |
| `--output_dir` | No | `data/vectors` | Base directory for CSV outputs |

Outputs go to `{output_dir}/{embedding}/` or `{output_dir}/{embedding}-svd{N}/` (e.g. `train.csv`, `test.csv` if `--split`).

**Example**

```bash
python scripts/preprocess.py --tokenize --stopwords --embedding tfidf-scikit-learn --svd 300 --split
```

---

### `scripts/cluster.py`

Runs K-Means on a vectors CSV and saves the fitted model.

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--input` | Yes | — | Path to vectors CSV (e.g. `data/vectors/tfidf-scikit-learn-svd300/train.csv`) |
| `--n_clusters` | No | 8 | Number of clusters \(k\) |
| `--distance` | No | cosine | `cosine` or `euclidean` |
| `--repeats` | No | 1 | Number of K-Means runs (best by inertia kept) |
| `--random_state` | No | 42 | Seed for centroid initialization |
| `--models_dir` | No | `data/models` | Directory for saved `.joblib` models |

Models are saved as `kmeans_v1.joblib`, `kmeans_v2.joblib`, … under `models_dir`.

**Example**

```bash
python scripts/cluster.py --input data/vectors/tfidf-scikit-learn-svd300/train.csv --n_clusters 8 --distance cosine
```

---

### `scripts/tune.py`

Sweeps over \(k\) and writes WCSS to CSV (for elbow plots or model selection).

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--input` | Yes | — | Path to vectors CSV |
| `--k_min` | No | 3 | Minimum \(k\) |
| `--k_max` | No | 10 | Maximum \(k\) (inclusive) |
| `--distance` | No | cosine | `cosine` or `euclidean` |
| `--repeats` | No | 1 | K-Means runs per \(k\) |
| `--random_state` | No | 42 | Seed |
| `--sample_ratio` | No | 1.0 | Fraction of rows to use (e.g. 0.5 for faster sweep) |
| `--output_dir` | No | `data/models/tune` | Where to write `wcss_results.csv` |

**Example**

```bash
python scripts/tune.py --input data/vectors/word2vec/train.csv --k_min 3 --k_max 12 --output_dir data/models/tune/word2vec
```

---

### `scripts/evaluate.py`

Loads a saved K-Means model and test vectors, then prints WCSS, Silhouette, and Davies–Bouldin.

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--model` | Yes | — | Path to `.joblib` model |
| `--input` | Yes | — | Path to test vectors CSV |

**Example**

```bash
python scripts/evaluate.py --model data/models/kmeans_v1.joblib --input data/vectors/tfidf-scikit-learn-svd300/test.csv
```

---

## Typical workflow

```bash
# 1. Preprocess and embed (with train/test split)
python scripts/preprocess.py --tokenize --stopwords --embedding tfidf-scikit-learn --svd 300 --split

# 2. (Optional) Tune k
python scripts/tune.py --input data/vectors/tfidf-scikit-learn-svd300/train.csv --k_min 3 --k_max 12

# 3. Train final model
python scripts/cluster.py --input data/vectors/tfidf-scikit-learn-svd300/train.csv --n_clusters 8 --distance cosine

# 4. Evaluate on test set
python scripts/evaluate.py --model data/models/kmeans_v1.joblib --input data/vectors/tfidf-scikit-learn-svd300/test.csv
```

## Dependencies

- Python 3
- `nltk`, `numpy`, `scikit-learn`, `gensim`, `joblib`, `pandas`

NLTK data (Reuters, stopwords, punkt) is downloaded into `data/nltk_data` on first run.
