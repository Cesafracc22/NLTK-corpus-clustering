# NLTK Corpus Clustering


## PLEASE READ `report.pdf` FOR FULL PROJECT REPORT.THIS README CONTAINS INFORMATION ABOUT PROGRAM USAGE AND CODE SPECIFICS. 

Clusters the **Reuters** corpus into a specified number of classes using K-Means, with document vectors from TF-IDF or Word2Vec and optional cosine or Euclidean distance.

## Overview

1. **Load** the Reuters corpus (NLTK).
2. **Preprocess** text (tokenize, optional stopword removal, optional stemming).
3. **Embed** documents with TF-IDF (custom or scikit-learn) or Word2Vec, optionally reduce dimensions with TruncatedSVD.
4. **Cluster** with K-Means; optionally **tune** the number of clusters \(k\).
5. **Evaluate** a saved model on test vectors (WCSS, Silhouette, Davies–Bouldin).

Outputs are CSV vectors and serialized models under `data/`.

### Environment setup / Install requirements

Install dependencies with **pip** or **conda** (run from the repository root).

**Option A – pip**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

**Option B – conda**

```bash
conda env create -f environment.yml
conda activate nlp-corpus-clustering
```

After that, download NLTK data once with:

```bash
python scripts/download_nltk_data.py
```

Then you can run the pipeline (preprocess, tune, cluster, evaluate) as described below.

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

**Examples**

```bash
# Custom TF-IDF + SVD 300 (with train/test split)
python scripts/preprocess.py --tokenize --stopwords --embedding tfidf-custom --svd 300 --split

# Scikit-learn TF-IDF + SVD 300
python scripts/preprocess.py --tokenize --stopwords --embedding tfidf-scikit-learn --svd 300 --split

# Word2Vec (no SVD; 100 dims)
python scripts/preprocess.py --tokenize --stopwords --embedding word2vec --split
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

**Examples**

```bash
# Train on custom TF-IDF (SVD=300), k=6, cosine, 5 repeats
python scripts/cluster.py --input data/vectors/tfidf-custom-svd300/train.csv --n_clusters 6 --distance cosine --repeats 5

# Train on scikit-learn TF-IDF (SVD=300)
python scripts/cluster.py --input data/vectors/tfidf-scikit-learn-svd300/train.csv --n_clusters 6 --distance cosine --repeats 5

# Train on Word2Vec
python scripts/cluster.py --input data/vectors/word2vec/train.csv --n_clusters 6 --distance cosine --repeats 5
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

**Examples**

```bash
# Tune k for custom TF-IDF vectors (writes data/models/tune/tfidf-custom/wcss_results.csv)
python scripts/tune.py --input data/vectors/tfidf-custom-svd300/train.csv --k_min 3 --k_max 12 --repeats 5 --output_dir data/models/tune/tfidf-custom

# Tune k for scikit-learn TF-IDF
python scripts/tune.py --input data/vectors/tfidf-scikit-learn-svd300/train.csv --k_min 3 --k_max 12 --repeats 5 --output_dir data/models/tune/tfidf-scikit-learn

# Tune k for Word2Vec
python scripts/tune.py --input data/vectors/word2vec/train.csv --k_min 3 --k_max 12 --repeats 5 --output_dir data/models/tune/word2vec
```

---

### `scripts/evaluate.py`

Loads a saved K-Means model and test vectors, then prints WCSS, Silhouette, and Davies–Bouldin.

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--model` | Yes | — | Path to `.joblib` model |
| `--input` | Yes | — | Path to test vectors CSV |

**Examples**

Use the same embedding (and SVD) for test vectors as for the model you trained.
Use the correct version of the joblib (v1,v2,...)
```bash
# Evaluate a model trained on custom TF-IDF (SVD=300)
python scripts/evaluate.py --model data/models/kmeans_v1.joblib --input data/vectors/tfidf-custom-svd300/test.csv

# Evaluate a model trained on scikit-learn TF-IDF (SVD=300)
python scripts/evaluate.py --model data/models/kmeans_v2.joblib --input data/vectors/tfidf-scikit-learn-svd300/test.csv

# Evaluate a model trained on Word2Vec
python scripts/evaluate.py --model data/models/kmeans_v3.joblib --input data/vectors/word2vec/test.csv
```

(Replace `kmeans_v1.joblib`, `kmeans_v2.joblib`, etc. with the actual model paths printed by `cluster.py`.)

---

## Full workflow example

Run from the repository root. Below: one pipeline for **custom TF-IDF**, one for **scikit-learn TF-IDF**, one for **Word2Vec**.

### Custom TF-IDF (with SVD)

```bash
# 1. Preprocess
python scripts/preprocess.py --tokenize --stopwords --embedding tfidf-custom --svd 300 --split

# 2. (Optional) Tune k for deciding best k (elbow method used in project)
python scripts/tune.py --input data/vectors/tfidf-custom-svd300/train.csv --k_min 3 --k_max 12 --repeats 5 --output_dir data/models/tune/tfidf-custom

# 3. Train (e.g. k=6)
python scripts/cluster.py --input data/vectors/tfidf-custom-svd300/train.csv --n_clusters 6 --distance cosine --repeats 5

# 4. Evaluate (use the path printed in step 3 for --model)
python scripts/evaluate.py --model data/models/kmeans_v1.joblib --input data/vectors/tfidf-custom-svd300/test.csv
```

### Scikit-learn TF-IDF (with SVD)

```bash
python scripts/preprocess.py --tokenize --stopwords --embedding tfidf-scikit-learn --svd 300 --split
python scripts/tune.py --input data/vectors/tfidf-scikit-learn-svd300/train.csv --k_min 3 --k_max 12 --repeats 5 --output_dir data/models/tune/tfidf-scikit-learn
python scripts/cluster.py --input data/vectors/tfidf-scikit-learn-svd300/train.csv --n_clusters 6 --distance cosine --repeats 5
python scripts/evaluate.py --model data/models/kmeans_v2.joblib --input data/vectors/tfidf-scikit-learn-svd300/test.csv
```

### Word2Vec

```bash
python scripts/preprocess.py --tokenize --stopwords --embedding word2vec --split
python scripts/tune.py --input data/vectors/word2vec/train.csv --k_min 3 --k_max 12 --repeats 5 --output_dir data/models/tune/word2vec
python scripts/cluster.py --input data/vectors/word2vec/train.csv --n_clusters 6 --distance cosine --repeats 5
python scripts/evaluate.py --model data/models/kmeans_v3.joblib --input data/vectors/word2vec/test.csv
```

## Dependencies

- Python 3
- `nltk`, `numpy`, `scikit-learn`, `gensim`, `joblib`, `pandas`

NLTK data (Reuters, stopwords, punkt) is downloaded into `data/nltk_data` on first run.
