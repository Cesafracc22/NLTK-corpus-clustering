"""Reuters corpus loading and NLTK resource management."""

import os
import nltk
from nltk.corpus import reuters
import pandas as pd

_NLTK_RESOURCES = {
    'stopwords': 'corpora/stopwords',
    'reuters': 'corpora/reuters',
    'punkt_tab': 'tokenizers/punkt_tab',
}


def ensure_nltk_data():
    """Download missing NLTK resources to project-local directory."""
    print("NLTK resources:")
    base = os.path.dirname(os.path.abspath(__file__))
    nltk_dir = os.path.abspath(os.path.join(base, '..', 'data', 'nltk_data'))
    os.makedirs(nltk_dir, exist_ok=True)
    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)

    for name, subpath in _NLTK_RESOURCES.items():
        print("\t-", name, end=" ")
        if os.path.exists(os.path.join(nltk_dir, subpath)):
            print("ok")
        else:
            print("downloading...", end=" ")
            nltk.download(name, quiet=True, download_dir=nltk_dir)
            print("ok")


def load_data() -> pd.DataFrame:
    """Load Reuters corpus. Returns DataFrame with ids, categories, text."""
    ensure_nltk_data()
    fids = reuters.fileids()
    categories = []
    text_list = []
    for f in fids:
        categories.append(reuters.categories(f))
        text_list.append(reuters.raw(f))
    return pd.DataFrame({
        'ids': fids,
        'categories': categories,
        'text': text_list,
    })
