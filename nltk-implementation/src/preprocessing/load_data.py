from nltk.corpus import reuters
import pandas as pd


def load_data():
    """Load Reuters corpus and return a DataFrame with ids, categories and text."""
    fileids = reuters.fileids()
    categories = [reuters.categories(f) for f in fileids]
    texts = [reuters.raw(f) for f in fileids]
    return pd.DataFrame({'ids': fileids, 'categories': categories, 'text': texts})
