from nltk.corpus import reuters
import pandas as pd

def load_data():
    fileids = reuters.fileids()
    categories = []
    text = []
    for file in fileids:
        categories.append(reuters.categories(file))
        text.append(reuters.raw(file))
    return pd.DataFrame({'ids': fileids, 'categories': categories, 'text': text})
