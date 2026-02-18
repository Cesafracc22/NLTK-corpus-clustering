from nltk.corpus import reuters
import pandas as pd

def load_data(): 
    fileids = reuters.fileids()

    categories = []
    text = []

    for file in fileids:
        categories.append(reuters.categories(file))
        text.append(reuters.raw(file))

    reutersDf = pd.DataFrame({'ids':fileids, 'categories':categories, 'text':text})
    reutersDf.to_csv('data/raw/reuters.csv', index=False)
    return reutersDf

if __name__ == "__main__":
    load_data()