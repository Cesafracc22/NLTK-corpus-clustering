from nltk.corpus import reuters
import pandas as pd
import numpy as np

def compute_tf(word, doc):
    tf = doc.count(word) / len(doc)
    return tf

def compute_idf(word, docs):
    idf = np.log(len(docs) / sum(1 for doc in docs if word in doc))
    return idf

def compute_tfidf(word, doc, docs):
    tfidf = compute_tf(word, doc) * compute_idf(word, docs)
    return tfidf

def compute_tfidf_matrix(docs):
    print(f"Computing TFIDF matrix for docs: {docs}")
    tfidf_matrix = []

    for doc in docs:
        tfidf_row = []

        for word in doc:
            tfidf_score = compute_tfidf(word, doc, docs)
            tfidf_row.append(tfidf_score)
            
        tfidf_matrix.append(tfidf_row)
    tfidf_matrix = np.array(tfidf_matrix)
    print(f"TFIDF matrix computed successfully")
    return tfidf_matrix