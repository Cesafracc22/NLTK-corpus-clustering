import numpy as np


def compute_tf(word, doc):
    """Compute term frequency for a word in a document."""
    return doc.count(word) / len(doc)


def compute_idf_dict(docs):
    """Compute IDF values for all unique words across documents."""
    all_words = set(word for doc in docs for word in doc)
    n_docs = len(docs)
    idf_dict = {}

    for word in all_words:
        doc_freq = sum(1 for doc in docs if word in doc)
        idf_dict[word] = np.log(n_docs / doc_freq) if doc_freq > 0 else 0.0

    return idf_dict


def compute_tfidf_matrix(docs, idf_dict=None):
    """Compute TF-IDF matrix. If idf_dict is None, computes IDF from docs."""
    if idf_dict is None:
        idf_dict = compute_idf_dict(docs)

    tfidf_matrix = []
    for doc in docs:
        row = []
        for word in doc:
            tf = compute_tf(word, doc)
            idf = idf_dict.get(word, 0.0)
            row.append(tf * idf)
        tfidf_matrix.append(row)

    return np.array(tfidf_matrix)
