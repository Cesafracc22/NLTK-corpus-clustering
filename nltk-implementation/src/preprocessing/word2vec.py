from gensim.models import Word2Vec
import numpy as np


def train_word2vec(sentences, vector_size=100, window=5, min_count=1, workers=4):
    """Train a Word2Vec model on tokenized sentences."""
    return Word2Vec(sentences, vector_size=vector_size, window=window,
                    min_count=min_count, workers=workers)


def documents_to_vectors(tokenized_docs, model):
    """Convert tokenized documents to normalized document vectors (mean of word vectors)."""
    vector_size = model.wv.vector_size
    doc_vectors = []

    for tokens in tokenized_docs:
        word_vecs = [model.wv[w] for w in tokens if w in model.wv]

        if not word_vecs:
            vec = np.zeros(vector_size)
        else:
            vec = np.mean(word_vecs, axis=0)

        # L2 normalize for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        doc_vectors.append(vec)

    return np.array(doc_vectors)
