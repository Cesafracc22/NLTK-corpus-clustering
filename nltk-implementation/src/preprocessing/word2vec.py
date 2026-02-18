from gensim.models import Word2Vec

def train_word2vec_model(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

def save_word2vec_model(model, filename):
    model.save(filename)
    return model

def load_word2vec_model(filename):
    model = Word2Vec.load(filename)
    return model