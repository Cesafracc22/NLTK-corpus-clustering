from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import dump, load

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that converts text to an embedding using Word2Vec.
    """
    
    def __init__(self, sentences):
        self.sentences = sentences
        self.model = Word2Vec(
            sentences=self.sentences,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4,
            sg=1,
            hs=0,
            negative=5,
            ns_exponent=0.75,
        )
        

    def fit(self, X):
        self.model.fit(X)
        return self
    
    def transform(self, X):
        return self.model.transform(X)
    
    def fit_transform(self, X):
        return self.model.fit_transform(X)

    def save(self):
        dump(self.model, 'src/models/word2vec.joblib')
        return self

    def load(self):
        self.model = load('src/models/word2vec.joblib')
        return self