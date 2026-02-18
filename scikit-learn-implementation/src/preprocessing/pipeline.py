import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import StopwordsRemover
from sklearn.feature_extraction.text import Stemmer
from sklearn.feature_extraction.text import Lemmatizer
import argparse
from src.preprocessing.word2vec import Word2VecTransformer
from nltk import reuters
from sklearn.model_selection import train_test_split

class PreprocessingPipeline:
    def __init__(self, args, X, y, split=None):
        self.args = args
        self.X = X
        self.y = y
        self.steps = []

        if getattr(args, "stopwords", None):
            self.steps.append(('stopwords', StopwordsRemover()))
        if getattr(args, "stemmer", None):
            self.steps.append(('stemmer', Stemmer()))
        if getattr(args, "lemmatizer", None):
            self.steps.append(('lemmatizer', Lemmatizer()))
        if getattr(args, "embedding", None) == "tfidf":
            self.steps.append(('tfidf', TfidfVectorizer()))
        if getattr(args, "embedding", None) == "word2vec":
            self.steps.append(('word2vec', Word2VecTransformer(reuters.sents())))
        if split is None:
            split_enabled = not getattr(args, "no_split", False)
        else:
            split_enabled = split

        if split_enabled:
            self.steps.append(('train_test_split', train_test_split(X, y, test_size=0.2, random_state=42)))
        
        self.pipeline = Pipeline(self.steps)

    def fit(self, X):
        if getattr(self.args, "stopwords", None):
            X = StopwordsRemover().fit(X)
        if getattr(self.args, "stemmer", None):
            X = Stemmer().fit(X)
        if getattr(self.args, "lemmatizer", None):
            X = Lemmatizer().fit(X)
        if getattr(self.args, "embedding", None) == "tfidf":
            X = TfidfVectorizer().fit(X)
        elif getattr(self.args, "embedding", None) == "word2vec":
            X = Word2VecTransformer(reuters.sents()).fit(X)
        return X
    
    def transform(self, X):
        if getattr(self.args, "stopwords", None):
            X = StopwordsRemover().transform(X)
        if getattr(self.args, "stemmer", None):
            X = Stemmer().transform(X)
        if getattr(self.args, "lemmatizer", None):
            X = Lemmatizer().transform(X)
        if getattr(self.args, "embedding", None) == "tfidf":
            X = TfidfVectorizer().transform(X)
        elif getattr(self.args, "embedding", None) == "word2vec":
            X = Word2VecTransformer(reuters.sents()).transform(X)
        return X
    
    def fit_transform(self, X):
        if getattr(self.args, "stopwords", None):
            X = StopwordsRemover().fit_transform(X)
        if getattr(self.args, "stemmer", None):
            X = Stemmer().fit_transform(X)
        if getattr(self.args, "lemmatizer", None):
            X = Lemmatizer().fit_transform(X)
        if getattr(self.args, "embedding", None) == "tfidf":
            X = TfidfVectorizer().fit_transform(X)
        elif getattr(self.args, "embedding", None) == "word2vec":
            X = Word2VecTransformer(reuters.sents()).fit_transform(X)
        return X
    
    def get_params(self):
        return self.pipeline.get_params()
    
    def set_params(self, **params):
        self.pipeline.set_params(**params)
    
