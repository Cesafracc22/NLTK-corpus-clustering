import random
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from load_data import load_data
from nltk_tfidf import compute_idf_dict, compute_tfidf_matrix
from word2vec import train_word2vec, documents_to_vectors


class NLTKPreprocessingPipeline:
    """NLTK preprocessing pipeline with sklearn-like fit/transform interface"""

    def __init__(self, args):
        """Initialize pipeline from argparse args"""
        self.args = args
        self.is_fitted = False
        self.stopwords_set = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

        self.idf_dict = None
        self.w2v_model = None

        self.train_data = None
        self.test_data = None

    def _tokenize(self, texts):
        """Tokenize raw strings into word lists"""
        return [word_tokenize(t) for t in texts]

    def _remove_stopwords(self, tokenized_texts):
        """Remove English stopwords from tokenized texts"""
        return [[w for w in doc if w not in self.stopwords_set] for doc in tokenized_texts]

    def _stem(self, tokenized_texts):
        """Apply Porter stemming to tokenized texts"""
        return [[self.stemmer.stem(w) for w in doc] for doc in tokenized_texts]

    def _preprocess(self, texts):
        """Apply all active preprocessing steps in order"""
        if getattr(self.args, 'tokenize', False):
            texts = self._tokenize(texts)
        if getattr(self.args, 'remove_stopwords', False):
            texts = self._remove_stopwords(texts)
        if getattr(self.args, 'stem', False):
            texts = self._stem(texts)
        return texts



    def _load(self, text=None):
        """Load data from Reuters corpus or use provided text"""
        if text is None:
            return load_data()['text'].tolist()
        if isinstance(text, str):
            return [text]
        return list(text)

    def _split(self, texts):
        """Split texts into train/test sets"""
        test_size = getattr(self.args, 'test_size', 0.2)
        seed = getattr(self.args, 'random_state', None)
        rng = random.Random(seed)

        shuffled = texts.copy()
        rng.shuffle(shuffled)

        n_test = int(len(shuffled) * test_size)
        self.test_data = shuffled[:n_test]
        self.train_data = shuffled[n_test:]
        return self.train_data



    def _apply_embedding(self, processed_texts):
        """Apply the active embedding (tfidf or word2vec) using fitted parameters"""
        if getattr(self.args, 'tfidf', False):
            return compute_tfidf_matrix(processed_texts, idf_dict=self.idf_dict)
        if getattr(self.args, 'word2vec', False):
            return documents_to_vectors(processed_texts, self.w2v_model)
        return processed_texts

    def fit(self, text=None):
        """Fit pipeline: learn IDF / train Word2Vec from training data."""
        texts = self._load(text)

        if text is None and getattr(self.args, 'split', False):
            texts = self._split(texts)

        processed = self._preprocess(texts)

        if getattr(self.args, 'tfidf', False):
            self.idf_dict = compute_idf_dict(processed)
        elif getattr(self.args, 'word2vec', False):
            self.w2v_model = train_word2vec(processed)

        self.is_fitted = True
        return self

    def transform(self, text=None):
        """Transform data using fitted pipeline. If text is None and split was used, transforms test data."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform().")

        if text is None:
            if self.test_data is not None:
                texts = self.test_data
            else:
                raise ValueError("No data to transform. Pass text or use --split.")
        elif isinstance(text, list):
            texts = text
        elif isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        processed = self._preprocess(texts)
        return self._apply_embedding(processed)

    def fit_transform(self, text=None):
        """Fit and transform training data in one step."""
        texts = self._load(text)

        if getattr(self.args, 'split', False):
            texts = self._split(texts)

        self.fit(texts)
        return self.transform(texts)


    def get_train_test_split(self):
        """Return (train_data, test_data) tuple."""
        return self.train_data, self.test_data
