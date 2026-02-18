from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import argparse
from nltk_tfidf import compute_tfidf_matrix
from word2vec import train_word2vec_model, save_word2vec_model, load_word2vec_model
from load_data import load_data


class NLTKPreprocessingPipeline:
    """
    Preprocessing pipeline class for text data using NLTK
    
    This class provides a structured way to apply various preprocessing steps
    including tokenization, stopword removal, stemming, TF-IDF computation,
    and Word2Vec (gensim) model training.
    """
    
    def __init__(self, args):
        """
        Initialize the preprocessing pipeline
        
        Parameters:
        -----------
        args : argparse.Namespace
            Arguments containing preprocessing options:
            - tokenize: apply tokenization
            - remove_stopwords: remove stopwords
            - stem: apply stemming
            - tfidf: compute TF-IDF matrix
            - word2vec: train Word2Vec model
        """
        self.args = args
        self.processed_texts = None
        self.tfidf_matrix = None
        self.word2vec_model = None
        self.stopwords_set = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def load_data(self, text=None):
        """
        Load text data from Reuters corpus or use provided text
        """
        if text is None:
            df = load_data()
            texts = df['text'].tolist()
        elif isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        self.processed_texts = texts.copy()
        return self.processed_texts
    
    def tokenize(self, texts):
        """
        Tokenize texts using NLTK word_tokenize
        """
        return [word_tokenize(text) for text in texts]
    
    def remove_stopwords(self, texts):
        """
        Remove stopwords from tokenized texts
        """
        return [[word for word in text if word not in self.stopwords_set] 
                for text in texts]
    
    def stem(self, texts):
        """
        Apply Porter stemming to tokenized texts
        """
        return [[self.stemmer.stem(word) for word in text] for text in texts]
    
    def compute_tfidf(self, texts):
        """
        Compute TF-IDF matrix for tokenized texts
        """
        self.tfidf_matrix = compute_tfidf_matrix(texts)
        return self.tfidf_matrix
    
    def train_word2vec(self, texts, model_path='word2vec.model'):
        """
        Train Word2Vec model on tokenized texts
        """
        self.word2vec_model = train_word2vec_model(texts)
        save_word2vec_model(self.word2vec_model, model_path)
        return self.word2vec_model
    
    def fit(self, text=None):
        """
        Execute the preprocessing pipeline based on args configuration
        """
        texts = self.load_data(text)
        
        if getattr(self.args, 'tokenize', False):
            texts = self.tokenize(texts)
        
        if getattr(self.args, 'remove_stopwords', False):
            texts = self.remove_stopwords(texts)
        
        if getattr(self.args, 'stem', False):
            texts = self.stem(texts)
        
        if getattr(self.args, 'tfidf', False):
            return self.compute_tfidf(texts)
        
        if getattr(self.args, 'word2vec', False):
            return self.train_word2vec(texts)
        
        self.processed_texts = texts
        return self.processed_texts
    
    def transform(self, text):
        """
        Transform new text using the fitted pipeline
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        if getattr(self.args, 'tokenize', False):
            texts = self.tokenize(texts)
        
        if getattr(self.args, 'remove_stopwords', False):
            texts = self.remove_stopwords(texts)
        
        if getattr(self.args, 'stem', False):
            texts = self.stem(texts)
        
        return texts
    
    def get_processed_texts(self):
        """Get the processed texts"""
        return self.processed_texts
    
    def get_tfidf_matrix(self):
        """Get the TF-IDF matrix if computed"""
        return self.tfidf_matrix
    
    def get_word2vec_model(self):
        """Get the Word2Vec model if trained"""
        return self.word2vec_model

def tokenize_text(text):
    """Tokenize a single text"""
    return word_tokenize(text)

def remove_stopwords(text):
    """Remove stopwords from a tokenized text"""
    return [word for word in text if word not in stopwords.words('english')]

def stem_text(text):
    """Stem a tokenized text"""
    return [PorterStemmer().stem(word) for word in text]

