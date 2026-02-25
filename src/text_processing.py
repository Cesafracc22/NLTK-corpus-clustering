"""Text preprocessing: tokenize, stopwords, stem (NLTK)."""

from typing import List
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

_STOPWORDS = None
_STEMMER = PorterStemmer()


def _get_stopwords() -> set:
    global _STOPWORDS
    if _STOPWORDS is None:
        _STOPWORDS = set(stopwords.words('english'))
    return _STOPWORDS


def tokenize(texts: List[str]) -> List[List[str]]:
    """Tokenize raw strings into word lists."""
    out = []
    for t in texts:
        out.append(word_tokenize(t))
    return out


def remove_stopwords(docs: List[List[str]]) -> List[List[str]]:
    """Remove English stopwords."""
    sw = _get_stopwords()
    out = []
    for doc in docs:
        row = []
        for w in doc:
            if w not in sw:
                row.append(w)
        out.append(row)
    return out


def stem(docs: List[List[str]]) -> List[List[str]]:
    """Apply Porter stemming."""
    out = []
    for doc in docs:
        row = []
        for w in doc:
            row.append(_STEMMER.stem(w))
        out.append(row)
    return out


def preprocess(
    texts: List[str],
    do_tokenize: bool = True,
    do_stopwords: bool = True,
    do_stem: bool = True,
) -> List[List[str]]:
    """Apply preprocessing steps in order. Returns tokenized documents."""
    docs = texts
    steps = []
    if do_tokenize:
        docs = tokenize(docs)
        steps.append("tokenize")
    if do_stopwords:
        docs = remove_stopwords(docs)
        steps.append("stopwords")
    if do_stem:
        docs = stem(docs)
        steps.append("stem")
    if steps:
        print("\t- Preprocessed:", ", ".join(steps))
    return docs
