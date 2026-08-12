"""
Feature Representation Strategy using PyThaiNLP Tokenization and TF-IDF FeatureUnion.
Guarantees strict zero data leakage by fitting exclusively on training folds within Pipeline.
"""

from typing import List
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def pythainlp_tokenizer(text: str) -> List[str]:
    """Thai word tokenizer using PyThaiNLP newmm engine."""
    if not text:
        return []
    return word_tokenize(str(text), engine="newmm", keep_whitespace=False)


def create_tfidf_vectorizer(
    word_max_features: int = 5000,
    char_max_features: int = 5000,
    use_hybrid: bool = True
) -> FeatureUnion:
    """
    Creates TF-IDF FeatureUnion combining:
    1. Word-level TF-IDF (1-2 ngrams via PyThaiNLP)
    2. Char-level TF-IDF (2-4 ngrams)
    """
    word_vec = TfidfVectorizer(
        tokenizer=pythainlp_tokenizer,
        ngram_range=(1, 2),
        max_features=word_max_features,
        token_pattern=None
    )
    
    if not use_hybrid:
        return word_vec
        
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=char_max_features
    )
    
    union = FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_vec),
            ("char_tfidf", char_vec)
        ]
    )
    
    return union
