"""
TF-IDF vectorization pipeline for the document classifier.
Returns fitted vectorizer and transformed matrices.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


def build_tfidf(
    ngram_range:  tuple[int, int] = (1, 2),
    max_features: int   = 10_000,
    min_df:       int   = 2,
    max_df:       float = 0.95,
    sublinear_tf: bool  = True,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b",
        stop_words="english",
    )


def encode_labels(labels: pd.Series) -> tuple[LabelEncoder, list[int]]:
    """Fit a LabelEncoder and return (encoder, encoded_labels)."""
    le = LabelEncoder()
    encoded = le.fit_transform(labels).tolist()
    return le, encoded


def decode_labels(le: LabelEncoder, encoded: list[int]) -> list[str]:
    return le.inverse_transform(encoded).tolist()
