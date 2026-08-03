"""
Train, evaluate, and predict with scikit-learn text classifiers.
Supports Logistic Regression and Multinomial Naive Bayes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.vectorizer import build_tfidf

ALGORITHMS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs"
    ),
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
}


@dataclass
class TrainResult:
    pipeline:       Pipeline
    label_encoder:  LabelEncoder
    classes:        list[str]
    X_train:        object      # sparse matrix
    X_test:         object
    y_train:        list[int]
    y_test:         list[int]
    y_pred:         list[int]
    test_accuracy:  float
    cv_scores:      list[float]
    cv_mean:        float
    cv_std:         float
    feature_names:  list[str]


def build_pipeline(algorithm: str, ngram_range=(1, 2), max_features=10_000,
                   min_df=2, max_df=0.95) -> Pipeline:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose: {list(ALGORITHMS)}")
    clf = ALGORITHMS[algorithm]
    tfidf = build_tfidf(ngram_range=ngram_range, max_features=max_features,
                        min_df=min_df, max_df=max_df)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def train(
    texts:       list[str],
    labels:      list[str],
    algorithm:   str   = "Logistic Regression",
    test_size:   float = 0.2,
    random_state: int  = 42,
    cv_folds:    int   = 5,
    ngram_range: tuple = (1, 2),
    max_features: int  = 10_000,
    min_df:      int   = 2,
    max_df:      float = 0.95,
) -> TrainResult:
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    classes = le.classes_.tolist()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    pipeline = build_pipeline(algorithm, ngram_range, max_features, min_df, max_df)
    pipeline.fit(X_train_raw, y_train)

    y_pred = pipeline.predict(X_test_raw).tolist()
    test_acc = float((np.array(y_pred) == np.array(y_test)).mean())

    # Cross-validation on full dataset
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_pipeline = build_pipeline(algorithm, ngram_range, max_features, min_df, max_df)
    cv_scores = cross_val_score(cv_pipeline, texts, y_enc, cv=cv, scoring="accuracy")

    tfidf_step = pipeline.named_steps["tfidf"]
    X_train_mat = tfidf_step.transform(X_train_raw)
    X_test_mat  = tfidf_step.transform(X_test_raw)

    return TrainResult(
        pipeline=pipeline,
        label_encoder=le,
        classes=classes,
        X_train=X_train_mat,
        X_test=X_test_mat,
        y_train=y_train.tolist(),
        y_test=y_test.tolist(),
        y_pred=y_pred,
        test_accuracy=round(test_acc, 4),
        cv_scores=cv_scores.round(4).tolist(),
        cv_mean=round(float(cv_scores.mean()), 4),
        cv_std=round(float(cv_scores.std()), 4),
        feature_names=tfidf_step.get_feature_names_out().tolist(),
    )


def predict_single(pipeline: Pipeline, le: LabelEncoder, text: str) -> dict:
    """Predict label and class probabilities for one text string."""
    proba = None
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        proba_arr = pipeline.predict_proba([text])[0]
        proba = {cls: round(float(p), 4) for cls, p in zip(le.classes_, proba_arr)}
    pred_enc = pipeline.predict([text])[0]
    pred_label = le.inverse_transform([pred_enc])[0]
    return {"predicted_label": pred_label, "probabilities": proba}


def top_features_per_class(result: TrainResult, top_n: int = 15) -> pd.DataFrame:
    """
    For Logistic Regression: extract the top_n highest-weight features per class.
    Returns: class, rank, term, weight.
    """
    clf = result.pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return pd.DataFrame(columns=["class", "rank", "term", "weight"])

    rows = []
    coef = clf.coef_
    # coef shape: (1, n_features) for binary; (n_classes, n_features) for multiclass
    if coef.shape[0] == 1:
        coef = np.vstack([-coef, coef])

    for i, cls in enumerate(result.classes):
        top_idx = coef[i].argsort()[::-1][:top_n]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append({
                "class": cls,
                "rank": rank,
                "term": result.feature_names[idx],
                "weight": round(float(coef[i][idx]), 5),
            })
    return pd.DataFrame(rows)
