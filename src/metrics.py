"""
Classification metrics: confusion matrix, per-class report, summary stats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def confusion_matrix_df(y_true: list[int], y_pred: list[int],
                        classes: list[str]) -> pd.DataFrame:
    """
    Return a labeled confusion matrix as a DataFrame.
    Rows = actual labels, columns = predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(cm, index=[f"Actual: {c}" for c in classes],
                      columns=[f"Predicted: {c}" for c in classes])
    return df


def per_class_report(y_true: list[int], y_pred: list[int],
                     classes: list[str]) -> pd.DataFrame:
    """
    Return per-class precision, recall, f1-score, support as a DataFrame.
    """
    report = classification_report(y_true, y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    rows = []
    for cls in classes:
        if cls in report:
            r = report[cls]
            rows.append({
                "class":     cls,
                "precision": round(r["precision"], 4),
                "recall":    round(r["recall"],    4),
                "f1_score":  round(r["f1-score"],  4),
                "support":   int(r["support"]),
            })
    df = pd.DataFrame(rows)
    return df


def summary_stats(y_true: list[int], y_pred: list[int],
                  classes: list[str]) -> dict:
    """
    Return macro-avg and weighted-avg stats plus overall accuracy.
    """
    report = classification_report(y_true, y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    accuracy = float((np.array(y_true) == np.array(y_pred)).mean())
    return {
        "accuracy":          round(accuracy, 4),
        "macro_precision":   round(report["macro avg"]["precision"],    4),
        "macro_recall":      round(report["macro avg"]["recall"],       4),
        "macro_f1":          round(report["macro avg"]["f1-score"],     4),
        "weighted_precision":round(report["weighted avg"]["precision"], 4),
        "weighted_recall":   round(report["weighted avg"]["recall"],    4),
        "weighted_f1":       round(report["weighted avg"]["f1-score"],  4),
    }


def misclassified_df(texts: list[str], y_true: list[int], y_pred: list[int],
                     classes: list[str]) -> pd.DataFrame:
    """
    Return a DataFrame of documents that were misclassified.
    Columns: text_preview, actual, predicted.
    """
    rows = []
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt != yp:
            preview = str(texts[i])[:120] + "…" if len(str(texts[i])) > 120 else str(texts[i])
            rows.append({
                "text_preview": preview,
                "actual":       classes[yt],
                "predicted":    classes[yp],
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["text_preview", "actual", "predicted"]
    )
