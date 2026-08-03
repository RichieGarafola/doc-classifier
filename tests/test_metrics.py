"""Tests for src/metrics.py"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metrics import (
    confusion_matrix_df,
    misclassified_df,
    per_class_report,
    summary_stats,
)

CLASSES = ["Alpha", "Beta", "Gamma"]

# Perfect predictions
Y_TRUE  = [0, 0, 0, 1, 1, 1, 2, 2, 2]
Y_PRED  = [0, 0, 0, 1, 1, 1, 2, 2, 2]

# Imperfect predictions (Beta predicted as Gamma twice)
Y_TRUE2 = [0, 0, 1, 1, 1, 2, 2, 2, 0]
Y_PRED2 = [0, 0, 2, 1, 1, 2, 2, 0, 0]

TEXTS   = [f"document {i}" for i in range(9)]


# ── confusion_matrix_df ───────────────────────────────────────────────────────

class TestConfusionMatrixDf:
    def test_returns_dataframe(self):
        df = confusion_matrix_df(Y_TRUE, Y_PRED, CLASSES)
        assert isinstance(df, pd.DataFrame)

    def test_shape(self):
        df = confusion_matrix_df(Y_TRUE, Y_PRED, CLASSES)
        assert df.shape == (3, 3)

    def test_perfect_diagonal(self):
        df = confusion_matrix_df(Y_TRUE, Y_PRED, CLASSES)
        import numpy as np
        assert (np.diag(df.values) == [3, 3, 3]).all()

    def test_row_labels_contain_actual(self):
        df = confusion_matrix_df(Y_TRUE, Y_PRED, CLASSES)
        assert all("Actual" in r for r in df.index)

    def test_col_labels_contain_predicted(self):
        df = confusion_matrix_df(Y_TRUE, Y_PRED, CLASSES)
        assert all("Predicted" in c for c in df.columns)

    def test_imperfect_predictions(self):
        df = confusion_matrix_df(Y_TRUE2, Y_PRED2, CLASSES)
        # Off-diagonal should have non-zero values
        import numpy as np
        off_diag = df.values - np.diag(np.diag(df.values))
        assert off_diag.sum() > 0

    def test_row_sums_equal_support(self):
        df = confusion_matrix_df(Y_TRUE, Y_PRED, CLASSES)
        assert (df.sum(axis=1) == 3).all()


# ── per_class_report ──────────────────────────────────────────────────────────

class TestPerClassReport:
    def test_returns_dataframe(self):
        df = per_class_report(Y_TRUE, Y_PRED, CLASSES)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        df = per_class_report(Y_TRUE, Y_PRED, CLASSES)
        for col in ["class", "precision", "recall", "f1_score", "support"]:
            assert col in df.columns

    def test_one_row_per_class(self):
        df = per_class_report(Y_TRUE, Y_PRED, CLASSES)
        assert len(df) == len(CLASSES)

    def test_perfect_scores(self):
        df = per_class_report(Y_TRUE, Y_PRED, CLASSES)
        assert (df["precision"] == 1.0).all()
        assert (df["recall"] == 1.0).all()
        assert (df["f1_score"] == 1.0).all()

    def test_support_sums_to_total(self):
        df = per_class_report(Y_TRUE, Y_PRED, CLASSES)
        assert df["support"].sum() == len(Y_TRUE)

    def test_imperfect_f1_below_1(self):
        df = per_class_report(Y_TRUE2, Y_PRED2, CLASSES)
        assert (df["f1_score"] <= 1.0).all()
        # at least one class has f1 < 1
        assert (df["f1_score"] < 1.0).any()

    def test_scores_between_0_and_1(self):
        df = per_class_report(Y_TRUE2, Y_PRED2, CLASSES)
        for col in ["precision", "recall", "f1_score"]:
            assert (df[col] >= 0).all()
            assert (df[col] <= 1).all()


# ── summary_stats ─────────────────────────────────────────────────────────────

class TestSummaryStats:
    def test_returns_dict(self):
        s = summary_stats(Y_TRUE, Y_PRED, CLASSES)
        assert isinstance(s, dict)

    def test_expected_keys(self):
        s = summary_stats(Y_TRUE, Y_PRED, CLASSES)
        for key in ["accuracy", "macro_f1", "weighted_f1", "macro_precision",
                    "macro_recall", "weighted_precision", "weighted_recall"]:
            assert key in s, f"Missing key: {key}"

    def test_perfect_accuracy(self):
        s = summary_stats(Y_TRUE, Y_PRED, CLASSES)
        assert s["accuracy"] == 1.0

    def test_imperfect_accuracy_below_1(self):
        s = summary_stats(Y_TRUE2, Y_PRED2, CLASSES)
        assert s["accuracy"] < 1.0

    def test_all_values_between_0_and_1(self):
        s = summary_stats(Y_TRUE2, Y_PRED2, CLASSES)
        assert all(0.0 <= v <= 1.0 for v in s.values())

    def test_binary_accuracy_correct(self):
        y_t = [0, 0, 1, 1]
        y_p = [0, 1, 1, 1]
        s = summary_stats(y_t, y_p, ["A", "B"])
        assert s["accuracy"] == 0.75


# ── misclassified_df ──────────────────────────────────────────────────────────

class TestMisclassifiedDf:
    def test_returns_dataframe(self):
        df = misclassified_df(TEXTS, Y_TRUE2, Y_PRED2, CLASSES)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        df = misclassified_df(TEXTS, Y_TRUE2, Y_PRED2, CLASSES)
        for col in ["text_preview", "actual", "predicted"]:
            assert col in df.columns

    def test_perfect_predictions_empty(self):
        df = misclassified_df(TEXTS, Y_TRUE, Y_PRED, CLASSES)
        assert len(df) == 0
        for col in ["text_preview", "actual", "predicted"]:
            assert col in df.columns

    def test_misclassified_count_correct(self):
        df = misclassified_df(TEXTS, Y_TRUE2, Y_PRED2, CLASSES)
        expected = sum(1 for a, b in zip(Y_TRUE2, Y_PRED2) if a != b)
        assert len(df) == expected

    def test_actual_predicted_are_class_names(self):
        df = misclassified_df(TEXTS, Y_TRUE2, Y_PRED2, CLASSES)
        assert set(df["actual"].unique()).issubset(set(CLASSES))
        assert set(df["predicted"].unique()).issubset(set(CLASSES))

    def test_long_text_truncated(self):
        long_texts = ["word " * 50 for _ in TEXTS]
        df = misclassified_df(long_texts, Y_TRUE2, Y_PRED2, CLASSES)
        if len(df):
            assert all(len(p) <= 122 for p in df["text_preview"])
