"""Tests for src/classifier.py"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.classifier import (
    ALGORITHMS,
    TrainResult,
    build_pipeline,
    predict_single,
    top_features_per_class,
    train,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

TEXTS = (
    [
        "The contractor shall provide program management support and deliverables",
        "Task order requires software development and agile lifecycle support",
        "Invoice submitted for labor hours and indirect costs during billing period",
        "This invoice covers direct labor travel and subcontractor expenses",
        "RFP solicitation for professional services technical and price proposal",
        "Offerors must submit proposals addressing all requirements in the PWS",
        "Contract modification extends period of performance and increases ceiling",
        "Modification P00001 adds scope and obligates additional funds",
    ] * 6  # 48 docs, 2 labels × 3 groups × 8 docs
)

LABELS = (
    ["SOW"] * 2 + ["Development"] * 2 + ["Invoice"] * 2 + ["RFP"] * 2
) * 6


@pytest.fixture
def result():
    return train(
        TEXTS, LABELS, algorithm="Logistic Regression",
        test_size=0.25, cv_folds=3, min_df=1, max_df=1.0,
    )


@pytest.fixture
def nb_result():
    return train(
        TEXTS, LABELS, algorithm="Multinomial Naive Bayes",
        test_size=0.25, cv_folds=3, min_df=1, max_df=1.0,
    )


# ── build_pipeline ────────────────────────────────────────────────────────────

class TestBuildPipeline:
    def test_valid_algorithm(self):
        p = build_pipeline("Logistic Regression", min_df=1, max_df=1.0)
        assert "tfidf" in p.named_steps
        assert "clf" in p.named_steps

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            build_pipeline("Random Forest")

    def test_all_algorithms_buildable(self):
        for alg in ALGORITHMS:
            p = build_pipeline(alg, min_df=1, max_df=1.0)
            assert p is not None


# ── train ─────────────────────────────────────────────────────────────────────

class TestTrain:
    def test_returns_train_result(self, result):
        assert isinstance(result, TrainResult)

    def test_classes_populated(self, result):
        assert set(result.classes) == {"SOW", "Development", "Invoice", "RFP"}

    def test_test_accuracy_between_0_and_1(self, result):
        assert 0.0 <= result.test_accuracy <= 1.0

    def test_cv_scores_length(self, result):
        assert len(result.cv_scores) == 3

    def test_cv_mean_matches_scores(self, result):
        import numpy as np
        assert abs(result.cv_mean - float(np.mean(result.cv_scores))) < 1e-3

    def test_feature_names_nonempty(self, result):
        assert len(result.feature_names) > 0

    def test_y_pred_same_length_as_y_test(self, result):
        assert len(result.y_pred) == len(result.y_test)

    def test_predictions_are_valid_class_indices(self, result):
        n_classes = len(result.classes)
        assert all(0 <= p < n_classes for p in result.y_pred)

    def test_naive_bayes_works(self, nb_result):
        assert 0.0 <= nb_result.test_accuracy <= 1.0

    def test_pipeline_fitted(self, result):
        pred = result.pipeline.predict(["invoice for labor hours"])
        assert len(pred) == 1

    def test_label_encoder_classes_match(self, result):
        assert list(result.label_encoder.classes_) == result.classes

    def test_distinguishable_corpus_high_accuracy(self):
        # Clearly distinct vocabulary per class -> should score > 0.7
        texts  = ["cloud infrastructure deployment automation"] * 20
        texts += ["invoice payment direct labor costs billing"] * 20
        labels = ["Cloud"] * 20 + ["Invoice"] * 20
        r = train(texts, labels, test_size=0.25, cv_folds=3, min_df=1, max_df=1.0)
        assert r.test_accuracy >= 0.7


# ── predict_single ────────────────────────────────────────────────────────────

class TestPredictSingle:
    def test_returns_dict(self, result):
        out = predict_single(result.pipeline, result.label_encoder,
                             "invoice for professional services")
        assert isinstance(out, dict)

    def test_predicted_label_is_valid_class(self, result):
        out = predict_single(result.pipeline, result.label_encoder,
                             "solicitation for proposals technical approach")
        assert out["predicted_label"] in result.classes

    def test_probabilities_sum_to_1(self, result):
        out = predict_single(result.pipeline, result.label_encoder,
                             "monthly status report and deliverable")
        if out["probabilities"] is not None:
            total = sum(out["probabilities"].values())
            assert abs(total - 1.0) < 1e-4

    def test_probabilities_keys_are_classes(self, result):
        out = predict_single(result.pipeline, result.label_encoder, "contract modification")
        if out["probabilities"] is not None:
            assert set(out["probabilities"].keys()) == set(result.classes)

    def test_nb_no_probabilities_key_still_present(self, nb_result):
        out = predict_single(nb_result.pipeline, nb_result.label_encoder, "invoice billing")
        assert "predicted_label" in out
        assert "probabilities" in out

    def test_empty_string_does_not_crash(self, result):
        out = predict_single(result.pipeline, result.label_encoder, "")
        assert "predicted_label" in out


# ── top_features_per_class ────────────────────────────────────────────────────

class TestTopFeaturesPerClass:
    def test_returns_dataframe(self, result):
        df = top_features_per_class(result, top_n=5)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, result):
        df = top_features_per_class(result, top_n=5)
        for col in ["class", "rank", "term", "weight"]:
            assert col in df.columns

    def test_top_n_per_class(self, result):
        top_n = 5
        df = top_features_per_class(result, top_n=top_n)
        if len(df):
            counts = df.groupby("class").size()
            assert (counts <= top_n).all()

    def test_rank_starts_at_1(self, result):
        df = top_features_per_class(result, top_n=5)
        if len(df):
            assert df.groupby("class")["rank"].min().eq(1).all()

    def test_nb_returns_empty_schema(self, nb_result):
        df = top_features_per_class(nb_result, top_n=5)
        for col in ["class", "rank", "term", "weight"]:
            assert col in df.columns
