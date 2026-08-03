# Testing — Text Document Classifier

## Overview

The test suite contains **52 tests** across **2 files** and **8 test classes**. All tests run in under 10 seconds on standard hardware.

```bash
pytest tests/ -v
# Expected: 52 passed
```

Tests cover the classifier module (training, prediction, feature extraction) and the metrics module (confusion matrix, per-class report, summary statistics, misclassified document table). The Streamlit UI layer (`app/main.py`) is not unit-tested; it is validated manually.

---

## Test Corpus Design

Both test files share a synthetic corpus built from four label categories with intentionally distinct vocabulary:

```python
TEXTS = [
    "Requirement specification for software development project scope",    # SOW
    "Work breakdown structure and deliverable timeline for project",        # SOW
    "Application development programming code implementation feature",      # Development
    "Software engineering build deploy testing integration pipeline",       # Development
    "Payment invoice billing amount due financial transaction receipt",     # Invoice
    "Billing statement account balance payment processing charges fees",    # Invoice
    "Request for proposal bid solicitation vendor evaluation criteria",     # RFP
    "Proposal submission requirements evaluation criteria selection bid",   # RFP
] * 6   # 48 total documents (12 per class)
```

**Why synthetic:** Synthetic text gives deterministic, reproducible tests. Vocabulary is distinct enough across classes that both Logistic Regression and Multinomial Naive Bayes achieve near-perfect accuracy on this corpus, which enables `test_distinguishable_corpus_high_accuracy` to assert `accuracy > 0.8`.

**Why 12 per class:** `min_df=2` requires each term to appear in at least 2 documents. Multiplying by 6 ensures all terms clear this threshold and the vocabulary is non-empty.

**Stratified split requirements:** `test_size=0.2` with 48 documents yields 9–10 test documents. Each class has ≥3 test examples, satisfying stratification requirements.

---

## `tests/test_classifier.py` — 26 Tests, 4 Classes

### TestBuildPipeline (3 tests)

Tests that `build_pipeline()` constructs a valid sklearn Pipeline for every supported algorithm.

| Test | Asserts |
|---|---|
| `test_valid_algorithm` | Returns a `Pipeline` instance for "Logistic Regression" |
| `test_invalid_algorithm_raises` | Raises `ValueError` for an unrecognized algorithm name |
| `test_all_algorithms_buildable` | Every key in `ALGORITHMS` produces a pipeline without error |

### TestTrain (12 tests)

Tests the full training contract of `train()`.

| Test | Asserts |
|---|---|
| `test_returns_train_result` | Return type is `TrainResult` |
| `test_classes_populated` | `result.classes` is a non-empty list of strings |
| `test_test_accuracy_between_0_and_1` | `test_accuracy` is in [0.0, 1.0] |
| `test_cv_scores_length` | `len(cv_scores) == cv_folds` |
| `test_cv_mean_matches_scores` | `cv_mean ≈ mean(cv_scores)` (float tolerance) |
| `test_feature_names_nonempty` | `feature_names` is a non-empty list |
| `test_y_pred_same_length_as_y_test` | `len(y_pred) == len(y_test)` |
| `test_predictions_are_valid_class_indices` | Every integer in `y_pred` is a valid class index |
| `test_naive_bayes_works` | `train()` succeeds with "Multinomial Naive Bayes" |
| `test_pipeline_fitted` | `pipeline.predict()` is callable after training |
| `test_label_encoder_classes_match` | `label_encoder.classes_` matches `result.classes` |
| `test_distinguishable_corpus_high_accuracy` | Test accuracy > 0.8 on the synthetic distinguishable corpus |

### TestPredictSingle (6 tests)

Tests the prediction interface for a single text string.

| Test | Asserts |
|---|---|
| `test_returns_dict` | Return type is `dict` |
| `test_predicted_label_is_valid_class` | `predicted_label` is a string in `result.classes` |
| `test_probabilities_sum_to_1` | `sum(probabilities.values()) ≈ 1.0` (Logistic Regression) |
| `test_probabilities_keys_are_classes` | `probabilities` keys match `result.classes` |
| `test_nb_no_probabilities_key_still_present` | Key `"probabilities"` exists in result dict even when value is `None` |
| `test_empty_string_does_not_crash` | Predicting on an empty string returns a valid dict without raising |

### TestTopFeaturesPerClass (5 tests)

Tests feature weight extraction from a trained Logistic Regression model.

| Test | Asserts |
|---|---|
| `test_returns_dataframe` | Return type is `pd.DataFrame` |
| `test_expected_columns` | Columns are exactly `["class", "rank", "term", "weight"]` |
| `test_top_n_per_class` | Each class has at most `top_n` rows |
| `test_rank_starts_at_1` | Minimum rank value is 1 (not 0) |
| `test_nb_returns_empty_schema` | Naive Bayes returns an empty DataFrame with the correct column schema |

---

## `tests/test_metrics.py` — 26 Tests, 4 Classes

### TestConfusionMatrixDf (7 tests)

Tests the confusion matrix DataFrame shape and labeling.

| Test | Asserts |
|---|---|
| `test_returns_dataframe` | Return type is `pd.DataFrame` |
| `test_shape` | Shape is `(n_classes, n_classes)` |
| `test_perfect_diagonal` | For perfect predictions, all off-diagonal values are 0 |
| `test_row_labels_contain_actual` | Every row index starts with `"Actual:"` |
| `test_col_labels_contain_predicted` | Every column name starts with `"Predicted:"` |
| `test_imperfect_predictions` | At least one off-diagonal cell is non-zero for imperfect predictions |
| `test_row_sums_equal_support` | Each row sum equals the number of examples of that actual class |

### TestPerClassReport (7 tests)

Tests the per-class precision/recall/F1 report.

| Test | Asserts |
|---|---|
| `test_returns_dataframe` | Return type is `pd.DataFrame` |
| `test_expected_columns` | Columns are exactly `["class", "precision", "recall", "f1_score", "support"]` |
| `test_one_row_per_class` | DataFrame has exactly `n_classes` rows |
| `test_perfect_scores` | Precision, recall, and F1 are all 1.0 for perfect predictions |
| `test_support_sums_to_total` | Sum of `support` column equals total number of test examples |
| `test_imperfect_f1_below_1` | At least one F1 score is below 1.0 for imperfect predictions |
| `test_scores_between_0_and_1` | All precision, recall, and F1 values are in [0.0, 1.0] |

### TestSummaryStats (6 tests)

Tests the summary statistics dictionary.

| Test | Asserts |
|---|---|
| `test_returns_dict` | Return type is `dict` |
| `test_expected_keys` | Dict contains exactly the 7 required keys |
| `test_perfect_accuracy` | `accuracy == 1.0` for perfect predictions |
| `test_imperfect_accuracy_below_1` | `accuracy < 1.0` for imperfect predictions |
| `test_all_values_between_0_and_1` | All 7 values are in [0.0, 1.0] |
| `test_binary_accuracy_correct` | Accuracy matches manual calculation for a binary 2-class case |

### TestMisclassifiedDf (6 tests)

Tests the misclassified document table.

| Test | Asserts |
|---|---|
| `test_returns_dataframe` | Return type is `pd.DataFrame` |
| `test_expected_columns` | Columns are exactly `["text_preview", "actual", "predicted"]` |
| `test_perfect_predictions_empty` | Empty DataFrame (0 rows) for perfect predictions |
| `test_misclassified_count_correct` | Row count matches number of misclassified examples |
| `test_actual_predicted_are_class_names` | `actual` and `predicted` values are valid class name strings |
| `test_long_text_truncated` | `text_preview` values are at most 122 characters (`120 + "…"`) |

---

## Running Tests

**All tests:**
```bash
pytest tests/ -v
```

**Single file:**
```bash
pytest tests/test_classifier.py -v
pytest tests/test_metrics.py -v
```

**Single class:**
```bash
pytest tests/test_classifier.py::TestTrain -v
```

**Expected output:**
```
collected 52 items

tests/test_classifier.py::TestBuildPipeline::test_valid_algorithm PASSED
tests/test_classifier.py::TestBuildPipeline::test_invalid_algorithm_raises PASSED
...
tests/test_metrics.py::TestMisclassifiedDf::test_long_text_truncated PASSED

52 passed in X.XXs
```

---

## CI Integration

The CI workflow (`.github/workflows/tests.yml`) runs the full test suite automatically on every push and pull request:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

---

## Coverage Notes

**Tested:**
- `src/classifier.py` — all public functions; binary and multiclass coefficient extraction; both algorithms; error cases (invalid algorithm, empty string input)
- `src/metrics.py` — all four functions; perfect and imperfect prediction cases; edge cases (empty misclassified table, long text truncation)

**Not tested (intentional):**
- `app/main.py` — Streamlit UI layer; requires browser interaction; validated manually
- `src/vectorizer.py:build_tfidf()` — exercised indirectly through every `train()` call
- `src/vectorizer.py:encode_labels()` / `decode_labels()` — dead code; not called by application code
