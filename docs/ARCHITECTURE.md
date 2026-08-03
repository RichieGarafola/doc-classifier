# Architecture — Text Document Classifier

## Overview

The Text Document Classifier is a four-module Streamlit application that wraps a scikit-learn supervised learning pipeline. Users supply a labeled CSV; the application fits a TF-IDF + classifier pipeline, evaluates it with cross-validation and test-set metrics, and provides a live prediction panel for new text. All computation is local — no external APIs.

The codebase is organized into a presentation layer (`app/`), a library layer (`src/`), and supporting assets (`data/`, `scripts/`, `tests/`). The library layer exposes a stable public API via `src/__init__.py`.

---

## Module Map

```
doc-classifier/
├── app/
│   └── main.py                  # Streamlit UI — all tab rendering and user interaction
├── src/
│   ├── __init__.py              # Public API (11 exports)
│   ├── classifier.py            # Training pipeline, TrainResult dataclass, prediction
│   ├── metrics.py               # Evaluation DataFrames and summary statistics
│   └── vectorizer.py            # TF-IDF configuration helper
├── data/
│   └── sample/
│       └── labeled_documents.csv  # ~380-row labeled sample (6 categories)
├── scripts/
│   └── generate_sample_data.py  # Regenerates the sample CSV
└── tests/
    ├── test_classifier.py       # 26 tests — classifier module
    └── test_metrics.py          # 26 tests — metrics module
```

---

## Architecture Diagram

```
User uploads CSV
        │
        ▼
app/main.py (Streamlit)
        │
        ├─ sidebar config ──▶ algorithm, test_size, cv_folds,
        │                      ngram_range, max_features, min_df
        │
        ▼
src/classifier.py:train(texts, labels, algorithm, ...)
        │
        ├─ LabelEncoder.fit_transform(labels)
        │         ──▶ y_encoded (integer targets)
        │         ──▶ classes (sorted string list)
        │
        ├─ train_test_split (stratified, random_state=42)
        │         ──▶ X_train, X_test, y_train, y_test
        │
        ├─ build_pipeline(algorithm, ngram_range, max_features, min_df, max_df)
        │         ──▶ Pipeline([("tfidf", TfidfVectorizer), ("clf", classifier)])
        │
        ├─ Pipeline.fit(X_train, y_train)
        │         ├─ TfidfVectorizer.fit_transform(X_train) ──▶ sparse matrix
        │         └─ Classifier.fit(sparse_matrix, y_train)
        │
        ├─ Pipeline.predict(X_test) ──▶ y_pred (integer encoded)
        │
        ├─ cross_val_score(pipeline, all_texts, y_all,
        │       cv=StratifiedKFold(cv_folds), scoring="accuracy")
        │         ──▶ cv_scores (array of per-fold accuracy)
        │
        └─ TrainResult (13 fields)
                 │
                 ├─ metrics.py:confusion_matrix_df(y_test, y_pred, classes)
                 │         ──▶ DataFrame (rows "Actual: X", cols "Predicted: X")
                 │
                 ├─ metrics.py:per_class_report(y_test, y_pred, classes)
                 │         ──▶ DataFrame (class, precision, recall, f1_score, support)
                 │
                 ├─ metrics.py:summary_stats(y_test, y_pred, classes)
                 │         ──▶ dict (7 keys: accuracy, macro/weighted P/R/F1)
                 │
                 ├─ metrics.py:misclassified_df(texts_test, y_test, y_pred, classes)
                 │         ──▶ DataFrame (text_preview, actual, predicted)
                 │
                 └─ classifier.py:top_features_per_class(result, top_n)
                           ──▶ DataFrame (class, rank, term, weight)
                               [empty schema for Naive Bayes]
```

---

## sklearn Pipeline Pattern

`build_pipeline()` wraps the TF-IDF vectorizer and classifier inside a single `Pipeline`:

```python
Pipeline([("tfidf", TfidfVectorizer(...)), ("clf", clf)])
```

**Why Pipeline instead of separate steps:**
- A single `Pipeline.fit()` call transforms and trains atomically — no risk of fitting on transform output from the wrong split.
- `cross_val_score` applies the entire pipeline (vectorize → train → evaluate) in each CV fold. If vectorization were done outside the pipeline, test-fold tokens would leak into the vocabulary fitted on training data, inflating CV scores.
- `Pipeline.predict()` accepts raw text strings directly. Callers never handle sparse matrices.

---

## Training Flow

```
train(texts, labels, algorithm="Logistic Regression", test_size=0.2,
      random_state=42, cv_folds=5, ngram_range=(1,2), max_features=10_000,
      min_df=2, max_df=0.95)
```

1. **Label encoding** — `LabelEncoder.fit_transform(labels)` maps each unique class name to a stable integer. The encoder is stored in `TrainResult.label_encoder` so predictions can be decoded.

2. **Stratified split** — `train_test_split(..., stratify=y_encoded)` preserves class proportions in both splits. `random_state=42` makes the split reproducible.

3. **Pipeline construction** — `build_pipeline()` instantiates the TF-IDF vectorizer and the selected algorithm from the `ALGORITHMS` dict, then wraps them in a `Pipeline`.

4. **Fitting** — `pipeline.fit(X_train, y_train)` fits the vectorizer vocabulary on the training texts and trains the classifier on the resulting feature matrix.

5. **Test evaluation** — `pipeline.predict(X_test)` produces encoded integer predictions. Test accuracy is `(y_pred == y_test).mean()`.

6. **Cross-validation** — `cross_val_score` runs `StratifiedKFold(n_splits=cv_folds)` on the **full dataset** (X_all, y_all). Each fold re-fits the entire pipeline internally. See [CV Methodology](#stratified-cv-methodology) for the known limitation this introduces.

7. **Feature names** — `pipeline.named_steps["tfidf"].get_feature_names_out()` extracts the fitted vocabulary for use in `top_features_per_class()`.

8. **Return** — `TrainResult` packages all outputs for downstream consumers (metrics module and Streamlit tabs).

---

## TrainResult Dataclass

`TrainResult` is a frozen dataclass with 13 typed fields returned by `train()`.

| Field | Type | Description |
|---|---|---|
| `pipeline` | `Pipeline` | Fitted sklearn pipeline (TF-IDF + classifier). Pass to `predict_single()`. |
| `label_encoder` | `LabelEncoder` | Fitted encoder; maps integers ↔ class name strings. |
| `classes` | `list[str]` | Sorted list of unique class labels as strings. |
| `X_train` | sparse matrix | TF-IDF feature matrix for the training split. |
| `X_test` | sparse matrix | TF-IDF feature matrix for the test split. |
| `y_train` | `list[int]` | Integer-encoded training labels. |
| `y_test` | `list[int]` | Integer-encoded ground-truth test labels. |
| `y_pred` | `list[int]` | Integer-encoded predictions on the test split. |
| `test_accuracy` | `float` | Fraction of test documents correctly classified. Range: [0.0, 1.0]. |
| `cv_scores` | `list[float]` | Per-fold accuracy from StratifiedKFold cross-validation. Length = `cv_folds`. |
| `cv_mean` | `float` | Mean of `cv_scores`. |
| `cv_std` | `float` | Standard deviation of `cv_scores`. |
| `feature_names` | `list[str]` | Vocabulary terms from the fitted TF-IDF vectorizer. |

---

## Prediction Flow

```python
predict_single(pipeline, le, text) -> dict
```

1. `pipeline.predict([text])` — vectorizes the input and returns an integer-encoded label.
2. `le.inverse_transform(y_encoded)` — decodes the integer to a class name string.
3. For Logistic Regression: `pipeline.predict_proba([text])` — returns a 1×n_classes probability array. The result dict includes `"probabilities": {class_name: prob, ...}`.
4. For Multinomial Naive Bayes: `predict_proba` is available but is not called by the current implementation; `"probabilities"` is set to `None`.

**Return schema:**
```python
{
    "predicted_label": str,         # decoded class name
    "probabilities": dict | None    # {class_name: float} for LR; None for NB
}
```

---

## Metrics Module

`metrics.py` provides four pure functions that compute evaluation outputs from raw prediction arrays. All functions accept integer-encoded labels and a `classes` list for display.

| Function | Output | Key Notes |
|---|---|---|
| `confusion_matrix_df(y_true, y_pred, classes)` | DataFrame | Rows = `"Actual: X"`, columns = `"Predicted: X"`, values = integer counts |
| `per_class_report(y_true, y_pred, classes)` | DataFrame | Columns: `class`, `precision`, `recall`, `f1_score`, `support`; `zero_division=0` |
| `summary_stats(y_true, y_pred, classes)` | dict | 7 keys: `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `weighted_precision`, `weighted_recall`, `weighted_f1` |
| `misclassified_df(texts, y_true, y_pred, classes)` | DataFrame | Columns: `text_preview` (≤122 chars), `actual`, `predicted`; empty DataFrame for perfect predictions |

The metrics module has no dependency on the classifier module and can be used independently with any integer prediction arrays.

---

## Coefficient Extraction

`top_features_per_class(result, top_n=15)` extracts discriminative terms from Logistic Regression coefficients.

**Multiclass case (n_classes > 2):**
`clf.coef_` shape is `(n_classes, n_features)`. Each row is the coefficient vector for one class. The top-`top_n` features by weight are extracted per row.

**Binary case (n_classes == 2):**
sklearn stores a single coefficient row for the positive class. `coef_.shape[0] == 1` triggers:

```python
coef = np.vstack([-coef, coef])
```

This reconstructs two rows — one for each class — matching the multiclass format. The negative class coefficients are the inverse of the positive class coefficients, which is correct for binary logistic regression.

**Naive Bayes:**
`MultinomialNB` does not expose a `coef_` attribute. The function returns an empty DataFrame with the schema `columns=["class", "rank", "term", "weight"]` so the UI can detect the empty result and display an info message.

---

## Stratified CV Methodology

Cross-validation uses `StratifiedKFold` with `cv_folds` splits. The fit call is:

```python
cross_val_score(pipeline, X_all, y_all, cv=StratifiedKFold(n_splits=cv_folds))
```

where `X_all` and `y_all` are the **complete dataset** — not the training split alone.

**Known limitation:** The test-split rows are included in the CV folds. This means the CV mean accuracy and the held-out test accuracy are not fully independent estimates. On small datasets (<500 rows), this typically produces a slight upward bias in CV scores. The limitation is documented in the README and is acceptable for the portfolio use case, where dataset sizes are expected to be small and the goal is workflow demonstration rather than production-grade model selection.

---

## Algorithm Configuration

Supported algorithms are defined in the `ALGORITHMS` module-level dict in `classifier.py`:

| Key | Class | Parameters |
|---|---|---|
| `"Logistic Regression"` | `LogisticRegression` | `max_iter=1000, C=1.0, solver="lbfgs"` |
| `"Multinomial Naive Bayes"` | `MultinomialNB` | `alpha=0.1` |

**Parameter rationale:**
- `max_iter=1000` — prevents convergence warnings on mid-sized vocabularies; the default of 100 is too low for TF-IDF feature spaces.
- `C=1.0` — sklearn default; moderate L2 regularization.
- `solver="lbfgs"` — efficient for dense multinomial problems; supports multiclass natively.
- `alpha=0.1` — mild additive smoothing; tighter than the sklearn default of 1.0, reducing over-smoothing on clean text data.

**sklearn 1.5+ compatibility:** The `multi_class` parameter was removed from `LogisticRegression` in sklearn 1.5. It is absent from the `ALGORITHMS` dict — no migration action required.

---

## Public API

`src/__init__.py` exports 11 symbols:

```python
from src import (
    # classifier
    TrainResult, train, predict_single, top_features_per_class,
    # metrics
    confusion_matrix_df, per_class_report, summary_stats, misclassified_df,
    # vectorizer
    build_tfidf, encode_labels, decode_labels,
)
```

`app/main.py` imports directly from submodules rather than from `src` directly, which is equivalent. The `__init__.py` exports exist for downstream callers and testing convenience.

---

## Dead Code

`vectorizer.py` contains two functions that are not called anywhere in the application:

- `encode_labels(labels)` — wraps `LabelEncoder.fit_transform`; returns `(le, encoded)` tuple
- `decode_labels(le, encoded)` — wraps `le.inverse_transform`

These were likely written before the current `LabelEncoder` usage was integrated directly into `classifier.py:train()`. They are retained as-is because the source is frozen. They are documented here for completeness and to prevent confusion during code review.

---

## Design Decisions and Trade-offs

| Decision | Alternative Considered | Reason for Choice |
|---|---|---|
| sklearn `Pipeline` for vectorizer + classifier | Separate `TfidfVectorizer` and classifier calls | Pipeline enables correct CV (no leakage), clean predict interface, and one-call training |
| `LabelEncoder` outside the pipeline | `LabelBinarizer` or `OrdinalEncoder` inside pipeline | Keeps encoder accessible for decoding in predict and metrics; pipelines can't easily expose the encoder object post-fit |
| CV on full dataset | CV on training split only | More folds available; stable estimates for small datasets; limitation explicitly documented |
| `sublinear_tf=True` | Raw term frequency | Log-scaling reduces influence of high-frequency terms without stop-word dependency |
| `ngram_range=(1,2)` as default | Unigrams only | Bigrams capture phrase patterns ("compliance document", "deliverable report") that improve classification of domain-specific text |
| `min_df=2` as default | `min_df=1` | Excludes hapax legomena (terms appearing once) that have high TF-IDF scores but low discriminative power |
| Empty `screenshots/` directory | Inline screenshots in README | Keeps repository clean; screenshots captured in a dedicated session after all 20 projects are modernized |
