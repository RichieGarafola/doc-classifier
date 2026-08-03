# Data Dictionary — Text Document Classifier

## Overview

This document describes all data structures produced by the `src/` library layer: the `TrainResult` dataclass, function output schemas, and the sample CSV format. All DataFrame columns are lowercase snake_case. All float values are standard Python `float` (64-bit).

---

## TrainResult Dataclass

`TrainResult` is returned by `src.classifier.train()`. It packages all outputs of a single training run for downstream consumption by the metrics module and the Streamlit UI.

```python
from src import TrainResult
```

| Field | Type | Range / Values | Description |
|---|---|---|---|
| `pipeline` | `sklearn.pipeline.Pipeline` | — | Fitted pipeline containing TF-IDF vectorizer and classifier. Pass to `predict_single()`. |
| `label_encoder` | `sklearn.preprocessing.LabelEncoder` | — | Fitted encoder; maps class name strings ↔ integer indices. Used to decode integer predictions. |
| `classes` | `list[str]` | non-empty | Sorted list of unique class label strings. Order matches `label_encoder.classes_`. |
| `X_train` | `scipy.sparse` matrix | — | TF-IDF feature matrix for the training split. Shape: `(n_train, n_features)`. |
| `X_test` | `scipy.sparse` matrix | — | TF-IDF feature matrix for the test split. Shape: `(n_test, n_features)`. |
| `y_train` | `list[int]` | `[0, n_classes)` | Integer-encoded ground-truth labels for the training split. |
| `y_test` | `list[int]` | `[0, n_classes)` | Integer-encoded ground-truth labels for the test split. |
| `y_pred` | `list[int]` | `[0, n_classes)` | Integer-encoded predictions on the test split. Same length as `y_test`. |
| `test_accuracy` | `float` | [0.0, 1.0] | Fraction of test documents where `y_pred == y_test`. |
| `cv_scores` | `list[float]` | [0.0, 1.0] | Per-fold accuracy from StratifiedKFold CV. Length equals `cv_folds` parameter. |
| `cv_mean` | `float` | [0.0, 1.0] | Arithmetic mean of `cv_scores`. |
| `cv_std` | `float` | ≥ 0.0 | Standard deviation of `cv_scores`. |
| `feature_names` | `list[str]` | non-empty | TF-IDF vocabulary terms as strings. Length equals `max_features` or vocabulary size (whichever is smaller). |

---

## `confusion_matrix_df()` Output

```python
from src import confusion_matrix_df
df = confusion_matrix_df(y_true, y_pred, classes)
```

| Property | Value |
|---|---|
| Type | `pd.DataFrame` |
| Shape | `(n_classes, n_classes)` |
| Index (rows) | `["Actual: class_0", "Actual: class_1", ...]` — prefixed with `"Actual: "` |
| Columns | `["Predicted: class_0", "Predicted: class_1", ...]` — prefixed with `"Predicted: "` |
| Cell values | `int` — count of documents with the row's actual class and column's predicted class |
| Perfect prediction | Diagonal values equal class support; all off-diagonal values are 0 |
| Row sums | Each row sum equals the number of test documents with that actual class |

**Example (3-class, 9 documents):**

|  | Predicted: A | Predicted: B | Predicted: C |
|---|---|---|---|
| Actual: A | 3 | 0 | 0 |
| Actual: B | 0 | 2 | 1 |
| Actual: C | 0 | 0 | 3 |

---

## `per_class_report()` Output

```python
from src import per_class_report
df = per_class_report(y_true, y_pred, classes)
```

| Column | Type | Range | Description |
|---|---|---|---|
| `class` | `str` | — | Class label name |
| `precision` | `float` | [0.0, 1.0] | Precision for this class: `TP / (TP + FP)`. `zero_division=0`. |
| `recall` | `float` | [0.0, 1.0] | Recall for this class: `TP / (TP + FN)`. `zero_division=0`. |
| `f1_score` | `float` | [0.0, 1.0] | Harmonic mean of precision and recall. `zero_division=0`. |
| `support` | `int` | ≥ 0 | Number of test documents with this actual class. |

**Row count:** Exactly `n_classes` rows, one per class. Sum of `support` equals total test document count.

---

## `summary_stats()` Output

```python
from src import summary_stats
stats = summary_stats(y_true, y_pred, classes)
```

Returns a `dict` with exactly 7 keys:

| Key | Type | Range | Description |
|---|---|---|---|
| `accuracy` | `float` | [0.0, 1.0] | Overall fraction of correctly classified documents |
| `macro_precision` | `float` | [0.0, 1.0] | Unweighted mean precision across all classes |
| `macro_recall` | `float` | [0.0, 1.0] | Unweighted mean recall across all classes |
| `macro_f1` | `float` | [0.0, 1.0] | Unweighted mean F1 across all classes |
| `weighted_precision` | `float` | [0.0, 1.0] | Support-weighted mean precision |
| `weighted_recall` | `float` | [0.0, 1.0] | Support-weighted mean recall (equals accuracy for single-label classification) |
| `weighted_f1` | `float` | [0.0, 1.0] | Support-weighted mean F1 |

**Note:** Macro averages treat all classes equally regardless of support. Weighted averages give more weight to larger classes. For balanced datasets the two will be approximately equal.

---

## `misclassified_df()` Output

```python
from src import misclassified_df
df = misclassified_df(texts, y_true, y_pred, classes)
```

| Column | Type | Max Length | Description |
|---|---|---|---|
| `text_preview` | `str` | 122 chars | First 120 characters of the document text, followed by `"…"` if truncated. Documents ≤120 chars are not truncated. |
| `actual` | `str` | — | Ground-truth class label string |
| `predicted` | `str` | — | Predicted class label string |

**Row count:** Number of documents where `y_pred != y_true`. Returns an empty DataFrame (0 rows, 3 columns) for perfect predictions.

**Note:** `texts` must be the test split texts (not all texts), aligned with `y_true` and `y_pred`.

---

## `predict_single()` Output

```python
from src import predict_single
result = predict_single(pipeline, le, text)
```

Returns a `dict` with exactly 2 keys:

| Key | Type | Values | Description |
|---|---|---|---|
| `predicted_label` | `str` | valid class name | Predicted class name for the input text |
| `probabilities` | `dict` or `None` | `{class_name: float}` or `None` | Per-class probability scores (Logistic Regression only). Keys are class names; values sum to approximately 1.0. `None` for Multinomial Naive Bayes. |

---

## `top_features_per_class()` Output

```python
from src import top_features_per_class
df = top_features_per_class(result, top_n=15)
```

| Column | Type | Values | Description |
|---|---|---|---|
| `class` | `str` | valid class name | Class label this feature is associated with |
| `rank` | `int` | 1 to `top_n` | Rank within the class (1 = highest weight) |
| `term` | `str` | — | TF-IDF vocabulary term (unigram or bigram) |
| `weight` | `float` | any real | Logistic Regression coefficient weight for this term in this class. Higher positive values → stronger association. |

**Row count:** At most `top_n × n_classes` rows (may be fewer if the vocabulary is small).

**Naive Bayes:** Returns an empty DataFrame with the schema above (0 rows, 4 columns). Check `df.empty` before rendering.

---

## Sample Data — `data/sample/labeled_documents.csv`

The sample dataset contains approximately 380 synthetic labeled contract documents across six categories. It is used as the default dataset in the application and is generated by `scripts/generate_sample_data.py`.

| Column | Type | Example | Description |
|---|---|---|---|
| `doc_id` | `str` | `"DOC-0001"` | Unique document identifier, zero-padded |
| `category` | `str` | `"RFP"` | Document class label |
| `text` | `str` | — | Free-text document content |

**Categories:**

| Category | Approx. Count | Description |
|---|---|---|
| `RFP` | ~65 | Request for Proposal |
| `SOW` | ~65 | Statement of Work |
| `Contract Modification` | ~60 | Amendment or modification to an existing contract |
| `Invoice` | ~65 | Billing or payment document |
| `Deliverable Report` | ~60 | Progress or completion report for a contract deliverable |
| `Compliance Document` | ~65 | Regulatory or audit compliance filing |

**Text generation:** Each document contains 3–5 phrases drawn from a category-specific phrase pool, joined with connectors ("as required by the contract", "per the approved project plan"). The vocabulary is intentionally distinctive across categories so that both supported classifiers achieve high accuracy on this sample.

---

## TF-IDF Configuration

`build_tfidf()` default parameters, as passed to `TfidfVectorizer`:

| Parameter | Default | Type | Description |
|---|---|---|---|
| `ngram_range` | `(1, 2)` | `tuple` | Include unigrams and bigrams |
| `max_features` | `10_000` | `int` | Vocabulary size cap |
| `min_df` | `2` | `int` | Minimum document frequency for a term to be included |
| `max_df` | `0.95` | `float` | Maximum document frequency fraction; excludes corpus-wide terms |
| `sublinear_tf` | `True` | `bool` | Apply `1 + log(tf)` scaling to term frequency |
| `strip_accents` | `"unicode"` | `str` | Normalize Unicode accents before tokenizing |
| `stop_words` | `"english"` | `str` | Remove English stop words |
| `token_pattern` | `r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b"` | `str` | Match tokens of ≥2 characters starting with a letter; hyphens allowed internally |
