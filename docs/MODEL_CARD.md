# Model Card — Text Document Classifier

## Project Objective

The Text Document Classifier is an end-to-end supervised machine learning workflow demonstrating explainable text classification using modern Python tooling. It covers every stage of a production ML pipeline — feature engineering, model training, cross-validation, prediction, and error analysis — within a single self-contained application.

**Explainability:** Logistic Regression coefficient weights are extracted per class and ranked, giving practitioners a direct view of which terms drive each prediction. The misclassified document table exposes model failures transparently rather than hiding them behind an aggregate accuracy score.

**Reproducibility:** A fixed `random_state=42` seed, stratified splitting, and a declarative `ALGORITHMS` configuration dict ensure that the same inputs produce the same outputs across runs. All dependencies are pinned in `requirements.txt`.

**Maintainability:** Business logic is isolated in three focused modules (`classifier.py`, `metrics.py`, `vectorizer.py`), each covered by a dedicated unit test file. The 52-test suite and CI workflow catch regressions before they reach the UI.

**Production-oriented engineering:** The sklearn `Pipeline` wraps vectorization and classification into a single estimable unit — the correct pattern for deployment, not just demonstration. The `TrainResult` dataclass structures all outputs with explicit types, making downstream consumers predictable and testable.

The tool allows any team to bring a labeled CSV of their own documents and produce a working classifier in minutes — without an external API, a GPU, or a large language model.

---

## Classification Task

**Task type:** Multi-class text classification (single-label; each document belongs to exactly one class)

**Input:** Free-text string — a document or document excerpt

**Output:** Predicted class label; optional per-class confidence probabilities (Logistic Regression only)

**Label source:** User-supplied labeled CSV. Labels are arbitrary strings; the classifier learns whatever categories the user defines. The included sample uses six government contracting document types.

**Minimum training requirements:**
- At least 2 unique classes
- At least 2 examples per class (required by `min_df=2` and stratified splitting)
- Reasonable vocabulary distinction between classes

---

## Supported Algorithms

Two algorithms are available and selectable from the Streamlit sidebar:

### Logistic Regression

| Parameter | Value | Rationale |
|---|---|---|
| `max_iter` | 1000 | Prevents convergence warnings on large TF-IDF vocabularies |
| `C` | 1.0 | Moderate L2 regularization (sklearn default) |
| `solver` | `"lbfgs"` | Efficient for dense multinomial problems; native multiclass support |

**Capabilities:** Supports `predict_proba` (per-class confidence scores). Supports `coef_` extraction for feature importance visualization. Generally the better choice for interpretability.

### Multinomial Naive Bayes

| Parameter | Value | Rationale |
|---|---|---|
| `alpha` | 0.1 | Mild additive (Laplace) smoothing; tighter than the default of 1.0 |

**Capabilities:** Fast and effective for high-dimensional sparse text. Does not expose coefficient weights; the Top Features tab shows an info message when Naive Bayes is selected. Probabilities are available via `predict_proba` but are not surfaced in the current UI.

---

## TF-IDF Feature Engineering

Documents are converted to numeric feature vectors using `TfidfVectorizer` with the following configuration:

| Parameter | Value | Effect |
|---|---|---|
| `ngram_range` | `(1, 2)` | Unigrams and bigrams; captures phrases like "compliance document" |
| `max_features` | 10,000 | Caps vocabulary size; prevents memory issues on large corpora |
| `min_df` | 2 | Excludes terms appearing in fewer than 2 documents; removes hapax legomena |
| `max_df` | 0.95 | Excludes near-universal terms; complements stop word removal |
| `sublinear_tf` | `True` | Log-scales term frequency: `1 + log(tf)`; reduces influence of repetitive terms |
| `stop_words` | `"english"` | Removes common English function words |
| `strip_accents` | `"unicode"` | Normalizes accented characters before tokenization |
| Token pattern | `r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b"` | Minimum 2-character tokens starting with a letter; allows hyphenated terms |

The vectorizer and classifier are combined inside a single sklearn `Pipeline`, which ensures the vocabulary is always fitted on training data only — no test-set vocabulary leakage.

---

## Training Workflow

1. **Label encoding** — Class name strings are mapped to integers via `LabelEncoder`. The encoder is stored alongside the pipeline so predictions can be decoded back to class names.

2. **Stratified split** — The labeled dataset is divided into training (default 80%) and test (default 20%) sets, preserving class proportions in both splits.

3. **Pipeline fitting** — The sklearn `Pipeline` is fit on the training split. This simultaneously fits the TF-IDF vocabulary and trains the classifier on the resulting feature matrix.

4. **Cross-validation** — Stratified K-fold CV (default 5 folds) is run across the full dataset. Each fold independently re-fits the pipeline (vectorization + training) and evaluates on the held-out fold. This produces per-fold accuracy scores and a mean ± std dev summary.

5. **Result packaging** — All outputs (fitted pipeline, encoder, splits, predictions, CV scores, feature names) are collected into a `TrainResult` dataclass and returned to the UI layer.

---

## Evaluation Methodology

The application reports four categories of evaluation output:

### Test-Set Accuracy
Fraction of held-out test documents classified correctly by the trained pipeline. This is a point estimate — variance can be high for small test sets.

### Cross-Validation Scores
Per-fold and mean accuracy from Stratified K-fold CV. CV mean is typically a more stable generalization estimate than test-set accuracy, particularly for datasets with fewer than 500 documents.

**Known limitation:** CV is run on the full dataset (not the training split alone), meaning test-set documents appear in CV folds. For small datasets this introduces a slight upward bias in CV scores. See [Known Limitations](#known-limitations).

### Confusion Matrix
A class × class count table showing where errors occur. Rows represent ground-truth classes; columns represent predicted classes. Diagonal cells are correct predictions; off-diagonal cells are errors. The confusion matrix is more informative than accuracy alone when class distributions are uneven.

### Per-Class Precision / Recall / F1
- **Precision:** Of all documents predicted as class X, what fraction actually are class X?
- **Recall:** Of all actual class X documents, what fraction were predicted as class X?
- **F1:** Harmonic mean of precision and recall. Balances both metrics.

All scores use `zero_division=0`, meaning classes with no predicted or actual examples receive a score of 0 rather than raising an error.

---

## Performance Characteristics

Performance depends heavily on dataset characteristics. The following observations apply to the included sample dataset (~380 documents, 6 categories):

- Both Logistic Regression and Multinomial Naive Bayes typically achieve **>95% test accuracy** on the sample data because the vocabulary is intentionally distinctive across categories.
- On real-world document datasets, expect **80–95% accuracy** for well-separated categories with sufficient training data (>50 examples per class).
- **Low recall** on a specific class typically indicates too few training examples or vocabulary overlap with other classes.
- **Low precision** on a specific class often indicates that class borrows vocabulary from others (e.g., "Invoice" documents that mention "deliverables").

The Top Features visualization is the primary diagnostic tool for understanding model behavior. If discriminative terms for a class include stop-word-like or generic terms, try reducing `max_df` or increasing `min_df`.

---

## Known Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| No model persistence | Trained pipeline is lost when the browser tab closes | Export to disk (joblib) is a planned enhancement |
| CV on full dataset | Slight upward bias in CV mean for small datasets | Acceptable for workflow demonstration; noted in results |
| No hyperparameter search | Default `C=1.0` / `alpha=0.1` may not be optimal | Try multiple values manually; grid search is a planned enhancement |
| `min_df=2` minimum | Empty vocabulary error if every term appears exactly once | Set `min_df=1` in the sidebar for very small datasets |
| Top Features for Logistic Regression only | No coefficient weights available for Naive Bayes | Use Logistic Regression if interpretability is required |
| TF-IDF + linear classifiers assume distinctive vocabulary | If classes share vocabulary, accuracy degrades | Consider a transformer-based model for subtle distinctions |
| No class imbalance handling | Minority classes may have low recall | Use class weights (planned) or oversample minority classes |
| Stratified split requires ≥2 examples per class | Very small classes cause a stratification error | Ensure all classes have at least 2 labeled examples |

---

## Intended Use Cases

This tool is designed for:

- **Document routing and triage** — Automatically classify incoming documents by type (contract, invoice, compliance filing) to route them to the correct team or workflow.
- **Model prototyping** — Quickly assess whether a linear text classifier is sufficient for a classification task before investing in a more complex model.
- **Analytics and reporting** — Categorize a corpus of documents by type to generate summary counts, identify patterns, or support audits.
- **Training data validation** — Use the confusion matrix and misclassified document table to identify mislabeled training examples or ambiguous category definitions.
- **Portfolio demonstration** — Illustrate the full supervised learning cycle (data ingestion → feature engineering → training → evaluation → prediction) in a self-contained, interactive application.

**Not intended for:**
- Production document classification without human review of model errors
- Tasks requiring semantic understanding rather than vocabulary-based discrimination
- Documents in languages other than English (stop word list is English-only)
- Highly sensitive or personally identifiable document classification without appropriate data handling

---

## Future Enhancement Opportunities

- **Model persistence** — Export the fitted pipeline to disk (`joblib.dump`) and reload on the next session, eliminating the need to retrain.
- **Hyperparameter grid search** — Expose a range of `C` values (Logistic Regression) and `alpha` values (Naive Bayes) and select the best via nested CV.
- **Additional algorithms** — Linear SVC (often the strongest linear text classifier), Random Forest, Gradient Boosting.
- **Class imbalance handling** — `class_weight="balanced"` for Logistic Regression; SMOTE-style oversampling.
- **Batch prediction** — Upload an unlabeled CSV and download a prediction column.
- **SHAP values** — Per-prediction explainability that goes beyond aggregate coefficient weights.
- **Multilingual support** — Swap English stop words for a language-specific list; support Unicode normalization for non-Latin scripts.
- **Confidence thresholding** — Flag low-confidence predictions (max probability below a threshold) for human review rather than automatic routing.

---

*Portfolio governance artifact — Model Card for P18 Text Document Classifier.*
