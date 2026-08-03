# Engineering Decisions — Text Document Classifier

---

## ED-1: sklearn Pipeline wraps TF-IDF and classifier as a single unit

**Decision:** `build_pipeline()` returns a `Pipeline([("tfidf", TfidfVectorizer(...)), ("clf", clf)])`. Training calls `pipeline.fit(X_train, y_train)`, prediction calls `pipeline.predict([text])`, and cross-validation passes the entire pipeline to `cross_val_score`.

**Alternative considered:** Fit the TF-IDF vectorizer outside the pipeline, then pass the sparse matrix to the classifier separately.

**Why Pipeline was chosen:** When cross-validation is done outside a Pipeline, the vocabulary is fitted on the entire dataset before folds are split. This causes test-fold tokens to appear in the vocabulary that the model sees during training — a form of data leakage that inflates CV accuracy scores. Wrapping TF-IDF and classifier in a Pipeline ensures that `cross_val_score` refits the vectorizer independently in each fold, using only the training split's tokens to build the vocabulary. Prediction also simplifies: callers pass a raw text string to `pipeline.predict()` without managing sparse matrices.

---

## ED-2: LabelEncoder maps string class names to integer targets

**Decision:** `LabelEncoder.fit_transform(labels)` converts string class names to stable integer indices before training. The fitted encoder is stored in `TrainResult.label_encoder` and used in `predict_single()` to decode integer predictions back to class name strings.

**Alternative considered:** Pass string labels directly to the sklearn classifier.

**Why LabelEncoder was chosen:** scikit-learn classifiers (`LogisticRegression`, `MultinomialNB`) accept string labels via internal encoding when called through a Pipeline, but the encoding is opaque — there is no guaranteed access to the mapping afterward. Explicit `LabelEncoder` usage stores the mapping in the result object, making label decoding in `predict_single()` and feature importance extraction in `top_features_per_class()` transparent and testable. The classes list produced by `label_encoder.classes_` defines the display order for the confusion matrix and per-class report.

---

## ED-3: TrainResult as a frozen dataclass with 13 typed fields

**Decision:** `train()` returns a `TrainResult` — a `@dataclass(frozen=True)` with 13 explicitly typed fields covering the fitted pipeline, label encoder, encoded splits, prediction arrays, test accuracy, CV scores, and vocabulary.

**Alternative considered:** Return a dict from `train()`, or separate the results into multiple function calls.

**Why TrainResult was chosen:** A frozen dataclass makes the return schema explicit and statically analyzable. The type annotations on all 13 fields document the contract between `classifier.py` and its callers (`app/main.py`, `metrics.py`) in a way that is machine-checkable rather than documentary. Immutability (frozen=True) guarantees that downstream consumers do not mutate shared training state. Tests can assert on specific named fields without parsing dict keys. A dict or multiple-function approach would require callers to know which keys exist and risk `KeyError` if the schema changes.

---

## ED-4: StratifiedKFold cross-validation runs on the full dataset

**Decision:** `cross_val_score` receives `X_all` and `y_all` (the complete corpus before the train/test split). The `cv_mean` and `cv_std` in the result reflect fold performance across all documents.

**Alternative considered:** Run cross-validation only on the training split, leaving the test split entirely unseen.

**Why full-dataset CV was chosen:** The application is an exploratory tool, not a production evaluation harness. Running CV on the full dataset gives users a more stable accuracy estimate by using all available data in validation. The known limitation — that test-set documents appear in CV folds — is documented explicitly in ARCHITECTURE.md and MODEL_CARD.md. For a portfolio demonstration tool intended to classify tens to hundreds of documents, the leakage is acknowledged and acceptable; users who need strict evaluation are advised to run train-only CV on a held-out corpus.

---

## ED-5: Top features per class extracted from Logistic Regression coefficients only

**Decision:** `top_features_per_class()` reads `pipeline.named_steps["clf"].coef_` to rank vocabulary terms by their classification weight per class. For Multinomial Naive Bayes, the function returns an empty DataFrame with the correct column schema rather than extracting log-probability features.

**Alternative considered:** Extract and display feature importances for both algorithms.

**Why LR-only extraction was chosen:** Logistic Regression's `coef_` array maps directly onto the vocabulary with weights that are interpretable as relative importance per class — a positive weight means the term pushes toward that class, a negative weight pushes away. Naive Bayes's `feature_log_prob_` expresses log-prior-adjusted probabilities, which are harder to communicate to a general-purpose audience and less comparable across classes of different sizes. The decision to return an empty DataFrame for NB is defensive: it allows the UI to conditionally display the feature importance tab without a separate branch for "not applicable," and preserves the function's consistent return schema for tests.

**Coefficient extraction note:** For binary Logistic Regression (`coef_.shape[0] == 1`), the single coefficient row is duplicated with sign inversion to produce one row per class via `np.vstack([-coef_, coef_])`, matching the multiclass schema expected by downstream display code.

---

## ED-6: Two-algorithm design — Logistic Regression and Multinomial Naive Bayes

**Decision:** The `ALGORITHMS` dict in `classifier.py` offers exactly two choices: `LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")` and `MultinomialNB(alpha=0.1)`. No other algorithms are offered in the UI.

**Alternative considered:** Expose the full scikit-learn classifier suite (SVM, Random Forest, k-NN).

**Why two algorithms were chosen:** The pair represents two distinct approaches to text classification that are genuinely instructive for the use case: LR is a discriminative linear model that learns decision boundaries directly from the feature space, while NB is a generative model that learns word likelihoods per class. Comparing their accuracy on the same corpus is a meaningful educational exercise. Adding more algorithms would increase training time per "Compare All" run, add display complexity, and shift the focus from "which approach is better" to "which of N algorithms wins" — a question better answered by a dedicated AutoML tool. The two-algorithm constraint keeps the application fast and the comparison interpretable.

---

*This document is a repository artifact and may be committed to the published repository.*
