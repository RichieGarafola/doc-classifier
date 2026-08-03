"""
Text Document Classifier
Train a TF-IDF + scikit-learn classifier on labeled text, evaluate with
cross-validation and a confusion matrix, and classify new documents in real time.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.classifier import ALGORITHMS, TrainResult, predict_single, top_features_per_class, train
from src.metrics import confusion_matrix_df, misclassified_df, per_class_report, summary_stats

st.set_page_config(page_title="Document Classifier", page_icon="📄", layout="wide")
st.title("📄 Document Classifier")
st.caption(
    "Train a TF-IDF + ML classifier on any labeled text dataset — "
    "evaluate accuracy, inspect top features, and classify new documents. No API required."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Data Source")
    upload = st.file_uploader("Upload labeled CSV", type="csv",
                               help="Must have a text column and a label column.")
    default_path = (
        Path(__file__).parent.parent / "data" / "sample" / "labeled_documents.csv"
    )

    if upload:
        raw_df = pd.read_csv(upload, dtype=str).fillna("")
        st.success(f"{len(raw_df):,} rows loaded")
    elif default_path.exists():
        raw_df = pd.read_csv(default_path, dtype=str).fillna("")
        st.info(f"Sample data — {len(raw_df):,} rows")
    else:
        st.warning("No data. Upload a CSV or run scripts/generate_sample_data.py")
        st.stop()

    all_cols = list(raw_df.columns)
    text_col = st.selectbox(
        "Text column",
        all_cols,
        index=all_cols.index("text") if "text" in all_cols else 0,
    )
    label_col = st.selectbox(
        "Label column",
        [c for c in all_cols if c != text_col],
        index=([c for c in all_cols if c != text_col].index("category")
               if "category" in all_cols else 0),
    )

    st.divider()
    st.subheader("Model")
    algorithm = st.selectbox("Algorithm", list(ALGORITHMS.keys()))
    test_size  = st.slider("Test split %", 10, 40, 20) / 100
    cv_folds   = st.slider("CV folds", 3, 10, 5)

    st.divider()
    st.subheader("TF-IDF")
    ngram_min    = st.slider("Min n-gram", 1, 3, 1)
    ngram_max    = st.slider("Max n-gram", 1, 3, 2)
    if ngram_min > ngram_max:
        ngram_max = ngram_min
    max_features = st.slider("Max features", 500, 20_000, 10_000, step=500)
    min_df       = st.slider("Min doc frequency", 1, 10, 2)
    top_n_feat   = st.slider("Top features per class", 5, 30, 15)

    train_btn = st.button("Train Model", type="primary", use_container_width=True)


# ── Guard ─────────────────────────────────────────────────────────────────────

if text_col not in raw_df.columns or label_col not in raw_df.columns:
    st.error("Selected columns not found in data.")
    st.stop()

texts  = raw_df[text_col].tolist()
labels = raw_df[label_col].tolist()
label_counts = raw_df[label_col].value_counts()

# ── Train (cached in session state) ──────────────────────────────────────────

if train_btn or "result" not in st.session_state:
    with st.spinner(f"Training {algorithm}…"):
        try:
            result: TrainResult = train(
                texts, labels,
                algorithm=algorithm,
                test_size=test_size,
                cv_folds=cv_folds,
                ngram_range=(ngram_min, ngram_max),
                max_features=max_features,
                min_df=min_df,
            )
            st.session_state["result"] = result
            st.session_state["texts_test"] = [
                texts[i] for i in range(len(texts))
                if i >= int(len(texts) * (1 - test_size))
            ]
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

result: TrainResult = st.session_state["result"]

# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "Overview", "Confusion Matrix", "Per-Class Metrics",
    "Top Features", "Misclassified", "Predict"
])

# ┌─ Tab 1: Overview ───────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Training Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Accuracy", f"{result.test_accuracy:.1%}")
    col2.metric("CV Mean Accuracy", f"{result.cv_mean:.1%}")
    col3.metric("CV Std Dev", f"±{result.cv_std:.3f}")
    col4.metric("Classes", len(result.classes))

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Cross-Validation Scores**")
        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(result.cv_scores))],
            "Accuracy": result.cv_scores,
        })
        fig_cv = px.bar(
            cv_df, x="Fold", y="Accuracy",
            title=f"{cv_folds}-Fold CV Accuracy",
            range_y=[0, 1],
            color="Accuracy",
            color_continuous_scale="Blues",
        )
        fig_cv.add_hline(y=result.cv_mean, line_dash="dash",
                         annotation_text=f"Mean: {result.cv_mean:.3f}")
        fig_cv.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cv, use_container_width=True)

    with col_b:
        st.markdown("**Class Distribution**")
        dist_df = label_counts.reset_index()
        dist_df.columns = ["Category", "Count"]
        fig_dist = px.bar(dist_df, x="Count", y="Category", orientation="h",
                          title="Documents per Class",
                          color="Count", color_continuous_scale="Teal")
        fig_dist.update_layout(yaxis={"autorange": "reversed"},
                                coloraxis_showscale=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    stats = summary_stats(result.y_test, result.y_pred, result.classes)
    st.markdown("**Weighted Averages (test set)**")
    stats_df = pd.DataFrame([{
        "Metric": k.replace("_", " ").title(), "Value": f"{v:.4f}"
    } for k, v in stats.items()])
    st.dataframe(stats_df, use_container_width=True, hide_index=True)


# ┌─ Tab 2: Confusion Matrix ───────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Confusion Matrix")

    cm_df = confusion_matrix_df(result.y_test, result.y_pred, result.classes)
    z_vals  = cm_df.values.tolist()
    x_labels = [c.replace("Predicted: ", "") for c in cm_df.columns]
    y_labels = [c.replace("Actual: ", "") for c in cm_df.index]

    fig_cm = ff.create_annotated_heatmap(
        z=z_vals,
        x=x_labels,
        y=y_labels,
        colorscale="Blues",
        showscale=True,
    )
    fig_cm.update_layout(
        title="Confusion Matrix — Test Set",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=450,
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    with st.expander("Raw confusion matrix"):
        st.dataframe(cm_df, use_container_width=True)


# ┌─ Tab 3: Per-Class Metrics ──────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Per-Class Precision / Recall / F1")

    pcr = per_class_report(result.y_test, result.y_pred, result.classes)
    st.dataframe(pcr, use_container_width=True, hide_index=True)

    fig_pcr = px.bar(
        pcr.melt(id_vars="class", value_vars=["precision", "recall", "f1_score"],
                 var_name="metric", value_name="score"),
        x="class", y="score", color="metric", barmode="group",
        range_y=[0, 1.05],
        title="Precision / Recall / F1 by Class",
        labels={"class": "Class", "score": "Score"},
        color_discrete_sequence=["#4C78A8", "#F58518", "#72B7B2"],
    )
    st.plotly_chart(fig_pcr, use_container_width=True)


# ┌─ Tab 4: Top Features ───────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Most Discriminative Terms per Class")
    st.caption("Logistic Regression coefficient weights — higher = stronger signal for that class.")

    feat_df = top_features_per_class(result, top_n=top_n_feat)
    if feat_df.empty:
        st.info("Feature weights not available for Multinomial Naive Bayes. Switch to Logistic Regression.")
    else:
        classes_to_show = st.multiselect(
            "Classes to display",
            result.classes,
            default=result.classes[:4] if len(result.classes) > 4 else result.classes,
        )
        for cls in classes_to_show:
            sub = feat_df[feat_df["class"] == cls]
            fig_f = px.bar(
                sub, x="weight", y="term", orientation="h",
                title=f"Top Terms — {cls}",
                labels={"weight": "LR Coefficient", "term": "Term"},
                color="weight", color_continuous_scale="RdBu",
                range_color=[-sub["weight"].abs().max(), sub["weight"].abs().max()],
            )
            fig_f.update_layout(yaxis={"autorange": "reversed"},
                                 coloraxis_showscale=False, height=380)
            st.plotly_chart(fig_f, use_container_width=True)


# ┌─ Tab 5: Misclassified ──────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Misclassified Documents")

    test_texts = [
        texts[i] for i in range(len(texts) - len(result.y_test), len(texts))
    ]
    mis_df = misclassified_df(test_texts, result.y_test, result.y_pred, result.classes)

    n_wrong = len(mis_df)
    n_total = len(result.y_test)
    st.metric("Misclassified", f"{n_wrong} / {n_total}",
              delta=f"{n_wrong/n_total:.1%} error rate", delta_color="inverse")

    if mis_df.empty:
        st.success("No misclassifications on the test set.")
    else:
        filter_class = st.selectbox("Filter by actual class", ["All"] + result.classes)
        display = mis_df if filter_class == "All" else mis_df[mis_df["actual"] == filter_class]
        st.dataframe(display, use_container_width=True, hide_index=True)


# ┌─ Tab 6: Predict ────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Classify a New Document")

    sample_texts = {
        "RFP": "Offerors are invited to submit proposals for professional services including technical approach and price.",
        "Invoice": "Invoice INV-2024-0099 for direct labor hours at negotiated rates during the billing period.",
        "SOW": "The contractor shall provide program management support and deliver monthly status reports.",
        "Contract Modification": "This modification extends the period of performance by 6 months and obligates additional funds.",
        "Compliance Document": "This System Security Plan describes NIST SP 800-53 controls implemented for FedRAMP authorization.",
        "Deliverable Report": "This Monthly Status Report covers activities completed deliverables submitted and risks identified.",
        "(blank)": "",
    }

    prefill = st.selectbox("Load a sample text", list(sample_texts.keys()))
    user_text = st.text_area("Document text", value=sample_texts[prefill], height=150)

    if st.button("Classify", type="primary"):
        if not user_text.strip():
            st.warning("Enter some text to classify.")
        else:
            out = predict_single(result.pipeline, result.label_encoder, user_text)
            st.success(f"**Predicted category: {out['predicted_label']}**")

            if out["probabilities"]:
                prob_df = (
                    pd.DataFrame(list(out["probabilities"].items()),
                                 columns=["Category", "Probability"])
                    .sort_values("Probability", ascending=False)
                )
                fig_p = px.bar(
                    prob_df, x="Probability", y="Category", orientation="h",
                    range_x=[0, 1],
                    title="Class Probabilities",
                    color="Probability", color_continuous_scale="Blues",
                )
                fig_p.update_layout(yaxis={"autorange": "reversed"},
                                     coloraxis_showscale=False)
                st.plotly_chart(fig_p, use_container_width=True)
