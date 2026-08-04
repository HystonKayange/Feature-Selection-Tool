"""Streamlit research UI for feature-selection comparisons.

Launch:
  feature-select ui
  streamlit run feature_selector/streamlit_app.py
"""

from __future__ import annotations

import io
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_diabetes, make_classification

from feature_selector.compare import DEFAULT_METHODS, compare_methods
from feature_selector.data import infer_task, load_dataset
from feature_selector.evaluate import evaluate_before_after
from feature_selector.selector import FeatureSelector
from feature_selector.stability import stability_selection


st.set_page_config(
    page_title="Feature Selector Tool",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Feature Selector Tool — Research UI")
st.caption(
    "Compare filter, model-based, and wrapper feature selection with "
    "before/after CV metrics and stability analysis."
)


@st.cache_data
def _demo_classification(n_samples=300, n_features=15, random_state=0) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=3,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y
    return df


@st.cache_data
def _sklearn_breast() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns={"target": "target"})
    return df


@st.cache_data
def _sklearn_diabetes() -> pd.DataFrame:
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    # target already named target
    return df


def _load_user_frame(uploaded, target_col=None):
    raw = uploaded.getvalue()
    df = pd.read_csv(io.BytesIO(raw))
    if target_col and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    X.columns = X.columns.astype(str)
    return X, y


with st.sidebar:
    st.header("Data")
    source = st.radio(
        "Source",
        ["Upload CSV", "Demo (synthetic clf)", "sklearn breast_cancer", "sklearn diabetes"],
    )
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("CSV file", type=["csv", "txt"])
        target_col = st.text_input("Target column (blank = last)", value="")
    else:
        target_col = "target"

    st.header("Selection")
    k = st.slider("k (features to keep)", min_value=1, max_value=30, value=5)
    task = st.selectbox("Task", ["auto", "classification", "regression"])
    mode = st.selectbox("Mode", ["Compare methods", "Single method", "Stability only"])

    methods = st.multiselect(
        "Methods",
        options=DEFAULT_METHODS,
        default=DEFAULT_METHODS,
    )
    single_method = st.selectbox(
        "Single / stability method",
        options=DEFAULT_METHODS,
        index=0,
    )
    cv = st.slider("CV folds (metrics)", 3, 10, 5)
    stab_splits = st.slider("Stability folds", 3, 10, 5)
    run_stability = st.checkbox("Run stability in compare", value=True)
    drop_corr = st.checkbox("Drop highly correlated features", value=True)
    corr_thr = st.slider("|corr| threshold", 0.8, 0.99, 0.95)
    seed = st.number_input("Random seed", value=42, step=1)
    go = st.button("Run", type="primary")


def _get_xy():
    if source == "Upload CSV":
        if uploaded is None:
            st.info("Upload a CSV to begin.")
            return None
        tc = target_col.strip() or None
        return _load_user_frame(uploaded, tc)
    if source == "Demo (synthetic clf)":
        df = _demo_classification()
        return df.drop(columns=["target"]), df["target"]
    if source == "sklearn breast_cancer":
        df = _sklearn_breast()
        return df.drop(columns=["target"]), df["target"]
    df = _sklearn_diabetes()
    # diabetes frame uses 'target'
    return df.drop(columns=["target"]), df["target"]


xy = _get_xy()
if xy is not None:
    X, y = xy
    suggested = infer_task(y)
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", len(X))
    c2.metric("Features", X.shape[1])
    c3.metric("Suggested task", suggested)
    with st.expander("Data preview"):
        st.dataframe(pd.concat([X, y.rename("target")], axis=1).head(20))

if go and xy is not None:
    X, y = xy
    k_eff = min(k, X.shape[1])
    corr_threshold = corr_thr if drop_corr else None

    with st.spinner("Running analysis…"):
        if mode == "Compare methods":
            if not methods:
                st.error("Select at least one method.")
                st.stop()
            result = compare_methods(
                X,
                y,
                methods=methods,
                k=k_eff,
                task=task,
                cv=cv,
                stability_splits=stab_splits,
                random_state=int(seed),
                correlation_threshold=corr_threshold,
                run_stability=run_stability,
            )
            st.subheader("Comparison summary")
            st.dataframe(result.summary, use_container_width=True)

            st.subheader("Selected features by method")
            for m, feats in result.selections.items():
                st.markdown(f"**{m}**")
                st.code(", ".join(feats))

            st.subheader("Jaccard similarity")
            st.dataframe(result.jaccard.style.background_gradient(cmap="Blues"), use_container_width=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(result.jaccard, annot=True, fmt=".2f", cmap="Blues", ax=ax, vmin=0, vmax=1)
            ax.set_title("Pairwise Jaccard")
            st.pyplot(fig)
            plt.close(fig)

            metric_col = next(
                (c for c in ("sel_accuracy", "sel_r2", "sel_f1_weighted") if c in result.summary.columns),
                None,
            )
            if metric_col:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                plot_df = result.summary.sort_values(metric_col, ascending=False)
                ax.bar(plot_df["method"], plot_df[metric_col], color="#3b6ea5")
                ax.set_ylabel(metric_col)
                ax.tick_params(axis="x", rotation=25)
                st.pyplot(fig)
                plt.close(fig)

            if result.stability:
                st.subheader("Stability frequencies")
                for m, frame in result.stability.items():
                    st.markdown(f"**{m}**")
                    st.dataframe(frame.head(20), use_container_width=True)

            if result.outlier_summary is not None and not result.outlier_summary.empty:
                st.subheader("Outlier report (IQR)")
                st.dataframe(result.outlier_summary.head(20), use_container_width=True)

            csv = result.summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download comparison_summary.csv",
                csv,
                file_name="comparison_summary.csv",
                mime="text/csv",
            )

        elif mode == "Single method":
            sel = FeatureSelector(
                k=k_eff,
                task=task,
                method=single_method,
                random_state=int(seed),
            )
            sel.fit(X, y)
            st.subheader(f"Selected features — {single_method}")
            st.code(", ".join(sel.selected_features_))
            st.dataframe(sel.get_feature_scores(), use_container_width=True)

            ba = evaluate_before_after(
                X, y, sel.selected_features_, task=task, cv=cv, random_state=int(seed)
            )
            st.subheader("Before / after CV metrics")
            col_a, col_b = st.columns(2)
            col_a.write("All features")
            col_a.json(ba["all_features"])
            col_b.write("Selected features")
            col_b.json(ba["selected"])
            st.write("Delta (selected − all)")
            st.json(ba["delta_selected_minus_all"])

            fig, ax = plt.subplots(figsize=(8, 4))
            scores = sel.get_feature_scores().head(k_eff)
            ax.bar(scores["feature"], scores["score"], color="#3b6ea5")
            ax.tick_params(axis="x", rotation=40)
            ax.set_title("Feature scores")
            st.pyplot(fig)
            plt.close(fig)

        else:
            stab = stability_selection(
                X,
                y,
                method=single_method,
                k=k_eff,
                task=task,
                n_splits=stab_splits,
                random_state=int(seed),
            )
            st.subheader(f"Stability — {stab.method}")
            st.metric("Mean stability", f"{stab.mean_stability:.3f}")
            st.write("Consensus features", stab.consensus_features)
            st.dataframe(stab.frequencies, use_container_width=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            top = stab.frequencies.head(min(20, len(stab.frequencies)))
            ax.barh(top["feature"][::-1], top["frequency"][::-1], color="#3b6ea5")
            ax.set_xlabel("Selection frequency")
            ax.set_title(f"Stability across {stab.n_splits} folds")
            st.pyplot(fig)
            plt.close(fig)

elif go and xy is None:
    st.warning("Load a dataset first.")
