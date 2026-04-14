"""
app_extended.py  —  Extended Streamlit front-end.

Strategy: import the ORIGINAL app.py's main() and run it, then append
new steps in the same Streamlit session.  All existing behaviour is
preserved; new sections are appended below Step 4.

HOW TO RUN
----------
    streamlit run app_extended.py

The file completely replaces app.py as the entry point.  Original app.py
and core.py are left untouched — this file only adds new code.
"""

from __future__ import annotations

# ── std library ─────────────────────────────────────────────────────────────
import io
import json
from typing import Any, Dict, List, Optional, Tuple

# ── third party ──────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

try:
    import shap as _shap
    SHAP_AVAILABLE = True
except ImportError:
    _shap = None           # type: ignore[assignment]
    SHAP_AVAILABLE = False

# ── ORIGINAL modules (untouched) ────────────────────────────────────────────
from core import (
    ALL_PATTERNS,
    FEATURE_NAMES,
    PATTERN_DESCRIPTIONS,
    apply_pattern_by_name,
    dataset_summary_stats,
    extract_meta_features,
    get_default_trained_model,
)

# ── NEW extension module ─────────────────────────────────────────────────────
plt.rcParams.update({
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.titlecolor": "white",
})
from extensions import (
    # Feature 1
    evaluate_model_on_dataset,
    compare_model_performance,
    # Feature 2
    plot_quality_ratio_comparison,
    # Feature 3
    get_top_k_patterns,
    # Feature 4
    run_ablation_study,
    plot_ablation_quality,
    plot_ablation_performance,
    # Feature 5
    generate_natural_language_explanation,
    # Feature 6
    load_alternative_model,
    compare_meta_models,
    # Feature 7
    plot_calibration_curve_ext,
    # Feature 8
    generate_report,
)

plt.switch_backend("Agg")

# ── Colour palette (copied from app.py; not importing to avoid side-effects) ─
_PALETTE = {
    "Internal Deficit":       "#E05C5C",
    "Interpretation Deficit": "#E8A838",
    "Operation Deficit":      "#4A90D9",
    "positive":               "#2980b9",
    "negative":               "#c0392b",
    "neutral":                "#95a5a6",
}
_PATTERN_EMOJI = {
    "Internal Deficit":       "🔴",
    "Interpretation Deficit": "🟠",
    "Operation Deficit":      "🔵",
}


# =============================================================================
# Cached resource loaders (NEW; do NOT modify originals)
# =============================================================================

@st.cache_resource(show_spinner="Training meta-model on synthetic benchmark…")
def _load_rf_pipeline() -> Tuple[Pipeline, Dict[str, Any]]:
    """Load the default RandomForest meta-model (same call as original)."""
    return get_default_trained_model(n_samples_per_class=400, model_type="rf")


@st.cache_resource(show_spinner="Training Gradient Boosting meta-model…")
def _load_gb_pipeline() -> Tuple[Pipeline, Dict[str, Any]]:
    """NEW — load the alternative Gradient Boosting meta-model."""
    return load_alternative_model(n_samples_per_class=400)


# =============================================================================
# Helper: unwrap calibrated classifier
# =============================================================================

def _get_raw_clf(pipeline: Pipeline):
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "calibrated_classifiers_"):
        return clf.calibrated_classifiers_[0].estimator
    return clf


def _get_classes(pipeline: Pipeline) -> List[str]:
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "classes_"):
        return list(clf.classes_)
    return list(_get_raw_clf(pipeline).classes_)


# =============================================================================
# Helpers copied from app.py (kept internal; app.py not modified)
# =============================================================================

def _read_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw), encoding_errors="replace")


def _confidence_bar(classes: List[str], probas: np.ndarray, predicted: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 2.4))
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")
    colours = [
        _PALETTE.get(c, "#888") if c == predicted else "#d0d0d0"
        for c in classes
    ]
    bars = ax.barh(classes, probas, color=colours, height=0.55, edgecolor="none")
    for bar, p in zip(bars, probas):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{p:.1%}", va="center", fontsize=10)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Confidence (calibrated probability)", fontsize=9)
    ax.tick_params(labelsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.axvline(0, color="#888", linewidth=0.7)
    fig.tight_layout(pad=0.5)
    return fig


def _feature_importance(pipeline: Pipeline) -> Optional[plt.Figure]:
    raw = _get_raw_clf(pipeline)
    if not hasattr(raw, "feature_importances_"):
        return None
    imp = raw.feature_importances_
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")
    ax.barh([FEATURE_NAMES[i] for i in order], imp[order],
            color="#4A90D9", height=0.55, edgecolor="none")
    ax.set_xlabel("Mean Decrease in Impurity (MDI)", fontsize=9)
    ax.set_title("Global feature importance (all patterns)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.5)
    return fig


def _shap_bar(pipeline: Pipeline, feats: pd.DataFrame, predicted: str):
    if not SHAP_AVAILABLE:
        return None, "", None
    raw_clf = _get_raw_clf(pipeline)
    classes = _get_classes(pipeline)
    if predicted not in classes:
        return None, "SHAP: class not found.", None
    Xv = feats.values.astype(float)
    cidx = classes.index(predicted)
    try:
        explainer = _shap.TreeExplainer(raw_clf)
        shap_raw  = explainer.shap_values(Xv)
    except Exception as exc:
        return None, f"SHAP failed: {exc}", None
    if isinstance(shap_raw, list):
        sv = shap_raw[cidx][0]
        base = explainer.expected_value
        base_val = float(base[cidx] if isinstance(base, np.ndarray) else base)
    else:
        sv = shap_raw[0]
        base_val = float(
            explainer.expected_value[cidx]
            if isinstance(explainer.expected_value, np.ndarray)
            else explainer.expected_value
        )
    order = np.argsort(np.abs(sv))[::-1]
    labels = [FEATURE_NAMES[i] for i in order]
    values = sv[order]
    feat_vals = [feats.iloc[0, i] for i in order]
    colours = [_PALETTE["negative"] if v < 0 else _PALETTE["positive"] for v in values]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")
    bars = ax.barh(labels[::-1], values[::-1], color=colours[::-1],
                   height=0.6, edgecolor="none")
    for bar, fv in zip(bars, feat_vals[::-1]):
        ax.text(0.003, bar.get_y() + bar.get_height() / 2,
                f"val={fv:.3f}", va="center", fontsize=7.5, color="#555")
    ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value", fontsize=9)
    ax.set_title(f"Why \"{predicted}\" was selected (instance-level SHAP)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.5)
    top_name = FEATURE_NAMES[order[0]]
    top_fv   = feats.iloc[0, order[0]]
    direction = "supports" if sv[0] > 0 else "opposes"
    narrative = (
        f"**Primary driver:** `{top_name}` = `{top_fv:.4f}` "
        f"— this feature {direction} selecting **{predicted}** "
        f"(SHAP = {sv[0]:+.4f}, baseline = {base_val:.4f})."
    )
    # Return raw SHAP values in FEATURE_NAMES order (for NL explanation)
    sv_full = np.zeros(len(FEATURE_NAMES))
    for i, idx in enumerate(order):
        sv_full[idx] = sv[i]
    return fig, narrative, sv_full


def _before_after_bar(before: Dict[str, Any], after: Dict[str, Any]) -> plt.Figure:
    metrics = {
        "Missing cells":  (before["missing_cells"],       after["missing_cells"]),
        "Duplicate rows": (before["duplicate_rows"],       after["duplicate_rows"]),
        "Outlier cells":  (before["outlier_cells_3sigma"], after["outlier_cells_3sigma"]),
        "Rows":           (before["rows"],                 after["rows"]),
    }
    keys = list(metrics.keys())
    b_vals = [metrics[k][0] for k in keys]
    a_vals = [metrics[k][1] for k in keys]
    x = np.arange(len(keys)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")
    ax.bar(x - w/2, b_vals, w, label="Before", color="#E05C5C", alpha=0.85, edgecolor="none")
    ax.bar(x + w/2, a_vals, w, label="After",  color="#4A90D9", alpha=0.85, edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Before vs After — key quality metrics", fontsize=11)
    ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.5)
    return fig


# =============================================================================
# NEW — Model performance chart helpers
# =============================================================================

def _plot_model_comparison_chart(
    perf_cmp: Dict[str, Any],
    metric: str = "accuracy",
) -> Optional[plt.Figure]:
    """
    Grouped bar chart: before/after per model for one metric.
    """
    if perf_cmp.get("error"):
        return None
    before = perf_cmp["before"]
    after  = perf_cmp["after"]
    models = [m for m in before if m in after]
    if not models:
        return None

    b_vals = [before[m][metric] for m in models]
    a_vals = [after[m][metric]  for m in models]
    x = np.arange(len(models)); w = 0.32

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_alpha(0.0); ax.set_facecolor("none")
    bars_b = ax.bar(x - w/2, b_vals, w, label="Before",
                    color="#E05C5C", alpha=0.85, edgecolor="none")
    bars_a = ax.bar(x + w/2, a_vals, w, label="After",
                    color="#4A90D9", alpha=0.85, edgecolor="none")
    for bar, v in [(b, v) for bars, vals in [(bars_b, b_vals), (bars_a, a_vals)]
                   for b, v in zip(bars, vals)]:
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=9)
    ax.set_title(f"Downstream model {metric.replace('_',' ')} — before vs after cleaning", fontsize=11)
    ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6)
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    # ── Page config ──────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="DQ Framework — Research Edition",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        "<style>.block-container{padding-top:1.4rem}"
        ".stMetric label{font-size:0.78rem!important}</style>",
        unsafe_allow_html=True,
    )

    # ── NEW: model selector in sidebar ───────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Meta-Model Card")
        # NEW FEATURE 6 — model selector
        model_choice = st.radio(
            "Select meta-model",
            options=["Random Forest (RF)", "Gradient Boosting (GB)"],
            index=0,
            help="RF is faster; GB may yield higher accuracy. "
                 "Switch to compare side-by-side below.",
        )

    use_gb = model_choice.startswith("Gradient")
    if use_gb:
        pipeline, train_metrics = _load_gb_pipeline()
    else:
        pipeline, train_metrics = _load_rf_pipeline()

    classes = _get_classes(pipeline)

    # ── Sidebar metrics ───────────────────────────────────────────────────────
    with st.sidebar:
        st.caption(
            f"{'Gradient Boosting' if use_gb else 'RandomForest (300 trees, calibrated)'} "
            "trained on 1 200 synthetic datasets — 400 per pattern class."
        )
        st.metric("Hold-out accuracy", f"{train_metrics['accuracy']:.3f}")
        st.metric("Macro precision",   f"{train_metrics['precision_macro']:.3f}")
        st.metric("Macro recall",      f"{train_metrics['recall_macro']:.3f}")
        st.metric(
            "5-fold CV accuracy",
            f"{train_metrics['cv_accuracy_mean']:.3f} ± {train_metrics['cv_accuracy_std']:.3f}",
        )
        with st.expander("Full classification report"):
            st.code(train_metrics["classification_report"], language=None)

        st.divider()
        st.subheader("📖 Pattern legend")
        for p in ALL_PATTERNS:
            st.markdown(f"**{_PATTERN_EMOJI.get(p,'•')} {p}**")
            st.caption(PATTERN_DESCRIPTIONS[p])

        st.divider()
        # NEW — ablation toggle
        ablation_on = st.toggle("🔬 Enable Ablation Study Mode", value=False)
        st.caption("SHAP: " + ("✅" if SHAP_AVAILABLE else "❌ pip install shap"))

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🧠 ML-Guided Data Quality Pattern Selection")
    st.markdown(
        """
        **Research proof-of-concept** for *"A Machine Learning-Based Adaptive Framework
        for Data Quality Pattern Selection in Big Data Systems."*

        Upload a CSV → meta-features → ML recommendation → SHAP + NL explanation →
        before/after comparison → model performance validation → ablation study.
        """
    )
    st.divider()

    # ── Step 0: Upload ────────────────────────────────────────────────────────
    st.subheader("📂 Step 0 — Upload your dataset")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.info("Awaiting CSV upload to begin analysis.")
        return

    try:
        df = _read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}"); return

    if df.empty:
        st.warning("The uploaded file appears to be empty."); return

    st.success(f"Loaded **{uploaded.name}** — {len(df):,} rows × {df.shape[1]} columns.")

    # ── Step 1: Meta-features ─────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Step 1 — Data quality profile (meta-features)")
    feats = extract_meta_features(df)

    r1 = st.columns(5)
    r1[0].metric("Missing ratio",      f"{feats['missing_ratio'].iloc[0]:.2%}")
    r1[1].metric("Duplicate ratio",    f"{feats['duplicate_ratio'].iloc[0]:.2%}")
    r1[2].metric("Outlier ratio",      f"{feats['outlier_ratio'].iloc[0]:.2%}")
    r1[3].metric("Entropy",            f"{feats['entropy'].iloc[0]:.3f}")
    r1[4].metric("Avg |skewness|",     f"{feats['skewness'].iloc[0]:.3f}")
    r2 = st.columns(4)
    r2[0].metric("Numeric ratio",      f"{feats['numeric_ratio'].iloc[0]:.2%}")
    r2[1].metric("Unique ratio",       f"{feats['unique_ratio'].iloc[0]:.2%}")
    r2[2].metric("Type consistency",   f"{feats['type_consistency'].iloc[0]:.2%}")
    r2[3].metric("Value range ratio",  f"{feats['value_range_ratio'].iloc[0]:.2f}")

    with st.expander("Raw meta-feature vector (for paper tables)"):
        st.dataframe(feats.T.rename(columns={0: "value"}), use_container_width=True)

    # ── Step 2: Prediction ────────────────────────────────────────────────────
    st.divider()
    st.subheader("🎯 Step 2 — Recommended remediation pattern")

    try:
        predicted: str = str(pipeline.predict(feats)[0])
        probas: Optional[np.ndarray] = None
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            probas = pipeline.predict_proba(feats)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}"); return

    emoji = _PATTERN_EMOJI.get(predicted, "•")

    # NEW FEATURE 3 — multi-pattern / low-confidence UI
    CONFIDENCE_THRESHOLD = 0.60
    top_k_list = get_top_k_patterns(probas, classes, k=2) if probas is not None else []
    top_conf   = top_k_list[0]["probability"] if top_k_list else 1.0

    col_pred, col_desc = st.columns([1, 2])
    with col_pred:
        st.markdown(
            f"<div style='background:{_PALETTE.get(predicted,'#aaa')}22;"
            f"border-left:6px solid {_PALETTE.get(predicted,'#aaa')};"
            f"padding:1rem 1.2rem;border-radius:6px'>"
            f"<h3 style='margin:0'>{emoji} {predicted}</h3></div>",
            unsafe_allow_html=True,
        )
        if probas is not None:
            conf_val = probas[classes.index(predicted)] if predicted in classes else 0.0
            st.metric("Confidence", f"{conf_val:.1%}")

        # NEW — low-confidence warning + top-2 display
        if top_conf < CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ Confidence is below {CONFIDENCE_THRESHOLD:.0%}. "
                "The top-2 candidates are shown. Consider using domain knowledge "
                "or the override selector below."
            )
            for entry in top_k_list:
                st.markdown(
                    f"- **{entry['pattern']}** — {entry['probability']:.1%}"
                )
            # NEW — user override
            override = st.selectbox(
                "Override pattern selection (optional)",
                options=["— use model recommendation —"] + ALL_PATTERNS,
            )
            if override != "— use model recommendation —":
                predicted = override
                emoji = _PATTERN_EMOJI.get(predicted, "•")
                st.info(f"Using override: **{predicted}**")

    with col_desc:
        st.info(PATTERN_DESCRIPTIONS.get(predicted, ""))

    if probas is not None:
        fig_conf = _confidence_bar(classes, probas, predicted)
        st.pyplot(fig_conf); plt.close(fig_conf)

    # ── Step 3: Explainability ────────────────────────────────────────────────
    st.divider()
    st.subheader("🔬 Step 3 — Explainability")

    tab_shap, tab_global = st.tabs(
        ["Instance SHAP (why this dataset?)", "Global feature importance"]
    )

    shap_sv_full: Optional[np.ndarray] = None   # passed to NL explanation below

    with tab_shap:
        st.caption(
            "SHAP attributes the model's margin to each meta-feature. "
            "Blue = supports this pattern; red = pushes away."
        )
        if not SHAP_AVAILABLE:
            st.warning("`shap` not installed. Run `pip install shap`.")
        else:
            try:
                fig_shap, narrative, shap_sv_full = _shap_bar(pipeline, feats, predicted)
                if fig_shap is not None:
                    st.pyplot(fig_shap); plt.close(fig_shap)
                    st.markdown(narrative)
                else:
                    st.warning(narrative)
            except Exception as exc:
                st.warning(f"SHAP visualisation skipped: {exc}")

        # NEW FEATURE 5 — natural language explanation (below SHAP, same tab)
        st.divider()
        nl_text = generate_natural_language_explanation(feats, predicted, shap_sv_full)
        st.markdown(nl_text)

    with tab_global:
        st.caption("Mean Decrease in Impurity (MDI) across all trees.")
        fig_imp = _feature_importance(pipeline)
        if fig_imp is not None:
            st.pyplot(fig_imp); plt.close(fig_imp)
        else:
            st.info("Feature importance not available for this classifier type.")

    # ── Step 4: Apply & Compare ───────────────────────────────────────────────
    st.divider()
    st.subheader("🔧 Step 4 — Apply & compare")
    st.caption(
        "Click to apply the recommended cleaning pipeline and compare quality metrics."
    )

    # Persistent session state so cleaned df survives re-runs
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None
    if "before_stats" not in st.session_state:
        st.session_state.before_stats = None
    if "after_stats" not in st.session_state:
        st.session_state.after_stats  = None

    if st.button(f"Apply  {emoji} {predicted}  pipeline", type="primary"):
        with st.spinner("Applying cleaning pipeline…"):
            cleaned = apply_pattern_by_name(df, predicted)
        st.session_state.cleaned_df    = cleaned
        st.session_state.before_stats  = dataset_summary_stats(df)
        st.session_state.after_stats   = dataset_summary_stats(cleaned)

    if st.session_state.cleaned_df is not None:
        cleaned = st.session_state.cleaned_df
        before  = st.session_state.before_stats
        after   = st.session_state.after_stats

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Missing cells",         after["missing_cells"],
                  delta=after["missing_cells"]       - before["missing_cells"],       delta_color="inverse")
        c2.metric("Duplicate rows",        after["duplicate_rows"],
                  delta=after["duplicate_rows"]      - before["duplicate_rows"],      delta_color="inverse")
        c3.metric("Outlier cells (3σ)",    after["outlier_cells_3sigma"],
                  delta=after["outlier_cells_3sigma"]- before["outlier_cells_3sigma"],delta_color="inverse")
        c4.metric("Rows",                  after["rows"],
                  delta=after["rows"]                - before["rows"],                delta_color="off")

        fig_ba = _before_after_bar(before, after)
        st.pyplot(fig_ba); plt.close(fig_ba)

        # NEW FEATURE 2 — normalised ratio chart (added below, not replacing)
        fig_ratio = plot_quality_ratio_comparison(before, after)
        st.pyplot(fig_ratio); plt.close(fig_ratio)
        st.caption("↑ Normalised ratio view — percentage improvement annotated above each pair.")

        col_b, col_a = st.columns(2)
        with col_b:
            st.markdown("**Before**"); st.json(before)
        with col_a:
            st.markdown("**After**");  st.json(after)

        st.markdown("**Cleaned data preview** (first 50 rows)")
        st.dataframe(cleaned.head(50), use_container_width=True)

        csv_bytes = cleaned.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download cleaned CSV", data=csv_bytes,
                           file_name=f"cleaned_{uploaded.name}", mime="text/csv")

        # ── NEW FEATURE 1 — Step 5: Model Performance Comparison ─────────────
        st.divider()
        st.subheader("📈 Step 5 — Downstream model performance comparison")
        st.caption(
            "Three sklearn models (Logistic Regression, Random Forest, Decision Tree) "
            "are trained on both raw and cleaned DataFrames to quantify the "
            "downstream value of cleaning.  Requires a detectable target column."
        )

        with st.spinner("Evaluating downstream model performance…"):
            perf_cmp = compare_model_performance(df, cleaned)

        if perf_cmp.get("error"):
            st.info(
                f"ℹ️ Downstream evaluation skipped: {perf_cmp['error']}\n\n"
                "Tip: add a column named `target`, `label`, or `class` with ≤20 "
                "unique values to enable this step."
            )
        else:
            tab_acc, tab_f1, tab_delta = st.tabs(
                ["Accuracy comparison", "F1 (macro) comparison", "Δ Delta table"]
            )
            with tab_acc:
                fig_acc = _plot_model_comparison_chart(perf_cmp, metric="accuracy")
                if fig_acc:
                    st.pyplot(fig_acc); plt.close(fig_acc)
            with tab_f1:
                fig_f1 = _plot_model_comparison_chart(perf_cmp, metric="f1_macro")
                if fig_f1:
                    st.pyplot(fig_f1); plt.close(fig_f1)
            with tab_delta:
                delta_rows = [
                    {
                        "Model": m,
                        "Δ Accuracy": f"{perf_cmp['delta'][m]['accuracy']:+.4f}",
                        "Δ F1 (macro)": f"{perf_cmp['delta'][m]['f1_macro']:+.4f}",
                    }
                    for m in perf_cmp["delta"]
                ]
                st.dataframe(pd.DataFrame(delta_rows), use_container_width=True)
                st.caption(
                    "Positive delta = cleaning improved downstream model performance."
                )

        # ── NEW FEATURE 4 — Ablation Study ────────────────────────────────────
        if ablation_on:
            st.divider()
            st.subheader("🔬 Step 6 — Ablation study")
            st.caption(
                "Three pipeline stages compared: Raw → Basic cleaning (median "
                "imputation + dedup) → Pattern-based cleaning (the framework's output)."
            )
            with st.spinner("Running ablation study…"):
                ablation = run_ablation_study(df, predicted)

            tab_q, tab_p = st.tabs(["Quality metrics", "Model performance"])
            with tab_q:
                fig_abl_q = plot_ablation_quality(ablation)
                st.pyplot(fig_abl_q); plt.close(fig_abl_q)
            with tab_p:
                fig_abl_p = plot_ablation_performance(ablation)
                if fig_abl_p:
                    st.pyplot(fig_abl_p); plt.close(fig_abl_p)
                else:
                    st.info("No usable target column for performance ablation.")

        # ── NEW FEATURE 6 — Meta-model comparison ────────────────────────────
        st.divider()
        st.subheader("⚖️ Step 7 — Meta-model comparison (RF vs GB)")
        st.caption(
            "Compare the two meta-classifier alternatives on the same synthetic "
            "benchmark.  Both are trained with identical data; only the algorithm differs."
        )
        if st.button("Run RF vs GB comparison", key="btn_mm_cmp"):
            with st.spinner("Training Gradient Boosting meta-model for comparison…"):
                _, rf_m  = _load_rf_pipeline()
                _, gb_m  = _load_gb_pipeline()
            fig_mm = compare_meta_models(rf_m, gb_m)
            st.pyplot(fig_mm); plt.close(fig_mm)

            col_rf, col_gb = st.columns(2)
            with col_rf:
                st.markdown("**Random Forest**")
                st.json({k: round(float(v), 4) for k, v in rf_m.items()
                         if isinstance(v, float)})
            with col_gb:
                st.markdown("**Gradient Boosting**")
                st.json({k: round(float(v), 4) for k, v in gb_m.items()
                         if isinstance(v, float)})

        # ── NEW FEATURE 7 — Calibration curve ────────────────────────────────
        st.divider()
        st.subheader("📐 Step 8 — Confidence calibration curve")
        st.caption(
            "A reliability diagram shows whether the model's predicted probabilities "
            "are well-calibrated.  The dashed line (y = x) is perfect calibration."
        )
        if st.button("Show calibration curve", key="btn_calib"):
            with st.spinner("Computing calibration curve on fresh synthetic data…"):
                fig_cal = plot_calibration_curve_ext(pipeline)
            st.pyplot(fig_cal); plt.close(fig_cal)
            st.info(
                "Points above the diagonal → model is under-confident; "
                "points below → over-confident.  CalibratedClassifierCV (isotonic) "
                "has been applied to the raw classifier to correct this."
            )

        # ── NEW FEATURE 8 — Export report ─────────────────────────────────────
        st.divider()
        st.subheader("📄 Step 9 — Export research report")
        st.caption(
            "Generate a structured JSON report suitable for supplementary materials "
            "or automated evaluation pipelines."
        )

        conf_val_for_report = (
            float(probas[classes.index(predicted)])
            if probas is not None and predicted in classes else 0.0
        )
        nl_for_report = generate_natural_language_explanation(
            feats, predicted, shap_sv_full
        )

        report_str = generate_report(
            df_name        = uploaded.name,
            before         = before,
            after          = after,
            prediction     = predicted,
            confidence     = conf_val_for_report,
            train_metrics  = train_metrics,
            top_k          = top_k_list,
            nl_explanation = nl_for_report,
            perf_comparison= perf_cmp if not perf_cmp.get("error") else None,
        )

        st.download_button(
            label     = "📥 Download JSON report",
            data      = report_str.encode("utf-8"),
            file_name = f"dq_report_{uploaded.name.replace('.csv','')}.json",
            mime      = "application/json",
        )
        with st.expander("Preview report"):
            st.code(report_str, language="json")


if __name__ == "__main__":
    main()
