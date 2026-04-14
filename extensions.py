"""
extensions.py  —  NEW MODULE (non-destructive additions only)

"A Research-Validated Adaptive Data Quality Framework with
 Performance Evaluation, Explainability, and Comparative Analysis"

All functions here are NEW.  Nothing in core.py or app.py is modified.
Import this module from app_extended.py alongside the originals.

New capabilities added
----------------------
1.  evaluate_model_on_dataset      — train multiple sklearn models on a df
2.  compare_model_performance      — run (1) on raw vs cleaned df, diff results
3.  plot_quality_ratio_comparison  — normalised ratio bar chart (new visual)
4.  get_top_k_patterns             — top-k patterns from probability vector
5.  run_ablation_study             — three-pipeline ablation comparison
6.  generate_natural_language_explanation — NL rationale from feats + SHAP
7.  load_alternative_model         — load GB meta-model for parallel comparison
8.  plot_calibration_curve         — reliability diagram for a pipeline
9.  generate_report                — structured JSON/text export
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ── re-use existing helpers without touching their source files ──────────────
from core import (
    ALL_PATTERNS,
    FEATURE_NAMES,
    apply_internal_deficit_fix,
    apply_interpretation_deficit_fix,
    apply_operation_deficit_fix,
    apply_pattern_by_name,
    dataset_summary_stats,
    extract_meta_features,
    generate_synthetic_training_data,
    get_default_trained_model,
    train_meta_model,
)

plt.switch_backend("Agg")

# ---------------------------------------------------------------------------
# Colour helpers (mirrors app.py palette without importing it)
# ---------------------------------------------------------------------------
_PALETTE = {
    "Internal Deficit":       "#E05C5C",
    "Interpretation Deficit": "#E8A838",
    "Operation Deficit":      "#4A90D9",
    "positive":               "#2ecc71",
    "negative":               "#e74c3c",
    "neutral":                "#95a5a6",
    "rf":                     "#4A90D9",
    "gb":                     "#E8A838",
    "lr":                     "#2ecc71",
    "dt":                     "#9b59b6",
}

_MODEL_REGISTRY = {
    "Random Forest":      lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":      lambda: DecisionTreeClassifier(max_depth=6, random_state=42),
}


# =============================================================================
# NEW FEATURE 1 — Model Performance Evaluation
# =============================================================================

def _prepare_supervised_data(
    df: pd.DataFrame,
) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """
    Auto-detect a usable target column from ``df``.

    Heuristic (in priority order):
      1. Column named 'target', 'label', 'class', 'y' (case-insensitive).
      2. Last column if it has ≤20 unique values and is not float.
      3. Return None if no usable target is found.

    Returns (X, y) or None.
    """
    candidate_names = {"target", "label", "class", "y", "outcome", "response"}
    for col in df.columns:
        if col.lower() in candidate_names:
            y = df[col]
            X = df.drop(columns=[col])
            return X, y

    # Fallback: last column with low cardinality
    last = df.columns[-1]
    if df[last].nunique() <= 20 and not pd.api.types.is_float_dtype(df[last]):
        return df.drop(columns=[last]), df[last]

    return None


def _build_supervised_pipeline(clf) -> Pipeline:
    """
    Wrap any sklearn classifier in a robust pipeline that handles
    missing values and scales features — safe for arbitrary user CSVs.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     clf),
    ])


def evaluate_model_on_dataset(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train Logistic Regression, Random Forest, and Decision Tree on ``df``
    and return accuracy + F1 (macro) for each.

    If no target column is detectable the function returns an empty dict
    with an explanatory ``error`` key so the caller can surface it gracefully.

    Parameters
    ----------
    df           : DataFrame (may contain non-numeric columns)
    test_size    : holdout fraction
    random_state : reproducibility seed

    Returns
    -------
    dict keyed by model name → {"accuracy": float, "f1_macro": float}
    or {"error": str} on failure.
    """
    result = _prepare_supervised_data(df)
    if result is None:
        return {"error": "No usable target column detected (need ≤20 unique values)."}

    X_raw, y_raw = result

    # Keep only numeric columns (safe fallback)
    X_num = X_raw.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        return {"error": "No numeric feature columns available for model evaluation."}

    # Encode target
    le = LabelEncoder()
    try:
        y_enc = le.fit_transform(y_raw.astype(str))
    except Exception as exc:
        return {"error": f"Target encoding failed: {exc}"}

    if len(np.unique(y_enc)) < 2:
        return {"error": "Target column has fewer than 2 unique classes."}

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_num, y_enc, test_size=test_size, random_state=random_state,
            stratify=y_enc if len(np.unique(y_enc)) > 1 else None,
        )
    except ValueError as exc:
        return {"error": f"Train/test split failed: {exc}"}

    scores: Dict[str, Dict[str, float]] = {}
    for name, factory in _MODEL_REGISTRY.items():
        pipe = _build_supervised_pipeline(factory())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
        scores[name] = {
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "f1_macro": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        }
    return scores


def compare_model_performance(
    df_raw: pd.DataFrame,
    df_cleaned: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Run evaluate_model_on_dataset on both the raw and cleaned DataFrames
    and return a structured diff so the UI can draw side-by-side charts.

    Returns
    -------
    {
        "before": {model_name: {"accuracy": …, "f1_macro": …}, …},
        "after":  {model_name: {"accuracy": …, "f1_macro": …}, …},
        "delta":  {model_name: {"accuracy": …, "f1_macro": …}, …},
        "error":  str | None
    }
    """
    before = evaluate_model_on_dataset(df_raw)
    after  = evaluate_model_on_dataset(df_cleaned)

    if "error" in before:
        return {"error": before["error"], "before": {}, "after": {}, "delta": {}}
    if "error" in after:
        return {"error": after["error"],  "before": before, "after": {}, "delta": {}}

    delta: Dict[str, Dict[str, float]] = {}
    for model in before:
        if model in after:
            delta[model] = {
                "accuracy": after[model]["accuracy"] - before[model]["accuracy"],
                "f1_macro": after[model]["f1_macro"] - before[model]["f1_macro"],
            }

    return {"before": before, "after": after, "delta": delta, "error": None}


# =============================================================================
# NEW FEATURE 2 — Normalised Quality Ratio Comparison Plot
# =============================================================================

def plot_quality_ratio_comparison(
    before: Dict[str, Any],
    after:  Dict[str, Any],
) -> plt.Figure:
    """
    Grouped bar chart of *normalised* quality ratios (missing, duplicate,
    outlier) with percentage-improvement annotations.

    Uses ratios rather than raw counts so the chart remains meaningful
    regardless of whether the cleaning step changed the number of rows.

    Parameters
    ----------
    before / after : dicts returned by dataset_summary_stats()
    """
    metrics = {
        "Missing ratio":   ("missing_ratio",   before.get("missing_ratio",   0),
                                                after.get("missing_ratio",    0)),
        "Duplicate ratio": ("duplicate_ratio",  before.get("duplicate_rows",  0)
                                                / max(before.get("rows", 1), 1),
                                                after.get("duplicate_rows",   0)
                                                / max(after.get("rows",  1), 1)),
        "Outlier ratio":   ("outlier_ratio",
                            before.get("outlier_cells_3sigma", 0)
                            / max(before.get("rows", 1) * max(before.get("numeric_columns", 1), 1), 1),
                            after.get("outlier_cells_3sigma",  0)
                            / max(after.get("rows",  1) * max(after.get("numeric_columns",  1), 1), 1)),
    }

    labels   = list(metrics.keys())
    b_vals   = [metrics[k][1] for k in labels]
    a_vals   = [metrics[k][2] for k in labels]
    x        = np.arange(len(labels))
    w        = 0.32

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    bars_b = ax.bar(x - w / 2, b_vals, w, label="Before",
                    color="#E05C5C", alpha=0.85, edgecolor="none")
    bars_a = ax.bar(x + w / 2, a_vals, w, label="After",
                    color="#4A90D9", alpha=0.85, edgecolor="none")

    # Annotate improvement %
    for bbar, abar, bv, av in zip(bars_b, bars_a, b_vals, a_vals):
        if bv > 0:
            pct = (bv - av) / bv * 100
            colour = "#2ecc71" if pct >= 0 else "#e74c3c"
            ax.text(
                (bbar.get_x() + abar.get_x() + abar.get_width()) / 2,
                max(bv, av) + 0.005,
                f"{pct:+.1f}%",
                ha="center", va="bottom", fontsize=8.5, color=colour, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Ratio (0 – 1)", fontsize=9)
    ax.set_title("Normalised quality-ratio comparison (before vs after)", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(max(b_vals + a_vals, default=0.1) * 1.25, 0.05))
    fig.tight_layout(pad=0.6)
    return fig


# =============================================================================
# NEW FEATURE 3 — Multi-Pattern Recommendation Helper
# =============================================================================

def get_top_k_patterns(
    probas: np.ndarray,
    classes: List[str],
    k: int = 2,
) -> List[Dict[str, Any]]:
    """
    Return the top-k patterns sorted by predicted probability (descending).

    Parameters
    ----------
    probas  : 1-D array of per-class probabilities (from predict_proba)
    classes : list of class names matching the probability vector
    k       : number of top patterns to return (clamped to len(classes))

    Returns
    -------
    list of {"pattern": str, "probability": float} dicts
    """
    k = min(k, len(classes))
    order = np.argsort(probas)[::-1]
    return [
        {"pattern": classes[i], "probability": float(probas[i])}
        for i in order[:k]
    ]


# =============================================================================
# NEW FEATURE 4 — Ablation Study
# =============================================================================

def _basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline pipeline: median imputation + duplicate removal only.
    Does NOT call any existing pattern-specific function — acts as a
    neutral baseline for the ablation comparison.
    """
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = [c for c in out.columns if c not in num_cols]
    if num_cols:
        imp = SimpleImputer(strategy="median")
        out[num_cols] = imp.fit_transform(out[num_cols])
    if obj_cols:
        imp_cat = SimpleImputer(strategy="most_frequent")
        out[obj_cols] = imp_cat.fit_transform(out[obj_cols].astype(str))
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def run_ablation_study(
    df: pd.DataFrame,
    predicted_pattern: str,
) -> Dict[str, Any]:
    """
    Run three pipelines and collect quality metrics + downstream model
    performance for each, enabling a controlled ablation comparison.

    Pipelines
    ---------
    1. Raw           — no cleaning at all
    2. Basic         — median imputation + dedup (neutral baseline)
    3. Pattern-based — existing apply_pattern_by_name (the framework's output)

    Returns
    -------
    {
        "quality": {stage: dataset_summary_stats dict},
        "performance": {stage: evaluate_model_on_dataset dict},
        "stages": ["Raw", "Basic Cleaning", "Pattern-Based Cleaning"]
    }
    """
    stages = {
        "Raw":                   df,
        "Basic Cleaning":        _basic_cleaning(df),
        "Pattern-Based Cleaning": apply_pattern_by_name(df, predicted_pattern),
    }

    quality:     Dict[str, Any] = {}
    performance: Dict[str, Any] = {}

    for stage_name, stage_df in stages.items():
        quality[stage_name]     = dataset_summary_stats(stage_df)
        performance[stage_name] = evaluate_model_on_dataset(stage_df)

    return {
        "quality":     quality,
        "performance": performance,
        "stages":      list(stages.keys()),
    }


def plot_ablation_quality(ablation: Dict[str, Any]) -> plt.Figure:
    """
    Grouped bar chart of missing_ratio, duplicate_ratio, outlier_ratio
    across the three ablation stages.
    """
    stages = ablation["stages"]
    q      = ablation["quality"]

    def _ratio(stats: Dict[str, Any], key: str, denom_key: str = "rows") -> float:
        denom = max(stats.get(denom_key, 1), 1)
        return stats.get(key, 0) / denom

    metrics_def = {
        "Missing ratio":    [q[s].get("missing_ratio", 0)            for s in stages],
        "Duplicate ratio":  [_ratio(q[s], "duplicate_rows")           for s in stages],
        "Outlier ratio":    [q[s].get("outlier_cells_3sigma", 0)
                             / max(q[s].get("rows", 1)
                                   * max(q[s].get("numeric_columns", 1), 1), 1)
                             for s in stages],
    }

    x   = np.arange(len(stages))
    n_m = len(metrics_def)
    w   = 0.22
    colours = ["#E05C5C", "#E8A838", "#4A90D9"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    for idx, (m_name, vals) in enumerate(metrics_def.items()):
        offset = (idx - n_m / 2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=m_name,
               color=colours[idx], alpha=0.85, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Ratio", fontsize=9)
    ax.set_title("Ablation study — quality metrics across pipeline stages", fontsize=11)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6)
    return fig


def plot_ablation_performance(ablation: Dict[str, Any]) -> Optional[plt.Figure]:
    """
    Grouped bar chart of model accuracy across the three ablation stages.
    Returns None if no supervised data was available.
    """
    stages = ablation["stages"]
    perf   = ablation["performance"]

    # Check for errors
    first = perf[stages[0]]
    if "error" in first:
        return None

    model_names = [m for m in _MODEL_REGISTRY if m in first]
    if not model_names:
        return None

    x  = np.arange(len(stages))
    w  = 0.22
    model_colours = ["#4A90D9", "#2ecc71", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    for idx, m_name in enumerate(model_names):
        vals = []
        for stage in stages:
            p = perf[stage]
            vals.append(p.get(m_name, {}).get("accuracy", 0) if "error" not in p else 0)
        offset = (idx - len(model_names) / 2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=m_name,
               color=model_colours[idx % len(model_colours)], alpha=0.85, edgecolor="none")
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_title("Ablation study — downstream model accuracy across pipeline stages", fontsize=11)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6)
    return fig


# =============================================================================
# NEW FEATURE 5 — Natural Language Explanation
# =============================================================================

# Threshold constants for NL generation
_NL_THRESHOLDS = {
    "missing_ratio":   {"high": 0.30, "medium": 0.10},
    "duplicate_ratio": {"high": 0.15, "medium": 0.05},
    "outlier_ratio":   {"high": 0.10, "medium": 0.03},
    "entropy":         {"high": 0.75, "medium": 0.45},
    "skewness":        {"high": 3.0,  "medium": 1.0},
    "numeric_ratio":   {"high": 0.80, "medium": 0.40},
    "unique_ratio":    {"high": 0.70, "medium": 0.30},
    "type_consistency":{"high": 0.50, "medium": 0.80},   # low = bad here
    "value_range_ratio":{"high": 20,  "medium": 8},
}

_FEATURE_PLAIN = {
    "missing_ratio":    "missing value ratio",
    "duplicate_ratio":  "duplicate row ratio",
    "outlier_ratio":    "outlier cell ratio",
    "entropy":          "entropy (cardinality) score",
    "skewness":         "average absolute skewness",
    "numeric_ratio":    "numeric column ratio",
    "unique_ratio":     "average unique-value ratio",
    "type_consistency": "type consistency score",
    "value_range_ratio":"value range ratio",
}

_PATTERN_NL_HINTS = {
    "Internal Deficit": (
        "missing_ratio", "duplicate_ratio"
    ),
    "Interpretation Deficit": (
        "skewness", "type_consistency", "entropy"
    ),
    "Operation Deficit": (
        "outlier_ratio", "duplicate_ratio", "value_range_ratio"
    ),
}


def generate_natural_language_explanation(
    feats: pd.DataFrame,
    prediction: str,
    shap_values: Optional[np.ndarray] = None,
) -> str:
    """
    Produce a human-readable rationale for the model's recommendation.

    The explanation is built in three layers:
      1. Opening statement naming the prediction and its primary driver.
      2. Per-feature sentences for the top signals (SHAP-ordered when
         shap_values is provided; otherwise sorted by absolute feature value).
      3. Closing remediation recommendation.

    Parameters
    ----------
    feats       : 1-row DataFrame from extract_meta_features()
    prediction  : predicted pattern name (str)
    shap_values : optional 1-D SHAP value array aligned to FEATURE_NAMES

    Returns
    -------
    Markdown-formatted explanation string.
    """
    row = feats.iloc[0]
    feature_names = list(feats.columns)

    # Order features by importance (SHAP magnitude if available, else raw val)
    if shap_values is not None and len(shap_values) == len(feature_names):
        order = np.argsort(np.abs(shap_values))[::-1]
    else:
        # Fall back to the pattern's known key features first
        key_feats = _PATTERN_NL_HINTS.get(prediction, feature_names)
        order_list = [feature_names.index(f) for f in key_feats if f in feature_names]
        order_list += [i for i in range(len(feature_names)) if i not in order_list]
        order = np.array(order_list)

    sentences: List[str] = []

    for idx in order[:4]:   # top 4 drivers
        fname = feature_names[idx]
        fval  = float(row[fname])
        plain = _FEATURE_PLAIN.get(fname, fname)
        thresh = _NL_THRESHOLDS.get(fname, {})
        high   = thresh.get("high",   float("inf"))
        medium = thresh.get("medium", float("inf"))

        # Direction of concern differs for type_consistency (low = bad)
        if fname == "type_consistency":
            if fval < 0.50:
                level = "critically low"
                impact = "strongly indicates mixed or corrupt column types"
            elif fval < 0.80:
                level = "moderately low"
                impact = "suggests some schema inconsistency"
            else:
                level = "healthy"
                impact = "columns appear consistently typed"
        else:
            if fval >= high:
                level = "high"
                impact = "is a strong indicator of this deficit pattern"
            elif fval >= medium:
                level = "moderate"
                impact = "contributes to the pattern diagnosis"
            else:
                level = "low"
                impact = "has limited influence on the diagnosis"

        shap_str = ""
        if shap_values is not None and len(shap_values) == len(feature_names):
            sv = float(shap_values[idx])
            shap_str = f" (SHAP contribution = {sv:+.4f})"

        sentences.append(
            f"- **{plain.capitalize()}** is **{level}** at `{fval:.4f}`{shap_str} — "
            f"this {impact}."
        )

    remedy = {
        "Internal Deficit":       "KNN / median **imputation** of missing values.",
        "Interpretation Deficit": "Type **coercion**, IQR Winsorising, and StandardScaling.",
        "Operation Deficit":      "**Deduplication**, high-cardinality column removal, and IQR outlier capping.",
    }.get(prediction, "the recommended cleaning pipeline.")

    header = (
        f"### 🗣️ Why the model selected **{prediction}**\n\n"
        f"The meta-classifier examined nine dataset-level features and "
        f"identified **{prediction}** as the dominant quality issue. "
        f"Key drivers:\n\n"
    )
    footer = (
        f"\n\n**Recommended remediation:** {remedy}\n\n"
        "_This explanation is generated from meta-feature values and SHAP "
        "attributions; it supplements but does not replace domain expertise._"
    )
    return header + "\n".join(sentences) + footer


# =============================================================================
# NEW FEATURE 6 — Alternative Meta-Model Loader
# =============================================================================

def load_alternative_model(
    random_state: int = 42,
    n_samples_per_class: int = 400,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train the Gradient Boosting meta-model on the same synthetic benchmark
    used by the default RF model — enabling a side-by-side comparison.

    Returns (pipeline, metrics) with the same structure as
    get_default_trained_model() so the UI can use it interchangeably.
    """
    return get_default_trained_model(
        random_state=random_state,
        n_samples_per_class=n_samples_per_class,
        model_type="gb",
    )


def compare_meta_models(
    rf_metrics: Dict[str, Any],
    gb_metrics: Dict[str, Any],
) -> plt.Figure:
    """
    Side-by-side grouped bar chart comparing RF vs GB on four metrics:
    accuracy, precision, recall, F1 (macro).
    """
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    labels      = ["Accuracy", "Precision", "Recall", "F1 (macro)"]

    rf_vals = [rf_metrics.get(k, 0) for k in metric_keys]
    gb_vals = [gb_metrics.get(k, 0) for k in metric_keys]
    x = np.arange(len(labels))
    w = 0.30

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    bars_rf = ax.bar(x - w / 2, rf_vals, w, label="Random Forest",
                     color="#4A90D9", alpha=0.88, edgecolor="none")
    bars_gb = ax.bar(x + w / 2, gb_vals, w, label="Gradient Boosting",
                     color="#E8A838", alpha=0.88, edgecolor="none")

    for bar, val in [(b, v) for bars, vals in [(bars_rf, rf_vals), (bars_gb, gb_vals)]
                     for b, v in zip(bars, vals)]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("Meta-model comparison: Random Forest vs Gradient Boosting", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6)
    return fig


# =============================================================================
# NEW FEATURE 7 — Calibration Curve
# =============================================================================

def _get_raw_classifier_ext(pipeline: Pipeline):
    """Unwrap CalibratedClassifierCV → base estimator (mirrors app.py helper)."""
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "calibrated_classifiers_"):
        return clf.calibrated_classifiers_[0].estimator
    return clf


def plot_calibration_curve_ext(
    pipeline: Pipeline,
    n_samples_per_class: int = 300,
    random_state: int = 42,
    n_bins: int = 10,
) -> plt.Figure:
    """
    Reliability diagram (calibration curve) for the meta-classifier.

    A well-calibrated model's curve lies close to the diagonal y = x.
    We generate a fresh synthetic test set (not used during training) to
    avoid data-leakage in the calibration estimate.

    Parameters
    ----------
    pipeline            : fitted Pipeline from get_default_trained_model()
    n_samples_per_class : size of the fresh evaluation set per class
    random_state        : different seed from training (42+1) to avoid overlap
    n_bins              : number of reliability bins
    """
    X_eval, y_eval = generate_synthetic_training_data(
        n_samples_per_class=n_samples_per_class,
        random_state=random_state + 1,
    )

    classes = list(_get_raw_classifier_ext(pipeline).classes_
                   if hasattr(_get_raw_classifier_ext(pipeline), "classes_")
                   else ALL_PATTERNS)

    # Use predict_proba for the full pipeline
    try:
        probas = pipeline.predict_proba(X_eval)
    except Exception:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "predict_proba not available.", ha="center", va="center")
        ax.axis("off")
        return fig

    colours = ["#E05C5C", "#E8A838", "#4A90D9"]
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")

    for idx, cls in enumerate(classes):
        binary_y = (y_eval == cls).astype(int)
        cls_prob  = probas[:, idx]
        try:
            frac_pos, mean_pred = calibration_curve(
                binary_y, cls_prob, n_bins=n_bins, strategy="uniform"
            )
            ax.plot(mean_pred, frac_pos,
                    marker="o", markersize=5,
                    color=colours[idx % len(colours)],
                    label=cls, linewidth=1.8)
        except Exception:
            continue

    ax.set_xlabel("Mean predicted probability", fontsize=9)
    ax.set_ylabel("Fraction of positives", fontsize=9)
    ax.set_title("Calibration (reliability) curve — meta-classifier", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6)
    return fig


# =============================================================================
# NEW FEATURE 8 — Report Generator
# =============================================================================

def generate_report(
    df_name: str,
    before: Dict[str, Any],
    after: Dict[str, Any],
    prediction: str,
    confidence: float,
    train_metrics: Dict[str, Any],
    top_k: List[Dict[str, Any]],
    nl_explanation: str,
    perf_comparison: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Produce a structured JSON report suitable for inclusion in a paper's
    supplementary materials or automated evaluation pipeline.

    Parameters
    ----------
    df_name        : filename of the uploaded CSV
    before / after : dataset_summary_stats() dicts
    prediction     : recommended pattern (str)
    confidence     : predicted class probability (float)
    train_metrics  : meta-model evaluation dict from get_default_trained_model()
    top_k          : list from get_top_k_patterns()
    nl_explanation : string from generate_natural_language_explanation()
    perf_comparison: dict from compare_model_performance() (optional)

    Returns
    -------
    Pretty-printed JSON string.
    """
    report = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "framework":    "ML-Based Adaptive DQ Pattern Selection",
            "dataset":      df_name,
        },
        "meta_model_evaluation": {
            "holdout_accuracy":   round(train_metrics.get("accuracy", 0), 4),
            "macro_precision":    round(train_metrics.get("precision_macro", 0), 4),
            "macro_recall":       round(train_metrics.get("recall_macro", 0), 4),
            "macro_f1":           round(train_metrics.get("f1_macro", 0), 4),
            "cv_accuracy_mean":   round(train_metrics.get("cv_accuracy_mean", 0), 4),
            "cv_accuracy_std":    round(train_metrics.get("cv_accuracy_std", 0), 4),
        },
        "recommendation": {
            "primary_pattern": prediction,
            "confidence":      round(confidence, 4),
            "top_k_patterns":  top_k,
        },
        "quality_metrics": {
            "before": before,
            "after":  after,
            "improvement": {
                k: round(
                    (float(before.get(k, 0)) - float(after.get(k, 0)))
                    / max(float(before.get(k, 1e-9)), 1e-9) * 100,
                    2,
                )
                for k in ["missing_cells", "duplicate_rows", "outlier_cells_3sigma"]
                if before.get(k, 0) > 0
            },
        },
        "natural_language_rationale": nl_explanation.replace("**", "").replace("`", ""),
    }

    if perf_comparison and perf_comparison.get("error") is None:
        report["downstream_model_performance"] = {
            "before": perf_comparison.get("before", {}),
            "after":  perf_comparison.get("after",  {}),
            "delta":  perf_comparison.get("delta",  {}),
        }

    return json.dumps(report, indent=2, default=str)
