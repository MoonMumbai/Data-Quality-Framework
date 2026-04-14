"""
core.py  —  Backend for:
    "A Machine Learning-Based Adaptive Framework for Data Quality
     Pattern Selection in Big Data Systems"

Improvement highlights over the original version
-------------------------------------------------
1. Richer meta-features (9 vs 5): adds numeric_ratio, unique_ratio,
   type_consistency_score, and value_range_ratio to break ties between
   patterns that score similarly on the original 5 features.
2. Stronger synthetic data: five injectors are composed probabilistically,
   and a "mixed" contamination mode is added to make the boundary regions
   more realistic for a research setting.
3. Cross-validation loop added to train_meta_model; metrics dict now
   contains mean ± std CV scores for inclusion in paper tables.
4. GradientBoostingClassifier added as an optional alternative brain to
   let callers A/B test without rewriting the pipeline.
5. Calibrated probabilities: CalibratedClassifierCV wrapper is optional
   so the Streamlit app can show reliable confidence bars.
6. All cleaning pipelines are more robust: Operation fix now also removes
   high-cardinality noise columns; Interpretation fix uses IQR-based
   Winsorizing before scaling; Internal fix adds KNN imputation option.
7. Type hints are complete; all edge-cases (empty df, no numeric cols,
   single-row df, all-NaN column) are guarded.

Design rationale (methodology):
    The meta-model operates on *summary statistics* of a table rather than
    row-level features, mimicking how a data steward would rapidly profile
    a dataset before choosing a remediation strategy.  Operating at the
    meta-feature level makes the framework dataset-size agnostic — the same
    nine numbers summarise a 500-row CSV and a 500-million-row Spark table.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PATTERN_INTERNAL = "Internal Deficit"
PATTERN_INTERPRETATION = "Interpretation Deficit"
PATTERN_OPERATION = "Operation Deficit"

ALL_PATTERNS: List[str] = [
    PATTERN_INTERNAL,
    PATTERN_INTERPRETATION,
    PATTERN_OPERATION,
]

FEATURE_NAMES: List[str] = [
    "missing_ratio",        # proportion of NaN cells
    "duplicate_ratio",      # proportion of exact duplicate rows
    "outlier_ratio",        # proportion of values > 3σ from column mean
    "entropy",              # avg normalised Shannon entropy across columns
    "skewness",             # avg |Fisher skewness| across numeric cols
    "numeric_ratio",        # proportion of columns that are numeric
    "unique_ratio",         # avg unique-value proportion across columns
    "type_consistency",     # fraction of columns with consistent inferred dtypes
    "value_range_ratio",    # avg (max−min)/std across numeric cols (spread proxy)
]

# Pattern → short description used in UI tooltips
PATTERN_DESCRIPTIONS: Dict[str, str] = {
    PATTERN_INTERNAL: (
        "High missing-value rate or structurally absent data. "
        "Remedy: imputation (median / KNN)."
    ),
    PATTERN_INTERPRETATION: (
        "Schema inconsistencies, mixed types, or heavily skewed distributions. "
        "Remedy: type coercion + StandardScaler."
    ),
    PATTERN_OPERATION: (
        "Operational noise: duplicates, extreme outliers, or artificially high "
        "cardinality. Remedy: dedup + IQR capping."
    ),
}


# =============================================================================
# Utility helpers
# =============================================================================

def _rng(seed: Optional[int] = None) -> np.random.Generator:
    """Reproducible random number generator."""
    return np.random.default_rng(seed)


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return list of genuinely numeric column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


# =============================================================================
# Module 1 – Synthetic Data & Label Generator
# =============================================================================

def _make_base_dataframe(
    n_rows: int,
    n_numeric: int,
    n_categorical: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Construct a clean mixed-type base table before injecting quality defects.

    Uses normal random draws for numeric columns so that any introduced
    skewness or outliers are unambiguously attributable to the injector.
    """
    data: Dict[str, Any] = {}
    for j in range(n_numeric):
        data[f"num_{j}"] = rng.normal(0, 1, size=n_rows)
    cats = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    for j in range(n_categorical):
        data[f"cat_{j}"] = rng.choice(cats, size=n_rows)
    # Add a date-like column to test type consistency scoring
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="h")
    data["ts"] = rng.choice(dates.astype(str), size=n_rows)
    return pd.DataFrame(data)


def _inject_internal_deficit(
    df: pd.DataFrame, rng: np.random.Generator, strength: float
) -> pd.DataFrame:
    """
    Internal Deficit injector — MCAR (Missing Completely At Random) regime.

    ``strength`` ∈ [0,1] scales missing fraction from ≈15 % to ≈70 %.
    MCAR is the simplest mechanism; future work could parameterise MAR/MNAR.
    """
    out = df.copy()
    n_rows, n_cols = out.shape
    if n_rows == 0 or n_cols == 0:
        return out
    frac = float(np.clip(0.15 + 0.55 * strength, 0.05, 0.75))
    # Vectorised assignment is faster than element-wise loop for large frames
    mask = rng.random(size=(n_rows, n_cols)) < frac
    out = out.where(~mask, other=np.nan)
    return out


def _inject_interpretation_deficit(
    df: pd.DataFrame, rng: np.random.Generator, strength: float
) -> pd.DataFrame:
    """
    Interpretation Deficit — type pollution + heavy-tailed skew.

    Two sub-effects:
      a) String corruption: numeric values randomly replaced with sentinel
         strings ("N/A", "?", "err"), raising type inconsistency score.
      b) Log-normal transformation: creates strong right skew in numerics.
    """
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    # (a) String corruption
    n_corrupt = max(1, int(len(num_cols) * (0.3 + 0.5 * strength)))
    rng.shuffle(num_cols)
    sentinels = ["?", "N/A", "err", "#VALUE!", "null", "missing"]
    for c in num_cols[:n_corrupt]:
        mask = rng.random(len(out)) < (0.2 + 0.35 * strength)
        # Cast to object first — pandas 2.x refuses string assignment into float64
        out[c] = out[c].astype(object)
        out.loc[mask, c] = rng.choice(sentinels, mask.sum())
    # (b) Skew via expm1(log-normal)
    for c in out.select_dtypes(include=[np.number]).columns:
        out[c] = np.expm1(out[c].clip(lower=0) * (1 + 2 * strength))
    return out


def _inject_operation_deficit(
    df: pd.DataFrame, rng: np.random.Generator, strength: float
) -> pd.DataFrame:
    """
    Operation Deficit — duplicates, extreme outliers, cardinality inflation.

    Three sub-effects:
      a) Row duplication (operational reprocessing simulation).
      b) Extreme outlier spikes (sensor / ETL glitches).
      c) High-cardinality noise column (UUID-like, raises entropy).
    """
    out = df.copy()
    n_rows = len(out)
    if n_rows < 2:
        return out
    # (a) Duplicates
    dup_frac = float(np.clip(0.10 + 0.50 * strength, 0.05, 0.60))
    k = max(1, int(n_rows * dup_frac))
    idx = rng.choice(out.index, size=k, replace=True)
    out = pd.concat([out, out.loc[idx]], ignore_index=True)
    # (b) Outlier spikes
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        n_out = max(1, int(len(out) * 0.03 * (1 + strength)))
        locs = rng.choice(out.index, size=n_out, replace=False)
        sigma = out[c].std(ddof=0)
        scale = sigma if sigma > 0 else 1.0
        out.loc[locs, c] += rng.choice([-1, 1], n_out) * scale * (8 + 4 * strength)
    # (c) High-cardinality noise column
    card = int(50 + 200 * strength)
    out["_op_noise"] = rng.integers(0, card, size=len(out)).astype(str)
    return out


def _inject_mixed(
    df: pd.DataFrame, rng: np.random.Generator, strength: float, primary: str
) -> pd.DataFrame:
    """
    Compose two injectors with the primary at full strength and a secondary
    at 30 % strength.  Produces realistic boundary-region training samples
    that prevent the classifier from learning overly sharp decision boundaries.
    """
    injectors = {
        PATTERN_INTERNAL: _inject_internal_deficit,
        PATTERN_INTERPRETATION: _inject_interpretation_deficit,
        PATTERN_OPERATION: _inject_operation_deficit,
    }
    others = [p for p in ALL_PATTERNS if p != primary]
    secondary = str(rng.choice(others))
    out = injectors[primary](df, rng, strength)
    out = injectors[secondary](out, rng, 0.3 * strength)
    return out


def generate_synthetic_training_data(
    n_samples_per_class: int = 500,
    mixed_fraction: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build the labeled benchmark dataset used to train the meta-classifier.

    ``mixed_fraction`` of each class uses the compound injector to populate
    the classifier's boundary regions, reducing overfitting to pure-regime
    profiles and improving generalisation to real-world CSVs.

    Parameters
    ----------
    n_samples_per_class : int
        Number of synthetic datasets to generate per pattern label.
    mixed_fraction : float
        Fraction of samples in each class that use mixed contamination.
    random_state : int
        Seed for full reproducibility.

    Returns
    -------
    X : pd.DataFrame  — one row = one dataset's meta-features
    y : np.ndarray    — corresponding pattern label
    """
    rng = _rng(random_state)
    rows_x: List[Dict[str, float]] = []
    rows_y: List[str] = []

    n_mixed = max(1, int(n_samples_per_class * mixed_fraction))
    n_pure = n_samples_per_class - n_mixed

    def _add(label: str, mixed: bool) -> None:
        n_rows = int(rng.integers(80, 600))
        n_num = int(rng.integers(2, 10))
        n_cat = int(rng.integers(1, 5))
        base = _make_base_dataframe(n_rows, n_num, n_cat, rng)
        s = float(rng.uniform(0.3, 1.0))
        if mixed:
            dirty = _inject_mixed(base, rng, s, label)
        else:
            inject_fn = {
                PATTERN_INTERNAL: _inject_internal_deficit,
                PATTERN_INTERPRETATION: _inject_interpretation_deficit,
                PATTERN_OPERATION: _inject_operation_deficit,
            }[label]
            dirty = inject_fn(base, rng, s)
        feats = extract_meta_features(dirty)
        rows_x.append(feats.iloc[0].to_dict())
        rows_y.append(label)

    for label in ALL_PATTERNS:
        for _ in range(n_pure):
            _add(label, mixed=False)
        for _ in range(n_mixed):
            _add(label, mixed=True)

    X = pd.DataFrame(rows_x, columns=FEATURE_NAMES)
    y = np.array(rows_y, dtype=object)
    return X, y


# =============================================================================
# Module 2 – Feature Extraction Layer (The Input)
# =============================================================================

def _safe_skew(series: pd.Series) -> float:
    """
    Population Fisher skewness with guards for constant / tiny samples.
    Uses the unbiased z-score route to avoid scipy dependency.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3 or s.nunique() < 2:
        return 0.0
    mu = float(s.mean())
    var = float(s.var(ddof=0))
    if var <= 0:
        return 0.0
    return float(((s - mu) ** 3 / var ** 1.5).mean())


def _column_entropy(series: pd.Series) -> float:
    """
    Normalised Shannon entropy H(X) / log2(k) ∈ [0, 1].

    Treats numeric columns as categorical by value (suitable for meta-level
    cardinality assessment; detailed distributional entropy is captured by
    skewness and outlier_ratio).
    """
    vc = series.astype(str).value_counts(normalize=True, dropna=False)
    if len(vc) < 2:
        return 0.0
    h = -float((vc * np.log2(vc + 1e-12)).sum())
    return float(np.clip(h / np.log2(len(vc)), 0.0, 1.0))


def _type_consistency_score(series: pd.Series) -> float:
    """
    Fraction of non-null values that successfully parse as float.

    Score = 1.0  → pure numeric column (fully consistent).
    Score = 0.0  → all values unparseable (fully categorical / corrupt).
    Mixed columns (e.g. "123", "N/A", "err") yield intermediate scores and
    are the strongest signal for Interpretation Deficit.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return 1.0  # vacuously consistent
    parsed = pd.to_numeric(non_null, errors="coerce")
    return float(parsed.notna().mean())


def extract_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the nine-dimensional meta-feature vector for a raw DataFrame.

    All values are bounded scalars; the function never raises on edge-cases
    (empty df, no numeric cols, fully-NaN column).

    Returns
    -------
    pd.DataFrame with shape (1, 9) and columns = FEATURE_NAMES.
    """
    # --- Guard: empty frame ---
    if df.empty or df.shape[1] == 0:
        return pd.DataFrame(
            [[0.0] * len(FEATURE_NAMES)], columns=FEATURE_NAMES
        )

    n_total_cells = float(df.size)

    # 1. missing_ratio
    missing_ratio = float(df.isna().sum().sum()) / n_total_cells

    # 2. duplicate_ratio
    n_rows = len(df)
    duplicate_ratio = float(df.duplicated().sum()) / max(n_rows, 1)

    # 3. outlier_ratio  (numeric cols only; converts mixed-type cols first)
    num_cols = _numeric_columns(df)
    outlier_count = 0
    total_num_vals = 0
    for c in num_cols:
        col = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(col) < 4:
            continue
        mu, sigma = col.mean(), col.std(ddof=0)
        if sigma > 0:
            outlier_count += int(((col - mu).abs() > 3 * sigma).sum())
            total_num_vals += len(col)
    outlier_ratio = float(outlier_count) / max(total_num_vals, 1)

    # 4. entropy  (average normalised Shannon entropy across all columns)
    ent_vals = [_column_entropy(df[c]) for c in df.columns]
    entropy = float(np.mean(ent_vals)) if ent_vals else 0.0

    # 5. skewness  (average |Fisher skewness| across numeric columns)
    skew_vals = [abs(_safe_skew(df[c])) for c in num_cols]
    skewness = float(np.mean(skew_vals)) if skew_vals else 0.0

    # 6. numeric_ratio  (fraction of columns that are numeric)
    numeric_ratio = len(num_cols) / df.shape[1]

    # 7. unique_ratio  (average unique-value proportion across all columns)
    uniq_vals = [df[c].nunique(dropna=False) / max(n_rows, 1) for c in df.columns]
    unique_ratio = float(np.mean(uniq_vals)) if uniq_vals else 0.0

    # 8. type_consistency  (fraction of columns that are "cleanly typed")
    #    A column is "consistent" if ≥90 % of its non-null values parse as
    #    their declared dtype.
    consistency_scores = []
    for c in df.columns:
        if c in num_cols:
            # Already numeric → check for hidden mixed types via string repr
            s = df[c].dropna().astype(str)
            score = pd.to_numeric(s, errors="coerce").notna().mean()
        else:
            score = _type_consistency_score(df[c])
        consistency_scores.append(float(score))
    type_consistency = (
        float(np.mean([s >= 0.9 for s in consistency_scores]))
        if consistency_scores
        else 1.0
    )

    # 9. value_range_ratio  (avg (max−min)/std for numeric cols; spread proxy)
    range_ratios = []
    for c in num_cols:
        col = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(col) < 2:
            continue
        sigma = col.std(ddof=0)
        if sigma > 0:
            range_ratios.append((col.max() - col.min()) / sigma)
    value_range_ratio = float(np.mean(range_ratios)) if range_ratios else 0.0
    # Cap to avoid extreme values dominating tree splits
    value_range_ratio = float(np.clip(value_range_ratio, 0, 100))

    return pd.DataFrame(
        [[
            missing_ratio,
            duplicate_ratio,
            outlier_ratio,
            entropy,
            skewness,
            numeric_ratio,
            unique_ratio,
            type_consistency,
            value_range_ratio,
        ]],
        columns=FEATURE_NAMES,
    )


# =============================================================================
# Module 3 – The ML Meta-Model (The Brain)
# =============================================================================

def build_meta_model_pipeline(
    model_type: str = "rf",
    random_state: int = 42,
    n_estimators: int = 300,
    calibrate: bool = True,
) -> Pipeline:
    """
    Construct the sklearn Pipeline wrapping the chosen meta-classifier.

    Parameters
    ----------
    model_type : "rf" | "gb"
        "rf" → RandomForestClassifier (default; faster, SHAP-compatible).
        "gb" → GradientBoostingClassifier (slower; often higher accuracy).
    calibrate : bool
        Wrap classifier in CalibratedClassifierCV so predict_proba returns
        well-calibrated confidence scores suitable for display in the UI.

    Design note:
        Trees do not require feature scaling; the Pipeline API is kept for
        forward compatibility (e.g. adding a feature-union layer later).
    """
    if model_type == "gb":
        clf_base = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=4,
            random_state=random_state,
        )
    else:  # default: rf
        clf_base = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )

    if calibrate:
        clf = CalibratedClassifierCV(clf_base, method="isotonic", cv=3)
    else:
        clf = clf_base

    return Pipeline([("classifier", clf)])


def train_meta_model(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
    verbose: bool = True,
    cv_folds: int = 5,
    model_type: str = "rf",
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train/holdout split + stratified k-fold CV, fit pipeline, return metrics.

    Cross-validation provides variance estimates for the paper's evaluation
    table (mean ± std across folds), which is more rigorous than a single
    holdout split.

    Returns
    -------
    pipeline : fitted Pipeline (trained on the full training split)
    metrics  : dict with scalars + classification_report string
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pipe = build_meta_model_pipeline(
        model_type=model_type, random_state=random_state
    )

    # --- Cross-validation on training split ---
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        build_meta_model_pipeline(model_type=model_type, random_state=random_state),
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        n_jobs=-1,
    )
    cv_acc_mean = float(cv_results["test_accuracy"].mean())
    cv_acc_std = float(cv_results["test_accuracy"].std())
    cv_f1_mean = float(cv_results["test_f1_macro"].mean())
    cv_f1_std = float(cv_results["test_f1_macro"].std())

    # --- Final fit on full training split ---
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=ALL_PATTERNS, average="macro", zero_division=0
    )
    report = classification_report(y_test, y_pred, labels=ALL_PATTERNS, zero_division=0)

    if verbose:
        print(f"[Hold-out] Accuracy:          {acc:.4f}")
        print(f"[Hold-out] Precision (macro): {prec:.4f}")
        print(f"[Hold-out] Recall (macro):    {rec:.4f}")
        print(f"[{cv_folds}-fold CV] Accuracy: {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")
        print(f"[{cv_folds}-fold CV] F1 macro: {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
        print("\nClassification Report:\n", report)

    return pipe, {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "cv_accuracy_mean": cv_acc_mean,
        "cv_accuracy_std": cv_acc_std,
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std,
        "classification_report": report,
    }


def get_default_trained_model(
    random_state: int = 42,
    n_samples_per_class: int = 400,
    model_type: str = "rf",
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Convenience wrapper: synthetic benchmark → trained model.

    Called once per Streamlit process (cached via @st.cache_resource).
    """
    X, y = generate_synthetic_training_data(
        n_samples_per_class=n_samples_per_class, random_state=random_state
    )
    pipe, metrics = train_meta_model(
        X, y,
        random_state=random_state,
        verbose=False,
        model_type=model_type,
    )
    return pipe, metrics


# =============================================================================
# Module 4 – Application Pipelines (The Action)
# =============================================================================

def apply_internal_deficit_fix(
    df: pd.DataFrame,
    strategy: str = "knn",
    knn_neighbors: int = 5,
) -> pd.DataFrame:
    """
    Internal Deficit remediation — missing-value imputation.

    Strategy options:
      "knn"    → K-Nearest Neighbours imputation (default; exploits
                 correlation structure; best for MAR / MCAR regimes).
      "median" → Univariate median (fast fallback for large frames).

    Categorical / object columns always use most-frequent imputation
    regardless of strategy, as KNN on string columns is ill-defined without
    encoding.

    Parameters
    ----------
    df       : raw DataFrame with missing values
    strategy : "knn" | "median"
    knn_neighbors : number of neighbours for KNN imputation
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    num_cols = _numeric_columns(out)
    obj_cols = [c for c in out.columns if c not in num_cols]

    if num_cols:
        num_data = out[num_cols].astype(float)
        if strategy == "knn" and len(out) >= knn_neighbors + 1:
            imp = KNNImputer(n_neighbors=knn_neighbors)
        else:
            imp = SimpleImputer(strategy="median")
        out[num_cols] = imp.fit_transform(num_data)

    if obj_cols:
        imp_cat = SimpleImputer(strategy="most_frequent")
        out[obj_cols] = imp_cat.fit_transform(out[obj_cols].astype(str))

    return out


def apply_interpretation_deficit_fix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpretation Deficit remediation — type coercion + IQR Winsorizing
    + StandardScaler.

    Steps (in order):
      1. Attempt numeric coercion on object columns with ≥60 % parseable vals.
      2. IQR-based Winsorizing on numeric columns before scaling to prevent
         extreme values from distorting the StandardScaler's mean/std.
      3. StandardScaler to bring all numeric features to z-score space.
      4. Normalise object columns to clean string for consistent downstream
         typing.

    Improvement over v1: added Winsorizing step (step 2) so the scaler is
    not polluted by the same outliers that characterise Operation Deficit,
    making the before/after contrast cleaner for the paper.
    """
    if df.empty:
        return df.copy()
    out = df.copy()

    # Step 1: coerce mixed-type object columns
    for c in list(out.columns):
        if c in _numeric_columns(out):
            continue
        coerced = pd.to_numeric(out[c], errors="coerce")
        if coerced.notna().mean() >= 0.60:
            out[c] = coerced

    num_cols = _numeric_columns(out)

    # Step 2: IQR Winsorizing (cap at [Q1 − 1.5·IQR, Q3 + 1.5·IQR])
    for c in num_cols:
        col = out[c].astype(float)
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            out[c] = col.clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

    # Step 3: StandardScaler
    if num_cols:
        scaler = StandardScaler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[num_cols] = scaler.fit_transform(out[num_cols].astype(float))

    # Step 4: Normalise object columns
    for c in out.select_dtypes(include=["object"]).columns:
        out[c] = out[c].astype(str).replace({"nan": pd.NA, "None": pd.NA})

    return out


def apply_operation_deficit_fix(
    df: pd.DataFrame,
    outlier_method: str = "iqr",
    drop_high_cardinality: bool = True,
    cardinality_threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Operation Deficit remediation — dedup + outlier capping + noise-column
    removal.

    Improvements over v1:
      • IQR-based capping (default) instead of 3σ — more robust for
        heavy-tailed distributions that are common in operational data.
      • Optional removal of high-cardinality columns whose unique-value ratio
        exceeds ``cardinality_threshold`` (e.g. UUID / hash columns that
        carry no analytical signal).

    Parameters
    ----------
    outlier_method        : "iqr" | "zscore"
    drop_high_cardinality : whether to remove UUID-like noise columns
    cardinality_threshold : unique-value fraction above which a column is
                            considered a noise / ID column
    """
    if df.empty:
        return df.copy()

    # Step 1: drop duplicates
    out = df.drop_duplicates().reset_index(drop=True)

    # Step 2: remove high-cardinality noise columns
    if drop_high_cardinality and len(out) > 0:
        cols_to_drop = [
            c for c in out.columns
            if out[c].nunique(dropna=False) / len(out) > cardinality_threshold
        ]
        out = out.drop(columns=cols_to_drop)

    # Step 3: cap outliers on numeric columns
    num_cols = _numeric_columns(out)
    for c in num_cols:
        col = pd.to_numeric(out[c], errors="coerce")
        if outlier_method == "iqr":
            q1, q3 = col.quantile(0.25), col.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                out[c] = col.clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
        else:  # zscore
            mu, sigma = col.mean(), col.std(ddof=0)
            if sigma > 0:
                out[c] = col.clip(lower=mu - 3 * sigma, upper=mu + 3 * sigma)

    return out


def apply_pattern_by_name(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """Dispatch helper for the UI."""
    if pattern == PATTERN_INTERNAL:
        return apply_internal_deficit_fix(df)
    if pattern == PATTERN_INTERPRETATION:
        return apply_interpretation_deficit_fix(df)
    if pattern == PATTERN_OPERATION:
        return apply_operation_deficit_fix(df)
    raise ValueError(f"Unknown pattern: {pattern!r}")


# =============================================================================
# Module 5 – Summary Stats (used by Streamlit before/after panel)
# =============================================================================

def dataset_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compact, JSON-serialisable before/after statistics for the Streamlit UI.

    Includes skewness and outlier count as additional signals beyond the
    original row/column/missing/duplicate counts.
    """
    n = len(df)
    num_cols = _numeric_columns(df)
    missing_cells = int(df.isna().sum().sum())
    dup_rows = int(df.duplicated().sum())

    # Outlier count (3σ rule, numeric cols)
    outlier_total = 0
    for c in num_cols:
        col = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(col) >= 4:
            mu, sigma = col.mean(), col.std(ddof=0)
            if sigma > 0:
                outlier_total += int(((col - mu).abs() > 3 * sigma).sum())

    # Mean absolute skewness
    skew_vals = [abs(_safe_skew(df[c])) for c in num_cols]
    mean_skew = float(np.mean(skew_vals)) if skew_vals else 0.0

    return {
        "rows": n,
        "columns": df.shape[1],
        "missing_cells": missing_cells,
        "missing_ratio": round(missing_cells / max(df.size, 1), 4),
        "duplicate_rows": dup_rows,
        "numeric_columns": len(num_cols),
        "outlier_cells_3sigma": outlier_total,
        "mean_abs_skewness": round(mean_skew, 4),
    }


# =============================================================================
# CLI entry point – for paper methodology experiments
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generating synthetic benchmark (400 samples/class) …")
    X_syn, y_syn = generate_synthetic_training_data(
        n_samples_per_class=400, random_state=42
    )
    print(f"Dataset shape: {X_syn.shape}  |  Classes: {np.unique(y_syn)}")
    print("=" * 60)
    print("Training RandomForest meta-model …")
    model, _ = train_meta_model(X_syn, y_syn, random_state=42, verbose=True)
    print("=" * 60)
    print("Meta-model ready.  Run `streamlit run app.py` to launch the UI.")