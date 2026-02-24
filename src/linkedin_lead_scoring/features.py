"""Shared feature engineering module for training and inference.

Pure functions — only depends on pandas and numpy.
Importable by both ``scripts/export_model.py`` (training) and the FastAPI
serving layer (``api/predict.py``) so the feature contract is kept in sync.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants (single source of truth — also used by export_model.py)
# ---------------------------------------------------------------------------

TEXT_COLS = ["summary", "skills", "jobtitle"]

LOW_CARDINALITY_CATS = [
    "llm_seniority",
    "llm_geography",
    "llm_business_type",
    "companysize",
    "companytype",
]

TARGET_ENCODE_CATS = [
    "llm_industry",
    "industry",
    "companyindustry",
    "languages",
    "location",
    "companylocation",
]

NUMERIC_COLS = [
    "llm_quality",
    "llm_engagement",
    "llm_decision_maker",
    "llm_company_fit",
    "companyfoundedon",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace raw text columns with engineered numeric features.

    Extracts completeness flags, length features, skills count, and job
    title role indicators.  Drops the original text columns.

    Args:
        df: DataFrame that may contain summary, skills, and jobtitle columns.

    Returns:
        DataFrame with text columns replaced by numeric features.
    """
    # Ensure text columns exist (may be absent for empty leads)
    for col in TEXT_COLS:
        if col not in df.columns:
            df[col] = None

    # Completeness flags
    df["has_summary"] = df["summary"].notna().astype(int)
    df["has_skills"] = df["skills"].notna().astype(int)
    df["has_jobtitle"] = df["jobtitle"].notna().astype(int)

    # Length features
    df["summary_length"] = df["summary"].str.len().fillna(0).astype(int)
    df["skills_count"] = (
        df["skills"]
        .str.split(",")
        .apply(lambda x: len(x) if isinstance(x, list) else 0)
    )
    df["jobtitle_length"] = df["jobtitle"].str.len().fillna(0).astype(int)

    # Job title role flags
    jobtitle = df["jobtitle"].fillna("")
    df["is_founder"] = jobtitle.str.contains(
        r"Founder|CEO|CTO|Co-founder|Co-Founder", case=False, regex=True,
    ).astype(int)
    df["is_director"] = jobtitle.str.contains(
        r"Director|VP|Vice President", case=False, regex=True,
    ).astype(int)
    df["is_manager"] = jobtitle.str.contains(
        r"Manager|Lead|Head of", case=False, regex=True,
    ).astype(int)
    df["is_sales"] = jobtitle.str.contains(
        r"Sales|Business Development", case=False, regex=True,
    ).astype(int)
    df["is_marketing"] = jobtitle.str.contains(
        r"Marketing|Growth|CMO", case=False, regex=True,
    ).astype(int)
    df["is_tech_role"] = jobtitle.str.contains(
        r"Engineer|Developer|Architect|CTO", case=False, regex=True,
    ).astype(int)

    df = df.drop(columns=[c for c in TEXT_COLS if c in df.columns])
    return df


# ---------------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------------

def fill_missing_values(
    df: pd.DataFrame,
    numeric_medians: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Fill missing values: numerics with training median, categoricals with "UNKNOWN".

    Args:
        df: DataFrame to fill.
        numeric_medians: Mapping of column name to median value from training.
            Falls back to 0 for any missing column or when ``None``.

    Returns:
        DataFrame with NaN values filled.
    """
    medians = numeric_medians or {}

    for col in NUMERIC_COLS:
        if col in df.columns:
            fill_val = medians.get(col, 0)
            df[col] = df[col].fillna(fill_val)

    all_cats = LOW_CARDINALITY_CATS + TARGET_ENCODE_CATS
    for col in all_cats:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    return df


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------

def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode low-cardinality categorical columns (``drop_first=True``).

    Boolean dummy columns are cast to int for XGBoost compatibility.
    """
    ohe_cols = [c for c in LOW_CARDINALITY_CATS if c in df.columns]
    if not ohe_cols:
        return df

    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # Cast boolean dummies to int
    bool_cols = df.select_dtypes(include="bool").columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)

    return df


# ---------------------------------------------------------------------------
# Column alignment
# ---------------------------------------------------------------------------

def align_columns(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Align DataFrame columns to the model's expected feature list.

    - Missing columns are added with value **0** (not NaN) because absent
      one-hot columns legitimately mean "not that category".
    - Extra columns are dropped.
    - Column order matches *feature_columns* exactly.
    """
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]


# ---------------------------------------------------------------------------
# Full inference pipeline
# ---------------------------------------------------------------------------

def preprocess_for_inference(
    df: pd.DataFrame,
    target_encoder,
    te_cols: list[str],
    feature_columns: list[str],
    numeric_medians: dict[str, float] | None = None,
) -> pd.DataFrame:
    """End-to-end preprocessing for a batch of raw leads.

    Orchestrates: text features → fill missing → target-encode → one-hot → align.

    Args:
        df: Raw lead DataFrame (one row per lead, raw field names).
        target_encoder: Fitted ``TargetEncoder`` instance or ``None``.
        te_cols: Columns the target encoder was fitted on.
        feature_columns: Ordered list of model input columns.
        numeric_medians: Training medians for numeric imputation.

    Returns:
        Numeric DataFrame ready for ``model.predict_proba()``.
    """
    df = df.copy()

    # 1. Text feature engineering
    df = extract_text_features(df)

    # 2. Fill missing values
    df = fill_missing_values(df, numeric_medians=numeric_medians)

    # 3. Target-encode high-cardinality categoricals (if encoder available)
    if target_encoder is not None and te_cols:
        present_te_cols = [c for c in te_cols if c in df.columns]
        if present_te_cols:
            df[present_te_cols] = target_encoder.transform(df[present_te_cols])

    # 4. One-hot encode low-cardinality categoricals
    df = one_hot_encode(df)

    # 5. Drop any remaining non-numeric columns (e.g. un-encoded categoricals
    #    when target_encoder is None).  align_columns will fill them with 0.
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        df = df.drop(columns=non_numeric)

    # 6. Align to expected feature columns (add missing=0, drop extra, reorder)
    df = align_columns(df, feature_columns)

    return df
