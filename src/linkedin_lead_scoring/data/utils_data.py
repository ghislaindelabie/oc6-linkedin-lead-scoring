"""
Utility functions for LinkedIn lead scoring data processing.

Adapted from OC4 HR attrition project with LinkedIn-specific extensions.
**MLflow integration from the start** for experiment tracking.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import Counter
import mlflow
from pathlib import Path


# ============================================================================
# MLFLOW SETUP & TRACKING
# ============================================================================

def find_project_root(marker_files: Tuple[str, ...] = ("pyproject.toml", ".git", "setup.py")) -> Path:
    """
    Find project root by looking for marker files.
    Handles paths with spaces correctly using pathlib.

    Parameters
    ----------
    marker_files : tuple of str
        Files that indicate the project root.

    Returns
    -------
    Path
        Absolute path to project root.

    Raises
    ------
    FileNotFoundError
        If project root cannot be found.
    """
    current = Path.cwd().resolve()

    # Start from current directory and traverse up
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in marker_files):
            return parent

    raise FileNotFoundError(
        f"Could not find project root. Looking for: {marker_files}\n"
        f"Started from: {current}"
    )


def configure_mlflow_tracking(tracking_dir: str = "mlruns") -> Path:
    """
    Configure MLflow to use project root for tracking.
    Automatically finds project root and sets tracking URI.

    Parameters
    ----------
    tracking_dir : str, default "mlruns"
        Name of the tracking directory (relative to project root).

    Returns
    -------
    Path
        Absolute path to the tracking directory.
    """
    project_root = find_project_root()
    tracking_path = project_root / tracking_dir

    # Use file:// URI scheme with proper path handling
    tracking_uri = tracking_path.as_uri()
    mlflow.set_tracking_uri(tracking_uri)

    print("✓ MLflow tracking configured")
    print(f"  Project root: {project_root}")
    print(f"  Tracking URI: {tracking_uri}")

    return tracking_path


def setup_mlflow(experiment_name: str = "linkedin-lead-scoring", auto_configure: bool = True) -> str:
    """
    Initialize MLflow experiment for tracking.

    Parameters
    ----------
    experiment_name : str, default "linkedin-lead-scoring"
        Name of the MLflow experiment.
    auto_configure : bool, default True
        Automatically configure tracking URI to project root.
        Set to False if you've already configured tracking manually.

    Returns
    -------
    str
        Experiment ID.

    Example
    -------
    >>> # Automatically finds project root and configures tracking
    >>> experiment_id = setup_mlflow(experiment_name="linkedin-lead-scoring")
    >>> mlflow.start_run(run_name="data_prep")
    """
    # Auto-configure tracking to use project root
    if auto_configure:
        configure_mlflow_tracking()

    # Set experiment (creates if doesn't exist)
    mlflow.set_experiment(experiment_name)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    print(f"✓ MLflow experiment: '{experiment_name}'")
    print(f"  Experiment ID: {experiment.experiment_id}")
    print(f"  Artifact location: {experiment.artifact_location}")

    return experiment.experiment_id


def log_data_profile(df: pd.DataFrame, stage: str = "raw") -> None:
    """
    Log dataset profile to current MLflow run.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to profile.
    stage : str, default "raw"
        Data processing stage (raw, labeled, cleaned, etc.).
    """
    if mlflow.active_run() is None:
        print("⚠️  No active MLflow run. Skipping logging.")
        return

    # Log basic metrics
    mlflow.log_param(f"{stage}_n_rows", len(df))
    mlflow.log_param(f"{stage}_n_cols", len(df.columns))
    mlflow.log_param(f"{stage}_memory_mb", df.memory_usage(deep=True).sum() / 1024**2)

    # Log column types
    type_counts = df.dtypes.value_counts().to_dict()
    for dtype, count in type_counts.items():
        mlflow.log_metric(f"{stage}_cols_{dtype}", count)

    print(f"✓ Logged {stage} data profile to MLflow")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_linkedin_csv(path: str, log_to_mlflow: bool = True) -> pd.DataFrame:
    """
    Load LinkedIn contact list CSV from LemList export.

    Parameters
    ----------
    path : str
        Path to CSV file.
    log_to_mlflow : bool, default True
        Log loading metadata to MLflow.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all columns.
    """
    df = pd.read_csv(path)
    print(f"✓ Loaded {len(df)} contacts with {len(df.columns)} columns")

    if log_to_mlflow and mlflow.active_run():
        mlflow.log_param("data_source", path)
        log_data_profile(df, stage="raw")

    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case, strip spaces.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Same DataFrame with renamed columns.
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# ============================================================================
# LABEL CREATION (LinkedIn-specific with MLflow logging)
# ============================================================================

def create_engagement_labels(df: pd.DataFrame, log_to_mlflow: bool = True) -> pd.DataFrame:
    """
    Create binary target variable based on LinkedIn engagement (reply or interest).

    Labeling logic:
    - Label = 1 (engaged): laststate == "linkedinReplied" OR "linkedinInterested"
    - Label = 0 (not engaged): status == "done" AND laststate != {"linkedinReplied" OR "linkedinInterested"}
    - Filter out: Everything else (inProgress, reviewed, etc.)

    Logs label distribution to MLflow.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'laststate' and 'status' columns (lowercase after standardization).
    log_to_mlflow : bool, default True
        Log labeling statistics to MLflow.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with 'engaged' column (0 or 1).
    """
    df = df.copy()

    # Define engagement states
    engaged_states = ['linkedinReplied', 'linkedinInterested']

    # Create label column
    df['engaged'] = np.nan

    # Positive class: replied or showed interest
    df.loc[df['laststate'].isin(engaged_states), 'engaged'] = 1

    # Negative class: done but didn't engage
    df.loc[
        (df['status'] == 'done') & (~df['laststate'].isin(engaged_states)),
        'engaged'
    ] = 0

    # Filter out unlabeled rows
    n_before = len(df)
    df = df[df['engaged'].notna()].copy()
    df['engaged'] = df['engaged'].astype(int)
    n_after = len(df)
    n_positive = (df['engaged'] == 1).sum()
    n_negative = (df['engaged'] == 0).sum()
    n_filtered = n_before - n_after

    print(f"✓ Created labels: {n_before} → {n_after} contacts")
    print(f"  Positive (engaged=1): {n_positive} contacts ({n_positive/n_after*100:.1f}%)")
    print(f"  Negative (engaged=0): {n_negative} contacts ({n_negative/n_after*100:.1f}%)")
    print(f"  Filtered out: {n_filtered} contacts (unlabeled)")

    # Log to MLflow
    if log_to_mlflow and mlflow.active_run():
        mlflow.log_metric("n_contacts_before_labeling", n_before)
        mlflow.log_metric("n_contacts_after_labeling", n_after)
        mlflow.log_metric("n_positive_labels", n_positive)
        mlflow.log_metric("n_negative_labels", n_negative)
        mlflow.log_metric("n_filtered_contacts", n_filtered)
        mlflow.log_metric("positive_rate", n_positive / n_after)
        mlflow.log_metric("class_imbalance_ratio", n_positive / n_negative if n_negative > 0 else 0)
        print("✓ Logged label statistics to MLflow")

    return df


def select_modeling_features(
    df: pd.DataFrame,
    drop_pii: bool = True,
    drop_ids: bool = True,
    log_to_mlflow: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select relevant features for modeling, removing PII and unnecessary columns.

    PII columns to drop: email, firstName, lastName, phone, linkedinUrl, picture, _id

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    drop_pii : bool, default True
        Drop personally identifiable information columns.
    drop_ids : bool, default True
        Drop ID columns (_id, emailStatus).
    log_to_mlflow : bool, default True
        Log feature selection to MLflow.

    Returns
    -------
    (df_clean, dropped_cols)
        Cleaned dataframe and list of dropped column names.
    """
    df = df.copy()
    dropped = []

    # PII columns
    pii_cols = [
        'email', 'firstname', 'lastname', 'phone', 'linkedinurl',
        'picture', 'companypicture', 'companylinkedinurl'
    ]

    # ID and status tracking columns
    id_cols = ['_id', 'emailstatus', 'laststate', 'status']

    # Combine based on flags
    to_drop = []
    if drop_pii:
        to_drop.extend(pii_cols)
    if drop_ids:
        to_drop.extend(id_cols)

    # Drop existing columns
    for col in to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            dropped.append(col)

    print(f"✓ Dropped {len(dropped)} columns")
    print(f"  Remaining: {len(df.columns)} columns")

    # Log to MLflow
    if log_to_mlflow and mlflow.active_run():
        mlflow.log_param("n_features_dropped", len(dropped))
        mlflow.log_param("n_features_remaining", len(df.columns) - 1)  # -1 for target
        mlflow.log_param("dropped_columns", str(dropped))
        log_data_profile(df, stage="feature_selected")
        print("✓ Logged feature selection to MLflow")

    return df, dropped


# ============================================================================
# DATA QUALITY ASSESSMENT (Adapted from OC4)
# ============================================================================

def assess_missingness(
    df: pd.DataFrame,
    quasi_constant_threshold: float = 0.98,
    high_missing_threshold: float = 0.80,
    moderate_missing_threshold: float = 0.40,
    log_to_mlflow: bool = True
) -> Dict[str, object]:
    """
    Profile missingness and low-variance columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to analyze.
    quasi_constant_threshold : float, default 0.98
        Minimum ratio of the most frequent value to flag as quasi-constant.
    high_missing_threshold : float, default 0.80
        Fraction of missing values to flag as high-missing.
    moderate_missing_threshold : float, default 0.40
        Fraction of missing values to flag as moderate-missing.
    log_to_mlflow : bool, default True
        Log quality metrics to MLflow.

    Returns
    -------
    Dict[str, object]
        {
          "profile": pd.DataFrame,
          "constant_cols": List[str],
          "quasi_constant_cols": List[str],
          "high_missing_cols": List[str],
          "moderate_missing_cols": List[str]
        }
    """
    n_rows = len(df)

    # Missingness overview
    missing = df.isna().sum().to_frame("n_missing")
    missing["pct_missing"] = missing["n_missing"] / n_rows if n_rows else 0.0

    # True-constant columns
    nunique = df.nunique(dropna=False)
    constant_cols: List[str] = nunique[nunique <= 1].index.tolist()

    # Quasi-constant columns
    def _top_ratio(s: pd.Series) -> float:
        if s.empty:
            return 1.0
        vc = s.value_counts(normalize=True, dropna=False)
        return float(vc.iloc[0]) if len(vc) else 1.0

    top_ratios = df.apply(_top_ratio)
    quasi_constant_cols = (
        top_ratios[top_ratios >= quasi_constant_threshold]
        .index.difference(constant_cols)
        .tolist()
    )

    # Build profiling table
    profile_rows = []
    for col in df.columns:
        vc = df[col].value_counts(dropna=False)
        first_val = vc.index[0] if len(vc) else np.nan
        first_cnt = int(vc.iloc[0]) if len(vc) else 0

        profile_rows.append({
            "column": col,
            "dtype": df[col].dtype,
            "nunique": int(nunique[col]),
            "n_missing": int(missing.loc[col, "n_missing"]),
            "pct_missing": float(missing.loc[col, "pct_missing"]),
            "top_value": first_val,
            "top_ratio": (first_cnt / n_rows) if n_rows else np.nan,
        })

    profile = pd.DataFrame(profile_rows).sort_values(
        ["pct_missing", "top_ratio"], ascending=[False, False]
    ).reset_index(drop=True)

    # Shortlists by thresholds
    high_missing_cols = missing.index[
        missing["pct_missing"] >= high_missing_threshold
    ].tolist()
    moderate_missing_cols = missing.index[
        (missing["pct_missing"] >= moderate_missing_threshold)
        & (missing["pct_missing"] < high_missing_threshold)
    ].tolist()

    # Log to MLflow
    if log_to_mlflow and mlflow.active_run():
        mlflow.log_metric("n_constant_cols", len(constant_cols))
        mlflow.log_metric("n_quasi_constant_cols", len(quasi_constant_cols))
        mlflow.log_metric("n_high_missing_cols", len(high_missing_cols))
        mlflow.log_metric("n_moderate_missing_cols", len(moderate_missing_cols))
        mlflow.log_metric("total_missing_pct", missing["pct_missing"].mean())
        print("✓ Logged data quality metrics to MLflow")

    return {
        "profile": profile,
        "constant_cols": constant_cols,
        "quasi_constant_cols": quasi_constant_cols,
        "high_missing_cols": high_missing_cols,
        "moderate_missing_cols": moderate_missing_cols,
    }


def value_frequencies(
    df: pd.DataFrame,
    column: str,
    split: bool = False,
    sep: str = ","
) -> Tuple[int, pd.DataFrame]:
    """
    Count value frequencies in a column.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    column : str
        Column name to analyze.
    split : bool, default False
        If True, split cell values by separator.
    sep : str, default ","
        Separator for split mode.

    Returns
    -------
    n_unique : int
        Number of unique values.
    freq_df : pd.DataFrame
        Frequency table sorted by count (descending).
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not in DataFrame")

    series = df[column].dropna().astype(str)

    # Flatten if needed
    if split:
        all_values = []
        for entry in series:
            parts = [v.strip() for v in entry.split(sep) if v.strip()]
            all_values.extend(parts)
    else:
        all_values = series.tolist()

    counter = Counter(all_values)
    total = sum(counter.values())

    freq_df = (
        pd.DataFrame(counter.items(), columns=[column, "Frequency"])
        .sort_values("Frequency", ascending=False)
        .reset_index(drop=True)
    )
    freq_df["Percentage"] = freq_df["Frequency"] / total * 100

    return freq_df.shape[0], freq_df


def engagement_rate_by_category(
    df: pd.DataFrame,
    cat_col: str,
    target: str = "engaged"
) -> pd.DataFrame:
    """
    Calculate engagement rate by categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with target column.
    cat_col : str
        Categorical column name.
    target : str, default 'engaged'
        Binary target column (0/1).

    Returns
    -------
    pd.DataFrame
        Table with columns [cat_col, 'n', 'engagement_rate'].
    """
    grouped = (
        df.groupby(cat_col)[target]
        .agg(n="count", engagement_rate=lambda x: (x == 1).mean())
        .reset_index()
        .sort_values("engagement_rate", ascending=False)
    )
    return grouped


# ============================================================================
# FEATURE ANALYSIS (Adapted from OC4)
# ============================================================================

def suggest_correlated_features(
    df: pd.DataFrame,
    cols: Optional[Iterable[str]] = None,
    *,
    threshold: float = 0.90,
    method: str = "pearson",
    protect: Optional[Iterable[str]] = None,
    return_pairs: bool = False,
    log_to_mlflow: bool = True
) -> Tuple[List[str], Optional[pd.DataFrame]]:
    """
    Suggest numeric features to drop based on correlation threshold.

    Parameters
    ----------
    df : pd.DataFrame
    cols : Iterable[str], optional
        Columns to analyze.
    threshold : float, default 0.90
        Correlation threshold.
    method : str, default "pearson"
        Correlation method.
    protect : Iterable[str], optional
        Columns to never drop.
    return_pairs : bool, default False
        Return correlation pairs dataframe.
    log_to_mlflow : bool, default True
        Log correlation analysis to MLflow.

    Returns
    -------
    (drop_list, pairs_df or None)
    """
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    else:
        cols = [c for c in cols if c in df.columns]
    cols = list(dict.fromkeys(cols))

    if len(cols) <= 1:
        return [], (pd.DataFrame(columns=["col_i","col_j","abs_corr"]) if return_pairs else None)

    corr = df[cols].corr(method=method).abs()
    np.fill_diagonal(corr.values, 0.0)

    iu = np.triu_indices_from(corr, k=1)
    pairs = [(cols[i], cols[j], corr.iat[i, j]) for i, j in zip(*iu) if corr.iat[i, j] >= threshold]

    if not pairs:
        if log_to_mlflow and mlflow.active_run():
            mlflow.log_metric("n_correlated_pairs", 0)
        return [], (pd.DataFrame(columns=["col_i","col_j","abs_corr"]) if return_pairs else None)

    pairs_df = pd.DataFrame(pairs, columns=["col_i","col_j","abs_corr"]).sort_values("abs_corr", ascending=False)

    mean_abs = corr.mean(axis=0)
    to_drop: set = set()
    protected = set(protect or [])

    for _, row in pairs_df.iterrows():
        a, b = row["col_i"], row["col_j"]
        if a in to_drop or b in to_drop:
            continue
        if a in protected and b in protected:
            continue
        if a in protected:
            to_drop.add(b)
            continue
        if b in protected:
            to_drop.add(a)
            continue
        drop_candidate = a if mean_abs[a] >= mean_abs[b] else b
        to_drop.add(drop_candidate)

    drop_list = [c for c in cols if c in to_drop]

    # Log to MLflow
    if log_to_mlflow and mlflow.active_run():
        mlflow.log_metric("n_correlated_pairs", len(pairs))
        mlflow.log_metric("n_features_to_drop_correlation", len(drop_list))
        mlflow.log_param("correlation_threshold", threshold)
        print(f"✓ Logged correlation analysis to MLflow ({len(drop_list)} features to drop)")

    return (drop_list, pairs_df) if return_pairs else (drop_list, None)


def split_feature_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    """
    Split features into numeric and categorical lists.

    Parameters
    ----------
    df : pd.DataFrame
    target : str
        Name of target column.

    Returns
    -------
    (numeric_features, categorical_features)
    """
    num_cols = df.drop(columns=[target]).select_dtypes(include="number").columns.tolist()
    cat_cols = df.drop(columns=[target]).select_dtypes(exclude="number").columns.tolist()
    return num_cols, cat_cols
