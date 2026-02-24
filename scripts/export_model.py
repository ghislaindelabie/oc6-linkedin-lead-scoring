"""
Export the best XGBoost model with preprocessing pipeline.

Re-trains the production model using known best hyperparameters on the full
cleaned dataset, then saves all artifacts needed for API serving and drift
monitoring.

Usage:
    python scripts/export_model.py [--data-path PATH] [--output-dir DIR]

Artifacts produced:
    model/xgboost_model.joblib     — trained XGBoost classifier
    model/preprocessor.joblib      — fitted preprocessing pipeline
    model/feature_columns.json     — ordered list of feature column names
    model/numeric_medians.json     — training medians for numeric imputation
    data/reference/training_reference.csv — first 100 rows of training data
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from linkedin_lead_scoring.features import (
    LOW_CARDINALITY_CATS,
    NUMERIC_COLS,
    TARGET_ENCODE_CATS,
    TEXT_COLS,
    extract_text_features,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BEST_PARAMS = {
    "n_estimators": 255,
    "max_depth": 3,
    "learning_rate": 0.121,
    "min_child_weight": 7,
    "subsample": 0.784,
    "colsample_bytree": 0.988,
    "gamma": 3.513,
    "scale_pos_weight": 2.501,
    "eval_metric": "logloss",
    "random_state": 42,
}

DATA_RELATIVE_PATHS = [
    Path("data/processed/linkedin_leads_clean.csv"),
    Path("../../data/processed/linkedin_leads_clean.csv"),
]


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def find_data_file(
    explicit_path: Optional[Path] = None,
    search_root: Optional[Path] = None,
) -> Path:
    """Locate the cleaned CSV data file.

    Args:
        explicit_path: If provided, use this path directly.
        search_root: Root directory to search from (default: cwd).

    Returns:
        Resolved path to the CSV file.

    Raises:
        FileNotFoundError: If the file cannot be found.
    """
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Data file not found at explicit path: {explicit_path}")

    root = Path(search_root) if search_root else Path.cwd()
    for rel in DATA_RELATIVE_PATHS:
        candidate = (root / rel).resolve()
        if candidate.exists():
            return candidate

    searched = [str((root / r).resolve()) for r in DATA_RELATIVE_PATHS]
    raise FileNotFoundError(
        "Could not find linkedin_leads_clean.csv. Searched:\n"
        + "\n".join(f"  {p}" for p in searched)
        + "\nPass --data-path explicitly."
    )


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame):
    """Full preprocessing pipeline: feature engineering + encoding + split.

    Args:
        df: Raw cleaned DataFrame (303 rows × 20 cols including target).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor_dict).
        preprocessor_dict contains the fitted TargetEncoder.
    """
    df = df.copy()

    # 1. Extract text features (drops summary, skills, jobtitle)
    df = extract_text_features(df)

    # 2. Separate target
    y = df.pop("engaged")
    X = df

    # 3. Fill missing values
    for col in NUMERIC_COLS:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())

    for col in LOW_CARDINALITY_CATS + TARGET_ENCODE_CATS:
        if col in X.columns:
            X[col] = X[col].fillna("UNKNOWN")

    # 4. One-hot encode low-cardinality categoricals
    ohe_cols = [c for c in LOW_CARDINALITY_CATS if c in X.columns]
    if ohe_cols:
        X = pd.get_dummies(X, columns=ohe_cols, drop_first=True)
        # Cast boolean dummy columns to int
        bool_cols = X.select_dtypes(include="bool").columns
        X[bool_cols] = X[bool_cols].astype(int)

    # 5. Train/test split (stratified, before target encoding to prevent leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Target-encode medium/high cardinality categoricals
    te_cols = [c for c in TARGET_ENCODE_CATS if c in X_train.columns]
    preprocessor = {"target_encoder": None, "te_cols": te_cols}
    if te_cols:
        encoder = TargetEncoder(cols=te_cols, smoothing=2.0)
        X_train[te_cols] = encoder.fit_transform(X_train[te_cols], y_train)
        X_test[te_cols] = encoder.transform(X_test[te_cols])
        preprocessor["target_encoder"] = encoder

    # 7. Ensure all columns are numeric (any remaining objects → drop)
    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  Dropping non-numeric columns: {non_numeric}", file=sys.stderr)
        X_train = X_train.drop(columns=non_numeric)
        X_test = X_test.drop(columns=non_numeric)

    # Reset index for clean DataFrames
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test, preprocessor


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------

def save_artifacts(
    model,
    preprocessor: dict,
    X_train: pd.DataFrame,
    model_dir: Path,
    reference_dir: Path,
) -> None:
    """Persist all model artifacts to disk.

    Args:
        model: Fitted XGBoost classifier.
        preprocessor: Dict with fitted TargetEncoder and column names.
        X_train: Training features (used to derive feature columns and reference data).
        model_dir: Directory to save model artifacts.
        reference_dir: Directory to save reference CSV for drift monitoring.
    """
    model_dir = Path(model_dir)
    reference_dir = Path(reference_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, model_dir / "xgboost_model.joblib")

    # Save preprocessor
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")

    # Save feature columns
    feature_columns = list(X_train.columns)
    with open(model_dir / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    # Save numeric medians for inference-time imputation
    medians = {
        col: float(X_train[col].median())
        for col in NUMERIC_COLS
        if col in X_train.columns
    }
    with open(model_dir / "numeric_medians.json", "w") as f:
        json.dump(medians, f, indent=2)

    # Save reference data (first 100 rows of training set for drift detection)
    reference_df = X_train.head(100)
    reference_df.to_csv(reference_dir / "training_reference.csv", index=False)

    print(f"  Saved model:         {model_dir / 'xgboost_model.joblib'}")
    print(f"  Saved preprocessor:  {model_dir / 'preprocessor.joblib'}")
    print(f"  Saved feature cols:  {model_dir / 'feature_columns.json'} ({len(feature_columns)} features)")
    print(f"  Saved medians:       {model_dir / 'numeric_medians.json'}")
    print(f"  Saved reference:     {reference_dir / 'training_reference.csv'} ({len(reference_df)} rows)")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def run_export(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    mlflow_tracking_uri: Optional[str] = None,
) -> None:
    """Full export pipeline: load → preprocess → train → evaluate → save → register.

    Args:
        data_path: Explicit path to linkedin_leads_clean.csv. Auto-discovered if None.
        output_dir: Project root for output paths (default: cwd).
        mlflow_tracking_uri: MLflow tracking URI (default: uses env var or local).
    """
    root = Path(output_dir) if output_dir else Path.cwd()
    model_dir = root / "model"
    reference_dir = root / "data" / "reference"

    # 1. Locate data
    print("Loading data...")
    csv_path = find_data_file(explicit_path=data_path)
    print(f"  Found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows × {len(df.columns)} columns")

    # 2. Preprocess
    print("\nPreprocessing...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess(df)
    print(f"  Train: {len(X_train)} rows, {len(X_train.columns)} features")
    print(f"  Test:  {len(X_test)} rows")

    # 3. Train
    print("\nTraining XGBoost with best hyperparameters...")
    model = XGBClassifier(**BEST_PARAMS)
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    print(f"\nTest metrics:")
    print(f"  F1:        {f1:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")

    # 5. Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, preprocessor, X_train, model_dir, reference_dir)

    # 6. Register in MLflow
    print("\nRegistering in MLflow...")
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_experiment("linkedin-lead-scoring-production")
    with mlflow.start_run(run_name="export-production-model"):
        mlflow.log_params(BEST_PARAMS)
        mlflow.log_metrics({"f1": f1, "precision": precision, "recall": recall})
        mlflow.log_artifact(str(model_dir / "xgboost_model.joblib"))
        mlflow.log_artifact(str(model_dir / "feature_columns.json"))

        # Register model
        mlflow.sklearn.log_model(
            model,
            artifact_path="xgboost-production",
            registered_model_name="linkedin-lead-scoring-xgboost",
        )
        print("  Model registered as 'linkedin-lead-scoring-xgboost'")

    print("\nExport complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export production XGBoost model.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to linkedin_leads_clean.csv (auto-discovered if not set)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Project root for output (default: current directory)",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: uses MLFLOW_TRACKING_URI env var)",
    )
    args = parser.parse_args()
    run_export(
        data_path=args.data_path,
        output_dir=args.output_dir,
        mlflow_tracking_uri=args.mlflow_uri,
    )


if __name__ == "__main__":
    main()
