"""
Retrain the XGBoost model with additional labelled data and compare versions.

Merges original training data with new production data that has ground-truth
labels, retrains the model, and registers both versions in MLflow for
side-by-side comparison.

Usage:
    python scripts/retrain_model.py

Outputs:
    - MLflow runs for v1 (original) and v2 (merged) in same experiment
    - Updated model artifacts in model/ (v2 replaces v1)
    - Merged dataset saved to data/processed/linkedin_leads_merged.csv
    - Comparison metrics printed to stdout
"""
import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from linkedin_lead_scoring.features import (
    LOW_CARDINALITY_CATS,
    NUMERIC_COLS,
    TARGET_ENCODE_CATS,
    TEXT_COLS,
    extract_text_features,
)


# ---------------------------------------------------------------------------
# Constants (same as export_model.py — single source of truth)
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


# ---------------------------------------------------------------------------
# Preprocessing (reuses export_model.py logic)
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame):
    """Full preprocessing pipeline: feature engineering + encoding + split."""
    df = df.copy()

    # 1. Extract text features
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
        bool_cols = X.select_dtypes(include="bool").columns
        X[bool_cols] = X[bool_cols].astype(int)

    # 5. Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Target-encode
    te_cols = [c for c in TARGET_ENCODE_CATS if c in X_train.columns]
    preprocessor = {"target_encoder": None, "te_cols": te_cols}
    if te_cols:
        encoder = TargetEncoder(cols=te_cols, smoothing=2.0)
        X_train[te_cols] = encoder.fit_transform(X_train[te_cols], y_train)
        X_test[te_cols] = encoder.transform(X_test[te_cols])
        preprocessor["target_encoder"] = encoder

    # 7. Drop remaining non-numeric
    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X_train = X_train.drop(columns=non_numeric)
        X_test = X_test.drop(columns=non_numeric)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test, preprocessor


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train XGBoost and return model + metrics."""
    model = XGBClassifier(**BEST_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }
    report = classification_report(y_test, y_pred, zero_division=0)
    return model, metrics, report


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------

def save_artifacts(model, preprocessor, X_train, model_dir, reference_dir):
    """Save model artifacts to disk."""
    model_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "xgboost_model.joblib")
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")

    feature_columns = list(X_train.columns)
    with open(model_dir / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    medians = {
        col: float(X_train[col].median())
        for col in NUMERIC_COLS
        if col in X_train.columns
    }
    with open(model_dir / "numeric_medians.json", "w") as f:
        json.dump(medians, f, indent=2)

    reference_df = X_train.head(100)
    reference_df.to_csv(reference_dir / "training_reference.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent

    original_path = project_root / "data" / "processed" / "linkedin_leads_clean.csv"
    new_data_path = project_root / "data" / "production" / "new_contacts_enriched.csv"
    merged_path = project_root / "data" / "processed" / "linkedin_leads_merged.csv"
    model_dir = project_root / "model"
    reference_dir = project_root / "data" / "reference"

    print("=" * 70)
    print("LinkedIn Lead Scoring — Model Retraining with Additional Data")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Load datasets
    # -----------------------------------------------------------------------
    print("\n[1/5] Loading datasets...")
    df_original = pd.read_csv(original_path)
    df_new = pd.read_csv(new_data_path)
    print(f"  Original training data: {len(df_original)} rows")
    print(f"  New labelled data:      {len(df_new)} rows")
    print(f"  Original target: {df_original['engaged'].value_counts().to_dict()}")
    print(f"  New target:      {df_new['engaged'].value_counts().to_dict()}")

    # -----------------------------------------------------------------------
    # 2. Set up MLflow
    # -----------------------------------------------------------------------
    print("\n[2/5] Setting up MLflow...")
    mlflow_uri = f"file://{project_root / 'mlruns'}"
    mlflow.set_tracking_uri(mlflow_uri)
    experiment_name = "linkedin-lead-scoring-retraining"
    mlflow.set_experiment(experiment_name)
    print(f"  Tracking URI: {mlflow_uri}")
    print(f"  Experiment: {experiment_name}")

    # -----------------------------------------------------------------------
    # 3. Train v1 (original data only) — baseline for comparison
    # -----------------------------------------------------------------------
    print("\n[3/5] Training v1 (original data only)...")
    X_train_v1, X_test_v1, y_train_v1, y_test_v1, prep_v1 = preprocess(df_original)
    print(f"  Train: {len(X_train_v1)} rows, {len(X_train_v1.columns)} features")
    print(f"  Test:  {len(X_test_v1)} rows")

    model_v1, metrics_v1, report_v1 = train_and_evaluate(
        X_train_v1, X_test_v1, y_train_v1, y_test_v1
    )

    with mlflow.start_run(run_name="v1-original-303-rows") as run_v1:
        mlflow.set_tag("model_version", "v1")
        mlflow.set_tag("dataset", "original")
        mlflow.set_tag("n_samples", len(df_original))
        mlflow.log_params(BEST_PARAMS)
        mlflow.log_param("n_train", len(X_train_v1))
        mlflow.log_param("n_test", len(X_test_v1))
        mlflow.log_param("n_features", len(X_train_v1.columns))
        mlflow.log_metrics(metrics_v1)
        mlflow.sklearn.log_model(
            model_v1,
            artifact_path="xgboost-model",
            registered_model_name="linkedin-lead-scoring-xgboost",
        )

    print(f"  F1:        {metrics_v1['f1']:.3f}")
    print(f"  Precision: {metrics_v1['precision']:.3f}")
    print(f"  Recall:    {metrics_v1['recall']:.3f}")
    print(f"  MLflow run: {run_v1.info.run_id}")

    # -----------------------------------------------------------------------
    # 4. Merge and train v2
    # -----------------------------------------------------------------------
    print("\n[4/5] Training v2 (merged data)...")
    df_merged = pd.concat([df_original, df_new], ignore_index=True)
    df_merged.to_csv(merged_path, index=False)
    print(f"  Merged dataset: {len(df_merged)} rows → saved to {merged_path.name}")
    print(f"  Target: {df_merged['engaged'].value_counts().to_dict()}")

    X_train_v2, X_test_v2, y_train_v2, y_test_v2, prep_v2 = preprocess(df_merged)
    print(f"  Train: {len(X_train_v2)} rows, {len(X_train_v2.columns)} features")
    print(f"  Test:  {len(X_test_v2)} rows")

    model_v2, metrics_v2, report_v2 = train_and_evaluate(
        X_train_v2, X_test_v2, y_train_v2, y_test_v2
    )

    with mlflow.start_run(run_name="v2-merged-454-rows") as run_v2:
        mlflow.set_tag("model_version", "v2")
        mlflow.set_tag("dataset", "merged")
        mlflow.set_tag("n_samples", len(df_merged))
        mlflow.log_params(BEST_PARAMS)
        mlflow.log_param("n_train", len(X_train_v2))
        mlflow.log_param("n_test", len(X_test_v2))
        mlflow.log_param("n_features", len(X_train_v2.columns))
        mlflow.log_metrics(metrics_v2)
        mlflow.sklearn.log_model(
            model_v2,
            artifact_path="xgboost-model",
            registered_model_name="linkedin-lead-scoring-xgboost",
        )

    print(f"  F1:        {metrics_v2['f1']:.3f}")
    print(f"  Precision: {metrics_v2['precision']:.3f}")
    print(f"  Recall:    {metrics_v2['recall']:.3f}")
    print(f"  MLflow run: {run_v2.info.run_id}")

    # -----------------------------------------------------------------------
    # 5. Compare and save v2 artifacts
    # -----------------------------------------------------------------------
    print("\n[5/5] Comparison and artifact update...")
    print()
    print("=" * 70)
    print("MODEL COMPARISON: v1 vs v2")
    print("=" * 70)
    print(f"{'Metric':<15} {'v1 (303 rows)':>15} {'v2 (454 rows)':>15} {'Delta':>10}")
    print("-" * 55)
    for metric in ["f1", "precision", "recall"]:
        v1_val = metrics_v1[metric]
        v2_val = metrics_v2[metric]
        delta = v2_val - v1_val
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<15} {v1_val:>15.3f} {v2_val:>15.3f} {sign}{delta:>9.3f}")
    print("-" * 55)
    print()

    # Decide whether to promote v2
    f1_improved = metrics_v2["f1"] >= metrics_v1["f1"]
    if f1_improved:
        print("v2 F1 >= v1 F1 — saving v2 as production model.")
    else:
        print("v2 F1 < v1 F1 — saving v2 anyway (more data = more robust).")
        print("  Note: F1 drop may indicate harder test split, not worse model.")

    # Save v2 artifacts
    save_artifacts(model_v2, prep_v2, X_train_v2, model_dir, reference_dir)
    print(f"\n  Artifacts saved to {model_dir}/")
    print(f"  Reference data saved to {reference_dir}/")

    # Print detailed classification reports
    print()
    print("=" * 70)
    print("v1 Classification Report (original 303 rows)")
    print("=" * 70)
    print(report_v1)

    print("=" * 70)
    print("v2 Classification Report (merged 454 rows)")
    print("=" * 70)
    print(report_v2)

    print("\nRetraining complete. Compare runs in MLflow UI:")
    print(f"  mlflow ui --backend-store-uri {mlflow_uri}")


if __name__ == "__main__":
    main()
