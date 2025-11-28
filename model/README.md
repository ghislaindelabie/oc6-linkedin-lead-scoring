# Model Directory

This directory contains trained model artifacts.

## Model Versioning

Models are tracked with MLflow and stored here for API deployment.

## Expected Files

- `linkedin_lead_scoring_v1.joblib` - XGBoost model
- `feature_columns.json` - List of feature names
- `preprocessing_pipeline.joblib` - Preprocessing steps (if needed)

## Model Metadata

See MLflow UI for complete model metadata:
- Training parameters
- Performance metrics
- Feature importance
- SHAP values
