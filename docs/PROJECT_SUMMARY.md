# Project Summary: LinkedIn Lead Scoring MLOps Pipeline

**Project**: OC6 - LinkedIn Lead Scoring
**Objective**: Predict LinkedIn contact engagement (reply/interest) using MLflow-tracked experiments
**Status**: Phase 1 Complete (Data Preparation) | Phase 2 Planned (Model Training)

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Data Preparation (Completed)](#phase-1-data-preparation-completed)
3. [Phase 2: Model Training (Planned)](#phase-2-model-training-planned)
4. [MLflow Integration Strategy](#mlflow-integration-strategy)
5. [Technical Architecture](#technical-architecture)
6. [Next Steps](#next-steps)

---

## Overview

### Business Goal
Predict which LinkedIn contacts will engage (reply or show interest) when sent connection invitations, enabling targeted outreach and improved conversion rates.

### Target Metric
- **Current acceptance rate**: ~57% (from Project Brief)
- **Goal**: Build classifier to identify high-probability leads

### Data Source
LemList campaign exports containing LinkedIn contact information and interaction history.

---

## Phase 1: Data Preparation (Completed)

### Notebook: `01_linkedin_data_prep.ipynb`

#### What Was Implemented

1. **MLflow Experiment Tracking (From the Start)**
   - All data operations logged to MLflow from cell 1
   - Automatic project root detection (handles paths with spaces)
   - Centralized tracking in `<project-root>/mlruns/`
   - Parameters, metrics, and artifacts tracked throughout

2. **Data Loading & Standardization**
   - Load LinkedIn contact CSV from LemList export
   - Standardize column names to lowercase (avoid camelCase issues)
   - Log raw data profile to MLflow (rows, columns, memory usage)

3. **Label Creation**
   - **Engaged (1)**: `laststate == "linkedinReplied"` OR `"linkedinInterested"`
   - **Not Engaged (0)**: `status == "done"` AND `laststate NOT IN {engaged_states}`
   - **Filtered out**: In-progress contacts, reviewed, etc.
   - Logged label distribution to MLflow (class balance, imbalance ratio)

4. **Data Quality Assessment**
   - Missing value analysis (high/moderate/low missing)
   - Constant and quasi-constant column detection
   - Cardinality analysis (unique values per feature)
   - All quality metrics logged to MLflow

5. **Feature Selection**
   - Removed PII columns: email, phone, firstname, lastname, linkedinurl, picture
   - Removed ID columns: _id, emailstatus
   - Removed metadata: status, laststate (used for labeling only)
   - Logged dropped columns and remaining feature count

6. **Exploratory Data Analysis**
   - Numeric vs categorical feature split
   - Value frequency distributions for categorical variables
   - Engagement rate by category (industry, job title, company size, etc.)
   - Correlation analysis for numeric features (threshold: 0.80)
   - Target distribution visualization (saved as artifact)

7. **Dataset Export**
   - Saved cleaned dataset: `data/processed/linkedin_leads_clean.csv`
   - Logged as MLflow artifact for reproducibility
   - Ready for feature engineering and modeling

#### How It Was Done

**Key Design Decisions:**

1. **MLflow-First Approach**
   - Every helper function includes `log_to_mlflow=True` parameter
   - `setup_mlflow()` auto-configures tracking URI using `pathlib` (handles spaces)
   - Created `find_project_root()` utility to detect project root from any notebook location
   - All operations logged: parameters, metrics, artifacts

2. **Modular Helper Functions** (`utils_data.py`)
   - Reused and adapted functions from OC4 HR Attrition project
   - Added LinkedIn-specific functions (label creation, engagement analysis)
   - MLflow logging integrated into each function
   - Functions handle edge cases (missing values, empty dataframes, etc.)

3. **Robust Path Handling**
   - Used `pathlib.Path` throughout (handles spaces in paths automatically)
   - Dynamic project root detection (looks for `pyproject.toml`, `.git`)
   - No hardcoded paths in notebooks
   - Works from any subdirectory (notebooks/, scripts/, etc.)

4. **Data Anonymization**
   - Dropped all PII columns after initial loading
   - Only keep aggregated/categorical features for modeling
   - Compliant with data minimization principles

**Expected Results:**
- ~293 labeled contacts (based on 477 total in CSV)
- Binary classification target: `engaged` (0 or 1)
- ~198 contacts with positive engagement (replied/interested)
- ~95 contacts with negative engagement (done but no reply)
- Remaining ~184 filtered out (in-progress)

---

## Phase 2: Model Training (Planned)

### Notebook: `02_linkedin_model_training.ipynb`

#### Objectives

1. Build multiple classification models to predict engagement
2. Track all experiments, hyperparameters, and metrics in MLflow
3. Handle class imbalance (even if not severe currently)
4. Select best model based on business metric (precision vs recall trade-off)
5. Register winning model in MLflow Model Registry

#### Implementation Plan

### 1. Feature Engineering

**To be added to notebook 01 (next iteration):**

- **Categorical Encoding**:
  - Target encoding for high-cardinality features (industry, jobtitle)
  - One-hot encoding for low-cardinality features (companysize, etc.)
  - Frequency encoding for location-based features

- **Numeric Features** (if available):
  - Scaling/normalization (StandardScaler or RobustScaler)
  - Polynomial features for key predictors
  - Binning for continuous variables

- **LLM Enrichment** (future enhancement):
  - Use LLM to extract features from job titles (seniority, department)
  - Company description analysis (industry classification, company stage)
  - Generate engagement propensity scores from profile text

- **Feature Importance Tracking**:
  - Log feature engineering pipeline to MLflow
  - Track feature names and transformations as artifacts

### 2. Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Stratified split to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Log split info to MLflow
mlflow.log_param("test_size", 0.2)
mlflow.log_param("stratify", True)
mlflow.log_metric("train_size", len(X_train))
mlflow.log_metric("test_size", len(X_test))
mlflow.log_metric("train_positive_rate", y_train.mean())
mlflow.log_metric("test_positive_rate", y_test.mean())
```

### 3. Class Imbalance Handling

**KNOWN ISSUE**: `imbalanced-learn` has compatibility issues with scikit-learn 1.5+ in current setup.
**Status**: TEMPORARILY DISABLED in notebook 02.
**Workaround**: Using `class_weight='balanced'` parameter in models instead.
**Future fix**: Resolve conda/uv version conflict or use alternative balancing strategies.

**Even though current imbalance is moderate (~67% positive), plan for future scenarios:**

#### Strategy A: Data-Level Techniques (DISABLED - compatibility issue)

```python
# TEMPORARILY COMMENTED OUT - imbalanced-learn compatibility issue
# from imblearn.over_sampling import SMOTE, ADASYN
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.combine import SMOTETomek

# Option 1: SMOTE (Synthetic Minority Over-sampling)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Option 2: ADASYN (Adaptive Synthetic Sampling)
adasyn = ADASYN(random_state=42)
X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)

# Option 3: Combined approach (over + under sampling)
smotetomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smotetomek.fit_resample(X_train, y_train)

# Log balancing strategy to MLflow
mlflow.log_param("balancing_method", "SMOTE")
mlflow.log_metric("train_balanced_size", len(X_train_balanced))
mlflow.log_metric("train_balanced_positive_rate", y_train_balanced.mean())
```

#### Strategy B: Algorithm-Level Techniques

```python
# Use class_weight parameter (built into most sklearn classifiers)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Option 1: Auto-balance weights
rf_balanced = RandomForestClassifier(class_weight='balanced', random_state=42)

# Option 2: Custom weights
class_weights = {0: 1.0, 1: 1.5}  # Give more weight to minority class
xgb_weighted = XGBClassifier(scale_pos_weight=1.5, random_state=42)

# Log class weights to MLflow
mlflow.log_param("class_weight", "balanced")
mlflow.log_param("scale_pos_weight", 1.5)
```

#### Strategy C: Ensemble with Balanced Bagging

```python
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier

# Automatically balances each bootstrap sample
balanced_rf = BalancedRandomForestClassifier(
    n_estimators=100,
    random_state=42,
    sampling_strategy='auto'
)

balanced_rf.fit(X_train, y_train)
```

### 4. Model Selection & Training

**Train multiple models and track all with MLflow:**

#### Baseline Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier

models = {
    "Dummy (Baseline)": DummyClassifier(strategy="stratified", random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "Naive Bayes": GaussianNB(),
}

# Train and log each baseline
for model_name, model in models.items():
    with mlflow.start_run(run_name=f"baseline_{model_name}"):
        mlflow.log_param("model_type", model_name)
        model.fit(X_train, y_train)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Evaluate and log metrics (see section 5 below)
```

#### Tree-Based Models

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

tree_models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=1.5,  # Handle imbalance
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    ),
}

# Train and log each tree model
for model_name, model in tree_models.items():
    with mlflow.start_run(run_name=f"tree_{model_name}"):
        mlflow.log_param("model_type", model_name)
        mlflow.log_params(model.get_params())  # Log all hyperparameters

        model.fit(X_train, y_train)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Evaluate (see section 5)
```

### 5. Model Evaluation & Metrics

**Comprehensive evaluation tracked in MLflow:**

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Comprehensive model evaluation with MLflow tracking.
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Prediction probabilities
    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_train_proba = y_train_pred
        y_test_proba = y_test_pred

    # === Classification Metrics ===
    metrics = {
        # Train metrics
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "train_precision": precision_score(y_train, y_train_pred),
        "train_recall": recall_score(y_train, y_train_pred),
        "train_f1": f1_score(y_train, y_train_pred),
        "train_roc_auc": roc_auc_score(y_train, y_train_proba),

        # Test metrics (most important!)
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred),
        "test_recall": recall_score(y_test, y_test_pred),
        "test_f1": f1_score(y_test, y_test_pred),
        "test_roc_auc": roc_auc_score(y_test, y_test_proba),
        "test_avg_precision": average_precision_score(y_test, y_test_proba),
    }

    # Log all metrics to MLflow
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    # === Confusion Matrix ===
    cm = confusion_matrix(y_test, y_test_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')

    # Save and log to MLflow
    fig.savefig('/tmp/confusion_matrix.png', dpi=100, bbox_inches='tight')
    mlflow.log_artifact('/tmp/confusion_matrix.png', 'evaluation')
    plt.close()

    # Log confusion matrix values
    tn, fp, fn, tp = cm.ravel()
    mlflow.log_metric("test_true_negatives", tn)
    mlflow.log_metric("test_false_positives", fp)
    mlflow.log_metric("test_false_negatives", fn)
    mlflow.log_metric("test_true_positives", tp)

    # === ROC Curve ===
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_test_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC (AUC = {metrics["test_roc_auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend()
    ax.grid(True)

    fig.savefig('/tmp/roc_curve.png', dpi=100, bbox_inches='tight')
    mlflow.log_artifact('/tmp/roc_curve.png', 'evaluation')
    plt.close()

    # === Precision-Recall Curve ===
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_test_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f'PR (AP = {metrics["test_avg_precision"]:.3f})')
    ax.axhline(y=y_test.mean(), color='k', linestyle='--', label='Baseline')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve - {model_name}')
    ax.legend()
    ax.grid(True)

    fig.savefig('/tmp/pr_curve.png', dpi=100, bbox_inches='tight')
    mlflow.log_artifact('/tmp/pr_curve.png', 'evaluation')
    plt.close()

    # === Classification Report ===
    report = classification_report(y_test, y_test_pred)
    with open('/tmp/classification_report.txt', 'w') as f:
        f.write(report)
    mlflow.log_artifact('/tmp/classification_report.txt', 'evaluation')

    # === Feature Importance (if available) ===
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_train.columns

        # Sort by importance
        indices = np.argsort(importances)[::-1][:20]  # Top 20

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top 20 Feature Importances - {model_name}')
        ax.invert_yaxis()

        fig.savefig('/tmp/feature_importance.png', dpi=100, bbox_inches='tight')
        mlflow.log_artifact('/tmp/feature_importance.png', 'evaluation')
        plt.close()

        # Log top features as parameters
        top_5_features = [feature_names[i] for i in indices[:5]]
        mlflow.log_param("top_5_features", ",".join(top_5_features))

    print(f"✓ Evaluation complete for {model_name}")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.3f}")
    print(f"  Test Precision: {metrics['test_precision']:.3f}")
    print(f"  Test Recall: {metrics['test_recall']:.3f}")
    print(f"  Test F1: {metrics['test_f1']:.3f}")
    print(f"  Test ROC-AUC: {metrics['test_roc_auc']:.3f}")

    return metrics
```

### 6. Hyperparameter Tuning with Optuna

**Integrate Optuna with MLflow for automatic hyperparameter optimization:**

```python
import optuna
from optuna.integration.mlflow import MLflowCallback

def objective(trial, X_train, y_train, X_test, y_test):
    """
    Optuna objective function for XGBoost hyperparameter tuning.
    """
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 3.0),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    # Train model with suggested hyperparameters
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return roc_auc  # Optimize for ROC-AUC

# Create Optuna study with MLflow integration
mlflow_callback = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(),
    metric_name="test_roc_auc"
)

study = optuna.create_study(
    direction='maximize',
    study_name='xgboost_hyperparameter_tuning',
    sampler=optuna.samplers.TPESampler(seed=42)
)

# Run optimization
with mlflow.start_run(run_name="xgboost_optuna_tuning"):
    mlflow.log_param("optimization_method", "optuna")
    mlflow.log_param("n_trials", 50)

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_test, y_test),
        n_trials=50,
        callbacks=[mlflow_callback]
    )

    # Log best parameters and score
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_test_roc_auc", study.best_value)

    print(f"✓ Best ROC-AUC: {study.best_value:.4f}")
    print(f"✓ Best parameters: {study.best_params}")

    # Train final model with best parameters
    best_model = XGBClassifier(**study.best_params)
    best_model.fit(X_train, y_train)

    # Log best model
    mlflow.sklearn.log_model(best_model, "best_model")

    # Comprehensive evaluation
    evaluate_model(best_model, X_train, X_test, y_train, y_test, "XGBoost_Optimized")
```

### 7. Model Explainability with SHAP

**Add SHAP values for model interpretability:**

```python
import shap

def explain_model_shap(model, X_train, X_test, model_name):
    """
    Generate SHAP explanations and log to MLflow.
    """
    # Create SHAP explainer
    if isinstance(model, (XGBClassifier, LGBMClassifier)):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_train)

    # Calculate SHAP values for test set
    shap_values = explainer(X_test)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    fig.savefig('/tmp/shap_summary.png', dpi=100, bbox_inches='tight')
    mlflow.log_artifact('/tmp/shap_summary.png', 'explainability')
    plt.close()

    # Waterfall plot for first prediction
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(shap_values[0], show=False)
    plt.tight_layout()
    fig.savefig('/tmp/shap_waterfall.png', dpi=100, bbox_inches='tight')
    mlflow.log_artifact('/tmp/shap_waterfall.png', 'explainability')
    plt.close()

    print(f"✓ SHAP explanations generated for {model_name}")
```

### 8. Model Selection & Registration

**Select best model based on business criteria:**

```python
# Query MLflow to find best model
experiment = mlflow.get_experiment_by_name("linkedin-lead-scoring")
runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Filter for model training runs (exclude data prep)
model_runs = runs_df[runs_df['tags.mlflow.runName'].str.contains('tree_|baseline_|optuna', na=False)]

# Sort by test F1 score (or other business metric)
best_run = model_runs.sort_values('metrics.test_f1', ascending=False).iloc[0]

print(f"✓ Best model: {best_run['tags.mlflow.runName']}")
print(f"  Test F1: {best_run['metrics.test_f1']:.3f}")
print(f"  Test ROC-AUC: {best_run['metrics.test_roc_auc']:.3f}")
print(f"  Run ID: {best_run['run_id']}")

# Register best model
model_uri = f"runs:/{best_run['run_id']}/model"
model_version = mlflow.register_model(model_uri, "linkedin-lead-scorer")

# Transition to Production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="linkedin-lead-scorer",
    version=model_version.version,
    stage="Production"
)

print(f"✓ Model registered as 'linkedin-lead-scorer' version {model_version.version}")
print(f"✓ Transitioned to Production stage")
```

---

## MLflow Integration Strategy

### Why MLflow from the Start?

1. **Reproducibility**: Every data transformation, model training, and evaluation is tracked
2. **Comparison**: Easy to compare runs across different parameters and models
3. **Collaboration**: Team members can see all experiments in centralized UI
4. **Production**: Seamless transition from experimentation to production via Model Registry
5. **Versioning**: Automatic versioning of datasets, models, and code

### What Gets Logged?

#### Data Preparation (Notebook 01)
- **Parameters**: data source, preprocessing steps, feature selection criteria
- **Metrics**: data quality (missing %, class balance, feature counts)
- **Artifacts**: cleaned dataset, visualizations, data profile reports

#### Model Training (Notebook 02)
- **Parameters**: model type, all hyperparameters, balancing strategy
- **Metrics**: train/test accuracy, precision, recall, F1, ROC-AUC, confusion matrix values
- **Artifacts**: trained model, evaluation plots (ROC, PR curves), feature importance, SHAP explanations, classification reports
- **Tags**: model family (tree-based, linear), optimization method (manual, optuna)

### MLflow UI Workflow

1. **Experiments page**: Compare all runs side-by-side
2. **Run details**: Drill into specific run to see all logged data
3. **Model Registry**: Manage model versions and stage transitions (Staging → Production)
4. **Compare runs**: Select multiple runs to compare metrics visually
5. **Search/Filter**: Find runs by metric thresholds, parameters, tags

---

## Technical Architecture

### Project Structure

```
oc6-linkedin-lead-scoring/
├── data/
│   ├── raw/                          # LemList CSV exports
│   └── processed/                    # Cleaned datasets (MLflow artifacts)
├── notebooks/
│   ├── 01_linkedin_data_prep.ipynb   # [DONE] Data preparation
│   └── 02_linkedin_model_training.ipynb  # [PLANNED] Model training
├── src/
│   └── linkedin_lead_scoring/
│       ├── data/
│       │   └── utils_data.py         # [DONE] Data processing utilities
│       ├── models/
│       │   └── utils_models.py       # [TODO] Model training utilities
│       └── evaluation/
│           └── utils_eval.py         # [TODO] Evaluation utilities
├── mlruns/                           # MLflow tracking data
├── docs/
│   └── PROJECT_SUMMARY.md            # This document
├── environment.yml                    # Conda environment (numpy, pandas, sklearn)
└── pyproject.toml                    # uv packages (mlflow, xgboost, fastapi)
```

### Environment Setup

**Hybrid conda + uv approach:**
- **Conda**: Big scientific packages (numpy 2.x, pandas, scikit-learn, matplotlib, jupyter)
- **uv**: Specialized ML packages (mlflow, xgboost, lightgbm, optuna, shap, fastapi)

**Setup:**
```bash
conda env create -f environment.yml
conda activate oc6
uv pip install -e ".[dev]"
```

**Daily usage:**
```bash
conda activate oc6
mlflow ui --port 5000        # Terminal 1
jupyter lab                  # Terminal 2
```

---

## Next Steps

### Immediate (Notebook 01 Enhancement)

1. **Feature Engineering**
   - Categorical encoding (target, one-hot, frequency)
   - Numeric feature scaling
   - Derived features (engagement propensity indicators)
   - Log all transformations to MLflow

2. **LLM Enrichment** (Optional)
   - Extract seniority level from job titles using LLM
   - Classify company stage/type from descriptions
   - Generate engagement likelihood scores
   - Track LLM API calls and costs in MLflow

3. **Advanced EDA**
   - Interaction effects between features
   - Time-based patterns (if timestamps available)
   - Geographic clustering analysis

### Next Sprint (Notebook 02)

1. **Implement Model Training Pipeline**
   - Follow plan outlined in Phase 2
   - Train baseline → tree models → optimized models
   - Track everything in MLflow

2. **Handle Class Imbalance**
   - Test SMOTE, ADASYN, class weights
   - Compare balanced vs imbalanced training
   - Document findings in MLflow

3. **Model Selection**
   - Define business metric (precision vs recall trade-off)
   - Select best model from MLflow experiments
   - Register in Model Registry

4. **Model Explainability**
   - SHAP values for top predictions
   - Feature importance analysis
   - Document insights for business stakeholders

### Future Enhancements

1. **Real-Time Prediction API**
   - Load model from MLflow Model Registry
   - FastAPI endpoint for scoring new contacts
   - A/B testing framework

2. **Continuous Training Pipeline**
   - Automated retraining when new data arrives
   - Drift detection (data + model drift)
   - MLflow Projects for pipeline orchestration

3. **Production Monitoring**
   - Log prediction requests and outcomes
   - Track model performance degradation
   - Automatic alerts for metric drops

---

## Key Learnings

### What Worked Well

1. **MLflow from Day 1**: Having tracking integrated from the start made experimentation seamless
2. **Pathlib for paths**: Robust handling of paths with spaces using `pathlib.Path`
3. **Modular utilities**: Reusable functions from OC4 accelerated development
4. **Auto-configuration**: `setup_mlflow()` automatically finding project root eliminates path issues

### Challenges Solved

1. **NumPy 2.x compatibility**: Hybrid conda + uv environment ensures consistent package versions
2. **MLflow tracking location**: Auto-detection of project root prevents scattered mlruns directories
3. **Column name inconsistency**: Standardization to lowercase prevents KeyError bugs

### Recommendations

1. Always use `pathlib` for file operations (handles edge cases)
2. Log everything to MLflow - storage is cheap, missing context is expensive
3. Use meaningful run names (e.g., `"xgboost_balanced_v3"` vs `"run_42"`)
4. Document labeling logic explicitly (business rules change over time)
5. Start simple (baseline models) before optimizing (tree models + hyperparameter tuning)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-12
**Author**: Claude Code (with Ghislain de Labie)
**Status**: Phase 1 Complete, Phase 2 Ready to Implement
