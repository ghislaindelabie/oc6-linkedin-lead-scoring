# Data Drift in Machine Learning — Complete Guide

**Context**: LinkedIn Lead Scoring project (LeadGen scorer) — XGBoost classifier predicting engagement probability from LinkedIn profile data.

---

## Table of Contents

1. [What Is Drift?](#1-what-is-drift)
2. [Types of Drift](#2-types-of-drift)
3. [Why Drift Happens](#3-why-drift-happens)
4. [How Drift Is Measured](#4-how-drift-is-measured)
5. [How Drift Is Managed](#5-how-drift-is-managed)
6. [Evidently AI — The Framework](#6-evidently-ai--the-framework)
7. [Implementation in This Project](#7-implementation-in-this-project)
8. [Testing the Drift Pipeline](#8-testing-the-drift-pipeline)
9. [Dashboard Integration](#9-dashboard-integration)
10. [Decision Framework](#10-decision-framework)

---

## 1. What Is Drift?

A machine learning model is trained on a specific dataset at a specific point in time. It learns patterns — statistical relationships between input features and the target variable. **Drift** occurs when the real-world data the model encounters in production diverges from the data it was trained on.

The fundamental assumption behind any ML model is:

> The data the model will see in production follows the same distribution as the data it was trained on.

When this assumption breaks, model predictions become unreliable — even if the model code and weights haven't changed at all.

### A concrete example

Our lead scoring model was trained on 303 LinkedIn contacts from LemList campaigns in late 2024. It learned that:
- Contacts from SaaS companies (`llm_industry = "Technology - SaaS"`) had higher engagement
- C-Level executives (`llm_seniority = "C-Level"`) responded more often
- Companies founded after 2015 (`companyfoundedon > 2015`) correlated with engagement

Six months later, the sales team shifts strategy to target the healthcare sector. Suddenly:
- `llm_industry` distribution shifts from tech-heavy to healthcare-heavy
- `companysize` shifts from startups to large hospital groups
- The model has never seen these patterns — its predictions become meaningless

This is drift.

---

## 2. Types of Drift

### 2.1 Data Drift (Covariate Shift)

**Definition**: The distribution of input features P(X) changes, but the relationship between features and target P(Y|X) remains the same.

```
Training:    P_train(X) ≠ P_prod(X)    but    P_train(Y|X) = P_prod(Y|X)
```

**Example**: The sales team starts targeting German companies instead of French ones. The `location` and `companylocation` distributions shift. However, within each geography, the factors predicting engagement haven't changed — German CTOs at SaaS startups behave similarly to French ones.

**Impact**: Moderate. The model may still be accurate for the new population, but its predictions are extrapolating to a region of feature space it hasn't seen much of. Confidence should decrease.

**Detection**: Compare feature distributions between training data and production data using statistical tests (KS, chi-squared, PSI).

### 2.2 Concept Drift

**Definition**: The relationship between features and target P(Y|X) changes, even if the input distribution P(X) stays the same.

```
Training:    P_train(X) = P_prod(X)    but    P_train(Y|X) ≠ P_prod(Y|X)
```

**Example**: LinkedIn changes its messaging algorithm. Previously, short direct messages from executives got high response rates. Now, detailed messages with industry context perform better. The input profiles look the same, but the rules of engagement have changed.

**Impact**: Severe. The model is confidently wrong — it returns high scores for leads that no longer convert.

**Detection**: Monitor the actual prediction outcomes. Compare predicted engagement rate against actual reply rate from campaign tools (LemList). Concept drift is harder to detect because it requires ground-truth labels.

### 2.3 Prediction Drift (Label Drift)

**Definition**: The distribution of model outputs P(Y_hat) changes. This is a downstream consequence — either data drift or concept drift (or both) cause the predictions themselves to shift.

```
P_train(Y_hat) ≠ P_prod(Y_hat)
```

**Example**: Average predicted engagement score drops from 0.45 to 0.22 over two weeks. Either the incoming leads are lower quality (data drift) or the model's calibration is off (concept drift).

**Detection**: Compare score distributions between a reference period and the current production window using a KS test on predicted probabilities.

### 2.4 Summary Table

| Type | What changes | Root cause | Detection difficulty | Impact |
|------|-------------|------------|---------------------|--------|
| **Data drift** | P(X) | New population, targeting change | Easy (no labels needed) | Moderate |
| **Concept drift** | P(Y\|X) | External change (platform, market) | Hard (needs labels) | Severe |
| **Prediction drift** | P(Y_hat) | Consequence of data or concept drift | Easy (no labels needed) | Diagnostic signal |

### 2.5 Temporal Patterns

Drift isn't always sudden. It can manifest in several patterns:

- **Sudden drift**: Abrupt distribution change (new campaign launch, strategy pivot)
- **Gradual drift**: Slow, continuous shift over weeks/months (market evolution)
- **Recurring drift**: Periodic changes (seasonal hiring patterns, fiscal year effects)
- **Incremental drift**: Small, imperceptible steps that accumulate

In our project, the most likely pattern is **sudden drift** — a new sales campaign targeting a different industry or geography.

---

## 3. Why Drift Happens

### 3.1 Changes in the Data Source

| Cause | Example in our project |
|-------|----------------------|
| New campaign targeting | Sales team pivots from SaaS to healthcare |
| Geographic expansion | Campaigns in new countries (Asia, LATAM) |
| Platform changes | LinkedIn updates profile fields or formats |
| Data pipeline changes | LemList export format changes, new enrichment fields |

### 3.2 Changes in the Real World

| Cause | Example |
|-------|---------|
| Market dynamics | Economic downturn reduces response rates across the board |
| Seasonal effects | Q4 budget freeze reduces engagement for enterprise contacts |
| Competition | Competitors adopt similar outreach, reducing uniqueness |
| Regulation | GDPR enforcement changes how contacts can be approached |

### 3.3 Changes in the Model's Environment

| Cause | Example |
|-------|---------|
| Feature engineering changes | Upstream pipeline modifies how `llm_quality` is computed |
| Preprocessing pipeline update | Encoder retrained with new vocabulary |
| Dependency updates | XGBoost version change alters predict_proba output |
| Infrastructure changes | Different CPU affecting float precision in ONNX inference |

### 3.4 Training Data Limitations

| Cause | Example |
|-------|---------|
| Small training set | 303 contacts — poor coverage of rare industries |
| Selection bias | Training data only includes contacts from one campaign wave |
| Label noise | "Engaged" definition changed since data collection |
| Temporal bias | Model trained on Q4 data, deployed year-round |

---

## 4. How Drift Is Measured

### 4.1 Statistical Tests for Numeric Features

#### Kolmogorov-Smirnov (KS) Test

The KS test compares two empirical cumulative distribution functions (CDFs). It measures the maximum vertical distance between them.

```
KS statistic = max |F_ref(x) - F_prod(x)|
```

where F_ref and F_prod are the empirical CDFs of the reference and production distributions.

**How it works**:
1. Sort both samples and compute their CDFs
2. Find the point where the two CDFs are furthest apart
3. The KS statistic is this maximum distance (range: 0 to 1)
4. The p-value tells you the probability of observing this distance under the null hypothesis (same distribution)

**Interpretation**:
- KS statistic close to 0 → distributions are similar
- KS statistic close to 1 → distributions are very different
- p-value < 0.05 → reject null hypothesis → drift detected

**Strengths**: Non-parametric (no distribution assumptions), sensitive to location, spread, and shape differences.

**Weaknesses**: Less sensitive in the tails, power decreases with small sample sizes.

**Used in our project for**: `llm_quality`, `llm_engagement`, `llm_decision_maker`, `companyfoundedon`, and all other numeric features. Also used for prediction score drift (comparing P(Y_hat) distributions).

```python
from scipy import stats

ks_result = stats.ks_2samp(reference_scores, production_scores)
# ks_result.statistic = 0.73  → large distance
# ks_result.pvalue = 0.0001   → highly significant drift
```

#### Population Stability Index (PSI)

PSI quantifies how much a distribution has shifted by comparing bin proportions:

```
PSI = Σ (p_prod_i - p_ref_i) × ln(p_prod_i / p_ref_i)
```

where p_ref_i and p_prod_i are the proportions in bin i for reference and production.

**Interpretation**:
- PSI < 0.1 → no significant shift
- 0.1 ≤ PSI < 0.25 → moderate shift, investigate
- PSI ≥ 0.25 → significant shift, action needed

**Strengths**: Easy to interpret, widely used in credit scoring and financial ML.

**Weaknesses**: Sensitive to binning strategy, assumes discrete or discretized features.

### 4.2 Statistical Tests for Categorical Features

#### Chi-Squared Test

Compares observed vs expected category frequencies:

```
χ² = Σ (O_i - E_i)² / E_i
```

where O_i are observed production frequencies and E_i are expected frequencies based on the reference distribution.

**Used in our project for**: `llm_seniority`, `llm_industry`, `llm_geography`, `companysize`, `companytype`, `location`, etc.

**Interpretation**: p-value < 0.05 → category distribution has shifted.

#### Cramer's V

A normalized version of chi-squared that gives a value between 0 and 1:

```
V = sqrt(χ² / (n × min(r-1, c-1)))
```

**Interpretation**: V > 0.3 is typically considered meaningful drift for categorical features.

### 4.3 Multivariate Drift Detection

Individual feature tests can miss drift that only appears in feature interactions. For example, `industry = "tech"` and `companysize = "1-10"` separately might not drift, but their joint distribution (small tech startups becoming rare) could shift significantly.

Approaches include:
- **Domain classifier**: Train a binary classifier to distinguish reference from production data. High accuracy → drift detected.
- **Maximum Mean Discrepancy (MMD)**: Kernel-based test comparing distributions in a high-dimensional space.

Evidently's `DataDriftPreset` handles per-feature tests. For multivariate analysis, additional custom metrics would be needed.

### 4.4 Drift Thresholds

| Metric | No drift | Investigate | Alert | Retrain |
|--------|----------|------------|-------|---------|
| KS p-value (per feature) | > 0.05 | 0.01–0.05 | < 0.01 | Multiple features < 0.01 |
| Drift share (% features drifted) | < 20% | 20–50% | ≥ 50% | ≥ 50% sustained |
| PSI (per feature) | < 0.1 | 0.1–0.25 | ≥ 0.25 | Multiple ≥ 0.25 |
| KS on scores | > 0.05 | 0.01–0.05 | < 0.01 | < 0.01 sustained |

---

## 5. How Drift Is Managed

### 5.1 The Monitoring Pipeline

```
Training Data ──────┐
                     │
                     ▼
              ┌─────────────┐
              │  Reference   │  ← Saved at training time
              │  Distribution│     (data/reference/training_reference.csv)
              └──────┬───────┘
                     │
         compare     │     compare
         features    │     scores
                     │
              ┌──────┴───────┐
              │  Production  │  ← Collected in real-time
              │  Data        │     (logs/predictions.jsonl)
              └──────┬───────┘
                     │
                     ▼
              ┌─────────────┐
              │  Drift       │  ← Statistical tests
              │  Detector    │     (Evidently + scipy)
              └──────┬───────┘
                     │
           ┌─────────┼──────────┐
           ▼         ▼          ▼
        Dashboard  Alerts    Reports
       (Streamlit) (manual)  (HTML)
```

### 5.2 Response Strategies

**Level 1 — Monitor** (drift_share < 20%):
- Log the drift event
- Continue serving predictions
- Review during weekly monitoring check

**Level 2 — Investigate** (20% ≤ drift_share < 50%):
- Identify which features drifted and why
- Check if the drift is intentional (new campaign) or accidental (data pipeline bug)
- Assess model accuracy on recent predictions if labels are available
- Consider adjusting prediction thresholds

**Level 3 — Alert** (drift_share ≥ 50% or score drift p < 0.01):
- Flag to the data science team
- Collect new labeled data from the drifted distribution
- Evaluate model on the new population
- Begin retraining planning

**Level 4 — Retrain** (sustained alert or accuracy degradation confirmed):
- Collect and label new data from current production distribution
- Retrain with combined old + new data (or new data only if concept drift)
- Evaluate on held-out new data
- A/B test old vs new model if possible
- Deploy via MLflow model registry + CI/CD

### 5.3 The Reference Window

A key design decision is what constitutes the "reference" distribution:

| Strategy | Pros | Cons |
|----------|------|------|
| **Fixed reference** (training data) | Stable baseline, always comparable to original model assumptions | Doesn't account for intentional distribution changes |
| **Sliding window** (last N days) | Adapts to gradual changes | May mask slow drift, "boiling frog" effect |
| **Expanding window** (all historical) | Comprehensive view | Dilutes signal from recent changes |

**Our project uses fixed reference** — the training data saved as `data/reference/training_reference.csv` (first 100 rows of the processed training set). This is appropriate for our use case because:
- Small dataset (303 rows) — sliding windows would be too small
- We want to detect *any* deviation from what the model was trained on
- Retraining is manual, so we want explicit drift signals

---

## 6. Evidently AI — The Framework

### 6.1 What Is Evidently?

[Evidently AI](https://www.evidentlyai.com/) is an open-source Python library for ML monitoring. It provides:

- **Statistical drift detection** for numeric and categorical features
- **Data quality monitoring** (missing values, duplicates, outliers)
- **Model performance tracking** (when labels are available)
- **Interactive HTML reports** with distribution plots and test results
- **Programmatic API** for integration into pipelines and dashboards

### 6.2 Core Concepts (Evidently v0.7+)

#### Report

A `Report` is a collection of metrics computed on data. You configure it with one or more metrics or presets, then call `.run()` with reference and current data.

```python
from evidently import Report
from evidently.presets import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(
    reference_data=training_df,
    current_data=production_df
)
```

#### Snapshot

The result of `report.run()`. Contains computed metric results and can export to HTML.

```python
snapshot.save_html("reports/drift_report.html")
results = snapshot.metric_results  # dict of metric_id → result
```

#### Presets

Pre-configured collections of metrics for common use cases:

| Preset | What it measures |
|--------|-----------------|
| `DataDriftPreset` | Per-feature drift (KS, chi-squared, Jensen-Shannon) + overall drift summary |
| `TargetDriftPreset` | Drift in the target variable or prediction scores |
| `DataQualityPreset` | Missing values, duplicates, outliers, feature statistics |
| `RegressionPreset` | Regression model performance metrics |
| `ClassificationPreset` | Classification metrics (accuracy, F1, ROC-AUC) |

#### Individual Metrics

For fine-grained control:

| Metric | Purpose |
|--------|---------|
| `ValueDrift(column="X")` | Drift for a single column |
| `DriftedColumnsCount()` | Count and share of drifted columns |
| `MissingValuesCount()` | Missing value statistics |

### 6.3 How Evidently Selects Tests Automatically

Evidently chooses the statistical test based on feature type and sample size:

| Feature type | Small sample (< 1000) | Large sample (≥ 1000) |
|-------------|----------------------|----------------------|
| **Numeric** | Kolmogorov-Smirnov test | Wasserstein distance |
| **Categorical** | Chi-squared test | Jensen-Shannon divergence |

Thresholds are configurable but default to p-value < 0.05 for KS and chi-squared, and heuristic cutoffs for Wasserstein and Jensen-Shannon.

### 6.4 Evidently v0.7+ API Notes

The API changed significantly in v0.7 (compared to pre-0.4):

```python
# v0.7+ (current)
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.metrics import ValueDrift, DriftedColumnsCount

report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(current_data=prod, reference_data=ref)
snapshot.save_html("report.html")

# Pre-0.4 (deprecated)
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab
dashboard = Dashboard(tabs=[DataDriftTab()])
dashboard.calculate(ref, prod)
dashboard.save("report.html")
```

Key differences:
- `Report.run()` returns a `Snapshot` object (doesn't modify the report in place)
- `snapshot.save_html()` replaces `dashboard.save()`
- Metric results accessed via `snapshot.metric_results` dict
- Presets moved from `evidently.metric_preset` to `evidently.presets`

---

## 7. Implementation in This Project

### 7.1 Architecture

```
src/linkedin_lead_scoring/monitoring/
├── __init__.py            # Package exports
├── drift.py               # DriftDetector class (Evidently + scipy)
├── dashboard_utils.py     # Log loading, simulation, metrics calculation
├── profiler.py            # Inference latency profiling
└── onnx_optimizer.py      # ONNX conversion and benchmarking

streamlit_app.py           # Monitoring dashboard (reads logs, runs drift detection)
tests/test_drift.py        # Drift detection unit tests
```

### 7.2 The DriftDetector Class

Located in `src/linkedin_lead_scoring/monitoring/drift.py`:

```python
class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame) -> None:
        """Initialize with training/reference data.

        Automatically excludes the 'engaged' target column from drift analysis.
        Raises ValueError if reference_data is empty.
        """
        self.reference_data = reference_data.copy()
        self._feature_columns = [c for c in reference_data.columns if c != "engaged"]
```

#### Method: `detect_data_drift(production_data)`

Runs Evidently's drift analysis on all feature columns.

**How it works**:
1. Aligns production data columns to match reference features (drops extra columns, reorders)
2. Creates per-column `ValueDrift` metrics + a `DriftedColumnsCount` aggregate
3. Runs `Report.run()` to compute all statistical tests
4. Extracts which columns drifted and the overall drift share
5. Flags overall drift if ≥ 50% of features drifted

**Returns**:
```python
{
    "drift_detected": True,          # Overall drift flag
    "drifted_features": ["industry", "companyfoundedon"],  # Which features drifted
    "drifted_count": 2,              # Number of drifted features
    "total_features": 19,            # Total features analyzed
    "drift_share": 0.105             # Fraction drifted (2/19)
}
```

**Actual code path**:
```python
def detect_data_drift(self, production_data: pd.DataFrame) -> dict:
    ref = self.reference_data[self._feature_columns]
    prod = self._align_columns(production_data)

    per_col_metrics = [ValueDrift(column=col) for col in self._feature_columns]
    count_metric = DriftedColumnsCount()

    report = Report(metrics=[count_metric, *per_col_metrics])
    snapshot = report.run(current_data=prod, reference_data=ref)

    results = snapshot.metric_results
    drifted_features = self._extract_drifted_columns(results, per_col_metrics)
    # ... compute counts and share
```

#### Method: `detect_prediction_drift(reference_scores, production_scores)`

Compares two arrays of predicted probabilities using the scipy KS test (not Evidently, for simplicity and reliability).

**How it works**:
1. Takes two arrays of float scores
2. Runs `scipy.stats.ks_2samp()` — a two-sample Kolmogorov-Smirnov test
3. Returns drift flag based on p-value threshold (default 0.05)

**Returns**:
```python
{
    "drift_detected": True,
    "statistic": 0.73,     # Maximum CDF distance
    "p_value": 0.0001,     # Probability under null hypothesis
    "method": "ks_2samp"
}
```

**Actual code**:
```python
def detect_prediction_drift(self, reference_scores, production_scores, threshold=0.05):
    ref_arr = np.asarray(reference_scores, dtype=float)
    prod_arr = np.asarray(production_scores, dtype=float)
    ks_result = stats.ks_2samp(ref_arr, prod_arr)
    return {
        "drift_detected": bool(ks_result.pvalue < threshold),
        "statistic": float(ks_result.statistic),
        "p_value": float(ks_result.pvalue),
        "method": "ks_2samp",
    }
```

#### Method: `generate_report(production_data, output_path)`

Generates a full interactive HTML report using Evidently's `DataDriftPreset`.

**Output**: An HTML file with:
- Overall drift summary (drifted feature count, share)
- Per-feature distribution plots (reference overlay vs production)
- Statistical test results (test name, statistic, p-value, drift flag)
- Interactive filtering and sorting

```python
def generate_report(self, production_data, output_path):
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=prod, reference_data=ref)
    snapshot.save_html(output_path)
    return output_path
```

### 7.3 Reference Data

The reference distribution is saved at model export time:

- **File**: `data/reference/training_reference.csv`
- **Content**: First 100 rows of the processed training set
- **Columns**: 47 engineered features (after one-hot encoding, target encoding, and feature extraction)
- **Created by**: `scripts/export_model.py` (Session A)

These 47 features include:
- **LLM-scored numeric**: `llm_quality`, `llm_engagement`, `llm_decision_maker`
- **Target-encoded categorical**: `llm_industry`, `industry`, `companyindustry`, `languages`, `location`, `companylocation` (encoded as floats)
- **One-hot encoded**: `llm_seniority_*`, `llm_geography_*`, `llm_business_type_*`, `companysize_*`, `companytype_*`
- **Binary flags**: `has_summary`, `has_skills`, `has_jobtitle`, `is_founder`, `is_director`, etc.
- **Numeric extracted**: `summary_length`, `skills_count`, `jobtitle_length`

### 7.4 Production Data Collection

Predictions are logged by the FastAPI middleware to `logs/predictions.jsonl`:

```json
{
    "timestamp": "2026-02-20T10:30:00Z",
    "input": {"jobtitle": "CTO", "industry": "tech", "companysize": "11-50", ...},
    "score": 0.73,
    "label": "engaged",
    "inference_ms": 12.5,
    "model_version": "0.3.0"
}
```

The dashboard extracts input features from the nested `input` field and passes them to the DriftDetector for comparison against the reference data.

### 7.5 Simulation Mode

When production logs don't exist yet (development/demo), the dashboard falls back to synthetic data:

```python
def simulate_production_logs(n=100, seed=42):
    """Generate synthetic prediction logs with slight distribution shift."""
    rng = np.random.default_rng(seed)
    scores = rng.beta(a=2.5, b=3.5, size=n)  # Shifted vs training distribution
    inference_times = rng.gamma(shape=2.0, scale=5.0, size=n)  # Right-skewed latency
    # ... builds JSONL-compatible dicts
```

The Beta(2.5, 3.5) distribution produces scores skewed slightly lower than a typical training score distribution, creating visible (but mild) prediction drift in the dashboard.

---

## 8. Testing the Drift Pipeline

### 8.1 Test Strategy

The drift detection module is tested in `tests/test_drift.py` with controlled synthetic data:

| Test | Setup | Expected result |
|------|-------|----------------|
| No-drift case | Reference and production drawn from same distribution | `drift_detected = False` |
| Drift case | Production with extreme shifts (2060-2090 founding years, different cities/industries) | `drift_detected = True`, `drifted_features` non-empty |
| Result completeness | Any run | All required keys present in result dict |
| NaN handling | Production with 10% NaN values | No exception, valid result |
| Prediction drift (same) | Scores from same uniform distribution | `drift_detected = False` |
| Prediction drift (shifted) | Ref scores near 0, prod scores near 1 | `drift_detected = True`, `p_value < 0.05` |
| Report generation | Both same-distribution and shifted data | HTML file created at specified path |

### 8.2 Test Fixtures

```python
@pytest.fixture
def reference_df():
    """100 rows, 6 features: timezone, icebreaker, companyfoundedon,
    industry, companysize, location."""
    rng = np.random.default_rng(42)
    # ... numeric + categorical features

@pytest.fixture
def shifted_df():
    """60 rows with extreme shifts:
    - timezone: [5, 8, 10] vs reference [0, 1, 2]
    - icebreaker: all zeros vs reference uniform(0,1)
    - companyfoundedon: 2060-2090 vs reference 1990-2020
    - industry: [mining, agriculture, retail] vs [tech, finance, health]
    """
```

### 8.3 Edge Cases Tested

| Edge case | What it validates |
|-----------|------------------|
| Empty reference data | `ValueError` raised at init |
| Target column present | `engaged` excluded from `_feature_columns` |
| Single column | Works with just one numeric feature |
| Extra production columns | Silently ignored via `_align_columns()` |
| None metric result | `_extract_drifted_columns` skips gracefully |
| Missing widget attribute | `_is_column_drifted` returns False safely |

### 8.4 Running the Tests

```bash
conda activate oc6
python -m pytest tests/test_drift.py -v --tb=short
```

---

## 9. Dashboard Integration

### 9.1 Drift Section in Streamlit

The Streamlit dashboard (`streamlit_app.py`) includes a "Data Drift Analysis" section that:

1. Loads prediction logs (or simulates them)
2. Extracts input features from nested `input` dicts
3. Loads reference data from CSV
4. Instantiates `DriftDetector` with reference data
5. Runs `detect_data_drift()` on production features
6. Displays drift summary (status, drifted count, share) with color coding
7. Generates and embeds the full Evidently HTML report

```python
# Simplified flow from streamlit_app.py
reference_data = pd.read_csv(REFERENCE_DATA_PATH)
detector = DriftDetector(reference_data=reference_data)

prod_features = pd.json_normalize([log["input"] for log in prediction_logs])
drift_result = detector.detect_data_drift(production_data=prod_features)

# Display
if drift_result["drift_detected"]:
    st.error(f"DRIFT DETECTED — {drift_result['drift_share']:.0%} of features drifted")
else:
    st.success(f"No significant drift — {drift_result['drift_share']:.0%} of features drifted")

# Embed Evidently HTML report
report_path = detector.generate_report(prod_features, "reports/drift_report.html")
with open(report_path) as f:
    st.components.v1.html(f.read(), height=800, scrolling=True)
```

### 9.2 Dashboard Sections Related to Drift

| Section | What it shows | Drift relevance |
|---------|--------------|-----------------|
| Score Distribution | Histogram of predicted scores | Visual prediction drift |
| Performance Metrics | Inference latency distribution | Indirect — data changes may affect preprocessing time |
| Data Drift Analysis | Evidently report + summary | Direct drift detection |
| Recent Predictions | Last 20 predictions | Manual inspection of unusual patterns |

---

## 10. Decision Framework

### When to act on drift signals

```
                    ┌─────────────────────┐
                    │  Drift detected?     │
                    └──────┬──────────────┘
                           │
                    ┌──────┴──────┐
                    │  Yes        │  No → Continue monitoring
                    └──────┬──────┘
                           │
                    ┌──────┴──────────────┐
                    │  Intentional?        │
                    │  (new campaign,      │
                    │   geographic shift)  │
                    └──────┬──────────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
         Yes: Expected          No: Investigate
                │                     │
                ▼                     ▼
        Update reference      Check data pipeline
        data if model         for bugs/changes
        still accurate              │
                                    ▼
                            ┌───────────────┐
                            │ Accuracy       │
                            │ degraded?      │
                            └───┬───────────┘
                                │
                     ┌──────────┴──────────┐
                     │                     │
              No: Monitor             Yes: Retrain
              closely                       │
                                           ▼
                                    Collect new labels
                                    Retrain model
                                    Validate on new data
                                    Deploy via MLflow + CI/CD
```

### Retraining triggers for this project

| Trigger | Threshold | Measurement |
|---------|-----------|-------------|
| Feature drift (overall) | ≥ 50% features drifted | `drift_share >= 0.5` |
| Prediction drift | KS p-value < 0.01 | `detect_prediction_drift()` result |
| Engagement rate drop | Actual rate < 10% (baseline ~25%) | LemList campaign data |
| Model age | > 3 months | MLflow model registry `created_at` |
| New labeled data volume | > 500 new contacts | Data collection tracking |

### What this project does NOT (yet) cover

| Limitation | Why | Potential solution |
|------------|-----|-------------------|
| Automated retraining | Small dataset, manual process is sufficient | Airflow/Prefect pipeline triggered by drift alerts |
| Concept drift detection | Requires ground-truth labels from LemList | Delayed label pipeline (match replies back to predictions) |
| Multivariate drift | Evidently tests features independently | Domain classifier or MMD-based detection |
| A/B testing | Single model deployment | Shadow mode — serve both models, compare predictions |
| Real-time drift | Batch analysis on log files | Streaming with Kafka + Evidently monitors |

---

## References

- Evidently AI documentation: https://docs.evidentlyai.com/
- Sculley et al., "Hidden Technical Debt in Machine Learning Systems" (2015)
- Gama et al., "A Survey on Concept Drift Adaptation" (2014)
- Rabanser et al., "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift" (2019)
