# Monitoring Guide — LinkedIn Lead Scoring

**Audience**: MLOps engineers and data scientists maintaining the lead scoring system.

---

## 1. Accessing the Streamlit Dashboard

The monitoring dashboard is deployed as a separate Hugging Face Space (Streamlit).

### Local development

```bash
conda activate oc6
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

### Environment variables (configure before launch)

| Variable | Default | Description |
|----------|---------|-------------|
| `PREDICTIONS_LOG` | `logs/predictions.jsonl` | Path to prediction log file |
| `API_REQUESTS_LOG` | `logs/api_requests.jsonl` | Path to API request log file |
| `REFERENCE_DATA_PATH` | `data/reference/training_reference.csv` | Reference distribution CSV |
| `API_BASE_URL` | `http://localhost:7860` | FastAPI endpoint base URL |

If the log files do not exist, the dashboard falls back to **simulated data** (100 synthetic predictions) so the UI remains functional during development.

### Docker (HF Spaces)

```bash
docker build -f Dockerfile.streamlit -t monitoring-dashboard .
docker run -p 8501:8501 \
  -e PREDICTIONS_LOG=/data/predictions.jsonl \
  -e API_BASE_URL=https://your-api-space.hf.space \
  monitoring-dashboard
```

---

## 2. Interpreting Dashboard Metrics

### 2.1 API Health Overview

| Indicator | Green | Yellow | Red |
|-----------|-------|--------|-----|
| API status | `/health` returns 200 | — | `/health` timeout or non-200 |
| Success rate | ≥ 99% | 95–99% | < 95% |
| Total predictions | Any value | — | 0 (no traffic in 24h) |

**Success rate** is computed from `logs/api_requests.jsonl`: `(2xx responses) / (total requests)`.

### 2.2 Score Distribution

| Metric | Healthy range | Action if outside |
|--------|:-------------:|:-----------------:|
| Engagement rate | 15–40% | Check for distribution shift |
| p50 score | 0.3–0.7 | Review threshold calibration |
| p95 score | 0.7–1.0 | Normal (high-confidence positive) |

The **score histogram** compares the current production distribution against the reference (training set). A visible horizontal shift indicates prediction drift — review the Drift Analysis section.

### 2.3 Performance Metrics (Inference Latency)

| Percentile | Target (ONNX) | Target (joblib fallback) | Alert threshold |
|------------|:------------:|:------------------------:|:---------------:|
| p50 | < 1 ms | < 5 ms | — |
| p95 | < 5 ms | < 20 ms | > 50 ms |
| p99 | < 10 ms | < 50 ms | > 100 ms |

Latency is logged per-request in `inference_ms` field of `logs/predictions.jsonl`.

If p95 exceeds the alert threshold:
1. Check if the ONNX model is loaded (not joblib fallback).
2. Check host CPU load — the dashboard does not auto-scale.
3. If sustained, open a GitHub issue and tag the infra team.

### 2.4 Data Drift Analysis

The drift panel shows a full Evidently HTML report embedded in the dashboard.

**Overall drift status**:
- **Green** — fewer than 50% of features show statistical drift.
- **Red** — ≥ 50% of features drifted. Model retraining is recommended.

**Per-feature drift** (KS test for numeric, chi-squared for categorical):
- p-value < 0.05 → feature is flagged as drifted.
- The Evidently report shows distribution plots for each feature.

### 2.5 Recent Predictions Table

Shows the last 20 predictions with score, label, and inference time.
Use the score range slider to filter to high-confidence predictions (score > 0.7) or uncertain ones (0.4–0.6 range).

---

## 3. What Drift Indicators Mean

### Data drift (feature distribution shift)

Drift in input features means the leads reaching the API differ from the training population.

| Feature | Likely cause of drift | Action |
|---------|-----------------------|--------|
| `industry` | Targeting a new vertical | Collect labelled data from new vertical |
| `companysize` | Campaign targeting change | Recalibrate thresholds for new segment |
| `timezone` | Geographic expansion | No action if intentional |
| `icebreaker` | Changed copy/templates | Update training labels |

### Prediction drift (score distribution shift)

Detected via KS test on `score` values between reference and production windows.
- `drift_detected: True` with `p_value < 0.05` means the score distribution has shifted.
- This can happen even without feature drift (model concept drift).
- Compare predicted engagement rate vs actual engagement rate from LemList reply data.

### Covariate shift vs concept drift

| Type | Symptom | Root cause |
|------|---------|------------|
| **Covariate shift** | Feature drift detected, but model still accurate | Input population changed |
| **Concept drift** | Prediction drift, accuracy degraded | Target relationship changed |
| **Both** | All drift flags active | Campaign / market change |

Run `DriftDetector.detect_data_drift()` and `detect_prediction_drift()` together for diagnosis.

---

## 4. When to Retrain the Model

Trigger model retraining when **any** of the following conditions are met:

| Trigger | Threshold | How to measure |
|---------|-----------|----------------|
| Overall data drift | ≥ 50% features drifted | `drift_share >= 0.5` in drift report |
| Prediction drift | KS p-value < 0.01 | `p_value < 0.01` in prediction drift result |
| Engagement rate drop | Actual rate < 10% (was ~25%) | Compare LemList replies vs `engagement_rate` |
| Model age | > 3 months since last training | Check MLflow model registry `created_at` |
| Data volume | > 500 new labelled examples | Sufficient for meaningful update |

### Retraining workflow

```bash
# 1. Export new labelled data from LemList
python scripts/export_lemlist.py --output data/raw/new_batch.csv

# 2. Merge with existing training set and retrain
python scripts/train_model.py --data data/processed/linkedin_leads_clean.csv

# 3. Evaluate and register in MLflow
python scripts/evaluate_model.py

# 4. Convert retrained model to ONNX
python scripts/optimize_model.py --model-path model/xgboost_model.joblib

# 5. Deploy to HF Spaces (Session A / CI/CD)
git push origin feature/your-branch  # triggers GitHub Actions
```

---

## 5. Alert Thresholds Reference

| Alert | Condition | Severity | Owner |
|-------|-----------|----------|-------|
| API down | `/health` fails 3× in 5 min | **Critical** | Infrastructure |
| High error rate | Error rate > 5% over 10 min | **High** | API / Backend |
| Slow inference | p95 > 50 ms over 5 min | **Medium** | ML Engineering |
| Data drift detected | `drift_share >= 0.5` | **Medium** | Data science |
| Prediction drift | `p_value < 0.01` | **Medium** | Data science |
| Model age > 3 months | Registry `created_at` | **Low** | Data science |
| Low engagement rate | `engagement_rate < 0.10` | **Low** | Business / data science |

Alerts are currently manual (check dashboard daily). For automated alerting, integrate Uptime Kuma with Slack webhooks on the `/health` endpoint.

---

## 6. Log Format Reference

### Prediction log (`logs/predictions.jsonl`)

```jsonc
{
  "timestamp": "2026-02-20T12:00:00",
  "input": {
    "jobtitle": "CTO",
    "industry": "tech",
    "companysize": "11-50"
    // ... full feature dict
  },
  "score": 0.73,           // P(engaged) from model
  "label": "engaged",      // "engaged" if score > 0.5
  "inference_ms": 0.008,   // ONNX Runtime latency
  "model_version": "0.3.0"
}
```

### API request log (`logs/api_requests.jsonl`)

```jsonc
{
  "timestamp": "2026-02-20T12:00:01",
  "endpoint": "/predict",
  "method": "POST",
  "status_code": 200,
  "response_ms": 5.2       // end-to-end API latency (includes preprocessing)
}
```

---

## 7. Useful Commands

```bash
# Run drift detection manually
python - <<'EOF'
import pandas as pd
from linkedin_lead_scoring.monitoring.drift import DriftDetector

ref = pd.read_csv("data/processed/linkedin_leads_clean.csv")
prod = pd.read_json("logs/predictions.jsonl", lines=True)
# Extract input features from nested 'input' column
prod_features = pd.json_normalize(prod["input"])

detector = DriftDetector(reference_data=ref)
result = detector.detect_data_drift(production_data=prod_features)
print(result)
detector.generate_report(production_data=prod_features, output_path="reports/drift_report.html")
EOF

# Benchmark inference latency
python scripts/optimize_model.py --n-calls 1000

# Profile API load
python scripts/profile_api.py --mode api --n-requests 200 --api-url http://localhost:7860
```
