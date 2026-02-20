# Session C — Monitoring, Drift Detection & Performance Optimization

**Branch**: `feature/monitoring`
**Worktree**: `worktrees/session-c`
**Role**: Evidently drift detection, Streamlit dashboard, performance profiling, ONNX optimization

---

## IMPORTANT RULES

1. **Work ONLY in this worktree directory** — never `cd` outside it
2. **Push ONLY to `feature/monitoring`** — never push to main or v0.3.0
3. **Own ONLY these files**: `src/linkedin_lead_scoring/monitoring/`, `streamlit_app.py`, `tests/test_monitoring*.py`, `tests/test_drift*.py`, `notebooks/03_*`, `scripts/profile_api.py`, `docs/PERFORMANCE_REPORT.md`, `docs/MONITORING_GUIDE.md`
4. **Do NOT touch**: `.github/`, `Dockerfile`, `src/linkedin_lead_scoring/api/`, `src/linkedin_lead_scoring/db/`
5. Run `python -m pytest tests/ -v --tb=short` before every commit
6. Read `CLAUDE.md` and `SESSION_COORDINATION.md` for full context

---

## Context: Data & Model

**Reference data** (training set): `data/processed/linkedin_leads_clean.csv` (303 rows, 20 cols)
- Use this as the reference distribution for drift detection
- Features: see CLAUDE.md for full list

**Production data format** (from Session B's logging):
Predictions are logged to `logs/predictions.jsonl` with format:
```json
{"timestamp": "...", "input": {feature_dict}, "score": 0.73, "label": "engaged", "inference_ms": 12.5, "model_version": "0.3.0"}
```

**If production logs don't exist yet**, simulate them:
- Generate synthetic production data by sampling from reference data with slight distribution shifts
- This demonstrates drift detection capability

---

## Task C.1: Create Drift Detection Module

Create `src/linkedin_lead_scoring/monitoring/__init__.py`
Create `src/linkedin_lead_scoring/monitoring/drift.py`:

1. **DriftDetector class**:
   ```python
   class DriftDetector:
       def __init__(self, reference_data: pd.DataFrame):
           """Initialize with training/reference data"""

       def detect_data_drift(self, production_data: pd.DataFrame) -> dict:
           """Run Evidently data drift analysis"""
           # Use evidently DataDriftPreset
           # Return: drift_detected (bool), drifted_features (list), report (HTML)

       def detect_prediction_drift(self, reference_scores, production_scores) -> dict:
           """Compare score distributions"""
           # Use evidently TargetDriftPreset or custom

       def generate_report(self, production_data, output_path) -> str:
           """Generate full HTML Evidently report"""
   ```

2. **Use Evidently AI** for drift detection:
   - `DataDriftPreset` for feature drift
   - `TargetDriftPreset` for prediction score drift
   - `DataQualityPreset` for data quality metrics
   - Generate HTML reports saved to `reports/`

3. **Metrics to track**:
   - Per-feature drift (KS test, PSI)
   - Score distribution shift
   - Missing value rate changes
   - Feature correlation changes

Commit: `feat: add Evidently-based data drift detection module`

## Task C.2: Build Streamlit Monitoring Dashboard

Create `streamlit_app.py` (root level — for HF Spaces deployment):

**Dashboard sections**:

1. **API Health Overview**:
   - Current status (calls /health endpoint)
   - Uptime percentage (from logs)
   - Total predictions served

2. **Score Distribution**:
   - Histogram of predicted scores (reference vs production)
   - Score percentiles (p25, p50, p75)
   - Engagement rate (% predicted as engaged)

3. **Performance Metrics**:
   - Inference time distribution (histogram + percentiles)
   - API response time trend (line chart over time)
   - Throughput (requests per minute)

4. **Data Drift Analysis**:
   - Overall drift status (green/yellow/red)
   - Per-feature drift indicators
   - Embedded Evidently HTML report (via st.components.html)

5. **Recent Predictions**:
   - Table of last 20 predictions
   - Filterable by score range

**Data sources**:
- `logs/predictions.jsonl` — prediction logs
- `logs/api_requests.jsonl` — API request logs
- `data/reference/training_reference.csv` — reference data for drift comparison
- Live API calls to `/health` endpoint

**If log files don't exist**, generate sample data for demo purposes.

Create `Dockerfile.streamlit` for separate HF Space deployment:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-streamlit.txt .
RUN pip install -r requirements-streamlit.txt
COPY streamlit_app.py .
COPY src/linkedin_lead_scoring/monitoring/ ./src/linkedin_lead_scoring/monitoring/
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Create `requirements-streamlit.txt` with Streamlit + Evidently deps.

Commit: `feat: build Streamlit monitoring dashboard with drift analysis`

## Task C.3: Implement Performance Profiling

Create `scripts/profile_api.py`:

1. **Inference profiling**:
   - Load model directly, measure predict_proba() time over 1000 calls
   - Use `time.perf_counter()` for accurate timing
   - Report: mean, p50, p95, p99 inference times

2. **API profiling** (if API is running):
   - Send 100 concurrent requests via httpx
   - Measure end-to-end response times
   - Report: latency distribution, throughput

3. **cProfile integration**:
   - Profile the full prediction pipeline
   - Identify bottleneck functions
   - Save profile report

Create `notebooks/03_performance_analysis.ipynb`:
- Load profiling results
- Visualize inference time distribution
- Compare model loading time vs inference time
- Identify optimization opportunities
- Document findings with plots

Commit: `feat: add performance profiling scripts and analysis notebook`

## Task C.4: ONNX Model Optimization

Create `scripts/optimize_model.py`:

1. **Convert XGBoost to ONNX**:
   ```python
   from skl2onnx import convert_sklearn
   # or use onnxmltools for XGBoost
   import onnxmltools
   onnx_model = onnxmltools.convert_xgboost(model, ...)
   onnxmltools.save_model(onnx_model, "model/model_optimized.onnx")
   ```

2. **Benchmark comparison**:
   - Run 1000 predictions with joblib model
   - Run 1000 predictions with ONNX Runtime
   - Compare: mean inference time, p95, p99
   - Measure memory usage

3. **Save results** as JSON for the dashboard and report

Commit: `feat: add ONNX model optimization with benchmark comparison`

## Task C.5: Write Monitoring Tests

Create `tests/test_drift.py`:
1. Test DriftDetector with identical data (no drift expected)
2. Test DriftDetector with shifted data (drift expected)
3. Test report generation creates HTML file
4. Test with missing values in production data

Create `tests/test_monitoring.py`:
1. Test Streamlit app imports correctly
2. Test metrics calculations
3. Test log parsing functions

Commit: `test: add unit tests for drift detection and monitoring`

## Task C.6: Performance Comparison Report

Create `docs/PERFORMANCE_REPORT.md`:
1. Baseline metrics (joblib model)
2. Optimized metrics (ONNX model)
3. Improvement percentage
4. Bottleneck analysis from cProfile
5. Recommendations for production
6. Include plots/screenshots

Create `docs/MONITORING_GUIDE.md`:
1. How to access the Streamlit dashboard
2. How to interpret each metric
3. What drift indicators mean
4. When to retrain the model
5. Alert thresholds

Commit: `docs: add performance report and monitoring guide`

---

## Final Step

Create PR from `feature/monitoring` → `v0.3.0` with:
- Summary of monitoring capabilities
- Dashboard screenshots
- Performance comparison results
- List of dependencies needed (evidently, streamlit, onnxruntime, onnxmltools)
