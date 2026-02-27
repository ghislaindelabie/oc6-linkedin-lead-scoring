# Drift Demonstration & Soutenance Preparation Plan

**Goal**: Demonstrate real drift detection, model evaluation on holdout data, and retraining lifecycle using actual LinkedIn contacts — ready for soutenance in 2 days.

**Branch**: `feature/drift-analysis` from `v0.3.1`

---

## Timeline (2 Days)

### Day 1 — Data & Analysis

| Block | Task | Who | Duration |
|-------|------|-----|----------|
| Morning | T1: Prepare raw data + run notebook 01 (LLM enrichment) | **You** (OpenAI API costs) | 1-2h |
| Morning | T2: Generate synthetic drift datasets | **Claude** (while you run T1) | 30min |
| Afternoon | T3: Split labelled data into holdout vs new training | **Together** | 15min |
| Afternoon | T4: Create drift analysis notebook (04) | **Claude** | 1-2h |
| Afternoon | T5: Send real data through API → populate logs | **Together** | 30min |

### Day 2 — Retraining & Polish

| Block | Task | Who | Duration |
|-------|------|-----|----------|
| Morning | T6: Evaluate model on holdout (measure real performance) | **Together** | 30min |
| Morning | T7: Retrain model with additional labelled data | **Together** | 1h |
| Afternoon | T8: Dashboard demo with real data | **Together** | 30min |
| Afternoon | T9: Update docs, final review | **Claude** | 30min |

---

## Task Details

### T1 — Prepare Real Contact Data (YOU — blocking)

**What**: Run your new LinkedIn contacts through notebook 01 to get LLM-enriched features.

**Input**: Raw CSV in LemList export format (same as `data/FR - LinkedIn Followers - Conversation starter.csv`)

**Steps**:
1. Place your raw CSV in `data/` directory
2. Open `notebooks/01_linkedin_data_prep.ipynb`
3. Update the data source path to point to your new file
4. Run the full notebook (LLM enrichment will call OpenAI API)
5. Save output as `data/production/new_contacts_enriched.csv`

**For labelled data** (contacts with known outcomes):
- Same process, but ensure the `engaged` column (or `laststate`/`status` columns) is present
- Save as `data/production/labelled_holdout_enriched.csv`

**Output**: Enriched CSV(s) with all 20 columns matching `linkedin_leads_clean.csv` schema

**Cost estimate**: ~$0.50-2.00 for 50-150 contacts via gpt-4o-mini

---

### T2 — Generate Synthetic Drift Datasets (CLAUDE)

**What**: Create synthetic profiles that deliberately trigger each type of drift, for comparison and testing.

**Scenarios to generate**:

| Dataset | What It Shows | How |
|---------|---------------|-----|
| `drift_sector_shift.csv` | **Covariate shift** — profiles from healthcare, agriculture, public sector (industries absent from training) | New industry/companyindustry values, different companysize distribution |
| `drift_seniority_shift.csv` | **Feature drift** — all C-Level/Executive profiles (training data is mostly Mid/Senior) | Skewed llm_seniority, higher llm_decision_maker, higher llm_quality |
| `drift_geography_shift.csv` | **Geographic drift** — profiles from Asia, Africa, South America (training is EU/US-heavy) | Different location, companylocation, llm_geography=other dominant |
| `drift_quality_degradation.csv` | **Data quality drift** — sparse profiles with missing fields | Low llm_quality, no summary, few skills, missing company info |
| `no_drift_baseline.csv` | **Control** — profiles similar to training distribution | Sample from same industries/seniorities as training data |

**Output**: 5 CSV files in `data/drift_scenarios/`, each with ~50 rows, in the 47-feature processed format (ready for drift detection, no re-enrichment needed)

---

### T3 — Split Labelled Data (TOGETHER — 15 min)

**What**: From your ~50-100 labelled contacts, split into:
- **Holdout evaluation set** (~70%): Used to measure real model performance on unseen data (T6)
- **Additional training data** (~30%): Added to training set for retraining (T7)

This split lets you demonstrate both:
- "Here's how the model performs on new real data" (holdout)
- "Here's how retraining with new data improves it" (retrain)

---

### T4 — Drift Analysis Notebook (CLAUDE — core deliverable)

**File**: `notebooks/04_drift_monitoring_analysis.ipynb`

**This IS your soutenance presentation for the monitoring section.**

**Sections**:

1. **Introduction & Context**
   - Business context: model deployed, new contacts coming in
   - What is drift and why it matters

2. **Load Data**
   - Training reference (100 rows from `data/reference/`)
   - Real production data (your new contacts from T1)
   - Synthetic drift scenarios (from T2)

3. **Data Drift Detection — Real Production Data**
   - Run Evidently `DataDriftPreset` on your real contacts vs training reference
   - Per-feature distribution comparison (histograms/violin plots)
   - KS test results table with p-values
   - Interpretation: which features shifted and business explanation

4. **Data Drift Detection — Synthetic Scenarios**
   - Run same analysis on each drift scenario
   - Side-by-side comparison: sector shift vs geography shift vs quality degradation
   - Show how drift_share increases with severity
   - Demonstrate alert thresholds (20%, 50%) triggering

5. **Prediction Drift Analysis**
   - Score distributions: training predictions vs new data predictions
   - KS test on prediction scores
   - Visualization: overlapping histograms

6. **Concept Drift — Ground Truth Analysis** (if labelled data available)
   - Actual vs predicted on holdout set
   - Confusion matrix, F1, precision, recall
   - Compare to training performance (F1=0.556)
   - This is the strongest evidence for/against concept drift

7. **Monitoring Dashboard Integration**
   - Show how this analysis connects to the live Streamlit dashboard
   - Reference the automated alerting thresholds

8. **Recommendations & Retraining Decision**
   - Based on findings: should we retrain?
   - What triggers retraining in production?
   - Link to T7 (actual retraining)

---

### T5 — Populate Real API Logs (TOGETHER — 30 min)

**What**: Send your real contacts through the deployed API to generate genuine prediction logs.

**Steps**:
1. Use `/predict/batch` endpoint with real contact data
2. This populates `logs/predictions.jsonl` with real entries
3. Dashboard switches from synthetic fallback to real data

**Script**: `scripts/generate_production_logs.py` — batch-sends contacts through API, handles pagination

**Why this matters for soutenance**: "Live demo" shows real data flowing through the system, not synthetic.

---

### T6 — Evaluate Model on Holdout Data (TOGETHER — 30 min)

**What**: Measure real model performance on your labelled holdout set.

**Steps**:
1. Load holdout data (from T3)
2. Run predictions (local, using `scripts/validate_pipeline.py` pattern)
3. Compare predicted labels vs actual `engaged` column
4. Compute: F1, precision, recall, confusion matrix
5. Log evaluation run to MLflow

**Key question this answers**: "Does the model still perform well on new, unseen data?"
- If F1 drops significantly → concept drift evidence → justifies retraining
- If F1 holds → model is stable → monitoring is working

---

### T7 — Retrain Model with Additional Data (TOGETHER — 1h)

**What**: Add labelled data to training set, retrain, compare with MLflow.

**Approach**: Use `scripts/export_model.py` (standalone, MLflow-tracked)

**Steps**:
1. Merge original training data + new labelled data (~30% from T3)
2. Run export_model.py with `--data-path` pointing to merged dataset
3. MLflow registers new model version (v2)
4. Compare v1 vs v2 metrics in MLflow UI:
   - Did F1 improve?
   - Did specific metrics change?
5. Optionally: evaluate v2 on the holdout set (remaining 70% from T3)

**Deliverables**:
- New model artifacts in `model/` (or versioned directory)
- MLflow experiment comparison screenshot
- Before/after metrics table

**For soutenance**: This demonstrates the complete MLOps lifecycle:
`monitor → detect drift → evaluate → retrain → compare → deploy`

---

### T8 — Dashboard Demo with Real Data (TOGETHER — 30 min)

**What**: Verify the Streamlit dashboard displays real monitoring data.

**Verify**:
- API Health section shows live status
- Score distribution shows real prediction histogram
- Performance metrics show real inference times
- Data drift section shows Evidently analysis on real data
- Recent predictions table shows real entries

**If issues**: Fix any dashboard bugs on `feature/drift-analysis` branch.

---

### T9 — Documentation Update (CLAUDE — 30 min)

**What**: Update docs to reflect drift analysis findings.

**Files to update**:
- `docs/MONITORING_GUIDE.md` — add real drift findings and thresholds observed
- `README.md` — add drift analysis section referencing notebook 04
- Commit all changes on feature/drift-analysis

---

## Files Created/Modified

| File | Action | Task |
|------|--------|------|
| `data/production/new_contacts_enriched.csv` | Created by user | T1 |
| `data/production/labelled_holdout_enriched.csv` | Created by user | T1 |
| `data/drift_scenarios/*.csv` (5 files) | Created by Claude | T2 |
| `scripts/generate_drift_scenarios.py` | New script | T2 |
| `scripts/generate_production_logs.py` | New script | T5 |
| `notebooks/04_drift_monitoring_analysis.ipynb` | New notebook | T4 |
| `docs/MONITORING_GUIDE.md` | Updated | T9 |
| `README.md` | Updated | T9 |

---

## Soutenance Checklist

### 1. Présentation des Livrables (15 min)

- [ ] **Monitoring results**: Notebook 04 with real drift graphs + metrics
- [ ] **Log analysis**: Dashboard showing real prediction logs + performance metrics
- [ ] **Performance optimization**: Notebook 03 (ONNX 26.5x speedup) — already done
- [ ] **GitHub structure**: Navigate repo, show Dockerfile, CI/CD workflows, README
- [ ] **API demo**: Send request with real contact → show score
- [ ] **CI/CD demo**: Push commit → show GitHub Actions triggering tests + deploy

### 2. Discussion Topics (10 min) — Be Ready For

- [ ] **Robustness**: Error handling (400 validation, 500 fallback, batch limits, missing fields)
- [ ] **Drift management**: Show notebook 04 findings, explain KS test, thresholds, response strategy
- [ ] **Concept drift**: Show holdout evaluation (T6), explain delayed labels challenge
- [ ] **Retraining**: Show MLflow v1 vs v2 comparison (T7), explain when to trigger
- [ ] **Scalability**: Batch endpoint (10K leads), ONNX optimization, async DB, Docker horizontal scaling

---

## Risks

| Risk | Mitigation |
|------|------------|
| LLM enrichment takes too long | Process in batches, budget for ~2h |
| OpenAI API costs | gpt-4o-mini is cheap (~$1-2 for 150 contacts) |
| Not enough labelled data | Even 30 labelled contacts is enough for basic evaluation |
| Model performs identically on new data | Still valuable — proves model is stable (no drift) |
| Dashboard issues with real data | Fix on feature branch, tested in T8 |

---

## Approval Needed

Before starting implementation:
1. ✅ Plan approved?
2. 📁 Where is your raw contact data? (file path)
3. 📁 Where is your labelled data? (file path, or same file with status column?)
4. 💰 OK with OpenAI API costs for LLM enrichment (~$1-2)?
5. 🔀 Confirm branch strategy: `feature/drift-analysis` from `v0.3.1`?
