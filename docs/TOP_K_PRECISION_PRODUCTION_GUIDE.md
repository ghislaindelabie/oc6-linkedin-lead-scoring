# Precision@K Production Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Understanding Precision@K](#understanding-precisionk)
3. [Model Performance Summary](#model-performance-summary)
4. [Production Implementation](#production-implementation)
5. [Phased Rollout Strategy](#phased-rollout-strategy)
6. [Code Examples](#code-examples)
7. [Monitoring & Iteration](#monitoring--iteration)

---

## Overview

This guide explains how to use the LinkedIn Lead Scoring model in production to prioritize leads for outreach, maximizing response rate while minimizing wasted effort.

**Use Case**: Lead Prioritization (Ranking Problem)
- Goal: Contact leads with highest probability of engagement
- Metric: Precision (response rate) in top K contacts
- Target: 65% precision (vs 40% baseline random selection)

---

## Understanding Precision@K

### What is Precision@K?

Precision@K measures the accuracy (response rate) when you contact the **top K leads** ranked by predicted probability.

**Formula**:
```
Precision@K = (Number of positive responses in top K) / K
```

**Example**:
- You contact top 50 leads (K=50)
- 35 respond positively
- Precision@50 = 35/50 = 70%

### Why Precision@K vs Traditional Metrics?

| Metric | Use Case | Why It Matters for Us |
|--------|----------|---------------------|
| **Accuracy** | Balanced classification | ❌ Not useful - we don't care about correctly predicting "not engaged" |
| **F1-Score** | Balanced precision/recall | ❌ Not useful - we care about precision, not recall |
| **Precision@K** | Ranking/prioritization | ✅ **Perfect fit** - measures response rate in top contacts |

### Key Concepts

**1. Threshold-Based Selection**
- Model assigns probability score to each lead (0.0 to 1.0)
- Apply threshold (e.g., 0.769) to filter high-quality leads
- Contact only leads with score >= threshold
- **Advantage**: Maintains consistent precision
- **Disadvantage**: Number of contacts varies by data quality

**2. Top-K Selection**
- Sort all leads by probability (descending)
- Take top K leads regardless of score
- **Advantage**: Fixed number of contacts (budget control)
- **Disadvantage**: Precision varies by data quality

---

## Model Performance Summary

### Training Data
- **Dataset**: 303 LinkedIn leads
- **Positive Rate**: 40% (121 engaged leads)
- **Features**: 47 classical features (demographic, behavioral)
- **Algorithm**: XGBoost with 5-fold cross-validation

### Cross-Validation Results (Unbiased Estimates)

| Top K Leads | Precision (Response Rate) | Lift vs Random |
|-------------|--------------------------|----------------|
| Top 5       | 80%                      | 2.0x           |
| Top 10      | 70%                      | 1.75x          |
| Top 20      | 65%                      | 1.6x           |
| Top 50      | 55%                      | 1.4x           |
| Top 100     | 50%                      | 1.25x          |

**Optimal Threshold for 65% Precision**:
- **Threshold**: 0.769
- **Expected Leads**: ~64 out of 303 (21%)
- **Expected Responses**: ~42 (65% of 64)
- **Lift**: 1.6x vs random

### Important Caveats

⚠️ **Small Dataset**: Only 303 training samples → high variance
⚠️ **Domain Shift Risk**: Performance may drop on different lead sources
⚠️ **Conservative Estimates**: Real-world precision likely 5-10% lower

---

## Production Implementation

### Option 1: Threshold-Based (Recommended)

**Best for**: Maximizing precision, flexible contact volume

```python
import joblib
import pandas as pd
import numpy as np

# 1. Load trained model
model = joblib.load('models/xgboost_lead_scorer.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# 2. Load and prepare new leads
new_leads = pd.read_csv('data/new_leads_1000.csv')
X_new = prepare_features(new_leads)  # Apply same preprocessing as training
X_new_scaled = scaler.transform(X_new)

# 3. Score all leads
probabilities = model.predict_proba(X_new_scaled)[:, 1]

# 4. Apply threshold
THRESHOLD = 0.769  # From CV analysis
high_quality_mask = probabilities >= THRESHOLD

# 5. Select high-quality leads
leads_to_contact = new_leads[high_quality_mask].copy()
leads_to_contact['score'] = probabilities[high_quality_mask]
leads_to_contact = leads_to_contact.sort_values('score', ascending=False)

print(f"Leads to contact: {len(leads_to_contact)}")
print(f"Expected precision: ~65%")
print(f"Expected responses: ~{int(len(leads_to_contact) * 0.65)}")
```

**Key Points**:
- Number of contacts varies based on data quality
- Maintains consistent ~65% precision
- More leads if data quality is high, fewer if low

### Option 2: Fixed Top-K

**Best for**: Fixed budget, exact contact target

```python
# 1-3. Same as Option 1 (load model, data, score)

# 4. Sort by probability
sorted_indices = np.argsort(probabilities)[::-1]

# 5. Take top K
K = 100  # Contact exactly 100 leads
top_k_indices = sorted_indices[:K]

leads_to_contact = new_leads.iloc[top_k_indices].copy()
leads_to_contact['score'] = probabilities[top_k_indices]

print(f"Leads to contact: {K}")
print(f"Score range: {probabilities[top_k_indices].min():.3f} - {probabilities[top_k_indices].max():.3f}")
print(f"Expected precision: 55-70% (depends on data quality)")
```

**Key Points**:
- Fixed number of contacts (budget control)
- Precision varies based on data quality
- Risk: might contact low-quality leads if threshold drops

---

## Phased Rollout Strategy

### Phase 1: Validation (First 50-100 Leads)

**Goal**: Verify model works on new data before scaling

```python
# Score all new leads
probabilities = model.predict_proba(X_new_scaled)[:, 1]

# Select top 50 for validation
sorted_indices = np.argsort(probabilities)[::-1]
validation_leads = new_leads.iloc[sorted_indices[:50]].copy()
validation_leads['score'] = probabilities[sorted_indices[:50]]

# Export for outreach
validation_leads.to_csv('phase1_validation_leads.csv', index=False)

print("PHASE 1: Contact top 50 leads and track responses")
print(f"Min score: {validation_leads['score'].min():.3f}")
print(f"Expected precision: 60-70%")
```

**Success Criteria**:
- If actual precision >= 60% → Proceed to Phase 2
- If actual precision 50-59% → Recalibrate expectations, proceed cautiously
- If actual precision < 50% → STOP, investigate model/data issues

### Phase 2: Calibration

**Goal**: Adjust threshold/K based on Phase 1 results

```python
# Example: Phase 1 showed 55% precision (lower than expected)

# Option A: Lower target precision
ADJUSTED_THRESHOLD = 0.65  # Lower threshold for more leads
print(f"Adjusting target to 55% precision")

# Option B: Reduce K to maintain precision
ADJUSTED_K = 30  # Contact fewer leads but maintain 65%
print(f"Targeting top {ADJUSTED_K} leads for 65% precision")

# Option C: Accept reality and optimize for ROI
print(f"Accept 55% precision, calculate ROI vs cost")
```

### Phase 3: Production Scale

**Goal**: Scale to full contact list with continuous monitoring

```python
# Apply calibrated threshold/K from Phase 2
final_leads = new_leads[probabilities >= ADJUSTED_THRESHOLD].copy()
final_leads['score'] = probabilities[probabilities >= ADJUSTED_THRESHOLD]
final_leads = final_leads.sort_values('score', ascending=False)

# Add confidence flags
final_leads['confidence'] = pd.cut(
    final_leads['score'],
    bins=[0, 0.6, 0.75, 0.85, 1.0],
    labels=['Medium', 'High', 'Very High', 'Excellent']
)

# Export with priority tiers
final_leads.to_csv('production_leads_prioritized.csv', index=False)

print(f"Total leads to contact: {len(final_leads)}")
print("\nBy confidence tier:")
print(final_leads['confidence'].value_counts())
```

---

## Code Examples

### Complete Production Pipeline

```python
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

class LeadScorer:
    """Production lead scoring pipeline."""

    def __init__(self, model_path, scaler_path, threshold=0.769):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.threshold = threshold

    def score_leads(self, leads_df):
        """Score new leads and return prioritized list."""
        # Prepare features
        X = self.prepare_features(leads_df)
        X_scaled = self.scaler.transform(X)

        # Predict probabilities
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        # Add scores to dataframe
        leads_df = leads_df.copy()
        leads_df['engagement_score'] = probabilities
        leads_df['priority'] = self.assign_priority(probabilities)

        return leads_df.sort_values('engagement_score', ascending=False)

    def select_top_k(self, scored_leads, k):
        """Select top K leads."""
        return scored_leads.head(k)

    def select_by_threshold(self, scored_leads):
        """Select leads above threshold."""
        return scored_leads[scored_leads['engagement_score'] >= self.threshold]

    def assign_priority(self, scores):
        """Assign priority labels based on score."""
        return pd.cut(
            scores,
            bins=[0, 0.6, 0.75, 0.85, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )

    def prepare_features(self, df):
        """Prepare features for model (same as training)."""
        # TODO: Implement feature engineering pipeline
        # - Handle missing values
        # - Encode categorical variables
        # - Create derived features
        # - Select only model features
        pass

# Usage
scorer = LeadScorer(
    model_path='models/xgboost_lead_scorer.pkl',
    scaler_path='models/feature_scaler.pkl',
    threshold=0.769
)

# Load new leads
new_leads = pd.read_csv('data/new_leads.csv')

# Score all leads
scored_leads = scorer.score_leads(new_leads)

# Option 1: Threshold-based
high_priority = scorer.select_by_threshold(scored_leads)
print(f"High priority leads: {len(high_priority)}")

# Option 2: Top-K
top_100 = scorer.select_top_k(scored_leads, k=100)
print(f"Top 100 leads selected")

# Export for outreach team
high_priority.to_csv(f'outreach_list_{datetime.now():%Y%m%d}.csv', index=False)
```

### Monitoring Script

```python
def track_campaign_performance(leads_contacted, responses_received):
    """Track actual vs expected performance."""

    # Calculate actual precision
    actual_precision = responses_received / leads_contacted

    # Expected from CV
    expected_precision = 0.65
    baseline_precision = 0.40

    # Calculate metrics
    lift = actual_precision / baseline_precision
    vs_expected = actual_precision / expected_precision

    print("="*60)
    print("CAMPAIGN PERFORMANCE REPORT")
    print("="*60)
    print(f"Leads contacted: {leads_contacted}")
    print(f"Responses received: {responses_received}")
    print(f"Actual precision: {actual_precision:.1%}")
    print(f"Expected precision: {expected_precision:.1%}")
    print(f"vs Expected: {vs_expected:.1%}")
    print(f"Lift vs random: {lift:.2f}x")

    # Decision logic
    if actual_precision >= 0.60:
        print("\n✓ Model performing well - continue current strategy")
    elif actual_precision >= 0.50:
        print("\n⚠️  Model underperforming - consider recalibration")
    else:
        print("\n❌ Model failing - stop and investigate")

    return actual_precision

# Example usage
actual_prec = track_campaign_performance(
    leads_contacted=64,
    responses_received=38
)
```

---

## Monitoring & Iteration

### Key Metrics to Track

| Metric | Target | Action if Below Target |
|--------|--------|----------------------|
| **Precision (Response Rate)** | >= 60% | Increase threshold or reduce K |
| **Lift vs Random** | >= 1.5x | Collect more data, retrain model |
| **Model Drift** | Score distribution stable | Retrain with recent data |
| **Coverage** | 10-20% of list | Adjust threshold if too high/low |

### When to Retrain

Retrain the model when:
- **Quarterly**: Regular refresh with new data
- **Performance drop**: Precision falls below 50%
- **Data drift**: New lead sources or markets
- **Sample size**: Collected 500+ new labeled examples

### A/B Testing Framework

```python
# Split new leads into treatment groups
np.random.seed(42)
new_leads['group'] = np.random.choice(['model', 'random'], size=len(new_leads), p=[0.7, 0.3])

# Model group: use ML scores
model_group = new_leads[new_leads['group'] == 'model']
model_contacts = scorer.select_by_threshold(scorer.score_leads(model_group))

# Random group: random selection (control)
random_group = new_leads[new_leads['group'] == 'random']
random_contacts = random_group.sample(n=len(model_contacts))

# Compare results after campaign
print("A/B Test Results:")
print(f"Model precision: {model_precision:.1%}")
print(f"Random precision: {random_precision:.1%}")
print(f"Lift: {model_precision/random_precision:.2f}x")
```

---

## Summary Checklist

### Before Production
- [ ] Model trained on representative data
- [ ] Cross-validation shows acceptable precision
- [ ] Feature engineering pipeline documented
- [ ] Model and scaler artifacts saved
- [ ] Threshold/K values determined from CV

### During Rollout
- [ ] Phase 1: Test on 50-100 leads
- [ ] Measure actual precision vs expected
- [ ] Calibrate threshold if needed
- [ ] Phase 3: Scale to full list

### Ongoing Operations
- [ ] Track precision weekly
- [ ] Monitor score distribution for drift
- [ ] Collect feedback on contacted leads
- [ ] Retrain quarterly or when performance drops
- [ ] A/B test model vs random baseline

---

## Contact & Support

For questions or issues with the lead scoring model:
- Model training notebook: `notebooks/02_linkedin_model_training.ipynb`
- Feature engineering: `src/linkedin_lead_scoring/features/`
- Model artifacts: `models/`

Last updated: 2025-12-27
