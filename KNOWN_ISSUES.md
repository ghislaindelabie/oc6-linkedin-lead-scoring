# Known Issues & Future Improvements

## Feature Engineering — Target Encoding Over-Reliance

**Status:** Open — identified during drift analysis (notebook 04)

**Problem:**
The model uses target encoding for 6 high-cardinality categorical features (`llm_industry`, `industry`, `companyindustry`, `languages`, `location`, `companylocation`). Target encoding replaces each category value with the mean engagement rate from training data. These 6 features account for ~67% of the model's feature importance, meaning the model primarily looks up historical engagement rates by category rather than learning generalizable patterns.

**Impact:**
- The model is fragile to concept drift: when production engagement rates differ from training, predictions degrade
- Small encoding differences (e.g., from different training sets) can cause large prediction shifts
- The model does not generalize well to categories unseen during training

**Recommended fix:**
Replace target encoding with grouped one-hot encoding for high-cardinality features:
- Keep the top-N most frequent categories as individual one-hot columns
- Group all remaining categories into an "other" bucket
- This avoids leaking target information into features while keeping interpretability

Example for `industry` (51 unique values):
```python
# Keep top 10 industries + "other"
top_10 = df["industry"].value_counts().head(10).index
df["industry_grouped"] = df["industry"].where(df["industry"].isin(top_10), "other")
# Then one-hot encode (11 columns instead of 51 or 1 target-encoded)
```

**Priority:** Medium — should be addressed before v2 production deployment

## Data Quality — companysize Inconsistency

**Status:** Open

**Problem:**
The original training data (`linkedin_leads_clean.csv`, 303 rows) has `companysize` as ranges (e.g., "2-10", "11-50", "51-200", "201-500"). Production data has raw employee counts (e.g., "1", "75", "40141"). Only 6 of 68 production values overlap with the original 7 range categories.

**Impact:**
- When one-hot encoding, production data creates many more dummy columns (97 features vs 47)
- v1 and v2 have incompatible feature spaces
- Most companysize dummy columns in v2 are noise (single-company indicators)

**Recommended fix:**
Normalize raw employee counts to the same range buckets used in the original data:
```python
bins = [0, 2, 10, 50, 200, 500, 1000, 5000, 10000, float("inf")]
labels = ["1", "2-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000", "5001-10000", "10001+"]
df["companysize"] = pd.cut(df["companysize"].astype(float), bins=bins, labels=labels)
```

**Priority:** High — required for clean v2 deployment
