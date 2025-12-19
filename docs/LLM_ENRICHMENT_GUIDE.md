# LLM Enrichment Guide

Complete guide for using OpenAI-powered feature enrichment in the LinkedIn Lead Scoring project.

---

## 🎯 Overview

The LLM enrichment module (`llm_enrichment.py`) provides automated feature generation using OpenAI's GPT models. It's designed for:

- **Batch processing**: Process multiple rows per API call to minimize costs
- **JSON-formatted outputs**: Structured data extraction
- **MLflow integration**: Automatic tracking of costs, tokens, and enrichment metrics
- **Cost safety**: Built-in spending limits and cost estimation

---

## 📦 Installation

### 1. Install Dependencies

```bash
# Activate your conda environment
conda activate oc6

# Install OpenAI packages via uv
uv pip install openai tiktoken

# Or reinstall the full project
uv pip install -e ".[dev]"
```

### 2. Configure OpenAI API Key

Add your OpenAI API key to `.env` file:

```bash
# Copy example file
cp .env.example .env

# Edit .env and add your key
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxx

# Optional: Set default model and limits
OPENAI_DEFAULT_MODEL=gpt-4o-mini
LLM_BATCH_SIZE=10
LLM_MAX_COST_USD=10.00
```

**Get your API key**: https://platform.openai.com/api-keys

---

## 🚀 Quick Start

### Basic Example

```python
import pandas as pd
from linkedin_lead_scoring.data.llm_enrichment import (
    enrich_column_with_llm,
    PROMPT_PROFILE_QUALITY,
    prepare_profile_text_column
)

# Load your LinkedIn leads data
df = pd.read_csv("data/linkedin_leads.csv")

# Step 1: Prepare profile text (combines firstname, lastname, title, company, etc.)
df = prepare_profile_text_column(df, output_column="profile_text")

# Step 2: Enrich with LLM
enrich_column_with_llm(
    df=df,
    input_column="profile_text",
    prompt_template=PROMPT_PROFILE_QUALITY,
    output_column="llm_profile_quality",
    batch_size=10,
    extract_field="quality_score"  # Extract just the score (0-100)
)

# Check results
print(df[['firstname', 'lastname', 'llm_profile_quality']].head())
```

**Output:**
```
================================================================================
LLM Enrichment: llm_profile_quality
================================================================================
✓ Using existing column: llm_profile_quality
✓ OpenAI Enricher initialized
  Model: gpt-4o-mini
  Max cost limit: $10.00
✓ Processing 293 rows in 30 batches
  Batch size: 10
  Model: gpt-4o-mini
  Estimated cost: $0.0156
  Batch 1/30 (rows 0-9)... ✓
  Batch 2/30 (rows 10-19)... ✓
  ...

================================================================================
Enrichment Complete: llm_profile_quality
================================================================================
  Rows processed: 293/293
  Failed batches: 0/30
  API calls: 30
  Total tokens: 28,456
  Total cost: $0.0147
  Duration: 45.2s
================================================================================
```

---

## 📋 Predefined Prompts

The module includes 5 ready-to-use LinkedIn enrichment prompts:

### 1. Profile Quality Score

Scores profile completeness and professionalism (0-100).

```python
from linkedin_lead_scoring.data.llm_enrichment import PROMPT_PROFILE_QUALITY

enrich_column_with_llm(
    df=df,
    input_column="profile_text",
    prompt_template=PROMPT_PROFILE_QUALITY,
    output_column="profile_quality_score",
    extract_field="quality_score"
)
```

**Output**: Integer score 0-100

### 2. Industry Categorization

Categorizes profiles into industry categories.

```python
from linkedin_lead_scoring.data.llm_enrichment import PROMPT_INDUSTRY_CATEGORY

enrich_column_with_llm(
    df=df,
    input_column="profile_text",
    prompt_template=PROMPT_INDUSTRY_CATEGORY,
    output_column="llm_industry",
    extract_field="industry"
)
```

**Output**: String like "Technology - SaaS", "Finance - Banking"

### 3. Seniority Level Detection

Determines career seniority from job title and profile.

```python
from linkedin_lead_scoring.data.llm_enrichment import PROMPT_SENIORITY_LEVEL

enrich_column_with_llm(
    df=df,
    input_column="title",
    prompt_template=PROMPT_SENIORITY_LEVEL,
    output_column="llm_seniority",
    extract_field="seniority"
)
```

**Output**: "Entry", "Mid", "Senior", "Executive", "C-Level"

### 4. Engagement Likelihood

Predicts how likely the lead is to respond to outreach.

```python
from linkedin_lead_scoring.data.llm_enrichment import PROMPT_ENGAGEMENT_LIKELIHOOD

enrich_column_with_llm(
    df=df,
    input_column="profile_text",
    prompt_template=PROMPT_ENGAGEMENT_LIKELIHOOD,
    output_column="llm_engagement_score",
    extract_field="engagement_score"
)
```

**Output**: Float 0.0-1.0

### 5. Decision Maker Probability

Assesses if the person is a decision-maker for B2B sales.

```python
from linkedin_lead_scoring.data.llm_enrichment import PROMPT_DECISION_MAKER_PROBABILITY

enrich_column_with_llm(
    df=df,
    input_column="profile_text",
    prompt_template=PROMPT_DECISION_MAKER_PROBABILITY,
    output_column="llm_decision_maker_score",
    extract_field="decision_maker_score"
)
```

**Output**: Float 0.0-1.0

---

## 🎨 Custom Prompts

### Creating Your Own Prompts

```python
CUSTOM_PROMPT = """
Analyze these {batch_size} LinkedIn profiles for tech stack mentions.
Extract programming languages, frameworks, and tools.

{batch_data}

Return ONLY a JSON array with exactly {batch_size} objects:
[
  {{"index": 0, "tech_stack": ["Python", "AWS", "React"], "confidence": 0.9}},
  {{"index": 1, "tech_stack": ["Java", "Spring"], "confidence": 0.7}}
]
"""

enrich_column_with_llm(
    df=df,
    input_column="profile_text",
    prompt_template=CUSTOM_PROMPT,
    output_column="tech_stack",
    extract_field="tech_stack"  # Will store list as JSON string
)
```

**Key rules for custom prompts:**

1. **Include placeholders**: `{batch_size}` and `{batch_data}`
2. **Request JSON**: Always ask for JSON array output
3. **Include index field**: Each object must have `"index": N`
4. **Match batch size**: Return exactly `{batch_size}` items
5. **Be specific**: Clear instructions = better results

---

## 💰 Cost Management

### Estimating Costs

The enricher automatically estimates costs before processing:

```python
# Enrichment will show:
# ✓ Processing 293 rows in 30 batches
#   Batch size: 10
#   Model: gpt-4o-mini
#   Estimated cost: $0.0156  ← Cost estimate
```

If estimated cost exceeds `max_cost_usd`, enrichment stops with a warning.

### Cost Control Options

```python
enrich_column_with_llm(
    df=df,
    input_column="profile_text",
    prompt_template=PROMPT_PROFILE_QUALITY,
    output_column="quality_score",
    batch_size=20,  # Larger batches = fewer API calls = lower cost
    max_cost_usd=5.00,  # Stop if cost exceeds $5
    model="gpt-4o-mini"  # Cheapest model
)
```

### Pricing Reference (Jan 2025)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|----------------------|
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4o | $5.00 | $15.00 |
| gpt-4-turbo | $10.00 | $30.00 |

**Tip**: Start with `gpt-4o-mini` for experimentation. Upgrade to `gpt-4o` if quality is insufficient.

---

## 📊 MLflow Integration

All LLM enrichments are automatically logged to MLflow (if active run exists):

```python
import mlflow
from linkedin_lead_scoring.data.utils_data import setup_mlflow

# Start MLflow run
setup_mlflow(experiment_name="linkedin-lead-scoring")
mlflow.start_run(run_name="llm_feature_engineering")

# Enrich (auto-logs to MLflow)
enrich_column_with_llm(df, "profile_text", PROMPT_PROFILE_QUALITY, "quality_score")

# View logged metrics
mlflow.log_param("llm_enrichment_llm_profile_quality_model", "gpt-4o-mini")
mlflow.log_metric("llm_enrichment_llm_profile_quality_cost_usd", 0.0147)
mlflow.log_metric("llm_enrichment_llm_profile_quality_rows", 293)

mlflow.end_run()
```

**Tracked metrics per enrichment:**
- `llm_enrichment_{column}_cost_usd`
- `llm_enrichment_{column}_rows`
- `llm_enrichment_{column}_duration_s`
- `llm_enrichment_{column}_failed_batches`
- `llm_api_calls` (session total)
- `llm_total_tokens` (session total)
- `llm_total_cost_usd` (session total)

---

## 🔧 Advanced Usage

### Batch Size Optimization

```python
# Small batches (5-10): Faster per-batch, more API calls
# Good for: Testing, unstable prompts
enrich_column_with_llm(df, ..., batch_size=5)

# Medium batches (10-20): Balanced cost/speed
# Good for: Production enrichment
enrich_column_with_llm(df, ..., batch_size=15)

# Large batches (20-50): Fewer API calls, slower per-batch
# Good for: Cost optimization
enrich_column_with_llm(df, ..., batch_size=30)
```

### Retry Configuration

```python
enrich_column_with_llm(
    df=df,
    input_column="profile_text",
    prompt_template=PROMPT_PROFILE_QUALITY,
    output_column="quality_score",
    max_retries=5  # Retry failed batches up to 5 times
)
```

### Using Different Models

```python
# Fast & cheap (default)
enrich_column_with_llm(df, ..., model="gpt-4o-mini")

# Better quality, higher cost
enrich_column_with_llm(df, ..., model="gpt-4o")

# Legacy model
enrich_column_with_llm(df, ..., model="gpt-4-turbo")
```

---

## 🐛 Troubleshooting

### Error: "OpenAI API key not found"

**Solution**: Set `OPENAI_API_KEY` environment variable

```bash
# Add to .env file
echo "OPENAI_API_KEY=sk-proj-xxxxx" >> .env

# Or export temporarily
export OPENAI_API_KEY=sk-proj-xxxxx
```

### Error: "Expected list in response"

**Cause**: LLM didn't return JSON array

**Solutions**:
1. Make prompt more explicit: "Return ONLY a JSON array"
2. Lower temperature: `temperature=0.0` for deterministic output
3. Add example in prompt

### Warning: "Expected N results, got M"

**Cause**: LLM returned wrong number of items

**Impact**: Some rows won't be enriched (partial failure)

**Solutions**:
1. Simplify prompt
2. Reduce batch size
3. Check for edge cases in data (very long/short profiles)

### Cost Limit Exceeded

```
WARNING: Estimated cost ($12.45) exceeds limit ($10.00)
Increase max_cost_usd parameter to proceed.
```

**Solution**:
```python
enrich_column_with_llm(df, ..., max_cost_usd=15.00)
```

---

## 📈 Best Practices

### 1. Test on Small Samples First

```python
# Test on 50 rows before full dataset
df_sample = df.head(50)
enrich_column_with_llm(df_sample, ..., batch_size=10)

# Check quality of results
df_sample[['profile_text', 'llm_output']].head(20)

# If satisfied, run on full dataset
enrich_column_with_llm(df, ..., batch_size=20)
```

### 2. Start with gpt-4o-mini

```python
# Much cheaper for experimentation
# Upgrade to gpt-4o only if quality insufficient
enrich_column_with_llm(df, ..., model="gpt-4o-mini")
```

### 3. Combine Multiple Enrichments

```python
# Enrich with multiple prompts
prepare_profile_text_column(df)

enrich_column_with_llm(df, "profile_text", PROMPT_PROFILE_QUALITY, "quality")
enrich_column_with_llm(df, "profile_text", PROMPT_SENIORITY_LEVEL, "seniority")
enrich_column_with_llm(df, "profile_text", PROMPT_ENGAGEMENT_LIKELIHOOD, "engagement")

# Use all as features in model
X_features = df[['quality', 'seniority', 'engagement', 'companysize', ...]]
```

### 4. Save Enriched Data

```python
# Save after enrichment to avoid re-processing
df.to_csv("data/processed/linkedin_leads_enriched.csv", index=False)

# Or save to MLflow as artifact
mlflow.log_artifact("data/processed/linkedin_leads_enriched.csv")
```

---

## 🎓 Example: Complete Workflow

```python
import pandas as pd
import mlflow
from dotenv import load_dotenv

from linkedin_lead_scoring.data.utils_data import setup_mlflow
from linkedin_lead_scoring.data.llm_enrichment import (
    prepare_profile_text_column,
    enrich_column_with_llm,
    PROMPT_PROFILE_QUALITY,
    PROMPT_SENIORITY_LEVEL,
    PROMPT_ENGAGEMENT_LIKELIHOOD
)

# Load environment
load_dotenv()

# Setup MLflow
setup_mlflow(experiment_name="linkedin-lead-scoring")
mlflow.start_run(run_name="llm_enrichment_demo")

# Load data
df = pd.read_csv("data/linkedin_leads.csv")
mlflow.log_param("original_rows", len(df))

# Prepare profile text
df = prepare_profile_text_column(df)

# Enrich with 3 different signals
enrich_column_with_llm(
    df, "profile_text", PROMPT_PROFILE_QUALITY,
    "profile_quality", extract_field="quality_score"
)

enrich_column_with_llm(
    df, "title", PROMPT_SENIORITY_LEVEL,
    "seniority", extract_field="seniority"
)

enrich_column_with_llm(
    df, "profile_text", PROMPT_ENGAGEMENT_LIKELIHOOD,
    "engagement_score", extract_field="engagement_score"
)

# Save enriched data
output_path = "data/processed/leads_llm_enriched.csv"
df.to_csv(output_path, index=False)
mlflow.log_artifact(output_path)

# Summary
print(f"\n✅ Enrichment complete!")
print(f"   Rows: {len(df)}")
print(f"   New columns: profile_quality, seniority, engagement_score")
print(f"   Saved: {output_path}")

mlflow.end_run()
```

---

## 📚 Next Steps

1. **Experiment with prompts**: Try different prompt templates
2. **Tune batch sizes**: Find optimal balance for your use case
3. **Integrate with models**: Use LLM features in training
4. **Monitor costs**: Track spending in MLflow
5. **Validate quality**: Compare LLM predictions with ground truth

**Need help?** Check the source code in `src/linkedin_lead_scoring/data/llm_enrichment.py`
