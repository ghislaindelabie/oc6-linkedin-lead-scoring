"""
LLM-based feature enrichment for LinkedIn lead scoring.

This module provides utilities for enriching LinkedIn profile data using OpenAI's API
with automatic batching, JSON-formatted outputs, and MLflow tracking.
"""

import os
import time
from typing import Optional, List, Dict, Any
import json

import pandas as pd
import mlflow
from openai import OpenAI
import tiktoken


# ============================================================================
# Predefined LinkedIn Enrichment Prompts
# ============================================================================

PROMPT_PROFILE_QUALITY = """
Analyze these {batch_size} LinkedIn profiles and score their completeness/quality (0-100).
Consider: profile completeness, professional experience relevance, activity level.

{batch_data}

Return ONLY a JSON array with exactly {batch_size} objects in this format:
[
  {{"index": 0, "quality_score": 85, "reasoning": "Complete profile with relevant experience"}},
  {{"index": 1, "quality_score": 60, "reasoning": "Minimal information provided"}}
]
"""

PROMPT_INDUSTRY_CATEGORY = """
Categorize these {batch_size} LinkedIn profiles into their primary industry category.

{batch_data}

Return ONLY a JSON array with exactly {batch_size} objects in this format:
[
  {{"index": 0, "industry": "Technology - SaaS", "confidence": 0.9}},
  {{"index": 1, "industry": "Finance - Banking", "confidence": 0.8}}
]
"""

PROMPT_SENIORITY_LEVEL = """
Determine the seniority level for these {batch_size} LinkedIn profiles.
Levels: Entry, Mid, Senior, Executive, C-Level.

{batch_data}

Return ONLY a JSON array with exactly {batch_size} objects in this format:
[
  {{"index": 0, "seniority": "Senior", "level_score": 7, "title_indicators": ["VP", "Director"]}},
  {{"index": 1, "seniority": "Mid", "level_score": 5, "title_indicators": ["Manager"]}}
]
"""

PROMPT_ENGAGEMENT_LIKELIHOOD = """
Predict engagement likelihood for these {batch_size} LinkedIn outreach targets.
Score from 0.0 (very unlikely) to 1.0 (very likely) based on profile signals.

{batch_data}

Return ONLY a JSON array with exactly {batch_size} objects in this format:
[
  {{"index": 0, "engagement_score": 0.75, "key_signals": ["recent activity", "open to opportunities"]}},
  {{"index": 1, "engagement_score": 0.45, "key_signals": ["inactive profile", "no clear signals"]}}
]
"""

PROMPT_DECISION_MAKER_PROBABILITY = """
Assess the probability that these {batch_size} profiles represent decision-makers for B2B sales.
Consider: job title, seniority, department, company role.

{batch_data}

Return ONLY a JSON array with exactly {batch_size} objects in this format:
[
  {{"index": 0, "decision_maker_score": 0.85, "reasoning": "C-level executive in target department"}},
  {{"index": 1, "decision_maker_score": 0.40, "reasoning": "Individual contributor role"}}
]
"""


# ============================================================================
# OpenAI Client Wrapper
# ============================================================================

class OpenAIEnricher:
    """
    Wrapper for OpenAI API with JSON-forced output and MLflow tracking.

    Uses GPT-4o-mini by default (fast, cost-effective GPT-4 class model).
    Automatically tracks API usage, costs, and errors to MLflow.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        log_to_mlflow: bool = True,
        max_cost_usd: float = 10.0
    ):
        """
        Initialize OpenAI enricher.

        Parameters:
        - api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        - model: Model to use (gpt-4o-mini, gpt-4o, gpt-4-turbo)
        - log_to_mlflow: Whether to track metrics in MLflow
        - max_cost_usd: Safety limit for total cost per session
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.log_to_mlflow = log_to_mlflow
        self.max_cost_usd = max_cost_usd

        # Pricing per 1M tokens (as of Jan 2025)
        self.pricing = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 5.00, "output": 15.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        }

        # Session tracking
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_calls = 0
        self.failed_calls = 0

        # Token counter for cost estimation
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        print(f"✓ OpenAI Enricher initialized")
        print(f"  Model: {model}")
        print(f"  Max cost limit: ${max_cost_usd:.2f}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))

    def estimate_cost(self, prompt: str, expected_output_tokens: int = 500) -> float:
        """
        Estimate cost for a single API call.

        Parameters:
        - prompt: Full prompt text
        - expected_output_tokens: Expected completion length

        Returns:
        - Estimated cost in USD
        """
        input_tokens = self.count_tokens(prompt)

        if self.model not in self.pricing:
            print(f"Warning: Pricing unknown for {self.model}, using gpt-4o-mini rates")
            pricing = self.pricing["gpt-4o-mini"]
        else:
            pricing = self.pricing[self.model]

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (expected_output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def call_with_json_output(
        self,
        prompt: str,
        max_retries: int = 3,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Call OpenAI API with JSON mode enabled.

        Parameters:
        - prompt: Prompt text
        - max_retries: Number of retries on failure
        - temperature: Sampling temperature (0.0-1.0)

        Returns:
        - Parsed JSON response

        Raises:
        - Exception if max_retries exceeded or cost limit reached
        """
        # Check cost limit
        if self.total_cost >= self.max_cost_usd:
            raise ValueError(
                f"Cost limit reached: ${self.total_cost:.4f} >= ${self.max_cost_usd:.2f}"
            )

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a data enrichment assistant. Always return valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=temperature
                )

                # Extract response
                content = response.choices[0].message.content
                result = json.loads(content)

                # Track usage
                usage = response.usage
                self.total_tokens += usage.total_tokens
                self.total_calls += 1

                # Calculate cost
                pricing = self.pricing.get(self.model, self.pricing["gpt-4o-mini"])
                cost = (
                    (usage.prompt_tokens / 1_000_000) * pricing["input"] +
                    (usage.completion_tokens / 1_000_000) * pricing["output"]
                )
                self.total_cost += cost

                # Log to MLflow
                if self.log_to_mlflow and mlflow.active_run():
                    mlflow.log_metric("llm_api_calls", self.total_calls)
                    mlflow.log_metric("llm_total_tokens", self.total_tokens)
                    mlflow.log_metric("llm_total_cost_usd", self.total_cost)

                return result

            except json.JSONDecodeError as e:
                print(f"  ⚠ JSON decode error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    self.failed_calls += 1
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

            except Exception as e:
                print(f"  ⚠ API error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    self.failed_calls += 1
                    raise
                time.sleep(2 ** attempt)

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of API usage for current session."""
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "model": self.model
        }


# ============================================================================
# Batch Enrichment Function
# ============================================================================

def enrich_column_with_llm(
    df: pd.DataFrame,
    input_column: str,
    prompt_template: str,
    output_column: str,
    batch_size: int = 10,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    log_to_mlflow: bool = True,
    api_key: Optional[str] = None,
    max_cost_usd: float = 10.0,
    extract_field: Optional[str] = None
) -> None:
    """
    Apply LLM enrichment to DataFrame column with batch processing.

    Modifies the DataFrame in-place by adding/updating the output_column.

    Parameters:
    - df: DataFrame to enrich (modified in-place)
    - input_column: Column name containing data to process
    - prompt_template: Prompt with {batch_size} and {batch_data} placeholders
    - output_column: Column name for results (created if doesn't exist)
    - batch_size: Number of rows per API call (default: 10)
    - model: OpenAI model to use
    - max_retries: Retry attempts for failed batches
    - log_to_mlflow: Track enrichment metrics in MLflow
    - api_key: OpenAI API key (optional, uses env var if None)
    - max_cost_usd: Maximum allowed cost for enrichment
    - extract_field: Field name to extract from JSON response (e.g., "quality_score")

    Returns:
    - None (modifies df in-place)

    Example:
        enrich_column_with_llm(
            df=leads_df,
            input_column="profile_text",
            prompt_template=PROMPT_PROFILE_QUALITY,
            output_column="llm_profile_quality",
            batch_size=10,
            extract_field="quality_score"
        )
    """
    print(f"\n{'='*70}")
    print(f"LLM Enrichment: {output_column}")
    print(f"{'='*70}")

    # Validate inputs
    if input_column not in df.columns:
        raise ValueError(f"Input column '{input_column}' not found in DataFrame")

    # Create output column if it doesn't exist
    if output_column not in df.columns:
        df[output_column] = None
        print(f"✓ Created new column: {output_column}")
    else:
        print(f"✓ Using existing column: {output_column}")

    # Initialize enricher
    enricher = OpenAIEnricher(
        api_key=api_key,
        model=model,
        log_to_mlflow=log_to_mlflow,
        max_cost_usd=max_cost_usd
    )

    # Calculate batches
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size

    print(f"✓ Processing {total_rows} rows in {num_batches} batches")
    print(f"  Batch size: {batch_size}")
    print(f"  Model: {model}")

    # Estimate cost
    sample_batch_data = "\n".join([
        f"{i}. {df[input_column].iloc[i] if i < len(df) else 'sample'}"
        for i in range(min(batch_size, len(df)))
    ])
    sample_prompt = prompt_template.format(
        batch_size=batch_size,
        batch_data=sample_batch_data
    )
    estimated_cost_per_batch = enricher.estimate_cost(sample_prompt, expected_output_tokens=300)
    total_estimated_cost = estimated_cost_per_batch * num_batches

    print(f"  Estimated cost: ${total_estimated_cost:.4f}")

    if total_estimated_cost > max_cost_usd:
        print(f"\n⚠ WARNING: Estimated cost (${total_estimated_cost:.4f}) exceeds limit (${max_cost_usd:.2f})")
        print("  Increase max_cost_usd parameter to proceed.")
        return

    # Process batches
    failed_batches = []
    start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_rows)
        actual_batch_size = batch_end - batch_start

        # Prepare batch data
        batch_rows = df.iloc[batch_start:batch_end]
        batch_data_items = []

        for idx, (row_idx, row) in enumerate(batch_rows.iterrows()):
            value = row[input_column]
            # Handle NaN values
            if pd.isna(value):
                value = "[No data]"
            batch_data_items.append(f"{idx}. {value}")

        batch_data_str = "\n".join(batch_data_items)

        # Format prompt
        prompt = prompt_template.format(
            batch_size=actual_batch_size,
            batch_data=batch_data_str
        )

        # Call LLM
        try:
            print(f"  Batch {batch_idx+1}/{num_batches} (rows {batch_start}-{batch_end-1})...", end=" ")

            response = enricher.call_with_json_output(prompt, max_retries=max_retries)

            # Parse response - expect a "results" or direct array
            if "results" in response:
                results = response["results"]
            elif isinstance(response, list):
                results = response
            else:
                # Response is likely the root object, extract array values
                results = list(response.values())[0] if response else []

            if not isinstance(results, list):
                raise ValueError(f"Expected list in response, got: {type(results)}")

            if len(results) != actual_batch_size:
                print(f"⚠ Expected {actual_batch_size} results, got {len(results)}")

            # Update DataFrame
            for result_item in results:
                if not isinstance(result_item, dict):
                    continue

                result_idx = result_item.get("index")
                if result_idx is None or result_idx >= actual_batch_size:
                    continue

                global_idx = batch_start + result_idx

                # Extract the requested field or store full result
                if extract_field and extract_field in result_item:
                    df.at[df.index[global_idx], output_column] = result_item[extract_field]
                else:
                    # Store full result as JSON string
                    df.at[df.index[global_idx], output_column] = json.dumps(result_item)

            print("✓")

        except Exception as e:
            print(f"✗ Error: {e}")
            failed_batches.append(batch_idx)

    # Summary
    duration = time.time() - start_time
    session_summary = enricher.get_session_summary()

    print(f"\n{'='*70}")
    print(f"Enrichment Complete: {output_column}")
    print(f"{'='*70}")
    print(f"  Rows processed: {total_rows - (len(failed_batches) * batch_size)}/{total_rows}")
    print(f"  Failed batches: {len(failed_batches)}/{num_batches}")
    print(f"  API calls: {session_summary['total_calls']}")
    print(f"  Total tokens: {session_summary['total_tokens']:,}")
    print(f"  Total cost: ${session_summary['total_cost_usd']:.4f}")
    print(f"  Duration: {duration:.1f}s")
    print(f"{'='*70}\n")

    # Log to MLflow
    if log_to_mlflow and mlflow.active_run():
        mlflow.log_param(f"llm_enrichment_{output_column}_model", model)
        mlflow.log_param(f"llm_enrichment_{output_column}_batch_size", batch_size)
        mlflow.log_metric(f"llm_enrichment_{output_column}_rows", total_rows)
        mlflow.log_metric(f"llm_enrichment_{output_column}_cost_usd", session_summary['total_cost_usd'])
        mlflow.log_metric(f"llm_enrichment_{output_column}_duration_s", duration)
        mlflow.log_metric(f"llm_enrichment_{output_column}_failed_batches", len(failed_batches))


# ============================================================================
# Helper Functions
# ============================================================================

def prepare_profile_text_column(
    df: pd.DataFrame,
    output_column: str = "profile_text",
    include_extended_fields: bool = False
) -> pd.DataFrame:
    """
    Combine LinkedIn profile fields into a single text column for LLM processing.

    Supports both snake_case and camelCase field names (LemList uses camelCase).

    Parameters:
    - df: DataFrame with LinkedIn profile data
    - output_column: Name for combined text column
    - include_extended_fields: If True, includes summary, skills, icebreaker, etc.

    Returns:
    - DataFrame with new profile_text column

    Example:
        df = prepare_profile_text_column(df)
        # Creates: "Name: John Doe\nTitle: CEO\nCompany: Acme Inc\n..."
    """
    df = df.copy()

    def get_field(options):
        """Try multiple field name variations, return first that exists."""
        for opt in options:
            if opt in df.columns:
                return df[opt].fillna("")
        return pd.Series("", index=df.index)

    # Build profile text row by row
    profile_parts = []
    fields_used = []

    # === Core Fields (Priority 1) ===

    # Name (try: firstName/lastName, firstname/lastname)
    first_name = get_field(['firstName', 'firstname'])
    last_name = get_field(['lastName', 'lastname'])
    if not first_name.str.strip().eq("").all() or not last_name.str.strip().eq("").all():
        profile_parts.append("Name: " + first_name + " " + last_name)
        fields_used.append("name")

    # Job Title (try: jobTitle, title)
    job_title = get_field(['jobTitle', 'title', 'position'])
    if not job_title.str.strip().eq("").all():
        profile_parts.append("Title: " + job_title)
        fields_used.append("jobTitle")

    # Company Name (try: companyName, companyname)
    company_name = get_field(['companyName', 'companyname'])
    if not company_name.str.strip().eq("").all():
        profile_parts.append("Company: " + company_name)
        fields_used.append("companyName")

    # Industry (try: companyIndustry, industry)
    company_industry = get_field(['companyIndustry', 'companyindustry', 'industry'])
    if not company_industry.str.strip().eq("").all():
        profile_parts.append("Industry: " + company_industry)
        fields_used.append("industry")

    # Company Size (try: companySize, companysize)
    company_size = get_field(['companySize', 'companysize'])
    if not company_size.str.strip().eq("").all():
        profile_parts.append("Company Size: " + company_size.astype(str))
        fields_used.append("companySize")

    # Location (consistent naming)
    location = get_field(['location'])
    if not location.str.strip().eq("").all():
        profile_parts.append("Location: " + location)
        fields_used.append("location")

    # === Extended Fields (Priority 2) - Rich context ===

    if include_extended_fields:
        # Tagline (LinkedIn headline)
        tagline = get_field(['tagline'])
        if not tagline.str.strip().eq("").all():
            profile_parts.append("Headline: " + tagline)
            fields_used.append("tagline")

        # Summary (About section)
        summary = get_field(['summary'])
        if not summary.str.strip().eq("").all():
            # Truncate long summaries for cost efficiency
            truncated_summary = summary.str[:500]
            profile_parts.append("Summary: " + truncated_summary)
            fields_used.append("summary")

        # Skills
        skills = get_field(['skills'])
        if not skills.str.strip().eq("").all():
            profile_parts.append("Skills: " + skills)
            fields_used.append("skills")

        # Icebreaker (personalized note from LemList)
        icebreaker = get_field(['icebreaker'])
        if not icebreaker.str.strip().eq("").all():
            profile_parts.append("Note: " + icebreaker)
            fields_used.append("icebreaker")

        # Company Description
        company_desc = get_field(['companyDescription'])
        if not company_desc.str.strip().eq("").all():
            truncated_desc = company_desc.str[:300]
            profile_parts.append("Company Info: " + truncated_desc)
            fields_used.append("companyDescription")

        # Company Specialties
        company_spec = get_field(['companySpecialties'])
        if not company_spec.str.strip().eq("").all():
            profile_parts.append("Company Focus: " + company_spec)
            fields_used.append("companySpecialties")

    # Combine all parts with newlines for each row
    if profile_parts:
        # Use pandas string concatenation with separator
        df[output_column] = profile_parts[0]
        for part in profile_parts[1:]:
            df[output_column] = df[output_column] + "\n" + part
    else:
        df[output_column] = "No profile data"

    print(f"✓ Created {output_column} column from {len(fields_used)} fields")
    print(f"  Fields included: {', '.join(fields_used[:10])}")
    if len(fields_used) > 10:
        print(f"  ... and {len(fields_used) - 10} more")

    return df
