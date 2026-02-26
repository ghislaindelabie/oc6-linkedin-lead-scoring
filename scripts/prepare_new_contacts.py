"""Process new LinkedIn contact CSVs through the full data preparation pipeline.

Replicates notebook 01 pipeline: load → standardize → label → LLM enrich → save.
Combines multiple LemList CSV exports into a single clean dataset.

Usage:
    python scripts/prepare_new_contacts.py

Requires OPENAI_API_KEY in .env file for LLM enrichment.
"""

import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from linkedin_lead_scoring.data.llm_enrichment import (
    enrich_column_with_llm,
    prepare_profile_text_column,
)
from linkedin_lead_scoring.data.utils_data import (
    create_engagement_labels,
    load_linkedin_csv,
    log_data_profile,
    select_modeling_features,
    setup_mlflow,
    standardize_columns,
)

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Configuration
INPUT_DIR = PROJECT_ROOT / "data" / "New data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "production"
OUTPUT_FILE = OUTPUT_DIR / "new_contacts_enriched.csv"

# LLM enrichment prompts (same as notebook 01)
# Import from llm_enrichment module if available, otherwise define inline
try:
    from linkedin_lead_scoring.data.llm_enrichment import (
        PROMPT_BUSINESS_TYPE,
        PROMPT_COMPANY_FIT,
        PROMPT_DECISION_MAKER_PROBABILITY,
        PROMPT_ENGAGEMENT_LIKELIHOOD,
        PROMPT_GEOGRAPHY,
        PROMPT_INDUSTRY_CATEGORY,
        PROMPT_PROFILE_QUALITY,
        PROMPT_SENIORITY_LEVEL,
    )
except ImportError:
    print("WARNING: Could not import all prompts from llm_enrichment module")
    print("Check that all PROMPT_* constants are defined in llm_enrichment.py")
    sys.exit(1)


def load_and_combine_csvs(input_dir: Path) -> pd.DataFrame:
    """Load all CSV files from directory and combine into one DataFrame."""
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["_source_file"] = f.name
        print(f"  Loaded {f.name}: {len(df)} rows x {len(df.columns)} cols")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(combined)} rows x {len(combined.columns)} cols")
    return combined


def run_llm_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    """Run all 8 LLM enrichment passes on the data."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Check your .env file.")

    # Prepare profile text for LLM
    print("\n  Preparing profile text...")
    df = prepare_profile_text_column(df, output_column="profile_text", include_extended_fields=True)

    enrichments = [
        ("llm_quality", PROMPT_PROFILE_QUALITY, "profile_text", "quality"),
        ("llm_seniority", PROMPT_SENIORITY_LEVEL, "profile_text", "seniority"),
        ("llm_engagement", PROMPT_ENGAGEMENT_LIKELIHOOD, "profile_text", "engagement"),
        ("llm_decision_maker", PROMPT_DECISION_MAKER_PROBABILITY, "profile_text", "decision_maker"),
        ("llm_industry", PROMPT_INDUSTRY_CATEGORY, "profile_text", "industry"),
        ("llm_geography", PROMPT_GEOGRAPHY, "profile_text", "geography"),
        ("llm_business_type", PROMPT_BUSINESS_TYPE, "profile_text", "business_type"),
        ("llm_company_fit", PROMPT_COMPANY_FIT, "profile_text", "company_fit"),
    ]

    for output_col, prompt, input_col, extract_field in enrichments:
        print(f"\n  Enriching: {output_col}...")
        enrich_column_with_llm(
            df=df,
            input_column=input_col,
            prompt_template=prompt,
            output_column=output_col,
            batch_size=10,
            model="gpt-4o-mini",
            max_retries=3,
            log_to_mlflow=True,
            api_key=api_key,
            max_cost_usd=5.0,
            extract_field=extract_field,
        )
        # Progress check
        non_null = df[output_col].notna().sum()
        print(f"    → {non_null}/{len(df)} rows enriched")

    return df


def run_pipeline(input_path: Path, output_path: Path, run_name: str, *, has_labels: bool = True):
    """Run the full data preparation pipeline on contact data.

    Args:
        input_path: Path to a single CSV or a directory of CSVs.
        output_path: Where to save the clean dataset.
        run_name: MLflow run name.
        has_labels: If True, create engagement labels from lastState/status.
                    If False, skip labeling (unlabelled production data).
    """
    print("=" * 60)
    print(f"LinkedIn Lead Scoring — Data Preparation ({run_name})")
    print("=" * 60)

    # Setup MLflow
    print("\n[1/7] Setting up MLflow tracking...")
    os.chdir(PROJECT_ROOT)
    setup_mlflow(experiment_name="linkedin-lead-scoring-data-prep")
    mlflow.start_run(run_name=run_name)
    mlflow.log_param("pipeline", "prepare_new_contacts.py")
    mlflow.log_param("has_labels", has_labels)

    # Load CSV(s)
    print("\n[2/7] Loading CSV files...")
    if input_path.is_dir():
        raw_df = load_and_combine_csvs(input_path)
        mlflow.log_metric("raw_n_files", len(list(input_path.glob("*.csv"))))
    else:
        raw_df = pd.read_csv(input_path)
        print(f"  Loaded {input_path.name}: {len(raw_df)} rows x {len(raw_df.columns)} cols")
        mlflow.log_metric("raw_n_files", 1)
    mlflow.log_metric("raw_n_rows", len(raw_df))

    # Standardize columns
    print("\n[3/7] Standardizing column names...")
    df = standardize_columns(raw_df)
    print(f"  Columns standardized: {len(df.columns)} columns")

    # Create engagement labels (only if data has outcome columns)
    if has_labels:
        print("\n[4/7] Creating engagement labels...")
        df = create_engagement_labels(df, log_to_mlflow=True)
        n_engaged = df["engaged"].sum() if "engaged" in df.columns else 0
        n_not = len(df) - n_engaged
        print(f"  Labeled: {len(df)} contacts ({n_engaged} engaged, {n_not} not engaged)")
    else:
        print("\n[4/7] Skipping labels (unlabelled production data)")
        mlflow.log_param("labeling", "skipped")

    # LLM Enrichment
    print("\n[5/7] Running LLM enrichment (this may take a few minutes)...")
    df = run_llm_enrichment(df)

    # Select modeling features (drop PII, IDs)
    print("\n[6/7] Selecting modeling features...")
    df_final, dropped = select_modeling_features(df, drop_pii=True, drop_ids=True, log_to_mlflow=True)
    print(f"  Kept {len(df_final.columns)} features, dropped {len(dropped)} columns")

    # Save
    print("\n[7/7] Saving clean dataset...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    mlflow.log_artifact(str(output_path), "processed_data")
    log_data_profile(df_final, stage="final")

    mlflow.end_run()

    print(f"\n{'=' * 60}")
    print(f"Done! Saved {len(df_final)} rows x {len(df_final.columns)} cols")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}")

    # Summary
    print("\nColumn summary:")
    llm_cols = [c for c in df_final.columns if c.startswith("llm_")]
    print(f"  LLM features: {llm_cols}")
    print(f"  Has 'engaged' label: {'engaged' in df_final.columns}")
    if "engaged" in df_final.columns:
        print(f"  Engagement rate: {df_final['engaged'].mean():.1%}")

    return df_final


def main():
    """Run pipeline on labelled data (3 files in data/New data/)."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare new LinkedIn contacts")
    parser.add_argument(
        "--unlabelled", type=str, default=None,
        help="Path to single CSV file with unlabelled contacts (no engagement outcomes)",
    )
    args = parser.parse_args()

    if args.unlabelled:
        # Unlabelled production data
        input_path = Path(args.unlabelled)
        output_path = OUTPUT_DIR / "unlabelled_contacts_enriched.csv"
        run_pipeline(input_path, output_path, "unlabelled_contacts", has_labels=False)
    else:
        # Labelled data from 3 files
        run_pipeline(INPUT_DIR, OUTPUT_FILE, "new_contacts_labelled", has_labels=True)


if __name__ == "__main__":
    main()
