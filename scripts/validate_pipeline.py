"""
Validate that local model predictions match the staging API.

Loads the same artifacts deployed on staging, runs predictions locally,
optionally calls the staging API, and compares scores.

Usage:
    python scripts/validate_pipeline.py [--staging-url URL] [--tolerance FLOAT] [--local-only]

Examples:
    # Local-only baseline (no API calls)
    python scripts/validate_pipeline.py --local-only

    # Full validation against staging
    python scripts/validate_pipeline.py --staging-url https://ghislaindelabie-oc6-bizdev-ml-api-staging.hf.space
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

# Ensure the package is importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from linkedin_lead_scoring.features import preprocess_for_inference

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
DEFAULT_STAGING_URL = "https://ghislaindelabie-oc6-bizdev-ml-api-staging.hf.space"
DEFAULT_TOLERANCE = 0.0001

# ---------------------------------------------------------------------------
# Sample leads for validation
# ---------------------------------------------------------------------------

SAMPLE_LEADS = [
    {
        "name": "full_lead",
        "data": {
            "llm_quality": 75,
            "llm_engagement": 0.8,
            "llm_decision_maker": 0.6,
            "llm_company_fit": 1,
            "companyfoundedon": 2015.0,
            "llm_seniority": "Senior",
            "llm_industry": "Technology - SaaS",
            "llm_geography": "international_hub",
            "llm_business_type": "leaders",
            "industry": "Information Technology & Services",
            "companyindustry": "Software Development",
            "companysize": "51-200",
            "companytype": "Privately Held",
            "languages": "English, French",
            "location": "Paris, Île-de-France, France",
            "companylocation": "Paris, France",
            "summary": "Experienced SaaS executive with 10+ years in B2B sales.",
            "skills": "Leadership, SaaS, B2B Sales, CRM",
            "jobtitle": "VP of Sales",
        },
    },
    {
        "name": "sparse_lead",
        "data": {
            "llm_quality": 50,
            "llm_seniority": "Mid",
            "jobtitle": "Software Engineer",
        },
    },
    {
        "name": "empty_lead",
        "data": {},
    },
    {
        "name": "none_text_fields",
        "data": {
            "llm_quality": 90,
            "llm_engagement": 0.9,
            "llm_decision_maker": 0.8,
            "llm_company_fit": 2,
            "companyfoundedon": 2020.0,
            "summary": None,
            "skills": None,
            "jobtitle": None,
        },
    },
    {
        "name": "low_quality_lead",
        "data": {
            "llm_quality": 10,
            "llm_engagement": 0.1,
            "llm_decision_maker": 0.1,
            "llm_company_fit": 0,
            "llm_seniority": "Entry",
            "llm_geography": "other",
            "llm_business_type": "workers",
            "companysize": "2-10",
        },
    },
    {
        "name": "executive_tech",
        "data": {
            "llm_quality": 95,
            "llm_engagement": 0.95,
            "llm_decision_maker": 0.9,
            "llm_company_fit": 2,
            "companyfoundedon": 2005.0,
            "llm_seniority": "Executive",
            "llm_industry": "Technology - Enterprise",
            "llm_geography": "international_hub",
            "llm_business_type": "leaders",
            "industry": "Computer Software",
            "companyindustry": "Software Development",
            "companysize": "501-1000",
            "companytype": "Public Company",
            "languages": "English",
            "location": "San Francisco, California, United States",
            "companylocation": "San Francisco, USA",
            "summary": "C-suite technology leader driving digital transformation.",
            "skills": "Strategy, Cloud, AI, Digital Transformation, Leadership",
            "jobtitle": "CTO",
        },
    },
]


# ---------------------------------------------------------------------------
# Local prediction
# ---------------------------------------------------------------------------


def load_artifacts(model_dir: Path) -> dict:
    """Load all model artifacts from disk."""
    model = joblib.load(model_dir / "xgboost_model.joblib")
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")

    with open(model_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)

    medians_path = model_dir / "numeric_medians.json"
    numeric_medians = None
    if medians_path.exists():
        with open(medians_path) as f:
            numeric_medians = json.load(f)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": feature_columns,
        "numeric_medians": numeric_medians,
    }


def predict_local(artifacts: dict, lead_data: dict) -> float:
    """Run a local prediction and return the engagement probability."""
    df = pd.DataFrame([lead_data])

    te = artifacts["preprocessor"].get("target_encoder")
    te_cols = artifacts["preprocessor"].get("te_cols", [])

    X = preprocess_for_inference(
        df,
        target_encoder=te,
        te_cols=te_cols,
        feature_columns=artifacts["feature_columns"],
        numeric_medians=artifacts["numeric_medians"],
    )

    proba = artifacts["model"].predict_proba(X)
    return float(proba[0, 1])


# ---------------------------------------------------------------------------
# API prediction
# ---------------------------------------------------------------------------


def predict_api(staging_url: str, lead_data: dict) -> float | None:
    """Call the staging /predict endpoint and return the score."""
    url = f"{staging_url.rstrip('/')}/predict"
    try:
        resp = requests.post(url, json=lead_data, timeout=30)
        resp.raise_for_status()
        return resp.json()["score"]
    except Exception as exc:
        print(f"    API error: {exc}")
        return None


def predict_batch_api(staging_url: str, leads: list[dict]) -> list[float | None]:
    """Call the staging /predict/batch endpoint."""
    url = f"{staging_url.rstrip('/')}/predict/batch"
    try:
        resp = requests.post(url, json={"leads": leads}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return [p["score"] for p in data["predictions"]]
    except Exception as exc:
        print(f"    Batch API error: {exc}")
        return [None] * len(leads)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_scores(
    results: list[dict],
    tolerance: float,
) -> bool:
    """Print comparison table and return True if all scores match within tolerance."""
    header = f"{'Lead':<20} {'Local':>8} {'API':>8} {'Delta':>8} {'Match':>6}"
    print(header)
    print("-" * len(header))

    all_match = True
    for r in results:
        local = r["local"]
        api = r.get("api")
        if api is not None:
            delta = abs(local - api)
            match = delta <= tolerance
            match_str = "OK" if match else "FAIL"
            if not match:
                all_match = False
            print(
                f"{r['name']:<20} {local:>8.4f} {api:>8.4f} {delta:>8.6f} {match_str:>6}"
            )
        else:
            print(f"{r['name']:<20} {local:>8.4f} {'N/A':>8} {'N/A':>8} {'SKIP':>6}")

    return all_match


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate local model predictions against staging API."
    )
    parser.add_argument(
        "--staging-url",
        type=str,
        default=DEFAULT_STAGING_URL,
        help=f"Staging API URL (default: {DEFAULT_STAGING_URL})",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Max acceptable score difference (default: {DEFAULT_TOLERANCE})",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Skip API calls, only run local predictions",
    )
    args = parser.parse_args()

    # Load artifacts
    print(f"Loading artifacts from {MODEL_DIR} ...")
    artifacts = load_artifacts(MODEL_DIR)
    te = artifacts["preprocessor"].get("target_encoder")
    print(f"  Model: {type(artifacts['model']).__name__}")
    print(f"  TargetEncoder: {'fitted' if te is not None else 'None'}")
    print(f"  Features: {len(artifacts['feature_columns'])} columns")
    print(f"  Numeric medians: {artifacts['numeric_medians']}")
    print()

    # Run local predictions
    print(f"Running local predictions on {len(SAMPLE_LEADS)} sample leads...")
    results = []
    for sample in SAMPLE_LEADS:
        score = predict_local(artifacts, sample["data"])
        results.append({"name": sample["name"], "local": score})
        print(f"  {sample['name']}: {score:.4f}")
    print()

    # Run API predictions (unless --local-only)
    if not args.local_only:
        print(f"Calling staging API at {args.staging_url} ...")

        # Individual /predict calls
        for i, sample in enumerate(SAMPLE_LEADS):
            api_score = predict_api(args.staging_url, sample["data"])
            results[i]["api"] = api_score
            status = f"{api_score:.4f}" if api_score is not None else "FAILED"
            print(f"  {sample['name']}: {status}")

        # Also test /predict/batch
        print("\nTesting /predict/batch ...")
        batch_leads = [s["data"] for s in SAMPLE_LEADS]
        batch_scores = predict_batch_api(args.staging_url, batch_leads)
        batch_match = True
        for i, (sample, batch_score) in enumerate(
            zip(SAMPLE_LEADS, batch_scores)
        ):
            individual_api = results[i].get("api")
            if batch_score is not None and individual_api is not None:
                delta = abs(batch_score - individual_api)
                ok = delta <= args.tolerance
                if not ok:
                    batch_match = False
                print(
                    f"  {sample['name']}: batch={batch_score:.4f} "
                    f"single={individual_api:.4f} delta={delta:.6f} "
                    f"{'OK' if ok else 'FAIL'}"
                )
            else:
                print(f"  {sample['name']}: batch={'N/A' if batch_score is None else f'{batch_score:.4f}'}")

        if batch_match:
            print("  Batch vs single: ALL MATCH")
        else:
            print("  Batch vs single: MISMATCH DETECTED")

        print()

    # Comparison table
    print("=" * 60)
    print("COMPARISON: Local vs API")
    print("=" * 60)
    all_match = compare_scores(results, args.tolerance)
    print()

    if args.local_only:
        print("Local-only mode — no API comparison performed.")
        print("Re-run without --local-only to compare against staging.")
        sys.exit(0)
    elif all_match:
        print(f"ALL SCORES MATCH (tolerance={args.tolerance})")
        sys.exit(0)
    else:
        print(f"MISMATCH DETECTED (tolerance={args.tolerance})")
        sys.exit(1)


if __name__ == "__main__":
    main()
