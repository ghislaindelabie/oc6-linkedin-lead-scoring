"""Generate synthetic drift scenario datasets for drift detection demonstration.

Creates 5 CSV files in data/drift_scenarios/, each with 50 rows in the
processed 47-feature format (matching training_reference.csv schema).
These datasets deliberately shift specific feature distributions to
trigger different types of drift detection.

Usage:
    python scripts/generate_drift_scenarios.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = PROJECT_ROOT / "data" / "reference" / "training_reference.csv"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "model" / "feature_columns.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "drift_scenarios"

N_SAMPLES = 50
SEED = 42


def load_reference():
    """Load reference data and feature columns."""
    ref = pd.read_csv(REFERENCE_PATH)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_columns = json.load(f)
    return ref, feature_columns


def make_base_dataframe(feature_columns: list[str], n: int = N_SAMPLES) -> pd.DataFrame:
    """Create a zero-filled dataframe with correct columns."""
    return pd.DataFrame(0.0, index=range(n), columns=feature_columns)


def generate_no_drift_baseline(ref: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Control dataset: resample from training distribution with small noise.

    This should NOT trigger drift detection — validates that the detector
    doesn't produce false positives.
    """
    # Bootstrap sample from reference + small gaussian noise on continuous cols
    df = ref.sample(n=N_SAMPLES, replace=True, random_state=SEED).reset_index(drop=True)

    # Add small noise to continuous features (not binary OHE columns)
    continuous_cols = [
        "llm_quality", "llm_engagement", "llm_decision_maker",
        "llm_company_fit", "companyfoundedon",
        "summary_length", "skills_count", "jobtitle_length",
    ]
    for col in continuous_cols:
        if col in df.columns:
            noise = rng.normal(0, df[col].std() * 0.05, size=len(df))
            df[col] = df[col] + noise

    # Clip to valid ranges
    df["llm_quality"] = df["llm_quality"].clip(0, 100).round().astype(int)
    df["llm_engagement"] = df["llm_engagement"].clip(0, 1)
    df["llm_decision_maker"] = df["llm_decision_maker"].clip(0, 1)
    df["llm_company_fit"] = df["llm_company_fit"].clip(0, 2).round().astype(int)
    df["summary_length"] = df["summary_length"].clip(0).round().astype(int)
    df["skills_count"] = df["skills_count"].clip(0).round().astype(int)
    df["jobtitle_length"] = df["jobtitle_length"].clip(0).round().astype(int)

    return df


def generate_sector_shift(feature_columns: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Covariate shift: profiles from healthcare, agriculture, public sector.

    Industries completely absent from training data. Shifts:
    - Target-encoded industry columns get extreme values (outside training range)
    - Company size shifts toward large enterprises
    - Company type shifts toward Government/Nonprofit
    - LLM features stay in normal range (these are profile-level, not sector-level)
    """
    df = make_base_dataframe(feature_columns)

    # Normal profile-level features (not sector-dependent)
    df["llm_quality"] = rng.integers(50, 85, size=N_SAMPLES)
    df["llm_engagement"] = rng.uniform(0.3, 0.7, size=N_SAMPLES).round(2)
    df["llm_decision_maker"] = rng.uniform(0.3, 0.7, size=N_SAMPLES).round(2)
    df["llm_company_fit"] = rng.choice([0, 0, 0, 1], size=N_SAMPLES)  # Low fit (new sectors)

    # Target-encoded columns: push OUTSIDE training range to simulate unknown sectors
    # Training range for llm_industry: ~[0.275, 0.500], industry: ~[0.343, 0.590]
    df["llm_industry"] = rng.uniform(0.15, 0.28, size=N_SAMPLES).round(6)  # Below training min
    df["industry"] = rng.uniform(0.60, 0.75, size=N_SAMPLES).round(6)  # Above training max
    df["companyindustry"] = rng.uniform(0.60, 0.75, size=N_SAMPLES).round(6)

    # Location stays somewhat normal
    df["languages"] = rng.uniform(0.38, 0.43, size=N_SAMPLES).round(6)
    df["location"] = rng.uniform(0.33, 0.47, size=N_SAMPLES).round(6)
    df["companylocation"] = rng.uniform(0.28, 0.53, size=N_SAMPLES).round(6)

    # Company founded: older, established institutions
    df["companyfoundedon"] = rng.uniform(1950, 1990, size=N_SAMPLES).round(0)

    # Text features: normal profiles
    df["has_summary"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.3, 0.7])
    df["has_skills"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.1, 0.9])
    df["has_jobtitle"] = 1
    df["summary_length"] = rng.integers(100, 1500, size=N_SAMPLES)
    df["skills_count"] = rng.integers(1, 5, size=N_SAMPLES)
    df["jobtitle_length"] = rng.integers(10, 40, size=N_SAMPLES)

    # Role indicators: more managers, fewer founders/tech
    df["is_manager"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.7, 0.3])
    df["is_director"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.8, 0.2])

    # Seniority: mostly Mid
    df["llm_seniority_Mid"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.4, 0.6])
    df["llm_seniority_Senior"] = 0

    # Geography: mostly other (non-tech-hubs)
    df["llm_geography_other"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.2, 0.8])

    # Business type: workers and others (not leaders)
    df["llm_business_type_workers"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.5, 0.5])
    df["llm_business_type_others"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.4, 0.6])

    # Company size: shift toward large enterprises (5001-10000, 1001-5000 absent from OHE)
    df["companysize_5001-10000"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.4, 0.6])
    df["companysize_201-500"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.7, 0.3])

    # Company type: Government and Nonprofit heavy (very different from training)
    df["companytype_Government Agency"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.4, 0.6])
    df["companytype_Nonprofit"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.6, 0.4])
    df["companytype_Privately Held"] = 0  # Almost none
    df["companytype_UNKNOWN"] = 0

    return df


def generate_seniority_shift(feature_columns: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Feature drift: all senior executives and C-level profiles.

    Training data is mostly Mid-level (40%) with some Senior (18%).
    This dataset is almost entirely Executive/Senior with high decision-maker scores.
    """
    df = make_base_dataframe(feature_columns)

    # High-quality executive profiles
    df["llm_quality"] = rng.integers(80, 100, size=N_SAMPLES)  # Training mean: 60
    df["llm_engagement"] = rng.uniform(0.7, 0.95, size=N_SAMPLES).round(2)  # Training mean: 0.54
    df["llm_decision_maker"] = rng.uniform(0.8, 1.0, size=N_SAMPLES).round(2)  # Training mean: 0.48
    df["llm_company_fit"] = rng.choice([1, 2, 2], size=N_SAMPLES)  # High fit

    # Target-encoded columns: within training range
    df["llm_industry"] = rng.uniform(0.35, 0.50, size=N_SAMPLES).round(6)
    df["industry"] = rng.uniform(0.40, 0.55, size=N_SAMPLES).round(6)
    df["companyindustry"] = rng.uniform(0.40, 0.55, size=N_SAMPLES).round(6)
    df["languages"] = rng.uniform(0.39, 0.43, size=N_SAMPLES).round(6)
    df["location"] = rng.uniform(0.33, 0.47, size=N_SAMPLES).round(6)
    df["companylocation"] = rng.uniform(0.28, 0.53, size=N_SAMPLES).round(6)

    df["companyfoundedon"] = rng.uniform(2000, 2020, size=N_SAMPLES).round(0)

    # Rich profiles: long summaries, many skills
    df["has_summary"] = 1  # All have summaries (training: 68%)
    df["has_skills"] = 1
    df["has_jobtitle"] = 1
    df["summary_length"] = rng.integers(800, 2500, size=N_SAMPLES)  # Training mean: 604
    df["skills_count"] = rng.integers(4, 8, size=N_SAMPLES)  # Training mean: 2.7
    df["jobtitle_length"] = rng.integers(15, 60, size=N_SAMPLES)

    # Role indicators: directors and founders, not managers or tech
    df["is_founder"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.5, 0.5])  # Training: 9%
    df["is_director"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.3, 0.7])  # Training: 5%
    df["is_tech_role"] = 0  # No tech roles

    # Seniority: almost all Executive or Senior (training: 2% Exec, 18% Senior)
    df["llm_seniority_Executive"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.3, 0.7])
    df["llm_seniority_Senior"] = 1 - df["llm_seniority_Executive"]
    df["llm_seniority_Mid"] = 0  # None mid-level

    # Geography: international hubs (executives at HQ)
    df["llm_geography_other"] = 0
    df["llm_geography_regional_hub"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.8, 0.2])

    # Business type: leaders (training: 23%)
    df["llm_business_type_leaders"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.1, 0.9])
    df["llm_business_type_others"] = 0
    df["llm_business_type_workers"] = 0

    # Company size: mid-size companies
    df["companysize_201-500"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.5, 0.5])
    df["companysize_51-200"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.6, 0.4])

    # Company type: mostly private
    df["companytype_Privately Held"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.3, 0.7])
    df["companytype_Public Company"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.7, 0.3])

    return df


def generate_geography_shift(feature_columns: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Geographic drift: profiles from Asia, Africa, South America.

    Training data is EU/US-heavy (international_hub dominant).
    This dataset shifts to regions outside the original geographic scope.
    """
    df = make_base_dataframe(feature_columns)

    # Profile quality: lower on average (different markets, different LinkedIn usage)
    df["llm_quality"] = rng.integers(30, 70, size=N_SAMPLES)  # Training mean: 60
    df["llm_engagement"] = rng.uniform(0.15, 0.50, size=N_SAMPLES).round(2)  # Lower engagement
    df["llm_decision_maker"] = rng.uniform(0.2, 0.6, size=N_SAMPLES).round(2)
    df["llm_company_fit"] = rng.choice([0, 0, 1], size=N_SAMPLES)

    # Target-encoded columns: location/companylocation are KEY drift signals
    df["llm_industry"] = rng.uniform(0.30, 0.45, size=N_SAMPLES).round(6)
    df["industry"] = rng.uniform(0.35, 0.50, size=N_SAMPLES).round(6)
    df["companyindustry"] = rng.uniform(0.35, 0.50, size=N_SAMPLES).round(6)
    # Different language mix (not French/English dominant)
    df["languages"] = rng.uniform(0.30, 0.37, size=N_SAMPLES).round(6)  # Below training range
    # Location encoding: far from training distribution
    df["location"] = rng.uniform(0.20, 0.30, size=N_SAMPLES).round(6)  # Below training min (0.333)
    df["companylocation"] = rng.uniform(0.15, 0.25, size=N_SAMPLES).round(6)  # Below training min

    df["companyfoundedon"] = rng.uniform(2005, 2023, size=N_SAMPLES).round(0)

    # Text features: shorter profiles (different LinkedIn culture)
    df["has_summary"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.5, 0.5])  # Less complete
    df["has_skills"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.3, 0.7])  # Fewer skills listed
    df["has_jobtitle"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.1, 0.9])
    df["summary_length"] = rng.integers(0, 400, size=N_SAMPLES)  # Shorter
    df["skills_count"] = rng.integers(0, 3, size=N_SAMPLES)  # Fewer
    df["jobtitle_length"] = rng.integers(5, 30, size=N_SAMPLES)

    # Seniority: more entry-level and mid
    df["llm_seniority_Mid"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.3, 0.7])
    df["llm_seniority_Senior"] = 0
    df["llm_seniority_Executive"] = 0

    # Geography: ALMOST ALL "other" (training: 26%)
    df["llm_geography_other"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.05, 0.95])
    df["llm_geography_regional_hub"] = 0

    # Business type: workers and experts
    df["llm_business_type_workers"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.4, 0.6])
    df["llm_business_type_others"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.5, 0.5])
    df["llm_business_type_leaders"] = 0

    # Company size: smaller companies
    df["companysize_2-10"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.4, 0.6])
    df["companysize_11-50"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.5, 0.5])

    # Company type: more self-employed and unknown
    df["companytype_Self-Employed"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.5, 0.5])
    df["companytype_UNKNOWN"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.4, 0.6])

    return df


def generate_quality_degradation(feature_columns: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Data quality drift: sparse profiles with many missing/default values.

    Simulates a scenario where the data source quality degrades — contacts
    with incomplete LinkedIn profiles, missing company info, no summaries.
    """
    df = make_base_dataframe(feature_columns)

    # Very low quality profiles
    df["llm_quality"] = rng.integers(10, 40, size=N_SAMPLES)  # Training mean: 60
    df["llm_engagement"] = rng.uniform(0.05, 0.30, size=N_SAMPLES).round(2)  # Very low
    df["llm_decision_maker"] = rng.uniform(0.05, 0.25, size=N_SAMPLES).round(2)  # Very low
    df["llm_company_fit"] = 0  # No fit

    # Target-encoded columns: mostly default/median values (UNKNOWN encoding)
    df["llm_industry"] = 0.40  # Near median — UNKNOWN gets average
    df["industry"] = 0.40
    df["companyindustry"] = 0.40
    df["languages"] = 0.40
    df["location"] = 0.33  # Minimum — default location encoding
    df["companylocation"] = 0.28  # Below min — unknown location

    # Company founded: missing → replaced by median in preprocessing
    df["companyfoundedon"] = 2016.5  # All at median (imputed value)

    # Almost no text content
    df["has_summary"] = 0  # Training: 68% have summary
    df["has_skills"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.7, 0.3])  # Training: 90%
    df["has_jobtitle"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.4, 0.6])  # Training: 84%
    df["summary_length"] = 0
    df["skills_count"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.6, 0.4])
    df["jobtitle_length"] = rng.integers(0, 15, size=N_SAMPLES)

    # No role indicators (nothing to detect from sparse profiles)
    # All stay at 0

    # Seniority: mostly unknown (mapped to baseline = Entry, no OHE column)
    # All OHE columns stay at 0 → represents Entry level

    # Geography: other (unknown)
    df["llm_geography_other"] = 1
    df["llm_geography_regional_hub"] = 0

    # Business type: workers or others
    df["llm_business_type_workers"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.5, 0.5])
    df["llm_business_type_others"] = 1 - df["llm_business_type_workers"]

    # Company size: mostly UNKNOWN
    df["companysize_UNKNOWN"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.1, 0.9])

    # Company type: mostly UNKNOWN
    df["companytype_UNKNOWN"] = rng.choice([0, 1], size=N_SAMPLES, p=[0.1, 0.9])

    return df


def main():
    """Generate all drift scenario datasets."""
    rng = np.random.default_rng(SEED)
    ref, feature_columns = load_reference()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "no_drift_baseline": generate_no_drift_baseline(ref, rng),
        "drift_sector_shift": generate_sector_shift(feature_columns, rng),
        "drift_seniority_shift": generate_seniority_shift(feature_columns, rng),
        "drift_geography_shift": generate_geography_shift(feature_columns, rng),
        "drift_quality_degradation": generate_quality_degradation(feature_columns, rng),
    }

    for name, df in scenarios.items():
        # Ensure column order matches feature_columns exactly
        df = df[feature_columns]
        output_path = OUTPUT_DIR / f"{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"Generated {output_path.name}: {len(df)} rows x {len(df.columns)} cols")

        # Quick sanity check
        assert list(df.columns) == feature_columns, f"Column mismatch in {name}"
        assert len(df) == N_SAMPLES, f"Row count mismatch in {name}"

    print(f"\nAll scenarios saved to {OUTPUT_DIR}/")
    print("Use these with DriftDetector to demonstrate drift detection.")


if __name__ == "__main__":
    main()
