"""Unit tests for linkedin_lead_scoring.features — shared feature engineering module."""
import numpy as np
import pandas as pd
import pytest

from linkedin_lead_scoring.features import (
    NUMERIC_COLS,
    TEXT_COLS,
    align_columns,
    extract_text_features,
    fill_missing_values,
    one_hot_encode,
    preprocess_for_inference,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_lead_df():
    """Single-row DataFrame with all raw fields populated."""
    return pd.DataFrame([{
        "llm_quality": 75,
        "llm_seniority": "Senior",
        "llm_engagement": 0.8,
        "llm_decision_maker": 0.6,
        "llm_industry": "Technology - SaaS",
        "llm_geography": "international_hub",
        "llm_business_type": "leaders",
        "llm_company_fit": 1,
        "industry": "IT Services",
        "companyindustry": "Software Development",
        "companysize": "51-200",
        "companytype": "Privately Held",
        "languages": "English, French",
        "location": "Paris, France",
        "companylocation": "Paris, FR",
        "companyfoundedon": 2015.0,
        "summary": "Experienced SaaS executive with 10+ years in B2B sales.",
        "skills": "Leadership, SaaS, B2B Sales, CRM",
        "jobtitle": "VP of Sales",
    }])


@pytest.fixture
def multi_lead_df():
    """Multi-row DataFrame with varied data, including nulls."""
    return pd.DataFrame([
        {
            "llm_quality": 75, "llm_seniority": "Senior", "llm_engagement": 0.8,
            "llm_decision_maker": 0.6, "llm_industry": "Technology - SaaS",
            "llm_geography": "international_hub", "llm_business_type": "leaders",
            "llm_company_fit": 1, "industry": "IT Services",
            "companyindustry": "Software Development", "companysize": "51-200",
            "companytype": "Privately Held", "languages": "English, French",
            "location": "Paris, France", "companylocation": "Paris, FR",
            "companyfoundedon": 2015.0,
            "summary": "Experienced SaaS executive.", "skills": "Python, ML, Leadership",
            "jobtitle": "CEO & Founder",
        },
        {
            "llm_quality": 30, "llm_seniority": "Entry", "llm_engagement": 0.2,
            "llm_decision_maker": 0.1, "llm_industry": "Unknown",
            "llm_geography": "local", "llm_business_type": "others",
            "llm_company_fit": 0, "industry": "Consulting",
            "companyindustry": None, "companysize": None,
            "companytype": None, "languages": "French",
            "location": "Lyon, France", "companylocation": None,
            "companyfoundedon": None,
            "summary": None, "skills": None, "jobtitle": None,
        },
        {
            "llm_quality": 60, "llm_seniority": "Mid", "llm_engagement": 0.5,
            "llm_decision_maker": 0.4, "llm_industry": "Finance",
            "llm_geography": "regional_hub", "llm_business_type": "experts",
            "llm_company_fit": 2, "industry": "Finance",
            "companyindustry": "Financial Services", "companysize": "201-500",
            "companytype": "Public Company", "languages": "English",
            "location": "London, UK", "companylocation": "London, UK",
            "companyfoundedon": 2010.0,
            "summary": "Marketing leader with 5 years experience in growth.",
            "skills": "Marketing, Growth, SEO, Analytics, Content",
            "jobtitle": "Marketing Manager",
        },
    ])


@pytest.fixture
def sample_medians():
    """Sample numeric medians dict (mimics model/numeric_medians.json)."""
    return {
        "llm_quality": 50.0,
        "llm_engagement": 0.5,
        "llm_decision_maker": 0.4,
        "llm_company_fit": 1.0,
        "companyfoundedon": 2016.0,
    }


@pytest.fixture
def sample_feature_columns():
    """Feature columns list matching model/feature_columns.json structure."""
    return [
        "llm_quality", "llm_engagement", "llm_decision_maker",
        "llm_industry", "llm_company_fit", "industry", "companyindustry",
        "languages", "location", "companylocation", "companyfoundedon",
        "has_summary", "has_skills", "has_jobtitle",
        "summary_length", "skills_count", "jobtitle_length",
        "is_founder", "is_director", "is_manager", "is_sales",
        "is_marketing", "is_tech_role",
        "llm_seniority_Executive", "llm_seniority_Mid", "llm_seniority_Senior",
        "llm_geography_other", "llm_geography_regional_hub",
        "llm_business_type_leaders", "llm_business_type_others",
        "llm_business_type_salespeople", "llm_business_type_workers",
        "companysize_11-50", "companysize_2-10", "companysize_201-500",
        "companysize_5001-10000", "companysize_501-1000", "companysize_51-200",
        "companysize_UNKNOWN",
        "companytype_Government Agency", "companytype_Nonprofit",
        "companytype_Partnership", "companytype_Privately Held",
        "companytype_Public Company", "companytype_Self-Employed",
        "companytype_Sole Proprietorship", "companytype_UNKNOWN",
    ]


# ---------------------------------------------------------------------------
# TestExtractTextFeatures
# ---------------------------------------------------------------------------

class TestExtractTextFeatures:
    def test_completeness_flags_present(self, single_lead_df):
        result = extract_text_features(single_lead_df.copy())
        assert "has_summary" in result.columns
        assert "has_skills" in result.columns
        assert "has_jobtitle" in result.columns

    def test_completeness_flags_correct_values(self, single_lead_df):
        result = extract_text_features(single_lead_df.copy())
        assert result["has_summary"].iloc[0] == 1
        assert result["has_skills"].iloc[0] == 1
        assert result["has_jobtitle"].iloc[0] == 1

    def test_length_features_present(self, single_lead_df):
        result = extract_text_features(single_lead_df.copy())
        assert "summary_length" in result.columns
        assert "skills_count" in result.columns
        assert "jobtitle_length" in result.columns

    def test_skills_count_correct(self, single_lead_df):
        result = extract_text_features(single_lead_df.copy())
        # "Leadership, SaaS, B2B Sales, CRM" → 4 skills
        assert result["skills_count"].iloc[0] == 4

    def test_role_detection_flags(self, single_lead_df):
        result = extract_text_features(single_lead_df.copy())
        # "VP of Sales" → is_director=1, is_sales=1
        assert result["is_director"].iloc[0] == 1
        assert result["is_sales"].iloc[0] == 1

    def test_drops_text_columns(self, single_lead_df):
        result = extract_text_features(single_lead_df.copy())
        for col in TEXT_COLS:
            assert col not in result.columns

    def test_handles_none_values(self):
        df = pd.DataFrame([{
            "summary": None, "skills": None, "jobtitle": None,
            "llm_quality": 50,
        }])
        result = extract_text_features(df)
        assert result["has_summary"].iloc[0] == 0
        assert result["has_skills"].iloc[0] == 0
        assert result["has_jobtitle"].iloc[0] == 0
        assert result["summary_length"].iloc[0] == 0
        assert result["skills_count"].iloc[0] == 0

    def test_multiple_rows(self, multi_lead_df):
        result = extract_text_features(multi_lead_df.copy())
        assert len(result) == 3
        # Row 0 has summary, row 1 doesn't
        assert result["has_summary"].iloc[0] == 1
        assert result["has_summary"].iloc[1] == 0

    def test_founder_detection(self):
        df = pd.DataFrame([{
            "summary": "test", "skills": "test", "jobtitle": "CEO & Founder",
        }])
        result = extract_text_features(df)
        assert result["is_founder"].iloc[0] == 1

    def test_tech_role_detection(self):
        df = pd.DataFrame([{
            "summary": "test", "skills": "test", "jobtitle": "Senior Engineer",
        }])
        result = extract_text_features(df)
        assert result["is_tech_role"].iloc[0] == 1


# ---------------------------------------------------------------------------
# TestFillMissingValues
# ---------------------------------------------------------------------------

class TestFillMissingValues:
    def test_numeric_filled_with_median(self, sample_medians):
        df = pd.DataFrame([{
            "llm_quality": None, "llm_engagement": None,
            "llm_decision_maker": 0.6, "llm_company_fit": None,
            "companyfoundedon": None,
        }])
        result = fill_missing_values(df, numeric_medians=sample_medians)
        assert result["llm_quality"].iloc[0] == 50.0
        assert result["llm_engagement"].iloc[0] == 0.5
        assert result["companyfoundedon"].iloc[0] == 2016.0

    def test_non_null_numerics_unchanged(self, sample_medians):
        df = pd.DataFrame([{"llm_quality": 80.0, "llm_engagement": 0.9}])
        result = fill_missing_values(df, numeric_medians=sample_medians)
        assert result["llm_quality"].iloc[0] == 80.0
        assert result["llm_engagement"].iloc[0] == 0.9

    def test_categorical_filled_with_unknown(self):
        df = pd.DataFrame([{
            "llm_seniority": None, "companysize": None, "companytype": "Public",
            "llm_industry": None, "location": None,
        }])
        result = fill_missing_values(df)
        assert result["llm_seniority"].iloc[0] == "UNKNOWN"
        assert result["companysize"].iloc[0] == "UNKNOWN"
        assert result["companytype"].iloc[0] == "Public"  # unchanged
        assert result["llm_industry"].iloc[0] == "UNKNOWN"
        assert result["location"].iloc[0] == "UNKNOWN"

    def test_numeric_fallback_to_zero_without_medians(self):
        df = pd.DataFrame([{"llm_quality": None, "llm_engagement": None}])
        result = fill_missing_values(df, numeric_medians=None)
        assert result["llm_quality"].iloc[0] == 0
        assert result["llm_engagement"].iloc[0] == 0


# ---------------------------------------------------------------------------
# TestOneHotEncode
# ---------------------------------------------------------------------------

class TestOneHotEncode:
    def test_creates_dummy_columns(self):
        df = pd.DataFrame([
            {"llm_seniority": "Senior", "companysize": "51-200"},
            {"llm_seniority": "Mid", "companysize": "11-50"},
        ])
        result = one_hot_encode(df)
        assert "llm_seniority" not in result.columns
        assert "companysize" not in result.columns
        # Should have dummies (drop_first removes one)
        seniority_dummies = [c for c in result.columns if c.startswith("llm_seniority_")]
        assert len(seniority_dummies) >= 1

    def test_drop_first(self):
        df = pd.DataFrame([
            {"llm_seniority": "A", "companysize": "X"},
            {"llm_seniority": "B", "companysize": "Y"},
            {"llm_seniority": "C", "companysize": "Z"},
        ])
        result = one_hot_encode(df)
        # With 3 categories and drop_first, expect 2 dummies each
        seniority_dummies = [c for c in result.columns if c.startswith("llm_seniority_")]
        assert len(seniority_dummies) == 2

    def test_boolean_cast_to_int(self):
        df = pd.DataFrame([
            {"llm_seniority": "Senior", "companysize": "51-200"},
            {"llm_seniority": "Mid", "companysize": "11-50"},
        ])
        result = one_hot_encode(df)
        # All dummy columns should be int, not bool
        for col in result.columns:
            if col.startswith(("llm_seniority_", "companysize_")):
                assert result[col].dtype in (np.int64, np.int32, int, np.uint8), \
                    f"{col} has dtype {result[col].dtype}"

    def test_preserves_non_cat_columns(self):
        df = pd.DataFrame([{
            "llm_quality": 75, "llm_seniority": "Senior",
        }])
        result = one_hot_encode(df)
        assert "llm_quality" in result.columns


# ---------------------------------------------------------------------------
# TestAlignColumns
# ---------------------------------------------------------------------------

class TestAlignColumns:
    def test_missing_columns_filled_with_zero(self, sample_feature_columns):
        df = pd.DataFrame([{"llm_quality": 75}])
        result = align_columns(df, sample_feature_columns)
        assert list(result.columns) == sample_feature_columns
        # Missing columns should be 0, not NaN
        assert result["has_summary"].iloc[0] == 0
        assert not np.isnan(result["has_summary"].iloc[0])

    def test_drops_extra_columns(self, sample_feature_columns):
        df = pd.DataFrame([{"llm_quality": 75, "extra_col": 999}])
        result = align_columns(df, sample_feature_columns)
        assert "extra_col" not in result.columns

    def test_preserves_order(self, sample_feature_columns):
        df = pd.DataFrame([{col: i for i, col in enumerate(reversed(sample_feature_columns))}])
        result = align_columns(df, sample_feature_columns)
        assert list(result.columns) == sample_feature_columns

    def test_no_nan_in_result(self, sample_feature_columns):
        df = pd.DataFrame([{"llm_quality": 75}])
        result = align_columns(df, sample_feature_columns)
        assert not result.isnull().any().any()


# ---------------------------------------------------------------------------
# TestPreprocessForInference
# ---------------------------------------------------------------------------

class TestPreprocessForInference:
    def test_full_pipeline_correct_shape(self, single_lead_df, sample_feature_columns, sample_medians):
        result = preprocess_for_inference(
            single_lead_df.copy(),
            target_encoder=None,
            te_cols=[],
            feature_columns=sample_feature_columns,
            numeric_medians=sample_medians,
        )
        assert result.shape[1] == len(sample_feature_columns)

    def test_no_nan_in_output(self, single_lead_df, sample_feature_columns, sample_medians):
        result = preprocess_for_inference(
            single_lead_df.copy(),
            target_encoder=None,
            te_cols=[],
            feature_columns=sample_feature_columns,
            numeric_medians=sample_medians,
        )
        assert not result.isnull().any().any()

    def test_all_numeric(self, single_lead_df, sample_feature_columns, sample_medians):
        result = preprocess_for_inference(
            single_lead_df.copy(),
            target_encoder=None,
            te_cols=[],
            feature_columns=sample_feature_columns,
            numeric_medians=sample_medians,
        )
        non_numeric = result.select_dtypes(exclude=[np.number]).columns.tolist()
        assert non_numeric == [], f"Non-numeric columns found: {non_numeric}"

    def test_single_row(self, single_lead_df, sample_feature_columns, sample_medians):
        result = preprocess_for_inference(
            single_lead_df.copy(),
            target_encoder=None,
            te_cols=[],
            feature_columns=sample_feature_columns,
            numeric_medians=sample_medians,
        )
        assert len(result) == 1

    def test_batch(self, multi_lead_df, sample_feature_columns, sample_medians):
        result = preprocess_for_inference(
            multi_lead_df.copy(),
            target_encoder=None,
            te_cols=[],
            feature_columns=sample_feature_columns,
            numeric_medians=sample_medians,
        )
        assert len(result) == 3
        assert not result.isnull().any().any()

    def test_empty_lead(self, sample_feature_columns, sample_medians):
        df = pd.DataFrame([{}])
        result = preprocess_for_inference(
            df,
            target_encoder=None,
            te_cols=[],
            feature_columns=sample_feature_columns,
            numeric_medians=sample_medians,
        )
        assert result.shape == (1, len(sample_feature_columns))
        assert not result.isnull().any().any()

    def test_with_target_encoder(self, single_lead_df, sample_feature_columns, sample_medians):
        """When a target_encoder is provided, it should be applied to te_cols."""
        class FakeEncoder:
            def transform(self, X):
                return X.fillna(0).apply(pd.to_numeric, errors="coerce").fillna(0)

        te_cols = ["llm_industry", "industry", "companyindustry",
                   "languages", "location", "companylocation"]
        result = preprocess_for_inference(
            single_lead_df.copy(),
            target_encoder=FakeEncoder(),
            te_cols=te_cols,
            feature_columns=sample_feature_columns,
            numeric_medians=sample_medians,
        )
        assert result.shape[1] == len(sample_feature_columns)
        assert not result.isnull().any().any()
