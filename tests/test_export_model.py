"""Tests for scripts/export_model.py preprocessing pipeline and artifact saving."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add scripts/ to path so we can import from export_model
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_raw_df():
    """Minimal synthetic DataFrame matching the 20-column cleaned CSV schema."""
    return pd.DataFrame({
        "llm_quality": [85, 75, 20, 80, 60, 50, 90, 30, 70, 45],
        "llm_seniority": ["Senior", "Mid", "Entry", "Mid", "Senior", "Entry", "Senior", "Mid", "Entry", "Mid"],
        "llm_engagement": [0.8, 0.65, 0.3, 0.55, 0.7, 0.4, 0.9, 0.2, 0.6, 0.35],
        "llm_decision_maker": [0.85, 0.4, 0.2, 0.5, 0.75, 0.3, 0.9, 0.15, 0.6, 0.25],
        "llm_industry": ["Technology - SaaS", "Technology - AI", "Unknown", "Consulting", "Technology - SaaS",
                         "Finance", "Technology - AI", "Unknown", "Consulting", "Finance"],
        "llm_geography": ["international_hub", "international_hub", "local", "international_hub", "local",
                          "international_hub", "local", "local", "international_hub", "local"],
        "llm_business_type": ["leaders", "experts", "others", "experts", "leaders",
                              "others", "leaders", "others", "experts", "others"],
        "llm_company_fit": [1, 2, 0, 1, 2, 0, 1, 0, 2, 1],
        "industry": ["IT Services", "Software", "IT Services", "Consulting", "Software",
                     "Finance", "IT Services", "Consulting", "Software", "Finance"],
        "companyindustry": ["Software Development", "Software Development", None, "Business Consulting",
                            "Software Development", "Financial Services", "Software Development",
                            "Business Consulting", None, "Financial Services"],
        "companysize": ["1001-5000", "201-500", None, "51-200", "201-500",
                        "11-50", "1001-5000", "51-200", "11-50", None],
        "companytype": ["Public Company", "Privately Held", None, "Partnership",
                        "Privately Held", "Public Company", "Public Company", None, "Privately Held", "Partnership"],
        "languages": ["English, French", "English", "French", "English, French",
                      "English", "French", "English, French", "English", "French", "English"],
        "location": ["Paris, France", "Paris, France", "Lyon, France", "Paris, France",
                     "Lyon, France", "Paris, France", "Paris, France", "Lyon, France", "Paris, France", "Lyon, France"],
        "companylocation": ["Paris, FR", "Paris, FR", None, "Lyon, FR",
                            "Paris, FR", None, "Paris, FR", "Lyon, FR", None, "Lyon, FR"],
        "companyfoundedon": [2020.0, 2015.0, None, 2010.0, 2018.0, 2012.0, 2019.0, None, 2016.0, 2011.0],
        "summary": ["Experienced leader in SaaS", None, "Junior developer", "Business consultant",
                    "Tech expert", None, "CEO and founder", "Marketing specialist", None, "Finance analyst"],
        "skills": ["Python, ML, Leadership", "Python, Data Science", None, "Strategy, Business",
                   "Python, React", None, "Strategy, Leadership, Python", None, "Marketing, SEO", "Finance, Excel"],
        "jobtitle": ["CEO & Founder", "Senior Data Scientist", "Junior Developer", "Director of Strategy",
                     "VP Engineering", "Marketing Manager", "Co-Founder", "Lead Data Analyst", "Developer", "Finance VP"],
        "engaged": [1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    })


# ---------------------------------------------------------------------------
# Tests: preprocessing helpers
# ---------------------------------------------------------------------------

class TestTextFeatureExtraction:
    def test_extract_text_features_has_summary(self, sample_raw_df):
        from export_model import extract_text_features
        result = extract_text_features(sample_raw_df.copy())
        assert "has_summary" in result.columns
        assert result["has_summary"].sum() == sample_raw_df["summary"].notna().sum()

    def test_extract_text_features_has_skills(self, sample_raw_df):
        from export_model import extract_text_features
        result = extract_text_features(sample_raw_df.copy())
        assert "has_skills" in result.columns

    def test_extract_text_features_jobtitle_flags(self, sample_raw_df):
        from export_model import extract_text_features
        result = extract_text_features(sample_raw_df.copy())
        assert "is_founder" in result.columns
        assert "is_director" in result.columns
        assert "is_manager" in result.columns
        # "CEO & Founder" → is_founder=1
        assert result.loc[0, "is_founder"] == 1
        # "Director of Strategy" → is_director=1
        assert result.loc[3, "is_director"] == 1

    def test_extract_text_features_drops_raw_text_cols(self, sample_raw_df):
        from export_model import extract_text_features
        result = extract_text_features(sample_raw_df.copy())
        assert "summary" not in result.columns
        assert "skills" not in result.columns
        assert "jobtitle" not in result.columns

    def test_extract_text_features_skills_count(self, sample_raw_df):
        from export_model import extract_text_features
        result = extract_text_features(sample_raw_df.copy())
        assert "skills_count" in result.columns
        # "Python, ML, Leadership" → 3 skills
        assert result.loc[0, "skills_count"] == 3


class TestPreprocessPipeline:
    def test_preprocess_returns_train_test_split(self, sample_raw_df):
        from export_model import preprocess
        X_train, X_test, y_train, y_test, preprocessor = preprocess(sample_raw_df.copy())
        assert len(X_train) + len(X_test) == len(sample_raw_df)
        assert len(y_train) + len(y_test) == len(sample_raw_df)

    def test_preprocess_no_missing_values_in_output(self, sample_raw_df):
        from export_model import preprocess
        X_train, X_test, y_train, y_test, preprocessor = preprocess(sample_raw_df.copy())
        assert not X_train.isnull().any().any(), "Training features must have no NaN"
        assert not X_test.isnull().any().any(), "Test features must have no NaN"

    def test_preprocess_target_not_in_features(self, sample_raw_df):
        from export_model import preprocess
        X_train, X_test, y_train, y_test, preprocessor = preprocess(sample_raw_df.copy())
        assert "engaged" not in X_train.columns
        assert "engaged" not in X_test.columns

    def test_preprocess_consistent_columns(self, sample_raw_df):
        from export_model import preprocess
        X_train, X_test, y_train, y_test, preprocessor = preprocess(sample_raw_df.copy())
        assert list(X_train.columns) == list(X_test.columns), "Train and test must share same columns"

    def test_preprocess_returns_preprocessor(self, sample_raw_df):
        from export_model import preprocess
        _, _, _, _, preprocessor = preprocess(sample_raw_df.copy())
        assert preprocessor is not None


# ---------------------------------------------------------------------------
# Tests: artifact saving
# ---------------------------------------------------------------------------

class TestArtifactSaving:
    """joblib.dump is patched so tests don't actually serialize objects."""

    def _call_save(self, tmp_path, sample_raw_df):
        """Helper: preprocess sample data and call save_artifacts with patched joblib."""
        from export_model import preprocess, save_artifacts
        X_train, X_test, y_train, y_test, preprocessor = preprocess(sample_raw_df.copy())
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        ref_dir = tmp_path / "data" / "reference"
        ref_dir.mkdir(parents=True)
        mock_model = MagicMock()
        with patch("export_model.joblib.dump"):
            save_artifacts(mock_model, preprocessor, X_train, model_dir, ref_dir)
        return X_train, preprocessor, model_dir, ref_dir

    def test_save_feature_columns_json(self, tmp_path, sample_raw_df):
        X_train, _, model_dir, _ = self._call_save(tmp_path, sample_raw_df)
        feature_cols_path = model_dir / "feature_columns.json"
        assert feature_cols_path.exists()
        with open(feature_cols_path) as f:
            cols = json.load(f)
        assert isinstance(cols, list)
        assert len(cols) == len(X_train.columns)
        assert cols == list(X_train.columns)

    def test_save_model_joblib(self, tmp_path, sample_raw_df):
        from export_model import preprocess, save_artifacts
        X_train, _, _, _, preprocessor = preprocess(sample_raw_df.copy())
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        ref_dir = tmp_path / "data" / "reference"
        ref_dir.mkdir(parents=True)
        mock_model = MagicMock()
        with patch("export_model.joblib.dump") as mock_dump:
            save_artifacts(mock_model, preprocessor, X_train, model_dir, ref_dir)
        # joblib.dump must have been called for model and preprocessor
        assert mock_dump.call_count == 2
        saved_paths = [str(call.args[1]) for call in mock_dump.call_args_list]
        assert any("xgboost_model.joblib" in p for p in saved_paths)
        assert any("preprocessor.joblib" in p for p in saved_paths)

    def test_save_reference_data_csv(self, tmp_path, sample_raw_df):
        _, _, model_dir, ref_dir = self._call_save(tmp_path, sample_raw_df)
        ref_csv = ref_dir / "training_reference.csv"
        assert ref_csv.exists()
        ref_df = pd.read_csv(ref_csv)
        assert len(ref_df) <= 100
        assert len(ref_df) > 0

    def test_feature_columns_match_model_input(self, tmp_path, sample_raw_df):
        """Feature columns JSON must match the columns of saved reference CSV."""
        X_train, _, model_dir, ref_dir = self._call_save(tmp_path, sample_raw_df)
        with open(model_dir / "feature_columns.json") as f:
            feature_cols = json.load(f)
        ref_df = pd.read_csv(ref_dir / "training_reference.csv")
        assert feature_cols == list(ref_df.columns)


# ---------------------------------------------------------------------------
# Tests: find_data_file helper
# ---------------------------------------------------------------------------

class TestFindDataFile:
    def test_find_data_file_local(self, tmp_path):
        from export_model import find_data_file
        csv_path = tmp_path / "data" / "processed" / "linkedin_leads_clean.csv"
        csv_path.parent.mkdir(parents=True)
        csv_path.write_text("col1,col2\n1,2\n")
        result = find_data_file(search_root=tmp_path)
        assert result == csv_path

    def test_find_data_file_not_found(self, tmp_path):
        from export_model import find_data_file
        with pytest.raises(FileNotFoundError):
            find_data_file(search_root=tmp_path)
