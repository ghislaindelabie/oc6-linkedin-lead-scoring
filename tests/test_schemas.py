"""Unit tests for API Pydantic schemas (Task B.1)"""
import pytest
from pydantic import ValidationError

from linkedin_lead_scoring.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    LeadInput,
    LeadPrediction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_LEAD = {
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
}

VALID_PREDICTION = {
    "score": 0.73,
    "label": "engaged",
    "confidence": "high",
    "model_version": "0.3.0",
    "inference_time_ms": 12.5,
}


# ---------------------------------------------------------------------------
# LeadInput — required/optional fields
# ---------------------------------------------------------------------------


class TestLeadInput:
    def test_valid_full_lead(self):
        lead = LeadInput(**VALID_LEAD)
        assert lead.llm_quality == 75
        assert lead.llm_engagement == 0.8

    def test_all_fields_optional_except_none_required(self):
        """LeadInput should accept an empty dict (all fields optional)."""
        lead = LeadInput()
        assert lead.llm_quality is None

    def test_llm_quality_upper_bound(self):
        """llm_quality must be <= 100."""
        with pytest.raises(ValidationError) as exc_info:
            LeadInput(llm_quality=101)
        assert "llm_quality" in str(exc_info.value)

    def test_llm_quality_lower_bound(self):
        """llm_quality must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            LeadInput(llm_quality=-1)
        assert "llm_quality" in str(exc_info.value)

    def test_llm_engagement_upper_bound(self):
        """llm_engagement must be <= 1."""
        with pytest.raises(ValidationError) as exc_info:
            LeadInput(llm_engagement=1.1)
        assert "llm_engagement" in str(exc_info.value)

    def test_llm_engagement_lower_bound(self):
        """llm_engagement must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            LeadInput(llm_engagement=-0.1)
        assert "llm_engagement" in str(exc_info.value)

    def test_llm_decision_maker_bounds(self):
        """llm_decision_maker must be in [0, 1]."""
        with pytest.raises(ValidationError):
            LeadInput(llm_decision_maker=1.5)
        with pytest.raises(ValidationError):
            LeadInput(llm_decision_maker=-0.1)

    def test_llm_company_fit_valid_values(self):
        """llm_company_fit must be 0, 1, or 2."""
        for v in (0, 1, 2):
            lead = LeadInput(llm_company_fit=v)
            assert lead.llm_company_fit == v

    def test_llm_company_fit_invalid(self):
        """llm_company_fit must be <= 2."""
        with pytest.raises(ValidationError):
            LeadInput(llm_company_fit=3)

    def test_wrong_type_for_numeric_field(self):
        """String where int expected should raise ValidationError."""
        with pytest.raises(ValidationError):
            LeadInput(llm_quality="high")

    def test_companyfoundedon_float(self):
        lead = LeadInput(companyfoundedon=2005.0)
        assert lead.companyfoundedon == 2005.0

    def test_string_fields_accept_none(self):
        lead = LeadInput(llm_seniority=None, jobtitle=None)
        assert lead.llm_seniority is None
        assert lead.jobtitle is None


# ---------------------------------------------------------------------------
# LeadPrediction — response schema
# ---------------------------------------------------------------------------


class TestLeadPrediction:
    def test_valid_prediction(self):
        pred = LeadPrediction(**VALID_PREDICTION)
        assert pred.score == 0.73
        assert pred.label == "engaged"
        assert pred.confidence == "high"

    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            LeadPrediction(**{**VALID_PREDICTION, "score": 1.5})
        with pytest.raises(ValidationError):
            LeadPrediction(**{**VALID_PREDICTION, "score": -0.1})

    def test_label_values(self):
        """label must be 'engaged' or 'not_engaged'."""
        pred_not = LeadPrediction(**{**VALID_PREDICTION, "label": "not_engaged"})
        assert pred_not.label == "not_engaged"

    def test_confidence_values(self):
        """confidence must be low/medium/high."""
        for c in ("low", "medium", "high"):
            pred = LeadPrediction(**{**VALID_PREDICTION, "confidence": c})
            assert pred.confidence == c

    def test_serialization(self):
        pred = LeadPrediction(**VALID_PREDICTION)
        d = pred.model_dump()
        assert set(d.keys()) == {"score", "label", "confidence", "model_version", "inference_time_ms"}


# ---------------------------------------------------------------------------
# BatchPredictionRequest
# ---------------------------------------------------------------------------


class TestBatchPredictionRequest:
    def test_valid_batch(self):
        req = BatchPredictionRequest(leads=[LeadInput(**VALID_LEAD)])
        assert len(req.leads) == 1

    def test_empty_batch_rejected(self):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(leads=[])

    def test_batch_over_limit_rejected(self):
        leads = [LeadInput(**VALID_LEAD)] * 101
        with pytest.raises(ValidationError):
            BatchPredictionRequest(leads=leads)


# ---------------------------------------------------------------------------
# BatchPredictionResponse
# ---------------------------------------------------------------------------


class TestBatchPredictionResponse:
    def test_valid_response(self):
        resp = BatchPredictionResponse(
            predictions=[LeadPrediction(**VALID_PREDICTION)],
            total_count=1,
            avg_score=0.73,
            high_engagement_count=1,
        )
        assert resp.total_count == 1
        assert resp.avg_score == 0.73


# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------


class TestHealthResponse:
    def test_health_includes_model_loaded(self):
        h = HealthResponse(
            status="healthy",
            service="linkedin-lead-scoring-api",
            version="0.3.0",
            model_loaded=True,
        )
        assert h.model_loaded is True

    def test_health_model_loaded_defaults_false(self):
        h = HealthResponse(
            status="healthy",
            service="linkedin-lead-scoring-api",
            version="0.3.0",
        )
        assert h.model_loaded is False


# ---------------------------------------------------------------------------
# ErrorResponse
# ---------------------------------------------------------------------------


class TestErrorResponse:
    def test_error_response_fields(self):
        err = ErrorResponse(
            error="validation_error",
            message="llm_quality must be between 0 and 100",
            detail={"field": "llm_quality", "value": 150},
        )
        assert err.error == "validation_error"
        assert "llm_quality" in err.message

    def test_error_response_optional_detail(self):
        err = ErrorResponse(error="not_found", message="Resource not found")
        assert err.detail is None
