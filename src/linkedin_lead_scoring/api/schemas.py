"""Pydantic models for API request/response validation"""
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class LeadInput(BaseModel):
    """Single lead prediction request — all 19 feature fields."""

    # Numeric features
    llm_quality: Optional[int] = Field(
        None, ge=0, le=100, description="Profile quality score (0-100)"
    )
    llm_engagement: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Engagement likelihood (0-1)"
    )
    llm_decision_maker: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Decision maker probability (0-1)"
    )
    llm_company_fit: Optional[int] = Field(
        None, ge=0, le=2, description="Company fit score (0, 1, or 2)"
    )
    companyfoundedon: Optional[float] = Field(
        None, description="Company founding year (e.g. 2010.0)"
    )

    # Categorical features
    llm_seniority: Optional[str] = Field(
        None,
        description="Seniority level: Entry / Mid / Senior / Executive / C-Level",
        examples=["Senior"],
    )
    llm_industry: Optional[str] = Field(
        None,
        description="LLM-inferred industry (e.g. 'Technology - SaaS')",
        examples=["Technology - SaaS"],
    )
    llm_geography: Optional[str] = Field(
        None,
        description="Geography type: international_hub / regional_hub / other",
        examples=["international_hub"],
    )
    llm_business_type: Optional[str] = Field(
        None,
        description="Business type: leaders / experts / salespeople / workers / others",
        examples=["leaders"],
    )
    industry: Optional[str] = Field(
        None,
        description="LinkedIn industry label (e.g. 'Information Technology & Services')",
        examples=["Information Technology & Services"],
    )
    companyindustry: Optional[str] = Field(
        None,
        description="Company industry (e.g. 'Software Development')",
        examples=["Software Development"],
    )
    companysize: Optional[str] = Field(
        None,
        description="Company size range: 1-10 / 11-50 / 51-200 / 201-500 / 501-1000 / 1001-5000 / 5001-10000 / 10001+",
        examples=["51-200"],
    )
    companytype: Optional[str] = Field(
        None,
        description="Company type (e.g. 'Privately Held', 'Public Company')",
        examples=["Privately Held"],
    )
    languages: Optional[str] = Field(
        None,
        description="Comma-separated languages (e.g. 'English, French')",
        examples=["English, French"],
    )
    location: Optional[str] = Field(
        None,
        description="Profile location (City, Region, Country)",
        examples=["Paris, Île-de-France, France"],
    )
    companylocation: Optional[str] = Field(
        None,
        description="Company location (City, Country)",
        examples=["Paris, France"],
    )

    # Text features
    summary: Optional[str] = Field(
        None,
        description="Professional summary text",
        examples=["Experienced SaaS executive with 10+ years in B2B sales."],
    )
    skills: Optional[str] = Field(
        None,
        description="Comma-separated list of skills",
        examples=["Leadership, SaaS, B2B Sales"],
    )
    jobtitle: Optional[str] = Field(
        None,
        description="Current job title",
        examples=["VP of Sales"],
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class LeadPrediction(BaseModel):
    """Single prediction response."""

    score: float = Field(..., ge=0.0, le=1.0, description="Engagement probability (0-1)")
    label: str = Field(..., description="'engaged' or 'not_engaged'")
    confidence: str = Field(..., description="Confidence level: low / medium / high")
    model_version: str = Field(..., description="Model version used for this prediction")
    inference_time_ms: float = Field(..., ge=0.0, description="Inference time in milliseconds")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request — 1 to 100 leads."""

    leads: list[LeadInput] = Field(
        ..., min_length=1, max_length=10_000, description="List of leads to score (max 10 000)"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response with summary statistics."""

    predictions: list[LeadPrediction] = Field(..., description="Per-lead predictions")
    total_count: int = Field(..., ge=0, description="Total number of leads scored")
    avg_score: float = Field(..., ge=0.0, le=1.0, description="Average engagement score")
    high_engagement_count: int = Field(
        ..., ge=0, description="Number of leads with score >= 0.5"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(False, description="Whether the ML model is loaded and ready")


class ErrorResponse(BaseModel):
    """Structured error response."""

    error: str = Field(..., description="Short error code / category")
    message: str = Field(..., description="Human-readable error description")
    detail: Optional[Any] = Field(None, description="Additional context (field names, values, etc.)")
