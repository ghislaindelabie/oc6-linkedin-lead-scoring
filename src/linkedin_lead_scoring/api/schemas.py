"""Pydantic models for API request/response validation"""
from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    service: str
    version: str


class LeadScoreRequest(BaseModel):
    """Lead scoring request (placeholder for future implementation)"""

    linkedin_url: str = Field(..., description="LinkedIn profile URL")
    # Additional fields will be added later:
    # - job_title, company, industry, etc.


class LeadScoreResponse(BaseModel):
    """Lead scoring response (placeholder for future implementation)"""

    score: float = Field(..., ge=0.0, le=1.0, description="Engagement probability (0-1)")
    risk_level: str = Field(..., description="Risk category: low/medium/high")
    model_version: Optional[str] = Field(None, description="Model version used")
