"""Pydantic models for the Vision-Reward-OS API.

This module defines the request and response schemas used by the
/evaluate/ab-test endpoint and related routes.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class InputImages(BaseModel):
    """Request payload for an A/B image evaluation."""

    image_a_url: Optional[str] = Field(
        default=None,
        description="Publicly accessible URL of candidate image A.",
        examples=["https://example.com/image_a.jpg"],
    )
    image_a_b64: Optional[str] = Field(
        default=None,
        description="Base-64 encoded bytes of candidate image A.",
    )
    image_b_url: Optional[str] = Field(
        default=None,
        description="Publicly accessible URL of candidate image B.",
        examples=["https://example.com/image_b.jpg"],
    )
    image_b_b64: Optional[str] = Field(
        default=None,
        description="Base-64 encoded bytes of candidate image B.",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Text prompt used to generate the images.",
        examples=["a cat sitting on a red couch"],
    )

    @field_validator("image_a_url", "image_b_url", mode="before")
    @classmethod
    def validate_url_scheme(cls, value: Optional[str]) -> Optional[str]:
        """Ensure image URLs use http or https."""
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("Image URLs must start with 'http://' or 'https://'.")
        return value

    @model_validator(mode="after")
    def validate_image_sources(self) -> "InputImages":
        """Ensure at least one source (URL or Base64) is provided for both A and B."""
        if not self.image_a_url and not self.image_a_b64:
            raise ValueError("Must provide either image_a_url or image_a_b64.")
        if not self.image_b_url and not self.image_b_b64:
            raise ValueError("Must provide either image_b_url or image_b_b64.")
        return self


class EvaluatorScore(BaseModel):
    """Score produced by a single evaluator for one candidate image."""

    evaluator_name: str = Field(description="Name of the evaluator model.")
    purpose: str = Field(
        default="General image evaluation.",
        description="The specific business or technical purpose of this evaluator's score."
    )
    score_a: float = Field(description="Raw score for image A.")
    score_b: float = Field(description="Raw score for image B.")
    preferred: str = Field(
        description="The preferred image: 'A' or 'B'.",
        pattern="^[AB]$",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence of the preference in the range [0, 1]. Optional for raw logit models.",
        ge=0.0,
        le=1.0,
    )

class MultiDimensionalReport(BaseModel):
    """Aggregated multi-evaluator report for an A/B image comparison."""

    overall_winner: str = Field(
        description="The overall preferred image: 'A' or 'B'.",
        pattern="^[AB]$",
    )
    overall_confidence: Optional[float] = Field(
        default=None,
        description="Aggregate confidence across all evaluators, in [0, 1].",
        ge=0.0,
        le=1.0,
    )
    evaluator_scores: list[EvaluatorScore] = Field(
        description="Per-evaluator score breakdown.",
    )
    reasoning_summary: Optional[str] = Field(
        default=None,
        description="Natural-language explanation from the VLM judge.",
    )
    prompt_used: Optional[str] = Field(
        default=None,
        description="The text prompt supplied with the request, if any.",
    )