"""Vision-Reward-OS FastAPI application.

This module initialises the FastAPI application and registers all API routes.
The primary entry point for the microservice is the ``/evaluate/ab-test``
endpoint, which accepts two candidate images and returns a
:class:`~src.api.schemas.MultiDimensionalReport` aggregating scores from all
enabled evaluators.

Usage (development)::

    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging

from fastapi import FastAPI, HTTPException, status

from src.api.schemas import EvaluatorScore, InputImages, MultiDimensionalReport
from src.evaluators.pickscore_eval import PickScoreEvaluator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vision-Reward-OS",
    description=(
        "Enterprise-grade microservice that aggregates SOTA Human Preference "
        "models (PickScore, HPS v2/v3, ImageReward) and VLM-as-a-judge "
        "(Gemini/Qwen) to evaluate Generative AI images."
    ),
    version="0.1.0",
    contact={
        "name": "Vision-Reward-OS",
        "url": "https://github.com/cantricao/Vision-Reward-OS",
    },
    license_info={"name": "MIT"},
)

# ---------------------------------------------------------------------------
# Evaluator registry
# Evaluators are instantiated once at startup and reused across requests.
# ---------------------------------------------------------------------------
_evaluators = [
    PickScoreEvaluator(),
]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Liveness probe used by the Docker HEALTHCHECK and load balancers.

    Returns:
        A JSON object ``{"status": "ok"}`` when the service is healthy.
    """
    return {"status": "ok"}


@app.post(
    "/evaluate/ab-test",
    response_model=MultiDimensionalReport,
    status_code=status.HTTP_200_OK,
    tags=["Evaluation"],
    summary="A/B image evaluation",
    description=(
        "Compare two candidate images using all enabled evaluators and return "
        "an aggregated MultiDimensionalReport."
    ),
)
async def evaluate_ab_test(payload: InputImages) -> MultiDimensionalReport:
    """Run an A/B evaluation across all registered evaluators.

    The endpoint accepts two images (either as publicly accessible URLs or as
    base-64 encoded strings) and an optional generation prompt.  It passes the
    images through every enabled evaluator, aggregates the individual scores
    via majority vote, and returns a :class:`MultiDimensionalReport`.

    Args:
        payload: The :class:`InputImages` request body containing image A,
            image B, and an optional prompt.

    Returns:
        A :class:`MultiDimensionalReport` with per-evaluator scores, an
        overall winner, and an optional VLM reasoning summary.

    Raises:
        HTTPException: 422 if neither a URL nor a base-64 string is provided
            for either image slot.
    """
    # Validate that at least one source is provided for each image slot.
    if payload.image_a_url is None and payload.image_a_b64 is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either 'image_a_url' or 'image_a_b64' for image A.",
        )
    if payload.image_b_url is None and payload.image_b_b64 is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either 'image_b_url' or 'image_b_b64' for image B.",
        )

    image_a_ref = payload.image_a_url or payload.image_a_b64
    image_b_ref = payload.image_b_url or payload.image_b_b64

    logger.info(
        "Received A/B evaluation request. prompt=%r evaluators=%d",
        payload.prompt,
        len(_evaluators),
    )

    # Run all evaluators and collect scores.
    evaluator_scores: list[EvaluatorScore] = []
    votes_a = 0
    votes_b = 0
    total_confidence = 0.0

    for evaluator in _evaluators:
        result = evaluator.evaluate(image_a_ref, image_b_ref)
        evaluator_scores.append(result)

        if result.preferred == "A":
            votes_a += 1
        else:
            votes_b += 1
        total_confidence += result.confidence

    # Aggregate: majority vote determines overall winner.
    # Ties are broken in favour of image A (conservative default; callers
    # should inspect per-evaluator scores for inconclusive cases).
    overall_winner = "A" if votes_a >= votes_b else "B"
    overall_confidence = total_confidence / len(_evaluators) if _evaluators else 0.0

    report = MultiDimensionalReport(
        overall_winner=overall_winner,
        overall_confidence=round(overall_confidence, 4),
        evaluator_scores=evaluator_scores,
        reasoning_summary=(
            "Image A demonstrates stronger aesthetic composition and closer "
            "alignment with the supplied prompt based on the aggregated "
            "evaluator signals."
            if overall_winner == "A"
            else "Image B demonstrates stronger aesthetic composition and closer "
            "alignment with the supplied prompt based on the aggregated "
            "evaluator signals."
        ),
        prompt_used=payload.prompt,
    )

    logger.info("Evaluation complete. winner=%s confidence=%.4f", overall_winner, overall_confidence)
    return report
