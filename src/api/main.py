"""Vision-Reward-OS FastAPI application.

This module initializes the FastAPI application and registers all API routes.
The primary entry point is the `/evaluate/ab-test` endpoint, which aggregates
scores from PickScore, ImageReward, HPS, and a VLM Judge.

Usage (development):
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import base64
import io
import requests
from PIL import Image
from fastapi import FastAPI, HTTPException, status

from src.api.schemas import EvaluatorScore, InputImages, MultiDimensionalReport
from src.evaluators.pickscore_eval import PickScoreEvaluator
from src.evaluators.imagereward_eval import ImageRewardEvaluator
from src.evaluators.hps_eval import HPSEvaluator
from src.evaluators.aesthetic_eval import AestheticEvaluator
from src.evaluators.simulacra_eval import SimulacraEvaluator
from src.evaluators.trending_eval import TrendingEvaluator
from src.evaluators.mps_eval import MPSEvaluator
from src.evaluators.vlm_judge_eval import VLMJudgeEvaluator

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vision-Reward-OS",
    description=(
        "Enterprise-grade microservice that aggregates SOTA Human Preference "
        "models and VLM-as-a-judge to evaluate Generative AI images."
    ),
    version="0.1.0",
    contact={
        "name": "Tristan (Tri)",
        "url": "https://github.com/cantricao/Vision-Reward-OS",
    },
    license_info={"name": "MIT"},
)

# ---------------------------------------------------------------------------
# Evaluator Registry
# Evaluators are instantiated once at startup (Singleton pattern) and reused.
# ---------------------------------------------------------------------------
logger.info("Initializing evaluation engines...")

# Instantiate the VLM Judge separately to extract its reasoning later
vlm_judge = VLMJudgeEvaluator()

_evaluators = [
    PickScoreEvaluator(),
    ImageRewardEvaluator(),
    HPSEvaluator(),
    AestheticEvaluator(),
    SimulacraEvaluator(),
    MPSEvaluator(),
    TrendingEvaluator(),
    vlm_judge,  # <-- The VLM joins the judging panel here
]

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def decode_image(url: str | None, b64: str | None) -> Image.Image:
    """Convert either a URL or a Base64 string into a PIL Image."""
    try:
        if b64:
            image_data = base64.b64decode(b64)
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        elif url:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            raise ValueError("No image data provided.")
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process image payload. Error: {str(e)}"
        )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Liveness probe used by Docker and load balancers."""
    return {"status": "ok"}


@app.post(
    "/evaluate/ab-test",
    response_model=MultiDimensionalReport,
    status_code=status.HTTP_200_OK,
    tags=["Evaluation"],
    summary="A/B Image Evaluation Pipeline",
)
async def evaluate_ab_test(payload: InputImages) -> MultiDimensionalReport:
    """Run an A/B evaluation across all registered evaluators."""
    logger.info(f"Received A/B evaluation request. Prompt: '{payload.prompt}'")

    # 1. Decode images once at the API layer
    img_a_pil = decode_image(payload.image_a_url, payload.image_a_b64)
    img_b_pil = decode_image(payload.image_b_url, payload.image_b_b64)

    # 2. Run all evaluators sequentially
    evaluator_scores: list[EvaluatorScore] = []
    votes_a = 0
    votes_b = 0
    total_confidence = 0.0

    prompt_str = payload.prompt or ""

    for evaluator in _evaluators:
        try:
            result = evaluator.evaluate(img_a_pil, img_b_pil, prompt_str)
            evaluator_scores.append(result)

            if result.preferred == "A":
                votes_a += 1
            else:
                votes_b += 1
            
            if result.confidence is not None:
                total_confidence += result.confidence

        except Exception as e:
            logger.error(f"Evaluator {evaluator.evaluator_name} failed: {e}")

    if not evaluator_scores:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="All evaluation engines failed to process the request."
        )

    # 3. Aggregate Results (Majority Vote)
    overall_winner = "A" if votes_a >= votes_b else "B"
    overall_confidence = total_confidence / len(_evaluators) if _evaluators else 0.0

    # Extract the natural language reasoning from our VLM Judge
    # Fallback to a generic message if the VLM failed or returned None
    dynamic_reasoning = vlm_judge.latest_reasoning if vlm_judge.latest_reasoning else (
        f"Image {overall_winner} demonstrates closer alignment with the supplied "
        "prompt based on the aggregated signals from the multi-model ensemble."
    )

    report = MultiDimensionalReport(
        overall_winner=overall_winner,
        overall_confidence=round(overall_confidence, 4),
        evaluator_scores=evaluator_scores,
        reasoning_summary=dynamic_reasoning,
        prompt_used=payload.prompt,
    )

    logger.info(f"Evaluation complete. Winner: {overall_winner}, Confidence: {overall_confidence:.4f}")
    return report