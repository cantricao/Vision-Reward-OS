"""PickScore evaluator implementation.

PickScore is a human preference model trained on the Pick-a-Pic dataset
(Kirstain et al., 2023).  It scores image-text pairs and has been shown to
correlate strongly with human judgements of image quality and prompt alignment.

Paper: https://arxiv.org/abs/2305.01569
Repository: https://github.com/yuvalkirstain/PickScore

.. note::
    This is a **stub/dummy implementation** for boilerplate purposes.
    The real implementation would load the PickScore checkpoint via
    ``transformers`` and run inference on the decoded images.  The dummy
    version returns deterministic mock scores so that the API is fully
    functional without requiring GPU resources or large model downloads.
"""

import logging

from src.api.schemas import EvaluatorScore
from src.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


class PickScoreEvaluator(BaseEvaluator):
    """Evaluator that wraps the PickScore human preference model.

    PickScore assigns a scalar aesthetic/preference score to each
    (image, prompt) pair.  In an A/B comparison the image with the higher
    score is considered the preferred output.

    Attributes:
        evaluator_name: Fixed identifier ``"PickScore"``.
        model_name: HuggingFace model hub identifier for the PickScore
            checkpoint.  Overridable via ``configs/eval_config.yaml``.

    Args:
        model_name: HuggingFace model hub path for the PickScore processor /
            model weights.  Defaults to the official checkpoint.
    """

    evaluator_name: str = "PickScore"

    def __init__(
        self,
        model_name: str = "yuvalkirstain/PickScore_v1",
    ) -> None:
        """Initialise the PickScore evaluator.

        In the full implementation this constructor would download and cache
        the PickScore model weights and processor from the HuggingFace Hub
        using ``transformers``.

        Args:
            model_name: HuggingFace Hub path for the PickScore checkpoint.
        """
        self.model_name = model_name
        logger.info(
            "Initialised %s (model=%s) — running in STUB mode.",
            self.evaluator_name,
            self.model_name,
        )
        # TODO: Load model weights here once GPU resources are available.
        # from transformers import AutoProcessor, AutoModel
        # self.processor = AutoProcessor.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name).eval()

    def evaluate(self, image_a: str, image_b: str) -> EvaluatorScore:
        """Score a pair of images using PickScore.

        In stub mode this method returns deterministic mock scores.  In the
        full implementation it would:

        1. Fetch / decode both images.
        2. Pre-process them with the PickScore processor.
        3. Run a forward pass through the CLIP-based reward model.
        4. Return the normalised scores as an :class:`EvaluatorScore`.

        Args:
            image_a: URL or base-64 string for candidate image A.
            image_b: URL or base-64 string for candidate image B.

        Returns:
            An :class:`~src.api.schemas.EvaluatorScore` with mock scores.
        """
        logger.debug(
            "%s.evaluate called. image_a=%r image_b=%r",
            self.evaluator_name,
            image_a[:40] if image_a else None,
            image_b[:40] if image_b else None,
        )

        # ------------------------------------------------------------------ #
        # STUB: Replace with real model inference in production.             #
        # ------------------------------------------------------------------ #
        mock_score_a: float = 0.72
        mock_score_b: float = 0.65

        preferred = "A" if mock_score_a >= mock_score_b else "B"
        # Confidence = normalised margin between the two scores.
        # Guard against division by zero when both scores are 0.
        total = mock_score_a + mock_score_b
        confidence = round(abs(mock_score_a - mock_score_b) / total, 4) if total > 0 else 0.0

        return EvaluatorScore(
            evaluator_name=self.evaluator_name,
            score_a=mock_score_a,
            score_b=mock_score_b,
            preferred=preferred,
            confidence=confidence,
        )
