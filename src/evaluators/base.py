"""Abstract base class for all image evaluators.

Every concrete evaluator in Vision-Reward-OS must inherit from
:class:`BaseEvaluator` and implement the :meth:`evaluate` method. This
enforces a uniform interface across all evaluators, making it trivial to add
new ones (PickScore, HPS v2/v3, ImageReward, VLM judges) without changing
the calling API code.
"""

from abc import ABC, abstractmethod
from PIL import Image

from src.api.schemas import EvaluatorScore

class BaseEvaluator(ABC):
    """Abstract interface that all image evaluators must implement."""

    evaluator_name: str = "BaseEvaluator"
    score_purpose: str = "Defines the specific evaluation criteria."

    def __init__(self):
        """Initiate Device and auto call load_model()."""
        self.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        # self.load_model()

    @abstractmethod
    def load_model(self):
        """Load model weights into memory and push to the appropriate device."""
        pass

    @abstractmethod
    def evaluate(self, image_a: Image.Image, image_b: Image.Image, prompt: str) -> EvaluatorScore:
        """Score a pair of candidate images against a text prompt.

        Args:
            image_a: The first candidate image (already decoded as PIL Image).
            image_b: The second candidate image (already decoded as PIL Image).
            prompt: The text prompt or instruction used to generate/edit the images.

        Returns:
            An :class:`~src.api.schemas.EvaluatorScore` containing:
            - ``evaluator_name``: the name of this evaluator.
            - ``score_a``: raw score for image A.
            - ``score_b``: raw score for image B.
            - ``preferred``: ``"A"`` or ``"B"``.
            - ``confidence``: preference confidence in [0, 1] (optional).
        """
        pass