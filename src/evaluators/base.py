"""Abstract base class for all image evaluators.

Every concrete evaluator in Vision-Reward-OS must inherit from
:class:`BaseEvaluator` and implement the :meth:`evaluate` method.  This
enforces a uniform interface across all evaluators, making it trivial to add
new ones (PickScore, HPS v2/v3, ImageReward, VLM judges, …) without changing
calling code.

Example::

    class MyCustomEvaluator(BaseEvaluator):
        def evaluate(self, image_a: str, image_b: str) -> EvaluatorScore:
            ...
"""

from abc import ABC, abstractmethod

from src.api.schemas import EvaluatorScore


class BaseEvaluator(ABC):
    """Abstract interface that all image evaluators must implement.

    Subclasses are responsible for loading their own model weights during
    ``__init__`` and implementing the :meth:`evaluate` method to score a pair
    of images.

    The :meth:`evaluate` contract:

    - Accepts two image references (URL strings **or** base-64 encoded strings).
    - Returns a single :class:`~src.api.schemas.EvaluatorScore` instance that
      captures the raw scores for both images, the preferred winner, and the
      evaluator's confidence.

    Attributes:
        evaluator_name: A human-readable identifier for the evaluator.  Must
            be set by every concrete subclass.
    """

    evaluator_name: str = "BaseEvaluator"

    @abstractmethod
    def evaluate(self, image_a: str, image_b: str) -> EvaluatorScore:
        """Score a pair of candidate images and return the evaluation result.

        Args:
            image_a: The first candidate image, provided as either a
                publicly accessible URL or a base-64 encoded string.
            image_b: The second candidate image, provided as either a
                publicly accessible URL or a base-64 encoded string.

        Returns:
            An :class:`~src.api.schemas.EvaluatorScore` containing:

            - ``evaluator_name``: the name of this evaluator.
            - ``score_a``: raw score for image A.
            - ``score_b``: raw score for image B.
            - ``preferred``: ``"A"`` or ``"B"``.
            - ``confidence``: preference confidence in [0, 1].
        """
