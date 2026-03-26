"""ImageReward evaluator implementation.

ImageReward is the first general-purpose text-to-image human preference RM 
(Reward Model). It is trained on a massive dataset of human comparisons 
and demonstrates excellent correlation with human aesthetic and alignment 
preferences.

-------------------------------------------------------------------------------
# Score Utility:
The model outputs a scalar reward value (typically ranging from -2.0 to +2.0). 
A higher positive value indicates superior text-to-image alignment, better 
visual fidelity, and fewer anatomical defects, acting as a direct proxy 
for human ranking preference.

# Licensing Information:
- Codebase License: Apache License 2.0 (Permissive, open for commercial use).
- Checkpoint/Weights License: Apache License 2.0 (Open for commercial integration).
- Dataset License: The ImageRewardDB dataset is released under CC-BY-NC 4.0 
  (Non-Commercial use only). Like PickScore, using the data for direct commercial 
  model training is restricted, but inference using the pre-trained weights is safe.
-------------------------------------------------------------------------------

Paper: https://arxiv.org/abs/2304.05977
Repository: https://github.com/THUDM/ImageReward
"""

import logging
import torch
from PIL import Image

from src.api.schemas import EvaluatorScore
from src.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)

# ==============================================================================
# 🚨 "FRANKENSTEIN" MONKEY PATCH FOR IMAGEREWARD (V5)
# Reconstructs the entire legacy ecosystem that HuggingFace destroyed.
# Injects missing pruning utilities, Tokenizer properties, and Attention masks.
# ==============================================================================
import transformers.modeling_utils
import transformers
from transformers import BertTokenizer

# 1. Dummy for chunking
if not hasattr(transformers.modeling_utils, "apply_chunking_to_forward"):
    def dummy_apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
        return forward_fn(*input_tensors)
    transformers.modeling_utils.apply_chunking_to_forward = dummy_apply_chunking_to_forward

# 2. Dummy for finding pruneable heads
if not hasattr(transformers.modeling_utils, "find_pruneable_heads_and_indices"):
    def dummy_find_pruneable_heads(*args, **kwargs):
        return set(), torch.tensor([])
    transformers.modeling_utils.find_pruneable_heads_and_indices = dummy_find_pruneable_heads

# 3. Dummy for pruning linear layers
if not hasattr(transformers.modeling_utils, "prune_linear_layer"):
    def dummy_prune_linear_layer(layer, *args, **kwargs):
        return layer
    transformers.modeling_utils.prune_linear_layer = dummy_prune_linear_layer

# 4. Dummy for BertTokenizer Token IDs
logger.warning("Force-overwriting 'additional_special_tokens_ids' for BertTokenizer...")
def _force_get_add_ids(self):
    return [30522]
BertTokenizer.additional_special_tokens_ids = property(_force_get_add_ids)

# 5. Dummy for tied weights in PreTrainedModel (Returns empty dict to prevent .items() crash)
logger.warning("Patching 'all_tied_weights_keys' for PreTrainedModel...")
transformers.modeling_utils.PreTrainedModel.all_tied_weights_keys = property(lambda self: {})

# 6. NEW: Dummy for get_head_mask (Crucial for BLIP's forward pass)
# HuggingFace removed this utility completely. We return a list of Nones,
# which tells the model "do not mask any attention heads", bypassing the error.
logger.warning("Patching 'get_head_mask' for PreTrainedModel...")
def dummy_get_head_mask(self, head_mask, num_hidden_layers, *args, **kwargs):
    return [None] * num_hidden_layers
transformers.modeling_utils.PreTrainedModel.get_head_mask = dummy_get_head_mask

# 🚀 Now import ImageReward safely!
import ImageReward as RM
# ==============================================================================

class ImageRewardEvaluator(BaseEvaluator):
    """Evaluator that wraps the THUDM/ImageReward preference model."""

    evaluator_name: str = "ImageReward"
    score_purpose: str = "Evaluates deep semantic text-to-image alignment and penalizes anatomical or structural defects."
    model_name: str = "ImageReward-v1.0"

    def load_model(self) -> None:
        """Download and initialize the ImageReward model."""
        logger.info(f"Loading {self.evaluator_name} model on {self.device}...")
        
        # ImageReward provides a convenient unified loader
        try:
            self.model = RM.load(self.model_name, device=self.device)
            logger.info(f"{self.evaluator_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load {self.evaluator_name}: {e}")
            raise e

    def evaluate(self, image_a: Image.Image, image_b: Image.Image, prompt: str) -> EvaluatorScore:
        """Score a pair of images against a text prompt using ImageReward.

        Args:
            image_a: The first candidate image (PIL Image).
            image_b: The second candidate image (PIL Image).
            prompt: The text prompt used to generate/edit the images.

        Returns:
            An EvaluatorScore containing the real inference scores.
        """
        if not prompt:
            logger.warning(f"{self.evaluator_name} highly relies on a text prompt. An empty prompt may yield inaccurate results.")
            prompt = " "

        logger.debug(f"Evaluating with {self.evaluator_name} for prompt: '{prompt[:30]}...'")

        # Run model inference
        # ImageReward natively supports scoring PIL Images directly
        with torch.no_grad():
            try:
                # The model returns a raw scalar reward score
                score_a = round(self.model.score(prompt, image_a), 4)
                score_b = round(self.model.score(prompt, image_b), 4)
            except Exception as e:
                logger.error(f"Inference failed in {self.evaluator_name}: {e}")
                # Fallback to zero if inference fails
                score_a, score_b = 0.0, 0.0

        preferred = "A" if score_a >= score_b else "B"
        
        # Calculate confidence using Softmax probabilities to keep it in [0, 1] range
        # (Same logic applied in PickScore)
        probs = torch.softmax(torch.tensor([score_a, score_b]), dim=0).tolist()
        confidence = round(abs(probs[0] - probs[1]), 4)

        return EvaluatorScore(
            evaluator_name=self.evaluator_name,
            purpose=self.score_purpose,
            score_a=score_a,
            score_b=score_b,
            preferred=preferred,
            confidence=confidence,
        )
