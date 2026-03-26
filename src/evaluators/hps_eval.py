"""HPS (Human Preference Score) evaluator implementation.

HPS v2/v3 evaluates the alignment and visual quality of AI-generated images
by utilizing a CLIP-based model fine-tuned on the Human Preference Dataset.

-------------------------------------------------------------------------------
# Score Utility:
The raw score reflects the alignment between the image and the text prompt, 
calibrated against human aesthetic preferences. Higher scores indicate that 
the image is more likely to be preferred by human annotators for that specific 
text prompt.

# Licensing Information:
- Codebase License: MIT License (Permissive, open for commercial use).
- Checkpoint/Weights License: CC-BY-NC 4.0 (Non-Commercial use for the weights 
  and the underlying HPD dataset). 
- Note for Enterprise: Commercial deployment of this specific checkpoint 
  requires obtaining an explicit commercial license from the original authors.
-------------------------------------------------------------------------------

Paper: https://arxiv.org/abs/2306.09341
Repository: https://github.com/tgxs002/HPSv2
"""

import logging
import torch
import hpsv2
from PIL import Image

from src.api.schemas import EvaluatorScore
from src.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)

import os
import urllib
# ==============================================================================
# 🚨 HPSV2 VOCABULARY PATCH
# The official hpsv2 PyPI package is notoriously missing a critical vocabulary 
# file for its internal OpenCLIP dependency. We dynamically download and inject 
# it into the library's installation directory at runtime to prevent fatal crashes.
# ==============================================================================
def _patch_hpsv2_vocab():
    """Silently patches the missing BPE vocab file in the hpsv2 package."""
    hps_dir = hpsv2.__path__[0]
    target_dir = os.path.join(hps_dir, "src", "open_clip")
    target_file = os.path.join(target_dir, "bpe_simple_vocab_16e6.txt.gz")
    
    url = "https://github.com/mlfoundations/open_clip/raw/main/src/open_clip/bpe_simple_vocab_16e6.txt.gz"
    
    os.makedirs(target_dir, exist_ok=True)
    
    if not os.path.exists(target_file):
        logger.warning(f"Missing OpenCLIP vocab file detected. Downloading to {target_dir}...")
        try:
            urllib.request.urlretrieve(url, target_file)
            logger.info("✅ Successfully patched HPSv2 vocabulary file!")
        except Exception as e:
            logger.error(f"❌ Failed to download HPSv2 vocab file: {e}")
            raise
    else:
        logger.info("✅ HPSv2 vocabulary file already exists. System ready.")

# Execute the patch before the class initializes
_patch_hpsv2_vocab()
# ==============================================================================

class HPSEvaluator(BaseEvaluator):
    """Evaluator that wraps the HPS v2.1 human preference model."""

    evaluator_name: str = "HPS_v2.1"
    score_purpose: str = "Calibrates the image against large-scale human aesthetic preference datasets for prompt fidelity."

    def load_model(self) -> None:
        """Initialize the HPS model environment.
        
        Note: The hpsv2 library automatically manages model weights caching 
        and lazy loading during the first inference call. We simply verify 
        the device setup here.
        """
        logger.info(f"Initializing {self.evaluator_name} environment on {self.device}...")

    def evaluate(self, image_a: Image.Image, image_b: Image.Image, prompt: str) -> EvaluatorScore:
        """Score a pair of images against a text prompt using HPSv2.

        Args:
            image_a: The first candidate image (PIL Image).
            image_b: The second candidate image (PIL Image).
            prompt: The text prompt used to generate the images.

        Returns:
            An EvaluatorScore containing the real inference scores.
        """
        if not prompt:
            logger.warning(
                f"{self.evaluator_name} strictly requires a text prompt. "
                "Using a blank space as fallback, which may degrade accuracy."
            )
            prompt = " "

        logger.debug(f"Evaluating with {self.evaluator_name} for prompt: '{prompt[:30]}...'")

        with torch.no_grad():
            try:
                # hpsv2.score accepts lists of PIL Images and a prompt string.
                # The hps_version parameter explicitly targets the v2.1 checkpoint.
                score_a_list = hpsv2.score([image_a], prompt, hps_version="v2.1")
                score_b_list = hpsv2.score([image_b], prompt, hps_version="v2.1")
                
                # Extract the raw scalar values from the returned lists
                score_a = round(score_a_list[0], 4)
                score_b = round(score_b_list[0], 4)
                
            except Exception as e:
                logger.error(f"Inference failed in {self.evaluator_name}: {e}")
                # Graceful degradation: assign 0.0 if the model crashes
                score_a, score_b = 0.0, 0.0

        preferred = "A" if score_a >= score_b else "B"
        
        # Calculate confidence using Softmax probabilities to keep it in [0, 1] range
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