"""PickScore evaluator implementation.

PickScore is a human preference model trained on the Pick-a-Pic dataset
(Kirstain et al., 2023). It scores image-text pairs and has been shown to
correlate strongly with human judgements of image quality and prompt alignment.

-------------------------------------------------------------------------------
# Score Utility:
The raw score represents the unnormalized logit (dot product) of human preference. 
A higher score indicates a higher probability that a human annotator would 
prefer this generated image for the given text prompt.

# Licensing Information:
- Codebase License: MIT License (Permissive, open for commercial use).
- Checkpoint/Weights License: Open for research and commercial use, following 
  the underlying LAION/CLIP model terms (OpenRAIL/MIT).
- Dataset License: The Pick-a-Pic training dataset is licensed under 
  CC-BY-NC 4.0 (Attribution-NonCommercial 4.0 International). Therefore, fine-tuning 
  commercial models directly on their raw data requires legal caution.
-------------------------------------------------------------------------------

Paper: https://arxiv.org/abs/2305.01569
Repository: https://github.com/yuvalkirstain/PickScore
"""

import logging
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image

from src.api.schemas import EvaluatorScore
from src.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


class PickScoreEvaluator(BaseEvaluator):
    """Evaluator that wraps the PickScore human preference model."""

    evaluator_name: str = "PickScore"
    score_purpose: str = "Measures general commercial aesthetic and how strongly human users prefer the image based on the prompt."
    model_name: str = "yuvalkirstain/PickScore_v1"

    def load_model(self) -> None:
        """Download and cache the PickScore model weights and processor."""
        logger.info(f"Loading {self.evaluator_name} model ({self.model_name}) on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).half().eval().to(self.device)
        logger.info(f"{self.evaluator_name} loaded successfully.")

    def evaluate(self, image_a: Image.Image, image_b: Image.Image, prompt: str) -> EvaluatorScore:
        """Score a pair of images using PickScore's CLIP-based reward model.

        Args:
            image_a: The first candidate image (PIL Image).
            image_b: The second candidate image (PIL Image).
            prompt: The text prompt used to generate the images.

        Returns:
            An EvaluatorScore containing the real inference scores.
        """
        logger.debug(f"Evaluating with {self.evaluator_name} for prompt: '{prompt[:30]}...'")

        images = [image_a, image_b]
        
        # Preprocess the images and text prompt
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        # Run model inference
        with torch.no_grad():
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
            
            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
            
            # Calculate the scores (Dot product)
            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            scores = scores.cpu().tolist()

        score_a = round(scores[0], 4)
        score_b = round(scores[1], 4)
        
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