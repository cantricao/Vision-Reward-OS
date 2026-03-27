import logging
import torch
import clip
from PIL import Image

from src.evaluators.base import BaseEvaluator
from src.api.schemas import EvaluatorScore
from src.evaluators.shared_backbones import BackboneRegistry

logger = logging.getLogger(__name__)

# ==============================================================================
# TRENDING EVALUATOR
# Calculates how closely an image aligns with high-quality aesthetic keywords
# (e.g., "trending on artstation"). Uses zero-shot CLIP text-image similarity.
# Costs 0MB extra VRAM because it purely reuses the shared CLIP backbone!
# ==============================================================================
class TrendingEvaluator(BaseEvaluator):
    evaluator_name: str = "Trending_Score"
    score_purpose: str = "Measures alignment with 'trending on artstation' and high-quality keywords."

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = None
        self.preprocess = None
        
        # Pre-tokenize our target aesthetic prompts
        self.target_prompts = [
            "trending on artstation",
            "masterpiece, best quality, highly detailed",
            "award winning photography, 4k resolution"
        ]
        self.text_tokens = None

    def load_model(self):
        """
        Connects to the shared ViT-L/14 backbone and pre-encodes the text prompts.
        """
        if self.clip_model is not None:
            return

        try:
            # 1. Fetch the SHARED Backbone (Zero extra VRAM cost!)
            logger.info(f"[{self.evaluator_name}] Connecting to shared ViT-L/14 backbone...")
            self.clip_model, self.preprocess = BackboneRegistry.get_vit_l_14()

            # 2. Pre-compute text embeddings in FP16 to save time during evaluation
            logger.info(f"[{self.evaluator_name}] Pre-computing text embeddings for trending keywords...")
            self.text_tokens = clip.tokenize(self.target_prompts).to(self.device)
            
            logger.info(f"[{self.evaluator_name}] Successfully warmed up. (0MB extra VRAM used).")

        except Exception as e:
            logger.error(f"[{self.evaluator_name}] Initialization failed: {e}")
            self.clip_model = None
            self.preprocess = None

    def _get_single_image_score(self, image: Image.Image) -> float:
        """
        Calculates the cosine similarity between the image and the trending text prompts.
        """
        # CRITICAL: Convert input image to FP16 (.half()) to match the backbone
        image_input = self.preprocess(image).unsqueeze(0).to(self.device).half()

        with torch.no_grad():
            # Extract image embeddings
            image_features = self.clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Extract text embeddings
            text_features = self.clip_model.encode_text(self.text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Calculate cosine similarity (dot product of normalized vectors)
            # We take the mean similarity across all our positive target prompts
            similarity = (image_features @ text_features.T).mean()

        return similarity.item()

    def evaluate(self, image_a: Image.Image, image_b: Image.Image, prompt: str) -> EvaluatorScore:
        """
        Compares two images based on their 'trending' similarity scores.
        """
        if self.clip_model is None:
            self.load_model()

        # Get raw cosine similarity scores (typically ranges from 0.15 to 0.40)
        score_a = self._get_single_image_score(image_a)
        score_b = self._get_single_image_score(image_b)

        preferred = "A" if score_a > score_b else "B"
        
        # Calculate confidence based on the margin of difference
        # In cosine similarity, a difference of 0.05 is highly significant
        score_diff = abs(score_a - score_b)
        confidence = min(score_diff * 10.0, 1.0) 

        # Normalize the raw scores into a percentage representation for the API
        total = score_a + score_b
        norm_a = score_a / total if total > 0 else 0.5
        norm_b = score_b / total if total > 0 else 0.5

        return EvaluatorScore(
            evaluator_name=self.evaluator_name,
            purpose=self.score_purpose,
            score_a=round(norm_a, 4),
            score_b=round(norm_b, 4),
            preferred=preferred,
            confidence=round(confidence, 4)
        )