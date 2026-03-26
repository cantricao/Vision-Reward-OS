"""Real-Time Google Trends 'Viral Potential' evaluator implementation.

This module dynamically fetches live trending topics from Google Trends using 
the `trendspyg` crawler. It engineers these trends into visual prompts and 
uses OpenAI's CLIP model to compute a contrastive probability. This acts as 
a dynamic 'Viral Potential' score, evaluating how closely the generated images 
align with current global market interests.

-------------------------------------------------------------------------------
# Score Utility:
Outputs a scalar viral potential score (0.0 to 1.0). This represents the sum 
of probabilities that the image matches current live trends rather than a 
generic, negative baseline.

# Licensing Information:
- Codebase: MIT License.
- Underlying CLIP Model: MIT License (OpenAI).
- Data Source: Google Trends RSS feeds (Publicly accessible).
-------------------------------------------------------------------------------
"""

import time
import logging
from typing import List
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

try:
    from trendspyg import download_google_trends_rss
except ImportError:
    download_google_trends_rss = None

from src.api.schemas import EvaluatorScore
from src.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


class TrendingEvaluator(BaseEvaluator):
    """Evaluator that calculates real-time viral potential using Google Trends and CLIP."""

    evaluator_name: str = "Live_Trending_Viral_Potential"
    score_purpose: str = "Quantifies the image's potential for virality by aligning it with real-time global market interests."
    model_id: str = "openai/clip-vit-base-patch32"

    def __init__(self):
        # Cache mechanism to prevent IP banning from Google
        self._cached_trends: List[str] = []
        self._last_fetch_time: float = 0.0
        self._cache_ttl: float = 3600.0  # 1 hour TTL
        super().__init__()

    def load_model(self) -> None:
        """Initialize the CLIP Engine on the available device."""
        if download_google_trends_rss is None:
            logger.error("Library 'trendspyg' is missing. Please install it.")
            
        logger.info(f"Initializing CLIP Engine ({self.model_id}) on {self.device}...")
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        logger.info(f"{self.evaluator_name} Engine Ready.")

    def _fetch_live_visual_trends(self, country_code: str = 'US', top_n: int = 3) -> List[str]:
        """Fetch live trends with a TTL cache to prevent rate-limiting."""
        current_time = time.time()
        
        # Return cached trends if they are still valid
        if self._cached_trends and (current_time - self._last_fetch_time < self._cache_ttl):
            logger.debug("Using cached Google Trends.")
            return self._cached_trends

        logger.info(f"[CRAWLER] Fetching live Google Trends ({country_code})...")
        try:
            trends = download_google_trends_rss(geo=country_code)
            raw_trends = [t["trend"] for t in trends][:top_n]
            logger.info(f"[CRAWLER] Raw trends captured: {raw_trends}")

            # Engineer visual prompts
            self._cached_trends = [
                f"{trend} aesthetic, trending photography, visual concept, high quality"
                for trend in raw_trends
            ]
            self._last_fetch_time = current_time
            return self._cached_trends

        except Exception as e:
            logger.error(f"[CRAWLER ERROR] Failed to fetch trends: {e}")
            # Fallback static trends for system stability
            return [
                "cyberpunk neon aesthetic, trending",
                "minimalist nature photography, viral",
                "cinematic lighting, masterpiece"
            ]

    def evaluate(self, image_a: Image.Image, image_b: Image.Image, prompt: str) -> EvaluatorScore:
        """Score the images against live market trends using contrastive probability.
        
        Note: The original 'prompt' is ignored. The model aligns images against 
        dynamically generated trending text.
        """
        logger.debug(f"Evaluating with {self.evaluator_name}...")

        # 1. Fetch live data (or hit cache)
        live_trends = self._fetch_live_visual_trends(top_n=3)
        
        # 2. Append the negative baseline
        evaluation_texts = live_trends.copy()
        evaluation_texts.append("generic, boring, bad aesthetic, irrelevant, out of style")

        # 3. Batch Process both images simultaneously for maximum throughput
        images = [image_a, image_b]
        inputs = self.processor(
            text=evaluation_texts, 
            images=images, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Apply softmax to get contrastive probabilities (shape: [2, num_texts])
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()

        # Score is the sum of probabilities of the positive trending prompts
        # (Excluding the last index which is the negative baseline)
        score_a = round(float(sum(probs[0][:-1])), 4)
        score_b = round(float(sum(probs[1][:-1])), 4)

        preferred = "A" if score_a >= score_b else "B"
        confidence = round(abs(score_a - score_b), 4)

        return EvaluatorScore(
            evaluator_name=self.evaluator_name,
            purpose=self.score_purpose,
            score_a=score_a,
            score_b=score_b,
            preferred=preferred,
            confidence=confidence,
        )