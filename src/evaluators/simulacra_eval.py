import math
import os
import logging
import torch
import torch.nn as nn
import clip
from PIL import Image
from huggingface_hub import hf_hub_download

from src.evaluators.base import BaseEvaluator
from src.api.schemas import EvaluatorScore
from src.evaluators.shared_backbones import BackboneRegistry

logger = logging.getLogger(__name__)

# ==============================================================================
# SIMULACRA AESTHETIC LINEAR HEAD
# A simple linear layer that maps 512-dim CLIP ViT-B/16 embeddings to a 1-10 score.
# ==============================================================================
class SimulacraAestheticHead(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# ==============================================================================
# SIMULACRA EVALUATOR CLASS
# Integrates the standalone script into our FastAPI A/B testing pipeline.
# ==============================================================================
class SimulacraEvaluator(BaseEvaluator):
    evaluator_name: str = " "
    score_purpose: str = "Raw aesthetic quality scoring (1-10 scale)."

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = None
        self.aesthetic_head = None
        self.preprocess = None

    def load_model(self):
        """
        Loads the heavier OpenAI CLIP ViT-L/14 model and the matching 768-dim aesthetic head.
        This provides a more nuanced aesthetic evaluation at the cost of higher VRAM usage.
        """
        if self.aesthetic_head is not None:
            return

        try:
            # 1. Load CLIP Backbone (Upgraded to ViT-L/14)
            # logger.info(f"[{self.evaluator_name}] Loading OpenAI CLIP (ViT-L/14)...")
            self.clip_model, self.preprocess = BackboneRegistry.get_vit_l_14()
        
            # 2. Download Matching Aesthetic Head Weights
            # Fetching the specific checkpoint trained for 768-dimensional embeddings
            logger.info(f"[{self.evaluator_name}] Fetching Simulacra ViT-L/14 weights...")
            weights_path = hf_hub_download(
                repo_id="feizhengcong/video-stable-diffusion",
                filename="deforum-stable-diffusion/models/sac_public_2022_06_29_vit_l_14_linear.pth"
            )

            # 3. Initialize and Load the Linear Head
            logger.info(f"[{self.evaluator_name}] Initializing 768-dim aesthetic linear head...")
            # CRITICAL: input_dim MUST be 768 to match ViT-L/14 output
            self.aesthetic_head = SimulacraAestheticHead(input_dim=768) 
            
            self.aesthetic_head.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.aesthetic_head.to(self.device)
            self.aesthetic_head.eval()

            logger.info(f"[{self.evaluator_name}] Successfully warmed up and ready.")

        except Exception as e:
            logger.error(f"[{self.evaluator_name}] Initialization failed: {e}")
            self.clip_model = None
            self.aesthetic_head = None

    def _get_single_image_score(self, image: Image.Image) -> float:
        """
        Calculates the raw aesthetic score for a single PIL Image.
        """
        # Preprocess and move to device
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Extract normalized CLIP embeddings
            image_features = self.clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Pass through the linear aesthetic head
            score = self.aesthetic_head(image_features.float())

        return score.item()

    def evaluate(self, image_a: Image.Image, image_b: Image.Image, prompt: str) -> EvaluatorScore:
        """
        Evaluates two images and returns the comparison metrics for the API.
        """
        if self.clip_model is None:
            self.load_model()

        # Get raw scores for both images (scale generally 1 to 10)
        raw_score_a = self._get_single_image_score(image_a)
        raw_score_b = self._get_single_image_score(image_b)

        # Determine preference
        preferred = "A" if raw_score_a > raw_score_b else "B"

        max_score = max(raw_score_a, raw_score_b)
        exp_a = math.exp(raw_score_a - max_score)
        exp_b = math.exp(raw_score_b - max_score)
        
        prob_a = exp_a / (exp_a + exp_b)
        prob_b = exp_b / (exp_a + exp_b)
        
        # Confidence is strictly the probability of the winning choice [0.5 to 1.0]
        confidence = max(prob_a, prob_b)

        return EvaluatorScore(
            evaluator_name=self.evaluator_name,
            purpose=self.score_purpose,
            score_a=round(raw_score_a, 4),
            score_b=round(raw_score_b, 4),
            preferred=preferred,
            confidence=round(confidence, 4)
        )