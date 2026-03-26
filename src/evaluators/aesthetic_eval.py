"""LAION Aesthetic Predictor evaluator implementation.

Unlike PickScore or ImageReward which evaluate text-to-image alignment, 
the LAION Aesthetic Predictor evaluates *pure visual aesthetics*. It uses 
a lightweight Multi-Layer Perceptron (MLP) trained on top of CLIP embeddings 
to predict human aesthetic ratings (1 to 10 scale).

-------------------------------------------------------------------------------
# Score Utility:
Outputs a raw scalar score typically between 1.0 and 10.0. A score > 6.0 
usually indicates a high-quality, visually pleasing image, regardless of 
the generation prompt.

# Licensing Information:
- Codebase & MLP Weights: MIT License.
- Underlying CLIP Model: MIT License (OpenAI's ViT-L/14).
- Safe for commercial enterprise deployment.
-------------------------------------------------------------------------------

Repository: https://github.com/christophschuhmann/improved-aesthetic-predictor
"""

import os
import urllib.request
import logging
import torch
import torch.nn as nn
from PIL import Image

# Requires: pip install clip-anytorch
import clip

from src.api.schemas import EvaluatorScore
from src.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """The lightweight linear model architecture used by LAION Aesthetic."""
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AestheticEvaluator(BaseEvaluator):
    """Evaluator that wraps the LAION Aesthetic Predictor."""

    evaluator_name: str = "LAION_Aesthetic"
    score_purpose: str = "Assesses pure visual appeal, artistic composition, and professional photography aesthetics blind to the text prompt."
    
    # Official pre-trained weights for the ViT-L/14 CLIP model
    mlp_url: str = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
    mlp_path: str = "configs/laion_aesthetic.pth"

    def load_model(self) -> None:
        """Load the CLIP model and the custom aesthetic MLP head."""
        logger.info(f"Loading {self.evaluator_name} on {self.device}...")
        
        # 1. Load the underlying CLIP model
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        
        # 2. Download the aesthetic MLP weights if they don't exist
        os.makedirs(os.path.dirname(self.mlp_path), exist_ok=True)
        if not os.path.exists(self.mlp_path):
            logger.info("Downloading LAION Aesthetic weights...")
            urllib.request.urlretrieve(self.mlp_url, self.mlp_path)
            
        # 3. Initialize and load the MLP
        self.mlp = MLP(768)  # ViT-L/14 output dimension is 768
        state_dict = torch.load(self.mlp_path, map_location=self.device)
        self.mlp.load_state_dict(state_dict)
        self.mlp.to(self.device)
        self.mlp.eval()
        
        logger.info(f"{self.evaluator_name} loaded successfully.")

    def evaluate(self, image_a: Image.Image, image_b: Image.Image, prompt: str) -> EvaluatorScore:
        """Score a pair of images based strictly on visual aesthetics.
        
        Note: The 'prompt' argument is intentionally ignored in this model 
        since it predicts pure visual appeal independent of text.
        """
        logger.debug(f"Evaluating with {self.evaluator_name} (Ignoring prompt)")

        # Preprocess images for CLIP
        img_a_input = self.preprocess(image_a).unsqueeze(0).to(self.device)
        img_b_input = self.preprocess(image_b).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Extract image features using CLIP
            feat_a = self.clip_model.encode_image(img_a_input)
            feat_b = self.clip_model.encode_image(img_b_input)
            
            # Normalize features
            feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
            feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)
            
            # Pass through the Aesthetic MLP to get the scalar score
            score_a = round(self.mlp(feat_a.float()).item(), 4)
            score_b = round(self.mlp(feat_b.float()).item(), 4)

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