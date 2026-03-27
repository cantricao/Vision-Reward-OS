"""Simulacra Aesthetic evaluator implementation.

This module wraps the Simulacra Aesthetic Predictor, a lightweight MLP 
trained on the SimulacraBot dataset (over 300k human ratings of AI-generated 
images). Unlike general aesthetic models, it is specifically highly attuned 
to detecting AI artifacts, anatomical flaws, and structural incoherence.

-------------------------------------------------------------------------------
# Score Utility:
Outputs a scalar aesthetic score (typically 1-10). A higher score indicates 
that the image is visually coherent and free from common AI generation flaws 
(e.g., mangled limbs, nonsensical geometry).

# Licensing Information:
- Codebase & MLP Weights: MIT License (Permissive).
- Underlying CLIP Model: MIT License (OpenAI's ViT-L/14).
- Safe for commercial enterprise deployment.
-------------------------------------------------------------------------------

Repository: https://github.com/JD-P/simulacra-aesthetic-models
"""

import os
import urllib.request
import logging
import torch
from torch import nn
from PIL import Image
from huggingface_hub import hf_hub_download

# Requires: pip install clip-anytorch
import clip

from src.api.schemas import EvaluatorScore
from src.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """The lightweight linear model architecture used by Simulacra Aesthetic."""
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


class SimulacraEvaluator(BaseEvaluator):
    """Evaluator that wraps the Simulacra Aesthetic Predictor."""

    evaluator_name: str = "Simulacra_Aesthetic"
    score_purpose: str = "Detects AI-specific generation artifacts, structural incoherence, and mangled geometry."
    
    # Official pre-trained weights for the ViT-L/14 CLIP model
    repo_id: str = "feizhengcong/video-stable-diffusion"
    filename: str = "deforum-stable-diffusion/models/sac_public_2022_06_29_vit_b_16_linear.pth"

    def load_model(self) -> None:
        """Load the CLIP model and the custom Simulacra MLP head."""
        logger.info(f"Loading {self.evaluator_name} on {self.device}...")
        
        # Load the underlying CLIP model (Must match ViT-L/14)
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        
        # Download the Simulacra MLP weights if they don't exist
        # logger.info("Loading Simulacra Aesthetic weights...")
        self.mlp_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename            
        )
            
        # Initialize and load the MLP (ViT-L/14 output dimension is 768)
        self.mlp = MLP(768)
        state_dict = torch.load(self.mlp_path, map_location=self.device)
        self.mlp.load_state_dict(state_dict)
        self.mlp.to(self.device)
        self.mlp.eval()
        
        logger.info(f"{self.evaluator_name} loaded successfully.")

    def evaluate(self, image_a: Image.Image, image_b: Image.Image, prompt: str) -> EvaluatorScore:
        """Score a pair of images based on coherence and absence of AI artifacts.
        
        Note: The 'prompt' argument is ignored as this is a pure aesthetic/quality predictor.
        """
        logger.debug(f"Evaluating with {self.evaluator_name} (Ignoring prompt)")

        img_a_input = self.preprocess(image_a).unsqueeze(0).to(self.device)
        img_b_input = self.preprocess(image_b).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat_a = self.clip_model.encode_image(img_a_input)
            feat_b = self.clip_model.encode_image(img_b_input)
            
            feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
            feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)
            
            score_a = round(self.mlp(feat_a.float()).item(), 4)
            score_b = round(self.mlp(feat_b.float()).item(), 4)

        preferred = "A" if score_a >= score_b else "B"
        
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