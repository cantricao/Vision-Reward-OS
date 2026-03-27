import os
import sys
import subprocess
import logging
import torch
import gdown
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.clip.modeling_clip import (
    CLIPTextTransformer, 
    CLIPConfig, 
    CLIPTextConfig, 
    CLIPVisionConfig
)

from src.api.schemas import EvaluatorScore
from src.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)

class MPSEvaluator(BaseEvaluator):
    evaluator_name: str = "Kwai-Kolors_MPS"
    score_purpose: str = "Multi-dimensional aesthetic preference scoring."

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        
        # Paths
        self.root_dir = os.getcwd()
        self.mps_repo_path = os.path.join(self.root_dir, "MPS")
        self.weights_dir = os.path.join(self.root_dir, "configs", "weights")
        self.ckpt_path = os.path.join(self.weights_dir, "MPS_overall_checkpoint.pth")
        
        # IDs and Prompts
        self.repo_url = "https://github.com/Kwai-Kolors/MPS.git"
        self.gdrive_id = "17qrK_aJkVNM75ZEvMEePpLj6L867MLkN"
        self.condition_prompt = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things."

    def _tokenize(self, caption: str):
        return self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        
    def _process_image(self, image) -> torch.Tensor:
        """
        Helper method to process a PIL image into a tensor format 
        that the MPS model understands, cast to FP16.
        """
        # Ensure the image is in RGB format before processing
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Use the HuggingFace image processor assigned during load_model
        inputs = self.image_processor(images=image, return_tensors="pt")
        
        # CRITICAL FIX: Extract pixel values, cast to FP16, and move to GPU
        pixel_values = inputs['pixel_values'].to(self.device)
        
        return pixel_values
    
    def _prepare_environment(self):
        """Automates 'git clone' and 'gdown' weight retrieval."""
        # 1. Clone Repo if missing
        if not os.path.exists(self.mps_repo_path):
            logger.info(f"[{self.evaluator_name}] Cloning MPS repository...")
            subprocess.run(["git", "clone", self.repo_url, "-q"], check=True)
        
        # 2. Inject to sys.path
        if self.mps_repo_path not in sys.path:
            sys.path.append(self.mps_repo_path)

        # 3. Download Weights if missing
        os.makedirs(self.weights_dir, exist_ok=True)
        if not os.path.exists(self.ckpt_path):
            logger.warning(f"[{self.evaluator_name}] Downloading 5GB checkpoint. Please wait...")
            url = f"https://drive.google.com/uc?id={self.gdrive_id}"
            gdown.download(url, self.ckpt_path, quiet=False)

    def load_model(self):
        """Bootstraps the environment and loads the patched CLIP model."""
        if self.model is not None:
            return

        # Step 1: Git/Download
        self._prepare_environment()

        # Step 2: Initialize Processors
        logger.info(f"[{self.evaluator_name}] Patching Transformers and loading model...")
        proc_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.image_processor = AutoProcessor.from_pretrained(proc_path)
        self.tokenizer = AutoTokenizer.from_pretrained(proc_path, trust_remote_code=True)

        # Step 3: Global Class Patches (The Tri-Protocol)
        CLIPTextTransformer.eos_token_id = self.tokenizer.eos_token_id
        CLIPConfig._output_attentions = False
        CLIPTextConfig._attn_implementation_internal = "eager"
        CLIPVisionConfig._attn_implementation_internal = "eager"

        # Step 4: Load PTH
        self.model = torch.load(self.ckpt_path, weights_only=False, map_location=self.device)
        
        if hasattr(self.model, 'text_model'):
            self.model.text_model.eos_token_id = self.tokenizer.eos_token_id

        self.model.eval().to(self.device)
        logger.info(f"[{self.evaluator_name}] Model loaded successfully.")

    def evaluate(self, image_a: Image.Image, image_b: Image.Image, prompt: str) -> EvaluatorScore:
        """Executes the scoring pipeline based on the human preference logic."""
        if not self.model:
            raise RuntimeError(f"{self.evaluator_name} is not loaded.")

        logger.info(f"[{self.evaluator_name}] Scoring pair for: '{prompt}'")
        
        try:
            with torch.no_grad():
                # Prepare inputs
                tensor_a = self._process_image(image_a).to(self.device)
                tensor_b = self._process_image(image_b).to(self.device)
                image_inputs = torch.cat([tensor_a, tensor_b], dim=0)
                
                text_inputs = self._tokenize(prompt).to(self.device)
                cond_inputs = self._tokenize(self.condition_prompt).to(self.device)

                # Forward pass
                text_f, img_0_f, img_1_f = self.model(text_inputs, image_inputs, cond_inputs)
                
                # Normalize
                img_0_f = img_0_f / img_0_f.norm(dim=-1, keepdim=True)
                img_1_f = img_1_f / img_1_f.norm(dim=-1, keepdim=True)
                text_f = text_f / text_f.norm(dim=-1, keepdim=True)
                
                # Calculate scores using logit scale
                scale = self.model.logit_scale.exp()
                score_a = scale * torch.diag(torch.einsum('bd,cd->bc', text_f, img_0_f))
                score_b = scale * torch.diag(torch.einsum('bd,cd->bc', text_f, img_1_f))
                
                probs = torch.softmax(torch.stack([score_a, score_b], dim=-1), dim=-1)[0]
                
            p_a, p_b = float(probs[0]), float(probs[1])
            
            return EvaluatorScore(
                evaluator_name=self.evaluator_name,
                purpose=self.score_purpose,
                score_a=p_a,
                score_b=p_b,
                preferred="A" if p_a > p_b else "B",
                confidence=abs(p_a - p_b)
            )
        except Exception as e:
            logger.error(f"[{self.evaluator_name}] Evaluation error: {e}")
            raise