"""Universal VLM (Vision Language Model) Judge evaluator implementation.

This module utilizes the OpenAI SDK standard to interface with ANY compatible 
Vision Language Model. By avoiding vendor-specific SDKs (like google.generativeai), 
this architecture prevents vendor lock-in and seamlessly supports:
1. Cloud Models (e.g., GPT-4o, GPT-4o-mini, Claude via proxies).
2. Local Open-Source Models (e.g., Qwen2-VL, LLaVA, InternVL) served via 
   vLLM, Ollama, or LM Studio running on localhost.

-------------------------------------------------------------------------------
# Score Utility:
Outputs a deterministic preference ("A" or "B") and a confidence score based on 
brutal, commercially-driven Chain-of-Thought (CoT) reasoning, parsing a strict 
JSON output.

# Licensing Information:
- Codebase License: MIT License.
- API Usage: Depends entirely on the backend configured via environment variables.
-------------------------------------------------------------------------------
"""

import os
import io
import json
import base64
import logging
from typing import Optional
from PIL import Image

# Require installing: pip install openai
from openai import OpenAI

from src.api.schemas import EvaluatorScore
from src.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


class VLMJudgeEvaluator(BaseEvaluator):
    """Evaluator that wraps any OpenAI-API-compatible Multimodal LLM."""

    evaluator_name: str = "Universal_VLM_Judge"
    score_purpose: str = "Delivers a definitive human-like verdict based on strict commercial standards and multi-dimensional visual analysis."
    
    def __init__(self):
        self.latest_reasoning: Optional[str] = None
        super().__init__()

    def load_model(self) -> None:
        """Initialize the Universal OpenAI-compatible client.
        
        Environment Variables needed:
        - VLM_MODEL_NAME: The model ID (e.g., "gemini-1.5-pro", "qwen2-vl-7b-instruct").
        - OPENAI_API_KEY: Your API key.
        - OPENAI_BASE_URL: To use Gemini, point to Google's OpenAI-compatible endpoint:
                           "https://generativelanguage.googleapis.com/v1beta/openai/"
        """
        api_key = os.getenv("OPENAI_API_KEY", "sk-dummy-key-for-local")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        self.model_name = os.getenv("VLM_MODEL_NAME", "gemini-1.5-pro")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        endpoint_type = f"Endpoint ({base_url})" if base_url else "OpenAI Cloud"
        logger.info(f"Initialized {self.evaluator_name} connecting to {endpoint_type} (Model: {self.model_name}).")

    def _encode_image_base64(self, image: Image.Image) -> str:
        """Convert a PIL Image to a base64 string formatted for the API payload."""
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def evaluate(self, image_a: Image.Image, image_b: Image.Image, prompt: str) -> EvaluatorScore:
        """Prompt the Universal VLM to compare two images and return a JSON decision."""
        self.latest_reasoning = None
        
        if not prompt:
            prompt = "[No specific prompt provided. Judge based on general visual quality and aesthetics]"

        logger.debug(f"Evaluating with {self.evaluator_name} (Model: {self.model_name}) for prompt: '{prompt[:30]}...'")

        # =======================================================================
        # THE SOTA PREFPO PROMPT (Extracted from the research pipeline)
        # =======================================================================
        system_instruction = f"""
        You are a RUTHLESS, elite AI Art Director. You are presented with two generated images (Image A and Image B) and the original text prompt.
        
        User Prompt: "{prompt}"

        Your task is to comparatively evaluate the two images across three specific dimensions:
        1. Alignment: How well does the image match the text prompt?
        2. Coherence: How logically consistent is it? (Punish AI glitches, extra fingers, distorted geometry).
        3. Style: How aesthetically appealing and commercially viable is it?

        SCORING RULE (ZERO-SUM):
        For EACH dimension, you must assign a relative score to Image A and Image B. 
        - Each score is a float between 0.0 and 1.0.
        - The scores for Image A and Image B MUST sum to exactly 1.0 for each dimension (e.g., A: 0.7, B: 0.3).
        - If an image is completely ruined by artifacts, give it 0.1 and the other 0.9. DO NOT be polite.

        OUTPUT FORMAT:
        You MUST output a valid JSON object strictly matching this schema. Write the 'rationale' first to ensure Chain-of-Thought reasoning.
        {{
            "rationale": "<Step 1: Brutal, analytical critique comparing both images across the 3 dimensions. Max 3 sentences.>",
            "preference": "<Step 2: Choose either 'A' or 'B' based on the overall winner>",
            "scores": {{
                "alignment": {{"A": <float>, "B": <float>}},
                "coherence": {{"A": <float>, "B": <float>}},
                "style": {{"A": <float>, "B": <float>}}
            }}
        }}
        """

        b64_img_a = self._encode_image_base64(image_a)
        b64_img_b = self._encode_image_base64(image_b)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Evaluate the design using the defined business pillars and output strictly valid JSON."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img_a}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img_b}"}}
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=300
            )
            
            # Parse the structured JSON output
            raw_content = response.choices[0].message.content
            result_data = json.loads(raw_content)
            
            preferred = result_data.get("preference", "A").upper()
            if preferred not in ["A", "B"]:
                preferred = "A"
                
            self.latest_reasoning = result_data.get("rationale", "Preferred based on multi-dimensional analysis.")
            
            # Extract scores safely
            scores = result_data.get("scores", {})
            align_a = scores.get("alignment", {}).get("A", 0.5)
            cohere_a = scores.get("coherence", {}).get("A", 0.5)
            style_a = scores.get("style", {}).get("A", 0.5)
            
            align_b = scores.get("alignment", {}).get("B", 0.5)
            cohere_b = scores.get("coherence", {}).get("B", 0.5)
            style_b = scores.get("style", {}).get("B", 0.5)

            # Average the 3 dimensions to get the final unified score
            score_a = round((align_a + cohere_a + style_a) / 3.0, 4)
            score_b = round((align_b + cohere_b + style_b) / 3.0, 4)
            
            # Confidence is the margin of victory in the unified score
            confidence = round(abs(score_a - score_b), 4)

            return EvaluatorScore(
                evaluator_name=self.evaluator_name,
                purpose=self.score_purpose,
                score_a=score_a,
                score_b=score_b,
                preferred=preferred,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"VLM Judge evaluation failed: {e}")
            self.latest_reasoning = "VLM evaluation failed due to an API or parsing error."
            return EvaluatorScore(
                evaluator_name=self.evaluator_name,
                purpose=self.score_purpose,
                score_a=0.0,
                score_b=0.0,
                preferred="A",
                confidence=0.0,
            )