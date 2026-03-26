"""Pytest fixtures for Vision-Reward-OS.

This file contains reusable testing components, such as dummy images 
and base64 encoders, to ensure tests run instantly without network I/O.
"""

import io
import base64
import pytest
from PIL import Image

@pytest.fixture
def dummy_pil_image() -> Image.Image:
    """Generate a tiny 10x10 red image in memory."""
    return Image.new("RGB", (10, 10), color="red")

@pytest.fixture
def dummy_b64_image(dummy_pil_image: Image.Image) -> str:
    """Encode the dummy PIL image to a Base64 string."""
    buffered = io.BytesIO()
    dummy_pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
