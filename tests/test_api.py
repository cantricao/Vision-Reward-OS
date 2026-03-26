"""Unit tests for the FastAPI routing and aggregation logic.

We rigorously test the /evaluate/ab-test endpoint. Crucially, we mock the 
heavy ML models (Evaluators) to ensure tests execute in milliseconds and 
do not require GPU resources or model weight downloads in CI/CD pipelines.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import status

from src.api.main import app
from src.api.schemas import EvaluatorScore

# Initialize the FastAPI test client
client = TestClient(app)

# =============================================================================
# 1. MOCKING THE ML ENGINES
# =============================================================================
class MockEvaluatorA:
    """A dummy evaluator that always prefers Image A."""
    evaluator_name = "Mock_Pro_A"
    score_purpose = "Test purpose A"
    
    def evaluate(self, image_a, image_b, prompt):
        return EvaluatorScore(
            evaluator_name=self.evaluator_name,
            purpose=self.score_purpose,
            score_a=0.9,
            score_b=0.1,
            preferred="A",
            confidence=0.8
        )

class MockEvaluatorB:
    """A dummy evaluator that always prefers Image B."""
    evaluator_name = "Mock_Pro_B"
    score_purpose = "Test purpose B"
    
    def evaluate(self, image_a, image_b, prompt):
        return EvaluatorScore(
            evaluator_name=self.evaluator_name,
            purpose=self.score_purpose,
            score_a=0.2,
            score_b=0.8,
            preferred="B",
            confidence=0.6
        )

# Inject our fake evaluators into the FastAPI app before every test
@pytest.fixture(autouse=True)
def override_evaluators(monkeypatch):
    """Replace the real ML models with our lightweight mocks."""
    # We simulate a scenario where 2 models vote for A, and 1 votes for B.
    # Therefore, the overall aggregated winner MUST be 'A'.
    mock_panel = [MockEvaluatorA(), MockEvaluatorA(), MockEvaluatorB()]
    
    # Also mock the VLM judge's latest_reasoning attribute
    class MockVLM:
        latest_reasoning = "Mocked reasoning for CI/CD."
        
    monkeypatch.setattr("src.api.main._evaluators", mock_panel)
    monkeypatch.setattr("src.api.main.vlm_judge", MockVLM())

# =============================================================================
# 2. TEST CASES
# =============================================================================

def test_health_check():
    """Ensure the liveness probe is operational."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}


def test_evaluate_ab_test_success(dummy_b64_image):
    """Test the happy path: API receives valid Base64 images and returns a full report."""
    payload = {
        "prompt": "A beautiful test prompt",
        "image_a_b64": dummy_b64_image,
        "image_b_b64": dummy_b64_image
    }
    
    response = client.post("/evaluate/ab-test", json=payload)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    
    # 1. Check Aggregation Logic (Majority Vote)
    # Since 2 mocks voted A and 1 voted B, overall winner must be A.
    assert data["overall_winner"] == "A"
    
    # 2. Check Confidence Math (Average of 0.8, 0.8, 0.6 = 0.7333)
    assert round(data["overall_confidence"], 4) == 0.7333
    
    # 3. Check Evaluator Details
    assert len(data["evaluator_scores"]) == 3
    assert data["evaluator_scores"][0]["evaluator_name"] == "Mock_Pro_A"
    assert "Mocked reasoning" in data["reasoning_summary"]
    assert data["prompt_used"] == payload["prompt"]


def test_evaluate_ab_test_missing_image_a():
    """Test API validation: Reject requests missing both URL and Base64 for Image A."""
    payload = {
        "prompt": "Missing image A",
        "image_b_b64": "dummy_b64_string"
    }
    response = client.post("/evaluate/ab-test", json=payload)
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "image_a" in response.text.lower()


def test_evaluate_ab_test_invalid_base64():
    """Test the image decoding error handling with corrupt data."""
    payload = {
        "prompt": "Corrupt image data",
        "image_a_b64": "this_is_not_valid_base64_!@#",
        "image_b_b64": "this_is_also_invalid"
    }
    response = client.post("/evaluate/ab-test", json=payload)
    
    # Our API should catch the decode error and return a 400 Bad Request
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Failed to process image payload" in response.json()["detail"]
