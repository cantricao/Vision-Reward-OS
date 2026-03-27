import pytest
import sys
import types
from unittest.mock import MagicMock

# ==============================================================================
# 1. THE GHOST MOCK PROTOCOL (Fixes the wandb.__spec__ error)
# We create a fake module type to satisfy the importlib checks in 'transformers' 
# and 'accelerate' without actually initializing the buggy wandb package.
# ==============================================================================
def mock_package(name):
    mock_mod = types.ModuleType(name)
    mock_mod.__spec__ = MagicMock() # <== The secret sauce that fixes Transformers
    sys.modules[name] = mock_mod
    return mock_mod

# Block wandb and its sub-modules early in the test lifecycle
mock_package("wandb")
mock_package("wandb.sdk")
mock_package("wandb.proto")
mock_package("wandb.proto.wandb_telemetry_pb2")

# ==============================================================================
# 2. IMMEDIATE CLASS-LEVEL PATCHING (Fixes HTTP 404 and heavy downloads)
# We MUST override these methods directly on the class BEFORE the FastAPI app 
# is imported. Fixtures are too slow and execute after the models try to load.
# ==============================================================================
from src.evaluators.base import BaseEvaluator
from src.evaluators.simulacra_eval import SimulacraEvaluator
from src.evaluators.mps_eval import MPSEvaluator
from src.evaluators.imagereward_eval import ImageRewardEvaluator
from src.api.schemas import EvaluatorScore

# Directly monkey-patch the base classes to skip network/GPU operations
BaseEvaluator.load_model = lambda self: print(f"[MOCK] Skipping load for {self.__class__.__name__}")
SimulacraEvaluator.load_model = lambda self: None
MPSEvaluator.load_model = lambda self: None
ImageRewardEvaluator.load_model = lambda self: None

# Force the evaluate method to return a dummy score instantly
def mock_evaluate(self, image_a, image_b, prompt):
    return EvaluatorScore(
        evaluator_name=getattr(self, "evaluator_name", "Mocked_Evaluator"),
        purpose="Unit Testing Isolation",
        score_a=0.9,
        score_b=0.1,
        preferred="A",
        confidence=0.8
    )

BaseEvaluator.evaluate = mock_evaluate

# ==============================================================================
# 3. FASTAPI TEST CLIENT
# ==============================================================================
@pytest.fixture
def client():
    """
    Provides a TestClient for the FastAPI app. 
    Because we patched the Evaluator classes above, the app will initialize safely.
    """
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)