# Vision-Reward-OS

An enterprise-grade, low-latency microservice for evaluating Generative AI images using SOTA Human Preference models and VLM judges.

---

## The Problem

Evaluating the quality of Generative AI images is a fundamentally hard problem. Traditional metrics (FID, CLIP score) often fail to capture what humans actually prefer. Teams building text-to-image pipelines are left without a reliable, automated signal to:

- Compare two candidate images in an A/B test (e.g., SDXL vs. Flux)
- Detect regressions when fine-tuning or swapping model checkpoints
- Rank outputs at scale without expensive human annotation

A single model is rarely sufficient — different evaluators capture different aspects of quality (aesthetics, alignment to prompt, coherence), and their signals must be aggregated intelligently.

---

## The Solution

**Vision-Reward-OS** is a composable, extensible evaluation microservice that aggregates signals from multiple SOTA Human Preference models and VLM-as-a-judge pipelines into a single, interpretable report.

**Supported evaluators (planned/integrated):**

| Evaluator | Type | What it measures |
|---|---|---|
| [PickScore](https://github.com/yuvalkirstain/PickScore) | Human Preference Model | Human preference over LAION aesthetics |
| [HPS v2 / v3](https://github.com/tgxs002/HPSv2) | Human Preference Model | Human preference on diverse prompts |
| [ImageReward](https://github.com/THUDM/ImageReward) | Human Preference Model | Text-image alignment + aesthetics |
| Gemini VLM Judge | VLM-as-a-Judge | Reasoning-based holistic quality |
| Qwen VLM Judge | VLM-as-a-Judge | Multi-lingual reasoning quality |

The service exposes a clean REST API built with **FastAPI**, following **OOP principles** with a pluggable `BaseEvaluator` interface. Complex multi-step reasoning flows (e.g., chain-of-thought VLM evaluation) are orchestrated via **LangGraph**.

---

## Architecture

```
┌────────────────────────────────────────────────────┐
│                   FastAPI Service                  │
│  POST /evaluate/ab-test                            │
│  ┌──────────────┐    ┌──────────────────────────┐  │
│  │ InputImages  │───▶│  EvaluationOrchestrator  │  │
│  │ (Pydantic)   │    │  (LangGraph pipeline)    │  │
│  └──────────────┘    └──────────┬───────────────┘  │
│                                 │                  │
│          ┌──────────────────────┼──────────────┐   │
│          ▼                      ▼              ▼   │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────┐  │
│  │ PickScore    │  │  HPS v2/v3       │  │ VLM  │  │
│  │ Evaluator    │  │  Evaluator       │  │Judge │  │
│  └──────────────┘  └──────────────────┘  └──────┘  │
│          │                      │              │   │
│          └──────────────────────┴──────────────┘   │
│                                 │                  │
│                    ┌────────────▼───────────┐      │
│                    │ MultiDimensionalReport │      │
│                    │ (Pydantic response)    │      │
│                    └────────────────────────┘      │
└────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **Plugin architecture**: Every evaluator inherits from `BaseEvaluator` and can be enabled/disabled via `configs/eval_config.yaml`.
- **Async-first**: FastAPI endpoints use `async def` for non-blocking I/O.
- **Config-driven**: Model weights, thresholds, and enabled evaluators are all controlled through YAML config.
- **OOP principles**: Abstract base classes enforce a consistent interface across all evaluators.

---

## Quick Start

### Using Docker (Recommended)

**1. Build the image:**
```bash
docker build -t vision-reward-os:latest .
```

**2. Run the container:**
```bash
docker run -p 8000:8000 vision-reward-os:latest
```

**3. Test the endpoint:**
```bash
curl -X POST "http://localhost:8000/evaluate/ab-test" \
  -H "Content-Type: application/json" \
  -d '{
    "image_a_url": "https://example.com/image_a.jpg",
    "image_b_url": "https://example.com/image_b.jpg",
    "prompt": "a cat sitting on a red couch"
  }'
```

**4. Interactive API docs:**

Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) for the auto-generated Swagger UI.

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the development server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Project Structure

```
Vision-Reward-OS/
├── configs/
│   └── eval_config.yaml        # Model weights, thresholds, enabled evaluators
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI application and route definitions
│   │   └── schemas.py          # Pydantic request/response models
│   └── evaluators/
│       ├── __init__.py
│       ├── base.py             # Abstract BaseEvaluator interface
│       └── pickscore_eval.py   # PickScore evaluator implementation
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## License

[MIT License](LICENSE)
