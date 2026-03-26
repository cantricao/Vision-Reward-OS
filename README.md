# 👁️ Vision-Reward-OS

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production_Ready-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Optimized-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

An enterprise-grade, low-latency microservice for evaluating Generative AI images using an ensemble of State-of-the-Art (SOTA) Human Preference models and Universal Vision Language Model (VLM) judges.

---

## 🛑 The Problem

Evaluating the quality of Generative AI images is a fundamentally hard problem. Traditional metrics (FID, CLIP score) often fail to capture what humans actually prefer. Teams building text-to-image pipelines are left without a reliable, automated signal to:

- Compare two candidate images in an A/B test (e.g., Midjourney vs. Flux).
- Detect visual regressions when fine-tuning or swapping model checkpoints.
- Rank outputs at scale without expensive and slow human annotation.

A single model is rarely sufficient. Different evaluators capture different aspects of quality (aesthetics, text alignment, structural coherence, viral potential). To achieve human-level QA, their signals must be aggregated intelligently.

---

## 💡 The Solution

**Vision-Reward-OS** is a composable, extensible evaluation microservice that aggregates signals from **8 distinct SOTA evaluators** into a single, highly interpretable JSON report. 

It leverages both mathematical alignment (CLIP-based reward models) and Agentic reasoning (VLM Chain-of-Thought) to provide rigorous, auditable QA for Generative AI outputs.

### 🏆 The Judging Panel (Integrated Engines)

| Evaluator | Type | Primary Purpose |
|---|---|---|
| **PickScore** | Human Preference | Measures general commercial aesthetic and human preference based on the prompt. |
| **ImageReward** | Semantic Alignment | Evaluates deep text-to-image alignment and penalizes structural defects. |
| **HPS v2.1** | Aesthetic Calibration | Calibrates the image against large-scale human aesthetic preference datasets. |
| **LAION Aesthetic** | Pure Aesthetic | Assesses pure visual appeal and artistic composition, blind to the text prompt. |
| **Simulacra Aesthetic** | Artifact Detection | Detects AI-specific generation artifacts, structural incoherence, and mangled geometry. |
| **Google Trends + CLIP** | Viral Potential | Quantifies the image's potential for virality by aligning it with real-time global market interests. |
| **Kwai-Kolors MPS** | Multi-Dimensional | Provides a balanced multi-dimensional assessment of both visual quality and prompt adherence. |
| **Universal VLM Judge** | Chain-of-Thought | Delivers a definitive human-like verdict and reasoning based on strict commercial standards. (Supports OpenAI, Gemini, and Local VLMs). |

The service exposes a clean REST API built with **FastAPI**, following strict **OOP principles** with a pluggable `BaseEvaluator` interface.

---

## 🏗️ Architecture Design

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Microservice                           │
│  POST /evaluate/ab-test                                                  │
│                                                                          │
│  ┌──────────────┐     ┌──────────────────────────────────────────┐       │
│  │ InputImages  │────▶│          Image Decoding Engine           │       │
│  │ (URL/Base64) │     │   (Decodes to PIL.Image once in RAM)     │       │
│  └──────────────┘     └─────────────────┬────────────────────────┘       │
│                                         │                                │
│          ┌──────────────────────────────┼─────────────────────────────┐  │
│          ▼                              ▼                             ▼  │
│ ┌──────────────────┐           ┌──────────────────┐          ┌─────────┐ │
│ │ Alignment Models │           │ Aesthetic Models │          │ Advanced│ │
│ ├──────────────────┤           ├──────────────────┤          ├─────────┤ │
│ │ 1. PickScore     │           │ 5. LAION Aesth.  │          │ 7.Trend │ │
│ │ 2. ImageReward   │           │ 6. Simulacra     │          │ 8. VLM  │ │
│ │ 3. HPS v2.1      │           └────────┬─────────┘          │  Judge  │ │
│ │ 4. Kwai MPS      │                    │                    └────┬────┘ │
│ └────────┬─────────┘                    │                         │      │
│          │                              │                         │      │
│          └──────────────────────────────┼─────────────────────────┘      │
│                                         ▼                                │
│                            ┌────────────────────────┐                    │
│                            │ MultiDimensionalReport │                    │
│                            │   (Aggregated JSON)    │                    │
│                            └────────────────────────┘                    │
└──────────────────────────────────────────────────────────────────────────┘
```
## Key Architectural Decisions:

---

* **Decode-Once Principle:** Images are downloaded/decoded at the API layer and passed as RAM-resident PIL.Image objects to prevent I/O bottlenecks.

* **Vendor-Agnostic VLM:** The Universal VLM Judge uses the standard OpenAI protocol. By changing environment variables, you can seamlessly swap between GPT-4o, Gemini 2.5 Pro, or a local Qwen2-VL running on vLLM.

* **Self-Documenting API:** Every evaluator defines a score_purpose which is directly injected into the final JSON output, making the response instantly understandable for frontend teams.

* **Fail-Safe Iteration:** If one model fails (e.g., OOM), the pipeline gracefully logs the error and continues with the remaining judges.

---

## ⚡ Quick Start
### 1. Master Environment Setup (Crucial)
Vision-Reward-OS relies on a specific matrix of PyTorch and Transformer versions to prevent conflicts between older models (ImageReward) and newer tools.

Run the automated setup script to install all dependencies and clone necessary remote repositories (e.g., Kwai-Kolors MPS):

```bash
python scripts/setup_env.py
```
_(Note: If you are running locally without the setup script, refer to requirements.txt and ensure you run `git clone https://github.com/Kwai-Kolors/MPS.git` into your working directory)._


### 2. Configure Environment Variables
To enable the Universal VLM Judge, configure your API keys in the terminal.

* For OpenAI Models:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export VLM_MODEL_NAME="gpt-4o-mini"
```

* For Google Gemini Models:

```bash
export OPENAI_API_KEY="your-gemini-api-key"
export OPENAI_BASE_URL="[https://generativelanguage.googleapis.com/v1beta/openai/](https://generativelanguage.googleapis.com/v1beta/openai/)"
export VLM_MODEL_NAME="gemini-1.5-pro"
```

* For Local Open-Source Models (via LM Studio / vLLM):

```bash
export OPENAI_API_KEY="sk-dummy"
export OPENAI_BASE_URL="http://localhost:1234/v1"
export VLM_MODEL_NAME="qwen2-vl-7b-instruct"
```

### 3. Run the Development Server
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Interactive API Docs (Swagger UI)
Navigate to http://127.0.0.1:8000/docs to access the auto-generated Swagger UI. You can test the A/B evaluation endpoint directly from your browser.

---

## 📡 API Usage Example
Send a POST request to /evaluate/ab-test:

```bash
curl -X POST "http://localhost:8000/evaluate/ab-test" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A futuristic cyberpunk city with flying cars at sunset, high quality, 4k",
    "image_a_url": "[https://example.com/generated_image_v1.jpg](https://example.com/generated_image_v1.jpg)",
    "image_b_url": "[https://example.com/generated_image_v2.jpg](https://example.com/generated_image_v2.jpg)"
  }'
```
The system will return a detailed MultiDimensionalReport containing individual engine scores, an aggregated overall winner, and a brutal, commercially-driven reasoning summary from the VLM Art Director.

---

## 📂 Project Structure
```plaintext
Vision-Reward-OS/
├── configs/                # Automatically stores downloaded .pth weights
├── src/
│   ├── api/
│   │   ├── main.py         # FastAPI application and Orchestrator
│   │   └── schemas.py      # Pydantic request/response models with embedded purposes
│   └── evaluators/
│       ├── base.py         # Abstract BaseEvaluator interface
│       ├── pickscore_eval.py 
│       ├── imagereward_eval.py
│       ├── hps_eval.py
│       ├── aesthetic_eval.py
│       ├── simulacra_eval.py
│       ├── trending_eval.py
│       ├── mps_eval.py
│       └── vlm_judge_eval.py
├── .env.example
├── Dockerfile
├── requirements.txt
└── README.md
```
---

## 📜 License
MIT License