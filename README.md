# 👁️ Vision-Reward-OS | Enterprise Image A/B Evaluator

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production_Ready-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Optimized-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

An enterprise-grade, low-latency framework designed to score, compare, and analyze AI-generated images. Instead of relying on a single metric, Vision-Reward-OS orchestrates a weighted ensemble of 8 State-of-the-Art (SOTA) evaluators, ranging from mathematical aesthetic scorers to reasoning Vision-Language Models (VLMs) to determine the true winner of an A/B test.

---

## 🛑 The Problem

Evaluating the quality of Generative AI images is fundamentally difficult. Traditional metrics (FID, raw CLIP scores) fail to capture what humans actually prefer. Teams building text-to-image pipelines lack a reliable, automated signal to:

- A/B Test Models: Objectively compare outputs
- Detect Regressions: Spot visual degradation when fine-tuning checkpoints.
- Rank Outputs: Sort generations at scale without expensive and slow human annotation.

A single model is rarely sufficient. A visually stunning image might completely ignore the text prompt, while a perfectly aligned image might look hideous. To achieve human-level QA, multiple signals must be intelligently aggregated.

---

## 💡 The Solution: Enterprise Weighted Voting

**Vision-Reward-OS** is a composable evaluation microservice that prevents "dumb" aesthetic models from outvoting "smart" semantic judges.

It aggregates signals from 8 distinct evaluators using an Enterprise Weighted Voting System, prioritizing prompt fidelity and human preference over raw pixel aesthetics. It outputs a highly interpretable JSON report and an interactive UI dashboard.

### 🏆 The Judging Panel (3 Phases of Evaluation)
The system categorizes evaluators into three evolutionary phases of Generative AI assessment, assigning higher voting weights to more advanced models:

| Weight | Evaluator | Phase | Primary Purpose |
| :---: | :--- | :--- | :--- |
| **0.5x** | **LAION Aesthetic** | Phase 1: Pure Aesthetics | Assesses pure visual appeal and artistic composition, *blind to the text prompt*. |
| **0.5x** | **Simulacra Aesthetic** | Phase 1: Pure Aesthetics | Evaluates raw aesthetic quality based on community-driven AI art preferences. |
| **0.8x** | **Trending Score** | Phase 1.5: Keyword Match | Quantifies viral potential by aligning with keywords like *"trending on artstation"*. |
| **0.8x** | **Kwai-Kolors MPS** | Phase 1.5: Multi-Dimensional | Provides a balanced assessment across clarity, contrast, and color. |
| **1.5x** | **HPS v2.1** | Phase 2: Human Preference | Calibrates the image against large-scale human aesthetic preference datasets. |
| **1.5x** | **ImageReward** | Phase 2: Semantic Alignment | Evaluates deep text-to-image alignment and penalizes anatomical/structural defects. |
| **2.0x** | **PickScore** | Phase 2: Advanced Preference | Highly advanced model predicting which image human users would prefer to download. |
| **3.0x** | **Universal VLM Judge** | Phase 3: Logic & Reasoning | Delivers a definitive verdict and Chain-of-Thought reasoning. (Supports GPT-4o, Gemini, Claude). |
---

## 🏗️ Architecture Design

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                           Vision-Reward-OS                               │
│                                                                          │
│  ┌──────────────┐     ┌──────────────────────────────────────────┐       │
│  │ InputImages  │───▶ │           Image Decoding Engine          │       │
│  │ (URL/Base64) │     │   (Decodes to PIL.Image once in RAM)     │       │
│  └──────────────┘     └─────────────────┬────────────────────────┘       │
│                                         │                                │
│          ┌──────────────────────────────┼─────────────────────────────┐  │
│          ▼                              ▼                             ▼  │
│ ┌──────────────────┐           ┌──────────────────┐          ┌─────────┐ │
│ │ Phase 1 Models   │           │ Phase 2 Models   │          │ Phase 3 │ │
│ │ (Low Weight)     │           │ (Medium Weight)  │          │ (High)  │ │
│ ├──────────────────┤           ├──────────────────┤          ├─────────┤ │
│ │ - LAION          │           │ - ImageReward    │          │ - VLM   │ │
│ │ - Simulacra      │           │ - PickScore      │          │   Judge │ │
│ │ - Trending       │           │ - HPS v2.1       │          │         │ │
│ │ - Kwai MPS       │           │                  │          │         │ │
│ └────────┬─────────┘           └────────┬─────────┘          └────┬────┘ │
│          │                              │                         │      │
│          └──────────────────────────────┼─────────────────────────┘      │
│                                         ▼                                │
│                            ┌────────────────────────┐                    │
│                            │    Weighted Voting     │                    │
│                            │   Aggregator Engine    │                    │
│                            └────────────┬───────────┘                    │
│                                         ▼                                │
│                            ┌────────────────────────┐                    │
│                            │ MultiDimensionalReport │                    │
│                            │  (JSON & Gradio UI)    │                    │
│                            └────────────────────────┘                    │
└──────────────────────────────────────────────────────────────────────────┘
```
## Key Architectural Decisions:

---

* **Decode-Once Principle:** Images are downloaded/decoded at the API layer and passed as RAM-resident `PIL.Image` objects to prevent I/O bottlenecks.

* **Vendor-Agnostic VLM:** The Universal VLM Judge uses the standard OpenAI protocol. By changing environment variables, you can seamlessly swap between GPT-4o, Gemini 2.5 Pro, or a local Qwen2-VL running on vLLM.

* **Self-Documenting API:** Every evaluator defines a score_purpose which is directly injected into the final JSON output, making the response instantly understandable for frontend teams.

* **Fail-Safe Iteration:** If one model fails (e.g., OOM), the pipeline gracefully logs the error and continues with the remaining judges.

---

## ⚡ Quick Start
### 1. Master Environment Setup (Crucial)
Vision-Reward-OS relies on a specific matrix of PyTorch and Transformer versions to prevent conflicts between older models (ImageReward) and newer tools.

Run the automated setup script to install all dependencies and clone necessary remote repositories (e.g., Kwai-Kolors MPS):

```bash
git clone https://github.com/yourusername/Vision-Reward-OS.git
cd Vision-Reward-OS
pip install -r requirements.txt
```
_(Note: The system will automatically download model weights on the first run, which may require several GBs of disk space)._


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
./run.sh
```

### 4. Run the REST API (FastAPI)
For production integration, run the headless microservice:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## 📡 API Usage Example
Evaluate an A/B test programmatically:
```bash
curl -X POST "http://localhost:8000/evaluate/ab-test" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A futuristic cyberpunk city with flying cars at sunset, highly detailed",
    "image_a_url": "https://example.com/candidate_a.jpg",
    "image_b_url": "https://example.com/candidate_b.jpg"
  }'
```

---

## 📂 Project Structure
```plaintext
Vision-Reward-OS/
├── src/
│   ├── api/
│   │   ├── main.py             # FastAPI application and Orchestrator
│   │   ├── gradio_ui.py        # Interactive Web Dashboard
│   │   └── schemas.py          # Pydantic data models
│   └── evaluators/
│       ├── base.py             # Abstract BaseEvaluator interface
│       ├── pickscore_eval.py 
│       ├── imagereward_eval.py
│       ├── hps_eval.py
│       ├── aesthetic_eval.py
│       ├── simulacra_eval.py
│       ├── trending_eval.py
│       ├── mps_eval.py
│       └── vlm_judge_eval.py
├── run.sh                      # Shell script to launch Gradio UI
├── requirements.txt            # Pinned dependencies for stability
└── README.md
```
---

## 📜 License
MIT License