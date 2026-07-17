# Adaptive Router (Embedding + Difficulty Classifier)

An adaptive LLM routing plugin for [LiteLLM](https://github.com/BerriAI/litellm), built on the skeleton of LiteLLM's [`litellm_internal_staging`](https://github.com/BerriAI/litellm/tree/litellm_internal_staging) branch and originally developed as part of an AI engineering internship. Extended independently with a fine-tuned difficulty classifier, a fine-tuned semantic embedding classifier, and a self-cognition fine-tuned small model.

The router sits in front of two backend models — a small, cheap model and a large, capable model — and decides per-request which one should handle a given prompt, aiming to cut inference cost without sacrificing quality on requests that need the larger model.

## Two routing strategies

This repo includes **two interchangeable routing strategies**, each with its own Gradio demo UI:

### 1. Difficulty classifier (`gradio_chat_ui_difficulty.py`)
A fine-tuned Qwen3-0.6B model (LoRA adapter in `classifiers/difficulty-classifier-final/`) classifies each prompt as `easy` / `medium` / `hard`. Easy and medium prompts route to the small model; hard prompts route to the large model. Classification happens **locally, in-process** — no LiteLLM proxy involved, no network round trip, low latency.

### 2. Semantic embedding + Thompson-sampling bandit (`gradio_chat_ui_embedding.py`)
Prompts are embedded with Qwen3-Embedding-0.6B (fine-tuned; see `qwen3-embedding-routing-finetuned/`) and matched via pgvector nearest-neighbor lookup to a `RequestType` category (coding, writing, factual lookup, etc.). A Thompson-sampling multi-armed bandit (`bandit.py`) then picks the best model for that category, learning and adapting its model preferences over time based on real usage signals. This strategy routes through the live LiteLLM proxy (`adaptive_router.py`'s `async_pre_routing_hook`), since the bandit's learned state lives there.

Both strategies route between the same two backends:
- **Small model**: a self-cognition fine-tuned Qwen2.5-7B-Instruct, served via a lightweight FastAPI wrapper (`qwen_router_ai_server.py`)
- **Large model**: served via [Ollama](https://ollama.com)

## Architecture

```
                    ┌─────────────────────┐
                    │   Gradio Chat UI     │
                    │ (difficulty variant  │
                    │  or embedding variant)│
                    └──────────┬───────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                  │
   difficulty classifier                 LiteLLM proxy
   (local, in-process)              (async_pre_routing_hook)
              │                                  │
              │                    ┌─────────────┴─────────────┐
              │                    │  embedding classifier      │
              │                    │  (Qwen3-Embedding + pgvector)│
              │                    │           │                 │
              │                    │  Thompson-sampling bandit    │
              │                    │       (bandit.py)            │
              │                    └─────────────┬─────────────┘
              │                                  │
              └────────────────┬─────────────────┘
                               │
                   ┌───────────┴───────────┐
                   │                       │
          small-model (FastAPI)    large-model (Ollama)
          Qwen2.5-7B + LoRA         (e.g. qwen3:8b)
          (self-cognition)
```

## What's included vs. excluded

**Included:**
- `adaptive_router.py` — custom LiteLLM routing strategy plugin (`async_pre_routing_hook`, `pick_model`, cold-start priors, session/owner caching, state persistence)
- `bandit.py` — Thompson sampling implementation (pure functions, no I/O)
- `classifiers/classifier_difficulty.py`, `classifiers/difficulty_classifier_inference.py` — difficulty classification pipeline
- `classifiers/classifier_embedder.py`, `classifiers/populate_vectors.py` — semantic embedding classification pipeline
- `qwen_router_ai_server.py` — FastAPI server for the fine-tuned small model
- `gradio_chat_ui_difficulty.py`, `gradio_chat_ui_embedding.py` — chat interfaces demonstrating each routing strategy
- Fine-tuned LoRA adapters (via Git LFS): `finetune-selfcognition/lora-adapter/`, `classifiers/difficulty-classifier-final/`
- `reports/` — benchmark and cost-analysis scripts and aggregated (non-identifying) results

**Excluded** (see below for how to obtain):
- **Base model weights** — `Qwen2.5-7B-Instruct` (public; download from Hugging Face)
- **Fine-tuned embedding model weights** — `qwen3-embedding-routing-finetuned/model.safetensors` (1GB+, too large for git; config/tokenizer files are included for reference)
- **Raw training/evaluation datasets** — excluded because some were derived from public prompt datasets containing incidental PII (emails, phone numbers) that weren't fully scrubbed at the source. If you want to reproduce this work, you'll need to source your own labeled difficulty dataset — the classifier scripts expect a CSV with `query`, `category`, and `difficulty` columns

## Setup

### 1. Download the base model

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir finetune-selfcognition/qwen2.5-7b-base
```

The fine-tuned LoRA adapter in `finetune-selfcognition/lora-adapter/` (included via Git LFS) loads on top of this base model — see `qwen_router_ai_server.py`.

### 2. Download the fine-tuned embedding model weights

The `qwen3-embedding-routing-finetuned/` folder includes all config/tokenizer files but not the ~1GB `model.safetensors`. If you want to reproduce the embedding-based routing strategy, you'll need to either fine-tune your own on top of [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), or substitute the base embedding model directly (with reduced routing accuracy, since it won't be fine-tuned on this project's category examples).

### 3. Difficulty classifier

The fine-tuned difficulty classifier adapter (`classifiers/difficulty-classifier-final/`) is included via Git LFS and loads on top of [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).

### 4. Set up pgvector (for the embedding classifier strategy only)

The embedding-based classifier queries a Postgres + pgvector instance for nearest-neighbor category lookup:

```bash
docker run -d --name pgvector -e POSTGRES_PASSWORD=<your-password> \
    -e POSTGRES_DB=routing -p 5432:5432 ankane/pgvector
```

Set the following environment variables (see `classifiers/classifier_embedder.py`):
```bash
export PGVECTOR_HOST=127.0.0.1
export PGVECTOR_PORT=5432
export PGVECTOR_DB=routing
export PGVECTOR_USER=litellm
export PGVECTOR_PASSWORD=<your-password>
```

Then populate the reference vectors:
```bash
python classifiers/populate_vectors.py
```

### 5. Install dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

*(Key deps: `litellm`, `transformers`, `peft`, `sentence-transformers`, `psycopg2`, `fastapi`, `uvicorn`, `gradio`, `httpx`)*

### 6. Start everything, in order

**Terminal 1 — Ollama** (large model backend):
```bash
ollama serve
ollama pull qwen3:8b   # or whichever large model you configure, just make sure the change the config.yaml model pointer as well
```

**Terminal 2 — FastAPI small-model server**:
```bash
python qwen_router_ai_server.py
```
Serves the fine-tuned small model on `http://localhost:8001`.

**Terminal 3 — LiteLLM proxy** (only needed for the embedding+bandit strategy):
```bash
litellm --config config.yaml --port 4000
```

**Terminal 4 — Gradio UI** (pick one):
```bash
python gradio_chat_ui_difficulty.py   # local difficulty classifier, no LiteLLM needed
# or
python gradio_chat_ui_embedding.py    # embedding + bandit, requires LiteLLM proxy running
```

Then open the URL Gradio prints (typically `http://127.0.0.1:7860`).

## Notes

- Difficulty classification runs in-process for low latency; the embedding+bandit strategy's routing decision lives inside the LiteLLM proxy since the bandit maintains learned state there.
- This project was developed and tested on CPU-only hardware. The small model (7B, fp16) can be slow on CPU for real-time use — GPU acceleration or a smaller/quantized model is recommended for production use.
- See `reports/` for benchmark methodology and aggregated cost/accuracy comparisons between routing strategies.
