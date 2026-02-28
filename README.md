# RKLLM API Server

## Introduction

A lightweight, high-performance API server for Rockchip NPUs (RKLLM), providing drop-in compatibility with **OpenAI API** and **Ollama API** formats. This allows you to seamlessly integrate locally hosted large language models on Rockchip hardware with existing AI tools, frontends, and frameworks.

### Features

* 🚀 **Hardware Optimized:** Leverages Rockchip's NPU for fast inference.
* 🔄 **Dual API Compatibility:** Supports both standard OpenAI (`/v1/chat/completions`) and Ollama API endpoints.
* 🌊 **Real-time Streaming:** Full support for Server-Sent Events (SSE) streaming token output.
* 🐳 **Docker Ready:** Minimal footprint containerization for easy deployment.
* 🛠️ **No External Tokenizers:** Operates independently without needing Hugging Face `transformers` or `AutoTokenizer`.

## Supported Platforms

* **Hardware:** RK3588 Series, RK3576 Series
* **RKNPU Driver Version:** `v0.9.8` (Recommended)

> **Note:** Check your RKNPU version before proceeding:
> ```bash
> cat /sys/kernel/debug/rknpu/version
> 
> ```
> 
> 
> *If this command returns no output, your Linux kernel does not currently support the RKNPU.*

---

## 🐳 Quickstart (Docker Recommended)

The easiest way to run the server is via Docker.

### Option A: Docker CLI

```bash
docker run -d \
  --name rkllm-server \
  --restart unless-stopped \
  --privileged \
  -p 8080:8080 \
  -v /dev:/dev \
  -v /YOUR/PATH/TO/MODELS:/rkllm_server/models \
  -e TARGET_PLATFORM=rk3588 \
  -e RKLLM_MODEL_PATH=YOUR_MODEL_FILE_NAME.rkllm \
  -e PORT=8080 \
  dukihiroi/rkllm-server:latest

```

### Option B: Docker Compose

Create a `docker-compose.yml` file:

```yaml
services:
  rkllm-server:
    image: dukihiroi/rkllm-server:latest
    container_name: rkllm-server
    restart: unless-stopped
    privileged: true
    ports:
      - "8080:8080"
    volumes:
      - /dev:/dev
      - ./models:/rkllm_server/models
    environment:
      - TARGET_PLATFORM=rk3588
      - RKLLM_MODEL_PATH=qwen3-vl-2b-instruct_w8a8_rk3588.rkllm
      - PORT=8080

```

Then start the server:

```bash
mkdir models # Place your .rkllm files here
docker compose up -d

```

**Test the deployment:**

```bash
curl http://localhost:8080/health

```

---

## 🛠️ Manual Installation

If you prefer to run the server directly on the host OS without Docker:

**1. Clone the repository:**

```bash
git clone https://github.com/anand34577/rkllm_openai.git
cd rkllm_openai

```

**2. Install RKLLM Dynamic Libraries:**

```bash
sudo cp lib/*.so /usr/lib
sudo ldconfig

```

**3. Install `uv` (Fast Python Package Installer):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```

**4. Sync Dependencies:**

```bash
uv sync

```

**5. Run the Server:**

```bash
uv run server.py \
  --rkllm_model_path=models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm \
  --target_platform=rk3588 \
  --port=8080

```

---

## 🔌 API Endpoints

Once running, the server listens on the configured port (default `8080`).

| API Type | Endpoint | Description |
| --- | --- | --- |
| **Server** | `GET /health` | Check server status and NPU availability. |
| **OpenAI** | `POST /v1/chat/completions` | Standard chat completion (supports `stream: true`). |
| **OpenAI** | `GET /v1/models` | Returns the currently loaded RKLLM model ID. |
| **Ollama** | `POST /api/chat` | Ollama-compatible chat completion. |
| **Ollama** | `GET /api/tags` | Ollama-compatible model listing. |

### Testing with the Built-in Client

You can test the OpenAI streaming implementation using the included Python client:

```bash
uv run client.py --host http://localhost:8080 --prompt "Explain quantum mechanics briefly." --stream

```

---

## ⚠️ Important Limitations & Notes

**Hardware Concurrency Limit**
Because the NPU handles one inference task at a time, **the server can only process one conversation at a time.** * Do not use this server for heavy background tasks (like bulk title/tag generation) if you also want it to remain responsive for interactive chat.

* If a new request arrives while the NPU is busy, the server will briefly wait. If the NPU does not free up, it will return an HTTP `503 Service Unavailable` error rather than crashing.

## 📦 Model Zoo

To download pre-converted `.rkllm` models, please refer to the official [Rockchip rknn-llm Model Zoo](https://github.com/airockchip/rknn-llm/tree/main#download).

---

## 📝 Changelog

* **2026-02-28:** Refactored for FastAPI, added Ollama support, implemented non-blocking hardware locks, and optimized Dockerfile.
* **2025-12-10:** Added request queueing; requests now wait up to 10s if the RKLLM runtime is busy instead of failing immediately.
* **2025-12-08:** Adapted to RKLLM v1.2.3. Optimized template logic (defaults to standard ChatML format).
* **2025-02-11:** Removed dependency on `AutoTokenizer`. An active internet connection to Hugging Face is no longer required to run the server.
* **2025-02-05:** Implemented `/v1/models`; manual configuration of the Model ID is no longer required.