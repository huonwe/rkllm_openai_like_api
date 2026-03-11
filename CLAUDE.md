# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Install dependencies:**
```bash
uv sync
```

**Run the server (manual/host):**
```bash
uv run server.py \
  --rkllm_model_path=models/YOUR_MODEL.rkllm \
  --target_platform=rk3588 \
  --port=8080
```

**Run the test client:**
```bash
# Chat (streaming)
uv run client.py --host http://localhost:8080 --prompt "Hello" --stream

# Chat (non-streaming)
uv run client.py --host http://localhost:8080 --prompt "Hello"

# Embeddings
uv run client.py --host http://localhost:8080 --prompt "Hello" --embeddings
```

**Docker:**
```bash
# Build
docker build -t rkllm-server .

# Run via compose (place .rkllm models in ./models/)
mkdir models
docker compose up -d
```

**Health check:**
```bash
curl http://localhost:8080/health
```

## Architecture

This is a single-process FastAPI server that wraps Rockchip's RKLLM runtime (NPU inference) behind OpenAI-compatible and Ollama-compatible REST APIs.

### Key Files

- **`server.py`** — FastAPI app, all HTTP endpoints, request/response models, startup logic (argument parsing, model load, uvicorn)
- **`rkllm.py`** — Low-level Python `ctypes` bindings to `lib/librkllmrt.so`. Contains the `RKLLM` class (init/run/abort/release), the C callback (`callback_impl`), and generator functions `get_RKLLM_output` (streaming tokens) and `get_RKLLM_embeddings`
- **`utils.py`** — `apply_chat_template` (ChatML format builder with vision/multimodal support) and `make_llm_response` (OpenAI response envelope)
- **`lib/librkllmrt.so`** — Rockchip's closed-source NPU runtime (ARM only, RK3588/RK3576)

### Request Flow

1. HTTP request → FastAPI endpoint in `server.py`
2. Messages formatted via `apply_chat_template()` in `utils.py` (ChatML format)
3. `get_RKLLM_output()` in `rkllm.py` spawns a thread calling `RKLLM.run()` which calls `rkllm_lib.rkllm_run()`
4. The C runtime fires `callback_impl` for each token → tokens placed on `output_queue`
5. Generator yields tokens back to FastAPI → streamed or accumulated per endpoint

### Hardware Concurrency

A global `hw_lock = threading.Lock()` in `server.py` serializes all NPU requests. The NPU can only run one inference at a time. Requests wait up to 30s for the lock; otherwise return HTTP 503.

### API Endpoints

| Endpoint | Protocol |
|---|---|
| `POST /v1/chat/completions` | OpenAI |
| `GET /v1/models` | OpenAI |
| `POST /v1/embeddings` | OpenAI |
| `POST /v1/messages` | Anthropic |
| `POST /api/chat` | Ollama |
| `GET /api/tags`, `/api/ps`, `/api/version` | Ollama |
| `GET /hello` | Quick smoke test |
| `GET /health` | Health check |

### Embeddings

Embeddings use `RKLLM_INFER_GET_LAST_HIDDEN_LAYER` mode — the NPU runs inference, the callback captures the hidden-layer vector (not text tokens), and the mode is switched back to `RKLLM_INFER_GENERATE` afterward.

### Vision / Multimodal

`apply_chat_template()` handles `image_url` content blocks: base64 data URIs are decoded to temp files; file paths and URLs are passed through. Images are wrapped in `<image>...</image>` tags for the model.

### Tool Calling

Tool calls are injected as a system prompt XML block (`inject_tool_prompt` in `server.py`). Model output with `<tool_call>` tags is parsed by `parse_model_output` and returned as structured tool call objects.

## Platform Notes

- Requires ARM hardware with Rockchip NPU (RK3588 or RK3576)
- RKNPU driver version `v0.9.8` recommended (`cat /sys/kernel/debug/rknpu/version`)
- The `.so` libraries in `lib/` must be loadable; for host installs run `sudo cp lib/*.so /usr/lib && sudo ldconfig`
- When `--isDocker y`, model paths are prefixed with `/rkllm_server/models/`
- For host installs, `fix_freq_rk3588.sh.d` is run at startup to stabilize CPU frequency
- Python 3.12+ required
