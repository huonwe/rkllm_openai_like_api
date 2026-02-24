# rkllm api server
[中文](https://github.com/huonwe/rkllm_openai_like_api/blob/main/README.zh.md)

## Introduction
An RKLLM server implementation compatible with the OpenAI API and Ollama API format.

## Supported Platforms
- RK3588 Series
- RK3576 Series
- RKNPU Driver Version: v0.9.8

## Quickstart
Check the rknpu version before using:
```bash
cat /sys/kernel/debug/rknpu/version
```
If no output, then your linux kernel doesn't support rknpu. It's recommended to be 0.9.8

use docker commandline:
```bash
docker run -d \
  --name rkllm-server \
  --restart unless-stopped \
  --privileged \
  -p 8080:8080 \
  -v /dev:/dev \
  -v YOUR/PATH/TO/MODELS:/rkllm_server/models \
  -e TARGET_PLATFORM=rk3588 \
  -e RKLLM_MODEL_PATH=YOUR_MODEL_FILE_NAME \
  dukihiroi/rkllm-server:latest
```

or use docker compose:
```bash
wget https://raw.githubusercontent.com/huonwe/rkllm_openai_like_api/refs/heads/main/docker-compose.yml

mkdir models
# Put your rkllm models here

docker compose up -d

# test
# Change the ip and port if needed
curl -N http://localhost:8080/hello
```

## Usage
```bash
git clone https://github.com/huonwe/rkllm_openai_like_api.git
cd rkllm_openai_like_api
````

Add the required dynamic libraries:

```bash
sudo cp lib/*.so /usr/lib
```

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh
```

Sync dependencies:

```bash
uv sync
```

Run the server:

```bash
uv run server.py
```

  - By default, the platform is set to `rk3588`, the model path is `models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm`, and the listening port is `11434`.
  - You can manually specify parameters, for example:
    `uv run server.py --rkllm_model_path=path/to/model.rkllm --target_platform=rk3588 --port=8080`

After startup, you can connect to OpenAI API via `http://your.ip:11434/api/v1`. Because only the `/v1/chat/completions` endpoint is implemented, not all features may function as expected.

### API List:
- Ollama API
  - `/api/chat`
  - `/api/tags`
- OpenAI API
  - `/v1/chat/completions`
  - `/v1/models`

You can test it using `client.py` (for OpenAI API format):

```bash
uv run client.py
```

## Notes

**Do not use the locally running RKLLM model for background tasks such as automatic title or tag generation.**

While such tasks are in progress, users will be unable to chat with the model. The server can only process one conversation at a time. If there is currently an unfinished conversation being processed, no other requests will be accepted.

## Changelog

  - [x] Implemented `/v1/models`; manual addition of the Model ID is no longer required. -- 2025-02-05
  - [x] Removed dependency on transformers's `AutoTokenizer`. Configuring a network environment to connect to Hugging Face is no longer necessary. -- 2025-02-11
  - [x] Adapted to RKLLM version 1.2.3. Optimized code logic. The default template now uses the ChatML format. -- 2025-12-08
  - [x] If RKLLM is busy, the request will wait for max 10s, rather than response error immediately. --2025-12-10
  - [x] Ollama is supported. Change to FastAPI. API URL is changed.

## Models

Please refer to the [rkllm_model_zoo](https://github.com/airockchip/rknn-llm/tree/main#download).
