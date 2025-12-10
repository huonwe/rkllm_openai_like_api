# rkllm openai like api server

## Introduction
An RKLLM server implementation compatible with the OpenAI API format.

## Supported Platforms
- RK3588 Series
- RK3576 Series
- RKNPU Driver Version: v0.9.8

## Quickstart
```bash
wget https://raw.githubusercontent.com/huonwe/rkllm_openai_like_api/refs/heads/main/docker-compose.yml

mkdir models
# Put your rkllm models here

docker compose up -d

# test
# Change the ip and port if needed
curl -N http://localhost:8080/test
```

## Usage
```bash
git clone [https://github.com/huonwe/rkllm_openai_like_api.git](https://github.com/huonwe/rkllm_openai_like_api.git)
cd rkllm_openai_like_api
````

Add the required dynamic libraries:

```bash
sudo cp lib/*.so /usr/lib
```

Install uv:

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

Sync dependencies:

```bash
uv sync
```

Run the server:

```bash
uv run server.py
```

  - By default, the platform is set to `rk3588`, the model path is `models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm`, and the listening port is `8080`.
  - You can manually specify parameters, for example:
    `uv run server.py --rkllm_model_path=path/to/model.rkllm --target_platform=rk3588 --port=8080`

After startup, you can connect to this service via `http://your.ip:8080/rkllm_chat/v1`. Since only the `/v1/chat/completions` endpoint is implemented, not all features may function as expected.

You can test it using `client.py`:

```bash
uv run client.py
```

## Notes

**Do not use the locally running RKLLM model for background tasks such as automatic title or tag generation.**

While such tasks are in progress, users will be unable to chat with the model. The server can only process one conversation at a time. If there is currently an unfinished conversation being processed, no other requests will be accepted.

## Changelog

  - [x] Implemented `/v1/models`; manual addition of the Model ID is no longer required. -- 2025-02-05
  - [x] Removed dependency on transformers' `AutoTokenizer`. Configuring a network environment to connect to Hugging Face is no longer necessary. -- 2025-02-11
  - [x] Adapted to RKLLM version 1.2.3. Optimized code logic. The default template now uses the ChatML format. -- 2025-12-08

## Models

Please refer to the [rkllm\_model\_zoo](https://github.com/airockchip/rknn-llm/tree/main#download).
