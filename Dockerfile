FROM python:3.12-slim


RUN apt-get update && apt-get install -y \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /rkllm_server

COPY ./lib /rkllm_server/lib
COPY ./rkllm.py /rkllm_server/rkllm.py
COPY ./server.py /rkllm_server/server.py
COPY ./utils.py /rkllm_server/utils.py
COPY ./pyproject.toml /rkllm_server/pyproject.toml
COPY ./uv.lock /rkllm_server/uv.lock

RUN cp lib/*.so /usr/lib/ && \
    ldconfig

RUN uv sync

ENV RKLLM_MODEL_PATH=default.rkllm
ENV TARGET_PLATFORM=rk3588

EXPOSE 8080

# CMD echo "Starting RKLLM server with model path: $RKLLM_MODEL_PATH and target platform: $TARGET_PLATFORM"
CMD uv run server.py \
    --rkllm_model_path "$RKLLM_MODEL_PATH" \
    --target_platform "$TARGET_PLATFORM" \
    --port 8080 \
    --isDocker y