FROM python:3.12-slim

# 1. Install ONLY required runtime dependencies (Removed curl & git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy `uv` directly from the official image (Saves space and avoids installing curl)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /rkllm_server

# 3. Optimize Caching: Copy ONLY dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies (ignoring dev dependencies for a smaller production image)
RUN uv sync --frozen --no-dev

# 4. Copy the rest of the application files
COPY ./lib ./lib
COPY ./rkllm.py ./server.py ./utils.py ./

# 5. Optimize Library Loading: Avoid copying .so files, just point to the directory
ENV LD_LIBRARY_PATH="/rkllm_server/lib:${LD_LIBRARY_PATH}"

# Environment Variables
ENV RKLLM_MODEL_PATH="default.rkllm"
ENV TARGET_PLATFORM="rk3588"
# 6. Default port set as an environment variable so it can be overridden
ENV PORT=8080

# Expose the configured port
EXPOSE $PORT

# 7. Start the server using the $PORT environment variable
CMD uv run server.py \
    --rkllm_model_path "$RKLLM_MODEL_PATH" \
    --target_platform "$TARGET_PLATFORM" \
    --port "$PORT" \
    --isDocker y