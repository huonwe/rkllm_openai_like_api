# RKLLM API Server

## 简介

一个为 Rockchip NPU (RKLLM) 设计的轻量级、高性能 API 服务器，提供与 **OpenAI API** 和 **Ollama API** 格式兼容的直接替代方案。这允许您将部署在 Rockchip 硬件上的本地大语言模型与现有的 AI 工具、前端和框架无缝集成。

### 特性

* 🚀 **硬件优化：** 利用 Rockchip 的 NPU 进行快速推理。
* 🔄 **双 API 兼容：** 同时支持标准 OpenAI (`/v1/chat/completions`) 和 Ollama API 端点。
* 🌊 **实时流式传输：** 全面支持服务器发送事件 (SSE) 流式 token 输出。
* 🐳 **Docker 就绪：** 最小占用的容器化设计，易于部署。
* 🛠️ **无需外部 Tokenizer：** 独立运行，无需 Hugging Face 的 `transformers` 或 `AutoTokenizer`。

## 支持的平台

* **硬件：** RK3588 系列，RK3576 系列
* **RKNPU 驱动版本：** `v0.9.8` (推荐)

> **注意：** 在继续之前，请检查您的 RKNPU 版本：
> ```bash
> cat /sys/kernel/debug/rknpu/version
> 
> ```
> 
> 
> *如果此命令没有输出，则您的 Linux 内核目前不支持 RKNPU。*

---

## 🐳 快速入门 (推荐使用 Docker)

运行服务器最简单的方法是通过 Docker。

### 选项 A: Docker CLI

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

### 选项 B: Docker Compose

创建一个 `docker-compose.yml` 文件：

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

然后启动服务器：

```bash
mkdir models # 将您的 .rkllm 文件放在这里
docker compose up -d

```

**测试部署：**

```bash
curl http://localhost:8080/health

```

---

## 🛠️ 手动安装

如果您更喜欢直接在主机操作系统上运行服务器而不是使用 Docker：

**1. 克隆仓库：**

```bash
git clone https://github.com/anand34577/rkllm_openai.git
cd rkllm_openai

```

**2. 安装 RKLLM 动态库：**

```bash
sudo cp lib/*.so /usr/lib
sudo ldconfig

```

**3. 安装 `uv` (快速的 Python 包安装器)：**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```

**4. 同步依赖：**

```bash
uv sync

```

**5. 运行服务器：**

```bash
uv run server.py \
  --rkllm_model_path=models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm \
  --target_platform=rk3588 \
  --port=8080

```

---

## 🔌 API 端点

运行后，服务器将在配置的端口 (默认 `8080`) 上监听。

| API 类型 | 端点 | 描述 |
| --- | --- | --- |
| **Server** | `GET /health` | 检查服务器状态和 NPU 可用性。 |
| **OpenAI** | `POST /v1/chat/completions` | 标准聊天补全 (支持 `stream: true`)。 |
| **OpenAI** | `GET /v1/models` | 返回当前加载的 RKLLM 模型 ID。 |
| **Ollama** | `POST /api/chat` | 兼容 Ollama 的聊天补全。 |
| **Ollama** | `GET /api/tags` | 兼容 Ollama 的模型列表。 |

### 使用内置客户端测试

您可以使用随附的 Python 客户端测试 OpenAI 流式传输实现：

```bash
uv run client.py --host http://localhost:8080 --prompt "简要解释一下量子力学。" --stream

```

---

## ⚠️ 重要限制与注意事项

**硬件并发限制**
因为 NPU 一次只处理一个推理任务，**所以服务器一次只能处理一个对话。** 
* 如果您也希望它在交互式聊天中保持响应，请不要将此服务器用于繁重的后台任务 (如批量生成标题/标签)。
* 如果在 NPU 繁忙时收到新请求，服务器将短暂等待。如果 NPU 仍未释放，它将返回 HTTP `503 Service Unavailable` 错误而不是崩溃。

## 📦 模型库

要下载预转换的 `.rkllm` 模型，请参考官方的 [Rockchip rknn-llm Model Zoo](https://github.com/airockchip/rknn-llm/tree/main#download)。

---

## 📝 更新日志

* **2026-02-28:** 使用 FastAPI 重构，增加 Ollama 支持，实现非阻塞硬件锁并优化 Dockerfile。
* **2025-12-10:** 增加了请求队列；如果 RKLLM 运行时繁忙，请求现在最多等待 10s 而不是立即失败。
* **2025-12-08:** 适配 RKLLM v1.2.3。优化了模板逻辑 (默认使用标准 ChatML 格式)。
* **2025-02-11:** 移除了对 `AutoTokenizer` 的依赖。运行服务器不再需要在 Hugging Face 保持活跃互联网连接。
* **2025-02-05:** 实现了 `/v1/models`；不再需要手动配置模型 ID。
