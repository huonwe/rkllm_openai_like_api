# rkllm openai like api server
[English](https://github.com/huonwe/rkllm_openai_like_api/blob/main/README.en.md)

## 介绍
兼容OpenAI API格式的rkllm server代码

## Support Platform
- RK3588 Series
- RK3576 Series
- RKNPU Driver Version: v0.9.8

## Quickstart
使用前先检查rknpu驱动版本
```bash
cat /sys/kernel/debug/rknpu/version
```
如果没有输出，则说明当前内核不支持rknpu. 建议版本为0.9.8

直接使用docker:
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

或者使用docker compose:
```bash
wget https://raw.githubusercontent.com/huonwe/rkllm_openai_like_api/refs/heads/main/docker-compose.yml

mkdir models
# Put your rkllm models here

docker compose up -d

# test
# Change the ip and port if needed
curl -N http://localhost:8080/hello
```

## 使用
```bash
git clone https://github.com/huonwe/rkllm_openai_like_api.git
cd rkllm_openai_like_api
```
添加需要用到的动态库:
```bash
sudo cp lib/*.so /usr/lib
```
安装uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv sync
```

运行:
```bash
uv run server.py
```
- 默认情况下，平台设置为rk3588，模型路径为`models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm`，监听端口为`8080`
- 你可以手动指定参数，如`uv run server.py --rkllm_model_path=path/to/model.rkllm --target_platform=rk3588 --port=8080`

之后，你可以通过`http://your.ip:8080/rkllm_chat/v1`来连接到本服务。由于只实现了`/v1/chat/completions`, 所以并不是所有功能都可以正常使用。

你可以用client.py测试:
```bash
uv run client.py
```

## 注意事项
不要使用rkllm本地运行的模型来进行自动生成标题、标签等任务。在进行此类任务时，用户将无法与模型聊天，同一时刻server只会处理一条对话。若当前存在对话未处理完成，则不会接受任何其他对话。

## 更新记录
- [x] 实现了/v1/models，现在无需手动添加模型ID.  --20250205
- [x] 删除了对transformers的AutoTokenizer的依赖, 现在无需配置网络环境以连接到huggingface.  --20250211
- [x] 适配了1.2.3的rkllm版本. 优化了代码逻辑. 默认模板使用ChatML格式. --20251208
- [x] 如果RKLLM忙碌，那么请求会等待最多10秒，而不是立即返回忙碌信息. --20251210

## 模型
请参照[rkllm_model_zoo](https://github.com/airockchip/rknn-llm/tree/main#download)

注意，自己转换的旧版本的rkllm模型在新版本的rkllm推理时会出现错误，比如无限循环
