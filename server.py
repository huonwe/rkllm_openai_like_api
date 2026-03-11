import sys
import os
import subprocess
import resource
import argparse

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from common import hw_lock, global_state
from rkllm import RKLLM, get_RKLLM_output
from utils import apply_chat_template

from api_openai import router as openai_router
from api_ollama import router as ollama_router
from api_claude import router as claude_router

app = FastAPI(title="RKLLM API Server", description="OpenAI and Ollama Compatible API (Vision & Embeddings)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(openai_router)
app.include_router(ollama_router)
app.include_router(claude_router)

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "state": "idle" if not hw_lock.locked() else "busy"}

@app.get("/hello")
def test():
    user_message = "Hello!"
    messages = [{'role':'user','content':user_message}]
    messages_formatted = apply_chat_template(messages)
    results = get_RKLLM_output(global_state.rkllm_model, messages_formatted)
    def stream_generator():
        for r in results:
            yield r
        yield '\n'
    return StreamingResponse(stream_generator(), media_type='text/event-stream')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', '-m', type=str, default="models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm")
    parser.add_argument('--max_context_len', '-c', type=int, default=4096)
    parser.add_argument('--target_platform', '-t', type=str, default="rk3588")
    parser.add_argument('--lora_model_path', '-lm', type=str)
    parser.add_argument('--prompt_cache_path', type=str)

    parser.add_argument('--host', type=str, default="0.0.0.0")
    parser.add_argument('--port', '-p', type=int, default=8080)
    parser.add_argument('--isDocker', type=str, default='n')
    args = parser.parse_args()

    rkllm_model_path = os.path.join("/rkllm_server/models/",
                                    args.rkllm_model_path) if args.isDocker.lower() == 'y' else args.rkllm_model_path

    if not os.path.exists(rkllm_model_path):
        print(f"[Error] RKLLM model path does not exist: {rkllm_model_path}")
        sys.exit(1)

    global_state.model_path = rkllm_model_path

    if args.isDocker.lower() != 'y':
        fix_req_file = f"fix_freq_{args.target_platform}.sh"
        if os.path.exists(fix_req_file):
            subprocess.run(f"sudo bash {fix_req_file}", shell=True)
            print("[Info] CPU frequency fixed via shell script.")

        if os.path.exists(fix_req_file+".d"):
            print(f"[Info] CPU frequency fix script will not be applied. Remove the suffix '.d' to enable it: {fix_req_file+'.d'}")

    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    config = {
        "max_context_len": args.max_context_len
    }

    print(f"[Info] RKLLM Model Path: {rkllm_model_path}")
    print(f"[Info] RKLLM Config: {config}")

    global_state.rkllm_model = RKLLM(config, rkllm_model_path, args.lora_model_path, args.prompt_cache_path, args.target_platform)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)

    global_state.rkllm_model.release()
