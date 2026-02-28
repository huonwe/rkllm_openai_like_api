import sys
import os
import subprocess
import resource
import threading
import argparse
import json
import time
import re
from datetime import datetime, timezone

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

from utils import apply_chat_template, make_llm_response
from rkllm import RKLLM, get_RKLLM_output, get_global_state

app = FastAPI(title="RKLLM API Server", description="OpenAI and Ollama Compatible API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global hardware lock to ensure RKLLM inference runs strictly one at a time
hw_lock = threading.Lock()
global_model = ""
rkllm_model = None


# --- Pydantic Models ---

class FunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


class ToolCall(BaseModel):
    function: FunctionCall


class ResponseMessage(BaseModel):
    role: str
    content: str = ""
    thinking: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: ResponseMessage
    done: bool


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    stream: Optional[bool] = False
    think: Optional[bool] = True


# --- Utilities ---

def inject_tool_prompt(messages: List[Dict], tools: List[Dict]) -> List[Dict]:
    tool_schemas = [t.get("function", t) for t in tools]
    system_content = (
        "You are a helpful assistant.\n\n"
        "# Tools\n"
        "You may call one or more functions to assist with the user query.\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        f"{json.dumps(tool_schemas, indent=2)}\n"
        "</tools>\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": "function_name", "arguments": {"arg_name": "value"}}\n'
        "</tool_call>"
    )
    new_messages = []
    has_system = False
    for msg in messages:
        if msg.get("role") == "system":
            new_messages.append({"role": "system", "content": system_content + "\n" + msg.get("content", "")})
            has_system = True
        else:
            new_messages.append(msg)
    if not has_system:
        new_messages.insert(0, {"role": "system", "content": system_content})
    return new_messages


def parse_model_output(text: str, enable_think: bool) -> tuple[str, str, List[Dict]]:
    """Extracts both <think> and <tool_call> tags from generated text."""
    thinking_content = ""
    clean_text = text

    # Extract Think Tags
    think_pattern = r"<think>(.*?)</think>"
    think_matches = list(re.finditer(think_pattern, clean_text, re.DOTALL))
    for match in think_matches:
        if enable_think:
            thinking_content += match.group(1).strip() + "\n"
        clean_text = clean_text.replace(match.group(0), "")

    # Extract Tools
    tool_calls = []
    tool_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_matches = list(re.finditer(tool_pattern, clean_text, re.DOTALL))
    for match in tool_matches:
        try:
            call_json = json.loads(match.group(1).strip())
            tool_calls.append({
                "function": {
                    "name": call_json.get("name", ""),
                    "arguments": call_json.get("arguments", {})
                }
            })
            clean_text = clean_text.replace(match.group(0), "")
        except json.JSONDecodeError:
            pass

    return clean_text.strip(), thinking_content.strip(), tool_calls


# --- FastAPI Endpoints ---

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "state": "idle" if not hw_lock.locked() else "busy"}


@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    # Non-blocking lock check to gracefully reject requests when busy
    if not hw_lock.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="RKLLM Hardware is currently processing another request.")

    try:
        messages = request.messages
        if request.tools:
            messages = inject_tool_prompt(messages, request.tools)

        messages_formatted = apply_chat_template(messages, thinking=request.think)
        results = get_RKLLM_output(rkllm_model, messages_formatted)

        if request.stream:
            def stream_generator():
                for r in results:
                    chunk = {
                        "model": request.model if hasattr(request, 'model') else "rkllm",
                        "created_at": datetime.now(timezone.utc).isoformat() + "Z",
                        "message": {"role": "assistant", "content": r},
                        "done": False
                    }
                    yield json.dumps(chunk) + "\n"

                yield json.dumps({
                    "model": request.model if hasattr(request, 'model') else "rkllm",
                    "created_at": datetime.now(timezone.utc).isoformat() + "Z",
                    "message": {"role": "assistant", "content": ""},
                    "done": True
                }) + "\n"

            return StreamingResponse(stream_generator(), media_type="application/x-ndjson")
        else:
            full_text = "".join(list(results))
            clean_content, thinking_content, _ = parse_model_output(full_text, request.think is not False)

            resp_msg = ResponseMessage(role="assistant", content=clean_content)
            if thinking_content:
                resp_msg.thinking = thinking_content

            response_data = ChatResponse(
                model=request.model if hasattr(request, 'model') else "rkllm",
                created_at=datetime.now(timezone.utc).isoformat() + "Z",
                message=resp_msg,
                done=True
            ).model_dump(exclude_none=True)

            return JSONResponse(content=response_data)

    finally:
        hw_lock.release()


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: ChatRequest):
    if not hw_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "Server is busy", "type": "server_error", "code": "server_busy"}}
        )

    try:
        messages = request.messages
        messages_formatted = apply_chat_template(messages)
        results = get_RKLLM_output(rkllm_model, messages_formatted)

        created_time = int(time.time())
        model_name = os.path.basename(global_model) if global_model else "rkllm"

        if request.stream:
            def stream_generator():
                for r in results:
                    chunk_data = {
                        "id": f"chatcmpl-{created_time}",
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": r},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            rkllm_output = "".join(list(results))
            response_data = make_llm_response(rkllm_output)
            response_data["created"] = created_time
            response_data["model"] = model_name
            return JSONResponse(content=response_data)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error", "code": "internal_error"}}
        )
    finally:
        hw_lock.release()


@app.get("/v1/models")
def list_openai_models():
    _model = os.path.basename(global_model) if global_model else "rkllm-model"
    return JSONResponse(content={
        "object": "list",
        "data": [{
            "id": f"rkllm/{_model}",
            "object": "model",
            "owned_by": "rkllm_server",
            "created": int(time.time())
        }]
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', '-m', type=str, default="models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm")
    parser.add_argument('--target_platform', '-t', type=str, default="rk3588")
    parser.add_argument('--lora_model_path', '-lm', type=str)
    parser.add_argument('--prompt_cache_path', type=str)
    parser.add_argument('--port', '-p', type=int, default=11434)
    parser.add_argument('--isDocker', type=str, default='n')
    args = parser.parse_args()

    rkllm_model_path = os.path.join("/rkllm_server/models/",
                                    args.rkllm_model_path) if args.isDocker.lower() == 'y' else args.rkllm_model_path

    if not os.path.exists(rkllm_model_path):
        print(f"[Error] RKLLM model path does not exist: {rkllm_model_path}")
        sys.exit(1)

    global_model = rkllm_model_path

    if args.isDocker.lower() != 'y':
        fix_req_file = f"fix_freq_{args.target_platform}.sh"
        if os.path.exists(fix_req_file):
            subprocess.run(f"sudo bash {fix_req_file}", shell=True)
            print("[Info] CPU frequency fixed via shell script.")

    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    rkllm_model = RKLLM(rkllm_model_path, args.lora_model_path, args.prompt_cache_path, args.target_platform)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)

    rkllm_model.release()