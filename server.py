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

app = FastAPI(title="RKLLM FastAPI Server with Ollama Compatibility")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lock = threading.Lock()
is_blocking = False
global_model = ""
rkllm_model = None

# --- 1. Pydantic Models ---

class FunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class ToolCall(BaseModel):
    function: FunctionCall

class ResponseMessage(BaseModel):
    role: str
    content: str = ""
    thinking: Optional[str] = None  # Native Ollama Think parameter
    tool_calls: Optional[List[ToolCall]] = None

class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: ResponseMessage
    done: bool

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    stream: Optional[bool] = False
    think: Optional[bool] = True    # Default True for Qwen3
    
    # OpenAI compatibility fields (ignored but allowed)
    # max_tokens: Optional[int] = None
    # temperature: Optional[float] = None
    # top_p: Optional[float] = None
    # stop: Optional[Union[str, List[str]]] = None
    # frequency_penalty: Optional[float] = 0
    # presence_penalty: Optional[float] = 0

# --- 2. Tool & Think Parsing Logic ---

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

    # 1. Extract Think
    think_pattern = r"<think>(.*?)</think>"
    think_matches = list(re.finditer(think_pattern, clean_text, re.DOTALL))
    for match in think_matches:
        if enable_think:
            thinking_content += match.group(1).strip() + "\n"
        clean_text = clean_text.replace(match.group(0), "")

    # 2. Extract Tools
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

# --- 3. FastAPI Endpoints ---

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    global is_blocking
    time_count = 100
    while (is_blocking or get_global_state() == 0):
        time.sleep(0.1)
        time_count -= 1
        if time_count <= 0:
            raise HTTPException(status_code=503, detail="RKLLM Server is Busy!")
    lock.acquire()
    try:
        is_blocking = True
        messages = request.messages
        if request.tools:
            messages = inject_tool_prompt(messages, request.tools)
        if request.think:
            messages_formatted = apply_chat_template(messages)
        else:
            messages_formatted = apply_chat_template(messages, thinking=False)
        # Get inference RKLLM output
        results = get_RKLLM_output(rkllm_model, messages_formatted)

        # Logic A: Request Contains Tools (Buffer and Parse)
        if request.tools:
            full_text = "".join(list(results))
            clean_content, thinking_content, extracted_tools = parse_model_output(full_text, request.think is not False)
            resp_msg = ResponseMessage(role="assistant", content=clean_content)
            if thinking_content:
                resp_msg.thinking = thinking_content
            if extracted_tools:
                resp_msg.tool_calls = [ToolCall(**tc) for tc in extracted_tools]
            response_data = ChatResponse(
                model=request.model,
                created_at=datetime.now(timezone.utc).isoformat() + "Z",
                message=resp_msg,
                done=True
            ).model_dump(exclude_none=True)
            if request.stream:
                def buffer_stream():
                    yield json.dumps(response_data) + "\n"
                return StreamingResponse(buffer_stream(), media_type="application/x-ndjson")
            else:
                return JSONResponse(content=response_data)
    
        # Logic B: Standard Chat Request
        if request.stream:
            def stream_generator():
                buffer = ""
                is_thinking = False
                def make_chunk(text_content: str, is_think: bool):
                    msg = {"role": "assistant", "content": ""}
                    if is_think:
                        msg["thinking"] = text_content
                    else:
                        msg["content"] = text_content
                    return {
                        "model": request.model,
                        "created_at": datetime.now(timezone.utc).isoformat() + "Z",
                        "message": msg,
                        "done": False
                    }
                # Sliding window parser for <think> tags over unaligned chunks
                for r in results:
                    buffer += r
                    while True:
                        if not is_thinking:
                            start_idx = buffer.find("<think>")
                            if start_idx != -1:
                                if start_idx > 0:
                                    yield json.dumps(make_chunk(buffer[:start_idx], False)) + "\n"
                                buffer = buffer[start_idx + 7:]
                                is_thinking = True
                            else:
                                safe_idx = buffer.rfind("<")
                                if safe_idx == -1:
                                    if buffer:
                                        yield json.dumps(make_chunk(buffer, False)) + "\n"
                                        buffer = ""
                                    break
                                else:
                                    if safe_idx > 0:
                                        yield json.dumps(make_chunk(buffer[:safe_idx], False)) + "\n"
                                        buffer = buffer[safe_idx:]
                                    break
                        else:
                            end_idx = buffer.find("</think>")
                            if end_idx != -1:
                                if end_idx > 0 and request.think is not False:
                                    yield json.dumps(make_chunk(buffer[:end_idx], True)) + "\n"
                                buffer = buffer[end_idx + 8:]
                                is_thinking = False
                            else:
                                safe_idx = buffer.rfind("<")
                                if safe_idx == -1:
                                    if buffer and request.think is not False:
                                        yield json.dumps(make_chunk(buffer, True)) + "\n"
                                    buffer = ""
                                    break
                                else:
                                    if safe_idx > 0 and request.think is not False:
                                        yield json.dumps(make_chunk(buffer[:safe_idx], True)) + "\n"
                                        buffer = buffer[safe_idx:]
                                    break
                if buffer:
                    if is_thinking and request.think is not False:
                        yield json.dumps(make_chunk(buffer, True)) + "\n"
                    elif not is_thinking:
                        yield json.dumps(make_chunk(buffer, False)) + "\n"
                yield json.dumps({
                    "model": request.model,
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
                model=request.model,
                created_at=datetime.now(timezone.utc).isoformat() + "Z",
                message=resp_msg,
                done=True
            ).model_dump(exclude_none=True)
            return JSONResponse(content=response_data)
    finally:
        is_blocking = False
        lock.release()

@app.get("/api/tags")
def show_models():
    global global_model
    _model = os.path.basename(global_model)
    return JSONResponse(content={
        "models": [{"name": _model, "model": _model, "details": {"family": "rkllm"}}]
    })

# --- 4. OpenAI Compatibility Endpoints ---

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: ChatRequest):
    global is_blocking
    
    # Wait if busy
    time_count = 100
    while (is_blocking or get_global_state() == 0):
        time.sleep(0.1)
        time_count -= 1
        if time_count <= 0:
             return JSONResponse(
                status_code=503, 
                content={"error": {"message": "Server is busy", "type": "server_error", "param": None, "code": "server_busy"}}
            )

    lock.acquire()
    try:
        is_blocking = True
        messages = request.messages
        # Format messages using existing utility
        messages_formatted = apply_chat_template(messages)
        
        # Get inference RKLLM output
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
                
                # Send done signal
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            rkllm_output = ""
            for r in results:
                rkllm_output += r
            
            # Use utility to structure the final response if available, or build it manually
            # The utils.make_llm_response often returns a specific dict structure compatible with OpenAI
            response_data = make_llm_response(rkllm_output)
            
            # Since make_llm_response logic is inside utils.py, we trust it returns the correct structure.
            # Usually it returns Dict. We just ensure it's returned as JSON.
            return JSONResponse(content=response_data)

    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": {"message": str(e), "type": "server_error", "param": None, "code": "internal_error"}}
        )
    finally:
        is_blocking = False
        lock.release()

@app.get("/v1/models")
def list_openai_models():
    global global_model
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

@app.get("/hello")
def test():
    user_message = "Hello!"
    messages = [{'role':'user','content':user_message}]
    messages_formatted = apply_chat_template(messages)
    results = get_RKLLM_output(rkllm_model, messages_formatted)
    def stream_generator():
        for r in results:
            yield r
        yield '\n'
    return StreamingResponse(stream_generator(), media_type='text/event-stream')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', '-m', type=str, default=f"models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm")
    parser.add_argument('--target_platform', '-t', type=str, default="rk3588")
    parser.add_argument('--lora_model_path', '-lm', type=str)
    parser.add_argument('--prompt_cache_path', type=str)
    parser.add_argument('--port', '-p', type=int, default=11434)
    parser.add_argument('--isDocker', type=str, default='n')
    args = parser.parse_args()

    if args.isDocker.lower() == 'y':
        rkllm_model_path = os.path.join("/rkllm_server/models/", args.rkllm_model_path)
    else:
        rkllm_model_path = args.rkllm_model_path

    if not os.path.exists(rkllm_model_path):
        print(f"Error: RKLLM model path does not exist: {rkllm_model_path}")
        sys.exit(1)

    global_model = rkllm_model_path

    if args.isDocker.lower() == 'y':  # Inside docker
        # Try to fix frequency inside container only if script exists
        # Usually easier to just do nothing if handled by host script or entrypoint
        pass
    else: # Outside docker
        fix_req_file = f"fix_freq_{args.target_platform}.sh"
        if os.path.exists(fix_req_file):
            subprocess.run(f"sudo bash {fix_req_file}", shell=True)
            print("Frequency fixed.")

    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    rkllm_model = RKLLM(rkllm_model_path, args.lora_model_path, args.prompt_cache_path, args.target_platform)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

    rkllm_model.release()
