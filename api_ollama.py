import os
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime, timezone
from common import ChatRequest, ChatResponse, ResponseMessage, hw_lock, global_state, inject_tool_prompt, parse_model_output
from utils import apply_chat_template
from rkllm import get_RKLLM_output

router = APIRouter()

@router.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    if request.stream:
        def stream_generator():
            if not hw_lock.acquire(blocking=True, timeout=30):
                yield json.dumps({"error": "Server busy"}) + "\n"
                return
            try:
                messages = request.messages
                if request.tools:
                    messages = inject_tool_prompt(messages, request.tools)
                messages_formatted = apply_chat_template(messages, thinking=request.think)
                results = get_RKLLM_output(global_state.rkllm_model, messages_formatted)
                for r in results:
                    yield json.dumps({
                        "model": request.model if hasattr(request, 'model') else "rkllm",
                        "created_at": datetime.now(timezone.utc).isoformat() + "Z",
                        "message": {"role": "assistant", "content": r},
                        "done": False
                    }) + "\n"
                yield json.dumps({
                    "model": request.model if hasattr(request, 'model') else "rkllm",
                        "created_at": datetime.now(timezone.utc).isoformat() + "Z",
                        "message": {"role": "assistant", "content": ""},
                        "done": True
                    }) + "\n"
            finally:
                hw_lock.release()
        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

    if not hw_lock.acquire(blocking=True, timeout=30):
        raise HTTPException(status_code=503, detail="RKLLM Hardware is currently processing another request.")
    try:
        messages = request.messages
        if request.tools:
            messages = inject_tool_prompt(messages, request.tools)
        messages_formatted = apply_chat_template(messages, thinking=request.think)
        results = get_RKLLM_output(global_state.rkllm_model, messages_formatted)
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

@router.get("/api/version")
def ollama_version():
    return JSONResponse(content={"version": "0.9.0"})

@router.get("/api/ps")
def ollama_ps():
    _model = os.path.basename(global_state.model_path) if global_state.model_path else "rkllm-model"
    model_size = os.path.getsize(global_state.model_path) if global_state.model_path and os.path.exists(global_state.model_path) else 0
    return JSONResponse(content={
        "models": [{
            "name": _model,
            "model": _model,
            "size": model_size,
            "digest": "",
            "expires_at": "0001-01-01T00:00:00Z",
            "size_vram": model_size,
            "details": {
                "format": "rkllm",
                "family": "rkllm",
                "families": ["rkllm"],
                "parameter_size": "",
                "quantization_level": ""
            }
        }]
    })

@router.get("/api/tags")
def ollama_list_models():
    _model = os.path.basename(global_state.model_path) if global_state.model_path else "rkllm-model"
    model_size = os.path.getsize(global_state.model_path) if global_state.model_path and os.path.exists(global_state.model_path) else 0
    return JSONResponse(content={
        "models": [{
            "name": _model,
            "model": _model,
            "modified_at": datetime.now(timezone.utc).isoformat() + "Z",
            "size": model_size,
            "digest": "",
            "details": {
                "format": "rkllm",
                "family": "rkllm",
                "families": ["rkllm"],
                "parameter_size": "",
                "quantization_level": ""
            }
        }]
    })