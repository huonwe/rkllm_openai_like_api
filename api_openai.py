import os
import json
import time
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime, timezone
from common import ChatRequest, EmbeddingRequest, hw_lock, global_state
from utils import apply_chat_template, make_llm_response
from rkllm import get_RKLLM_output, get_RKLLM_embeddings

router = APIRouter()

@router.post("/v1/embeddings")
def openai_embeddings(request: EmbeddingRequest):
    if not hw_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "Server busy", "type": "server_error", "code": "server_busy"}}
        )

    try:
        inputs = request.input if isinstance(request.input, list) else [request.input]
        data_results = []

        for idx, text in enumerate(inputs):
            vector = get_RKLLM_embeddings(global_state.rkllm_model, text)

            data_results.append({
                "object": "embedding",
                "embedding": vector,
                "index": idx
            })

        return JSONResponse(content={
            "object": "list",
            "data": data_results,
            "model": request.model,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0
            }
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error", "code": "internal_error"}}
        )
    finally:
        hw_lock.release()

@router.post("/v1/chat/completions")
async def openai_chat_completions(request: ChatRequest):
    created_time = int(time.time())
    model_name = os.path.basename(global_state.model_path) if global_state.model_path else "rkllm"

    if request.stream:
        def stream_generator():
            if not hw_lock.acquire(blocking=True, timeout=30):
                yield f"data: {json.dumps({'error': {'message': 'Server busy', 'type': 'server_error', 'code': 'server_busy'}})}\n\n"
                return
            try:
                messages_formatted = apply_chat_template(request.messages)
                results = get_RKLLM_output(global_state.rkllm_model, messages_formatted)
                for r in results:
                    yield f"data: {json.dumps({'id': f'chatcmpl-{created_time}', 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': r}, 'finish_reason': None}]})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                hw_lock.release()
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    if not hw_lock.acquire(blocking=True, timeout=30):
        return JSONResponse(status_code=503, content={"error": {"message": "Server is busy", "type": "server_error", "code": "server_busy"}})
    try:
        messages_formatted = apply_chat_template(request.messages)
        results = get_RKLLM_output(global_state.rkllm_model, messages_formatted)
        rkllm_output = "".join(list(results))
        response_data = make_llm_response(rkllm_output)
        response_data["created"] = created_time
        response_data["model"] = model_name
        return JSONResponse(content=response_data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": {"message": str(e), "type": "server_error", "code": "internal_error"}})
    finally:
        hw_lock.release()

@router.get("/v1/models")
def list_openai_models():
    _model = os.path.basename(global_state.model_path) if global_state.model_path else "rkllm-model"
    return JSONResponse(content={
        "object": "list",
        "data": [{
            "id": f"rkllm/{_model}",
            "object": "model",
            "owned_by": "rkllm_server",
            "created": int(time.time())
        }]
    })