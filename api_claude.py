import os
import json
import time
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from common import hw_lock, global_state
from utils import apply_chat_template
from rkllm import get_RKLLM_output

router = APIRouter()

@router.post("/v1/messages")
async def anthropic_messages(request: Request):
    body = await request.json()
    stream = body.get("stream", False)

    # Convert Anthropic messages format to flat list
    messages = []
    system_prompt = body.get("system", "")
    if system_prompt:
        if isinstance(system_prompt, list):
            system_prompt = " ".join(b.get("text", "") for b in system_prompt if isinstance(b, dict))
        messages.append({"role": "system", "content": system_prompt})
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        messages.append({"role": msg["role"], "content": content})

    model_name = body.get("model", os.path.basename(global_state.model_path) if global_state.model_path else "rkllm")
    msg_id = f"msg_{int(time.time())}"

    if stream:
        def stream_generator():
            if not hw_lock.acquire(blocking=True, timeout=30):
                yield f"event: error\ndata: {json.dumps({'type':'error','error':{'type':'overloaded_error','message':'Server busy'}})}\n\n"
                return
            try:
                messages_formatted = apply_chat_template(messages, thinking=False)
                results = get_RKLLM_output(global_state.rkllm_model, messages_formatted)
                yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','content':[],'model':model_name,'stop_reason':None,'usage':{'input_tokens':0,'output_tokens':1}}})}\n\n"
                yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
                yield "event: ping\ndata: {\"type\":\"ping\"}\n\n"
                output_tokens = 0
                try:
                    for token in results:
                        output_tokens += 1
                        yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':token}})}\n\n"
                except Exception:
                    pass
                finally:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
                    yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':output_tokens}})}\n\n"
                    yield "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
            finally:
                hw_lock.release()
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    if not hw_lock.acquire(blocking=True, timeout=30):
        return JSONResponse(status_code=529, content={"type": "error", "error": {"type": "overloaded_error", "message": "Server busy"}})
    try:
        messages_formatted = apply_chat_template(messages, thinking=False)
        results = get_RKLLM_output(global_state.rkllm_model, messages_formatted)
        full_text = "".join(list(results))
        return JSONResponse(content={
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": full_text}],
            "model": model_name,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": len(full_text.split())}
        })
    finally:
        hw_lock.release()