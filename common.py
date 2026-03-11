import json
import re
import threading
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

# Global hardware lock to ensure RKLLM inference runs strictly one at a time
hw_lock = threading.Lock()

class GlobalState:
    model_path: str = ""
    rkllm_model: Any = None

global_state = GlobalState()

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

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    stream: Optional[bool] = False
    think: Optional[bool] = True

class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: ResponseMessage
    done: bool

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "rkllm-model"

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
            new_messages.append({"role": "system", "content": system_content + "\n" + str(msg.get("content", ""))})
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

    think_pattern = r"<think>(.*?)</think>"
    think_matches = list(re.finditer(think_pattern, clean_text, re.DOTALL))
    for match in think_matches:
        if enable_think:
            thinking_content += match.group(1).strip() + "\n"
        clean_text = clean_text.replace(match.group(0), "")

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
