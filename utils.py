import re
import base64
import tempfile
import os
import uuid


def apply_chat_template(messages, thinking=True):
    """
    Applies the ChatML format, compatible with most models including Qwen.
    Now supports Multimodal (Vision) payload parsing.
    """
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    prompt = ""

    for msg in messages:
        role = msg['role']
        content = msg['content']
        text_content = ""

        # --- Multimodal / Vision Parsing ---
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text_content += item.get("text", "")
                elif item.get("type") == "image_url":
                    img_url = item["image_url"]["url"]
                    if img_url.startswith("data:image"):
                        # Decode Base64 and save to temp file
                        header, encoded = img_url.split(",", 1)
                        img_data = base64.b64decode(encoded)
                        ext = "jpg" if "jpeg" in header or "jpg" in header else "png"
                        tmp_path = os.path.join(tempfile.gettempdir(), f"rkllm_vis_{uuid.uuid4().hex}.{ext}")
                        with open(tmp_path, "wb") as f:
                            f.write(img_data)
                        text_content = f"<image>{tmp_path}</image>\n" + text_content
                    else:
                        # Standard URL or local file path
                        text_content = f"<image>{img_url}</image>\n" + text_content
        else:
            text_content = str(content)
        # -----------------------------------

        # Remove chain-of-thought content if we don't want the model to see previous thought processes
        text_content = re.sub(r'<think>.*?</think>', '', text_content, flags=re.DOTALL)

        if thinking:
            prompt += f"{im_start}{role}\n{text_content}{im_end}\n"
        else:
            prompt += f"{im_start}{role}\n{text_content} /nothink{im_end}\n"

    # Append the assistant start tag to prompt the model to generate content
    prompt += f"{im_start}assistant\n"

    return prompt


def make_llm_response(llm_output: str) -> dict:
    """
    Defines the standard OpenAI-compatible structure for the returned response.
    """
    rkllm_responses = {
        "id": "chatcmpl-rkllm",
        "object": "chat.completion",
        "created": None,
        "model": "rkllm-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": llm_output,
                },
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
    return rkllm_responses