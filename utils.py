import re


def apply_chat_template(messages, thinking=True):
    """
    Applies the ChatML format, compatible with most models including Qwen.
    RKLLM retains the original Tokenizer behavior unless specifically overridden.
    Standard special tokens for Qwen are <|im_start|> and <|im_end|>.
    """
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    prompt = ""

    for msg in messages:
        role = msg['role']
        content = msg['content']

        # Remove chain-of-thought content if we don't want the model to see previous thought processes
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # Format: <|im_start|>role\ncontent<|im_end|>\n
        # Append /nothink if thinking is disabled
        if thinking:
            prompt += f"{im_start}{role}\n{content}{im_end}\n"
        else:
            prompt += f"{im_start}{role}\n{content} /nothink{im_end}\n"

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