import re

def apply_chat_template(messages):
    """
    适用于大部分采用 ChatML 格式的模型。
    """
    # 定义标准 ChatML 令牌
    # 注意：RKLLM 转换时如果没有特殊指定，通常保留了原模型的 Tokenizer 行为
    # Qwen 系列的标准特殊 Token 是 <|im_start|> 和 <|im_end|>
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    
    # 2. 构建 Prompt
    prompt = ""
    
    for msg in messages:
        role = msg['role']
        content = msg['content']
        
        # 移除思维链内容（如果不想让模型看到之前的思考过程，保留此逻辑）
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # 格式：<|im_start|>role\ncontent<|im_end|>\n
        prompt += f"{im_start}{role}\n{content}{im_end}\n"

    # 追加助手角色的起始标记，等待模型生成内容
    prompt += f"{im_start}assistant\n"
    
    return prompt

def make_llm_response(llm_output: str) -> dict:
    # Define the structure for the returned response.
    rkllm_responses = {
        "id": "rkllm_chat",
        "object": "rkllm_chat",
        "created": None,
        "choices": [],
        "usage": {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None
        }
    }
    rkllm_responses["choices"].append(
        {"index": 0,
        "message": {
            "role": "assistant",
            "content": llm_output,
        },
        "logprobs": None,
        "finish_reason": "stop"
        }
    )
    return rkllm_responses