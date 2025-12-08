import requests
import json
import argparse
import sys

def chat_completions(host, prompt, stream=False):
    url = f"{host}/rkllm_chat/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    
    # 构造请求数据
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": stream
    }

    print(f"[-] Sending request to {url}...")
    print(f"[-] Stream mode: {'ON' if stream else 'OFF'}")
    print(f"[-] Prompt: {prompt}")
    print("-" * 40)

    try:
        response = requests.post(url, headers=headers, json=payload, stream=stream)
        
        if response.status_code != 200:
            print(f"[!] Error: Server returned status code {response.status_code}")
            print(response.text)
            return

        if stream:
            # 流式处理 (Streaming Mode)
            print("Response: ", end="", flush=True)
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    # 服务端发送格式为: data: {json_data}
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:] # 去掉 "data: " 前缀
                        
                        if data_str.strip() == "[DONE]":
                            print("\n[Done]")
                            break
                        
                        try:
                            data_json = json.loads(data_str)
                            # 获取 delta content
                            if "choices" in data_json and len(data_json["choices"]) > 0:
                                delta = data_json["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            continue
        else:
            # 非流式处理 (Non-Streaming Mode)
            data = response.json()
            # 适配服务端返回的结构
            if "choices" in data and len(data["choices"]) > 0:
                print("Response: " + data["choices"][0]["message"]["content"])
            else:
                print("Raw Response:", data)

    except requests.exceptions.ConnectionError:
        print(f"[!] Could not connect to server at {host}. Is it running?")
    except Exception as e:
        print(f"[!] An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RKLLM Chat Client Tester")
    
    parser.add_argument('--host', type=str, default="http://localhost:8081", 
                        help='Server address (default: http://localhost:8081)')
    parser.add_argument('--prompt', type=str, default="Hello, explain quantum mechanics briefly.", 
                        help='The input prompt to send to the model')
    parser.add_argument('--stream', action='store_true', 
                        help='Enable streaming mode')

    args = parser.parse_args()

    chat_completions(args.host, args.prompt, args.stream)