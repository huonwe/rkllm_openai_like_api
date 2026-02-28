import requests
import json
import argparse
import sys


def chat_completions(host, prompt, stream=False):
    """Tests the /v1/chat/completions endpoint."""
    url = f"{host}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream
    }

    print(f"[-] Sending Chat request to {url}...")
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
            print("Response: ", end="", flush=True)
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]

                        if data_str.strip() == "[DONE]":
                            print("\n[Done]")
                            break

                        try:
                            data_json = json.loads(data_str)
                            if "choices" in data_json and len(data_json["choices"]) > 0:
                                delta = data_json["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            continue
        else:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                print("Response: " + data["choices"][0]["message"]["content"])
            else:
                print("Raw Response:", data)

    except requests.exceptions.ConnectionError:
        print(f"[!] Could not connect to server at {host}. Is it running?")
    except Exception as e:
        print(f"[!] An error occurred: {e}")


def get_embeddings(host, prompt):
    """Tests the new /v1/embeddings endpoint."""
    url = f"{host}/v1/embeddings"
    headers = {"Content-Type": "application/json"}

    payload = {
        "input": prompt,
        "model": "rkllm-model"
    }

    print(f"[-] Sending Embeddings request to {url}...")
    print(f"[-] Input: {prompt}")
    print("-" * 40)

    try:
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            print(f"[!] Error: Server returned status code {response.status_code}")
            print(response.text)
            return

        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            vector = data["data"][0]["embedding"]
            print(f"Response: Received embedding vector of length {len(vector)}")
            print(f"First 5 values: {vector[:5]} ...")
        else:
            print("Raw Response:", data)

    except requests.exceptions.ConnectionError:
        print(f"[!] Could not connect to server at {host}. Is it running?")
    except Exception as e:
        print(f"[!] An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RKLLM API Client Tester")

    parser.add_argument('--host', type=str, default="http://localhost:8080",
                        help='Server address (default: http://localhost:8080)')
    parser.add_argument('--prompt', type=str, default="Hello, explain quantum mechanics briefly.",
                        help='The input text to send to the model')
    parser.add_argument('--stream', action='store_true',
                        help='Enable streaming mode for chat')
    parser.add_argument('--embeddings', action='store_true',
                        help='Test the embeddings endpoint instead of chat')

    args = parser.parse_args()

    if args.embeddings:
        get_embeddings(args.host, args.prompt)
    else:
        chat_completions(args.host, args.prompt, args.stream)