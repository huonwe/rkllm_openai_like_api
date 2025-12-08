import sys
import os
import subprocess
import resource
import threading
import argparse
import json
from flask import Flask, request, jsonify, Response
from flask_cors import cross_origin
from utils import apply_chat_template, make_llm_response
from rkllm import RKLLM, get_RKLLM_output, get_global_state

app = Flask(__name__)
# Create a lock to control multi-user access to the server.
lock = threading.Lock()
# Create a global variable to indicate whether the server is currently in a blocked state.
is_blocking = False
# Create a global variable to save the model path.
global_model = ""
# Create a function to receive data sent by the user using a request
@app.route('/rkllm_chat/v1/chat/completions', methods=['POST'])
@cross_origin()
def receive_message():
    # Link global variables to retrieve the output information from the callback function
    # global global_text, global_state
    global is_blocking

    # If the server is in a blocking state, return a specific response.
    if is_blocking or get_global_state()==0:
        resp = make_llm_response("⚠ RKLLM_Server 正忙碌! 请稍后再尝试. 这通常是由于Web UI前端正在执行自动生成标签、标题, 以及自动补齐等任务.")
        return jsonify(resp), 200
    
    lock.acquire()
    try:
        # Set the server to a blocking state.
        is_blocking = True

        # Get JSON data from the POST request.
        data = request.json
        if data and 'messages' in data:
            # Process the received data here.
            messages = data['messages']
            # messages.insert(0,{'role':'system','content':'You are a helpful assistant.'})
            # print("Received messages: ", messages)
            # tokenized = tokenizer.apply_chat_template(messages, tokenize=False)
            messages_formatted = apply_chat_template(messages)

            # 这里rkllm_model应当已经初始化完成并可用
            results = get_RKLLM_output(rkllm_model, messages_formatted)

            if data.get("stream", False):
                def stream_generator():
                    for r in results:
                        # print("streaming chunk: ", r)
                        yield f"data: {json.dumps({'choices':[
                            {'delta':{'content': r}}]})}\n\n"
                    yield f"data: [DONE]\n\n"
                return Response(stream_generator(), mimetype='text/event-stream')
            else:
                rkllm_output = ""
                for r in results:
                    rkllm_output += r
                rkllm_responses = make_llm_response(rkllm_output)
                return jsonify(rkllm_responses), 200

        else:
            return jsonify({'status': 'error', 'message': 'Invalid JSON data!'}), 400
    finally:
        lock.release()
        is_blocking = False

@app.route("/rkllm_chat/v1/models", methods=['GET'])
@cross_origin()
def show_models():
    global global_model
    info = json.dumps({"object": "list", "data": [{
        "id": f"{global_model}",
        "object": "model",
        "owned_by": "rkllm_server"
    }]})
    return Response(info, content_type="application/json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', type=str, default="models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm", help='Absolute path of the converted RKLLM model on the Linux board;')
    parser.add_argument('--target_platform', type=str, default="rk3588", help='Target platform: e.g., rk3588/rk3576;')
    parser.add_argument('--lora_model_path', type=str, help='Absolute path of the lora_model on the Linux board;')
    parser.add_argument('--prompt_cache_path', type=str, help='Absolute path of the prompt_cache file on the Linux board;')
    parser.add_argument('--port', type=int, default=8080, help='Port that the flask server will listen.')

    args = parser.parse_args()

    if not os.path.exists(args.rkllm_model_path):
        print("Error: Please provide the correct rkllm model path, and ensure it is the absolute path on the board.")
        sys.stdout.flush()
        exit()

    if not (args.target_platform in ["rk3588", "rk3576"]):
        print("Error: Please specify the correct target platform: rk3588/rk3576.")
        sys.stdout.flush()
        exit()

    if args.lora_model_path:
        if not os.path.exists(args.lora_model_path):
            print("Error: Please provide the correct lora_model path, and advise it is the absolute path on the board.")
            sys.stdout.flush()
            exit()

    if args.prompt_cache_path:
        if not os.path.exists(args.prompt_cache_path):
            print("Error: Please provide the correct prompt_cache_file path, and advise it is the absolute path on the board.")
            sys.stdout.flush()
            exit()

    # Fix frequency
    command = "sudo bash fix_freq_{}.sh".format(args.target_platform)
    subprocess.run(command, shell=True)

    # Set resource limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    # Initialize RKLLM model
    print("=========init....===========")
    sys.stdout.flush()
    global_model = args.rkllm_model_path
    rkllm_model = RKLLM(global_model, args.lora_model_path, args.prompt_cache_path, args.target_platform)
    print("RKLLM Model has been initialized successfully！")
    print("==============================")
    sys.stdout.flush()
        
    # Start the Flask application.
    app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)

    print("====================")
    print("RKLLM model inference completed, releasing RKLLM model resources...")
    rkllm_model.release()
    print("====================")
