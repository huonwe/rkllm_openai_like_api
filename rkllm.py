import ctypes
import sys
import os
import threading
import queue

# Set the dynamic library path
rkllm_lib = ctypes.CDLL('lib/librkllmrt.so')

# Define the structures from the library
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL = 0
LLMCallState.RKLLM_RUN_WAITING = 1
LLMCallState.RKLLM_RUN_FINISH = 2
LLMCallState.RKLLM_RUN_ERROR = 3

RKLLMInputType = ctypes.c_int
RKLLMInputType.RKLLM_INPUT_PROMPT = 0
RKLLMInputType.RKLLM_INPUT_TOKEN = 1
RKLLMInputType.RKLLM_INPUT_EMBED = 2
RKLLMInputType.RKLLM_INPUT_MULTIMODAL = 3

RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0
RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLMInferMode.RKLLM_INFER_GET_LOGITS = 2


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104)
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMLoraAdapter(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale", ctypes.c_float)
    ]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t)
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t)
    ]


class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t)
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput)
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", RKLLMInputType),
        ("input_data", RKLLMInputUnion)
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_name", ctypes.c_char_p)
    ]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p)
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int)
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float)
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat)
    ]


# Define thread-safe queue for retrieving generated tokens and embeddings
output_queue = queue.Queue()
global_state = -1


# Callback function to receive data from C++ runtime
def callback_impl(result, userdata, state):
    global global_state
    if state == LLMCallState.RKLLM_RUN_FINISH:
        global_state = state

        # Extract Embeddings if they exist in the payload
        if result and result.contents.last_hidden_layer.embd_size > 0:
            embd_size = result.contents.last_hidden_layer.embd_size
            hidden_states_ptr = ctypes.cast(
                result.contents.last_hidden_layer.hidden_states,
                ctypes.POINTER(ctypes.c_float * embd_size)
            )
            # Safely cast pointer array to Python list
            vector = [float(hidden_states_ptr.contents[i]) for i in range(embd_size)]
            output_queue.put({"embedding": vector})

        output_queue.put(None)  # Sentinel to mark end of generation
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        global_state = state
        output_queue.put(Exception("RKLLM Runtime Error"))
    elif state == LLMCallState.RKLLM_RUN_NORMAL:
        global_state = state
        text = result.contents.text.decode('utf-8')
        output_queue.put(text)
    return 0


callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)


class RKLLM(object):
    """
    RKLLM class handles initialization, inference, and release operations
    for the RKLLM model bound dynamically from librkllmrt.so.
    """

    def __init__(self, model_path, lora_model_path=None, prompt_cache_path=None, platform="rk3588"):
        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        rkllm_param.max_context_len = 4096
        rkllm_param.max_new_tokens = 4096
        rkllm_param.skip_special_token = True
        rkllm_param.n_keep = -1
        rkllm_param.top_k = 1
        rkllm_param.top_p = 0.9
        rkllm_param.temperature = 0.8
        rkllm_param.repeat_penalty = 1.1
        rkllm_param.frequency_penalty = 0.0
        rkllm_param.presence_penalty = 0.0
        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1
        rkllm_param.is_async = False

        # Set valid Vision tags
        rkllm_param.img_start = "<image>".encode('utf-8')
        rkllm_param.img_end = "</image>".encode('utf-8')
        rkllm_param.img_content = "".encode('utf-8')

        rkllm_param.extend_param.base_domain_id = 0
        rkllm_param.extend_param.embed_flash = 1
        rkllm_param.extend_param.n_batch = 1
        rkllm_param.extend_param.use_cross_attn = 0
        rkllm_param.extend_param.enabled_cpus_num = 4

        if platform.lower() in ["rk3576", "rk3588"]:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
        else:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)

        self.handle = RKLLM_Handle_t()

        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int

        ret = self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), callback)
        if ret != 0:
            print("\n[Error] RKLLM initialization failed\n")
            exit(1)
        else:
            print("\n[Success] RKLLM initialization successful!\n")

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam),
                                   ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int

        self.set_chat_template = rkllm_lib.rkllm_set_chat_template
        self.set_chat_template.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.set_chat_template.restype = ctypes.c_int

        self.set_function_tools_ = rkllm_lib.rkllm_set_function_tools
        self.set_function_tools_.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.set_function_tools_.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

        self.rkllm_abort = rkllm_lib.rkllm_abort

        rkllm_lora_params = None
        if lora_model_path:
            lora_adapter_name = "default_lora"
            lora_adapter = RKLLMLoraAdapter()
            ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
            lora_adapter.lora_adapter_path = ctypes.c_char_p((lora_model_path).encode('utf-8'))
            lora_adapter.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode('utf-8'))
            lora_adapter.scale = 1.0

            rkllm_load_lora = rkllm_lib.rkllm_load_lora
            rkllm_load_lora.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
            rkllm_load_lora.restype = ctypes.c_int
            rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))
            rkllm_lora_params = RKLLMLoraParam()
            rkllm_lora_params.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode('utf-8'))

        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self.rkllm_infer_params.lora_params = ctypes.pointer(rkllm_lora_params) if rkllm_lora_params else None
        self.rkllm_infer_params.keep_history = 0

        self.prompt_cache_path = None
        if prompt_cache_path:
            self.prompt_cache_path = prompt_cache_path
            rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
            rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
            rkllm_load_prompt_cache.restype = ctypes.c_int
            rkllm_load_prompt_cache(self.handle, ctypes.c_char_p((prompt_cache_path).encode('utf-8')))

        self.tools = None

    def set_function_tools(self, system_prompt, tools, tool_response_str):
        if self.tools is None or not self.tools == tools:
            self.tools = tools
            self.set_function_tools_(self.handle, ctypes.c_char_p(system_prompt.encode('utf-8')),
                                     ctypes.c_char_p(tools.encode('utf-8')),
                                     ctypes.c_char_p(tool_response_str.encode('utf-8')))

    def run(self, role, enable_thinking, prompt):
        rkllm_input = RKLLMInput()
        rkllm_input.role = role.encode('utf-8') if role is not None else "user".encode('utf-8')
        rkllm_input.enable_thinking = ctypes.c_bool(enable_thinking if enable_thinking is not None else False)
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode('utf-8'))
        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)

    def get_embedding(self, prompt):
        """Switches the NPU to embedding mode, extracts vectors, and switches back."""
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER

        rkllm_input = RKLLMInput()
        rkllm_input.role = b"user"
        rkllm_input.enable_thinking = False
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode('utf-8'))

        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)

        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE

    def abort(self):
        return self.rkllm_abort(self.handle)

    def release(self):
        self.rkllm_destroy(self.handle)


def get_RKLLM_output(rkllm_model, chat_formatted):
    """
    Generator function to stream tokens from the RKLLM runtime.
    """
    global global_state

    while not output_queue.empty():
        output_queue.get_nowait()

    global_state = -1

    model_thread = threading.Thread(target=rkllm_model.run, args=('system', True, chat_formatted,))
    model_thread.start()

    try:
        while True:
            item = output_queue.get()

            if item is None:
                break

            if isinstance(item, Exception):
                raise item

            print(item, end="", flush=True)
            yield item

    except GeneratorExit:
        print("\n[Info] Client disconnected! Aborting RKLLM inference...")
        rkllm_model.abort()
        raise

    except Exception as e:
        print(f"\n[Error] Inference error: {e}")
        rkllm_model.abort()
        raise

    finally:
        if model_thread.is_alive():
            model_thread.join()
        print("\n[Info] Inference thread finished.")


def get_RKLLM_embeddings(rkllm_model, text: str):
    """Blocking function to retrieve embeddings securely via the queue."""
    global global_state

    while not output_queue.empty():
        output_queue.get_nowait()

    global_state = -1

    thread = threading.Thread(target=rkllm_model.get_embedding, args=(text,))
    thread.start()

    embedding_vector = []
    try:
        while True:
            item = output_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            if isinstance(item, dict) and "embedding" in item:
                embedding_vector = item["embedding"]
    finally:
        thread.join()

    return embedding_vector


def get_global_state():
    return global_state