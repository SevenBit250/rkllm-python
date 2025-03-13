import ctypes
import platform
from ctypes import (
    c_void_p, c_int32, c_uint8, c_float, c_size_t, POINTER, Structure, Union, CFUNCTYPE, c_char_p, c_bool
)


# Define the structures from the library
RKLLM_Handle_t   = ctypes.c_void_p

LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL                = 0
LLMCallState.RKLLM_RUN_WAITING               = 1
LLMCallState.RKLLM_RUN_FINISH                = 2
LLMCallState.RKLLM_RUN_ERROR                 = 3
LLMCallState.RKLLM_RUN_GET_LAST_HIDDEN_LAYER = 4

RKLLMInputMode = ctypes.c_int
RKLLMInputMode.RKLLM_INPUT_PROMPT            = 0
RKLLMInputMode.RKLLM_INPUT_TOKEN             = 1
RKLLMInputMode.RKLLM_INPUT_EMBED             = 2
RKLLMInputMode.RKLLM_INPUT_MULTIMODAL        = 3

RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE              = 0
RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1

class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("reserved", ctypes.c_uint8 * 112)
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
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

class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t)
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput)
    ]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("input_mode", ctypes.c_int),
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
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam))
    ]

class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("size", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer)
    ]

LLMResultCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)


class RKLLM:
    def __init__(self, lib_path : str):
        if platform.system() != "Linux":
            raise Exception("Only Linux is supported for now.")
        self.lib = ctypes.CDLL(lib_path)
        self.handle = RKLLM_Handle_t()
        self._init_function_prototypes()

    def _init_function_prototypes(self):
        """Initialize function prototypes for the shared library."""
        self.lib.rkllm_createDefaultParam.restype = RKLLMParam

        self.lib.rkllm_init.argtypes = [POINTER(RKLLM_Handle_t), POINTER(RKLLMParam), LLMResultCallback]
        self.lib.rkllm_init.restype = c_int32

        self.lib.rkllm_load_lora.argtypes = [RKLLM_Handle_t, POINTER(RKLLMLoraAdapter)]
        self.lib.rkllm_load_lora.restype = c_int32

        self.lib.rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, c_char_p]
        self.lib.rkllm_load_prompt_cache.restype = c_int32

        self.lib.rkllm_release_prompt_cache.argtypes = [RKLLM_Handle_t]
        self.lib.rkllm_release_prompt_cache.restype = c_int32

        self.lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.lib.rkllm_destroy.restype = c_int32

        self.lib.rkllm_run.argtypes = [RKLLM_Handle_t, POINTER(RKLLMInput), POINTER(RKLLMInferParam), c_void_p]
        self.lib.rkllm_run.restype = c_int32

        self.lib.rkllm_run_async.argtypes = [RKLLM_Handle_t, POINTER(RKLLMInput), POINTER(RKLLMInferParam), c_void_p]
        self.lib.rkllm_run_async.restype = c_int32

        self.lib.rkllm_abort.argtypes = [RKLLM_Handle_t]
        self.lib.rkllm_abort.restype = c_int32

        self.lib.rkllm_is_running.argtypes = [RKLLM_Handle_t]
        self.lib.rkllm_is_running.restype = c_int32

    def create_default_param(self) -> RKLLMParam:
        """Create a default RKLLMParam structure."""
        return self.lib.rkllm_createDefaultParam()

    def init(self, param: RKLLMParam, callback) -> int:
        """Initialize the LLM with the given parameters."""
        return self.lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(param), callback)

    def load_lora(self, lora_adapter: RKLLMLoraAdapter) -> int:
        """Load a Lora adapter into the LLM."""
        return self.lib.rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))

    def load_prompt_cache(self, prompt_cache_path: str) -> int:
        """Load a prompt cache from a file."""
        return self.lib.rkllm_load_prompt_cache(self.handle, ctypes.c_char_p((prompt_cache_path).encode('utf-8')))

    def release_prompt_cache(self) -> int:
        """Release the prompt cache from memory."""
        return self.lib.rkllm_release_prompt_cache(self.handle)

    def destroy(self) -> int:
        """Destroy the LLM instance and release resources."""
        return self.lib.rkllm_destroy(self.handle)

    def run(self, rkllm_input: RKLLMInput, rkllm_infer_params: RKLLMInferParam, userdata) -> int:
        """Run an LLM inference task synchronously."""
        return self.lib.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), userdata)

    def run_async(self, rkllm_input: RKLLMInput, rkllm_infer_params: RKLLMInferParam, userdata) -> int:
        """Run an LLM inference task asynchronously."""
        return self.lib.rkllm_run_async(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), userdata)

    def abort(self) -> int:
        """Abort an ongoing LLM task."""
        return self.lib.rkllm_abort(self.handle)

    def is_running(self) -> int:
        """Check if an LLM task is currently running."""
        return self.lib.rkllm_is_running(self.handle)
