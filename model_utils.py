import gc
import os
from pathlib import Path
from threading import Thread
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed, TextIteratorStreamer
from peft import PeftModel


def _looks_like_local_path(path: str) -> bool:
    if path.startswith(".") or path.startswith("~"):
        return True
    if os.path.isabs(path):
        return True
    return any(sep in path for sep in (os.sep, "/"))


def load_model_and_tokenizer(config):
    base_model = config.base_model
    base_model_path = Path(base_model).expanduser()
    local_files_only = False

    if base_model_path.is_dir():
        base_model = str(base_model_path.resolve())
        local_files_only = True
    elif _looks_like_local_path(base_model):
        raise RuntimeError(
            f"Local model path does not exist: {base_model}\n"
            "Please download a compatible local model directory and point --model or LOCAL_MODEL_PATH to it.\n"
            "Example: --model models/qwen-3.5-9b"
        )

    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")
    if use_cuda and config.device_map == "auto":
        config.device_map = {"": "cuda:0"}
    elif not use_cuda:
        print("Warning: CUDA not available. The model will run on CPU and may be extremely slow.")

    # Pick compute dtype: prefer bfloat16 on supported GPUs when opted in, otherwise float16.
    if use_cuda and getattr(config, "compute_dtype_bfloat16", False) and torch.cuda.is_bf16_supported():
        _compute_dtype = torch.bfloat16
        print("Using bfloat16 compute dtype.")
    else:
        _compute_dtype = torch.float16

    print(f"Loading tokenizer from {base_model} (local_files_only={local_files_only})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=False,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load tokenizer for base model. "
            "If you are using a gated Hugging Face repo, either authenticate or set LOCAL_MODEL_PATH to a local model directory. "
            f"Original error: {exc}"
        ) from exc

    print(f"Loading model from {base_model} (4-bit={config.load_in_4bit})")
    _attn_impl = getattr(config, "attn_implementation", "sdpa") or None
    try:
        if config.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_compute_dtype,
            )
            load_kwargs = dict(
                quantization_config=quant_config,
                device_map=config.device_map,
                torch_dtype=_compute_dtype,
                local_files_only=local_files_only,
            )
        else:
            load_kwargs = dict(
                device_map=config.device_map,
                torch_dtype=_compute_dtype,
                local_files_only=local_files_only,
            )
        if _attn_impl:
            load_kwargs["attn_implementation"] = _attn_impl
        try:
            model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
            if _attn_impl:
                print(f"Using attention implementation: {_attn_impl}")
        except (TypeError, ValueError, NotImplementedError):
            # Fall back gracefully if the model or transformers version doesn't support the attn param.
            load_kwargs.pop("attn_implementation", None)
            print(f"Warning: attn_implementation='{_attn_impl}' not supported; using model default.")
            model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load model weights. "
            "If you are using a gated Hugging Face repo, either authenticate or set LOCAL_MODEL_PATH to a local model directory. "
            f"Original error: {exc}"
        ) from exc

    print(f"Loaded model. device_map={getattr(model, 'hf_device_map', None)}, device={getattr(model, 'device', 'unknown')}")

    if getattr(config, "use_torch_compile", False):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile.")
        except Exception as exc:
            print("Warning: torch.compile failed and will be skipped:", exc)

    if os.path.isdir(config.adapter_dir):
        print(f"Applying adapter from {config.adapter_dir}")
        model = PeftModel.from_pretrained(
            model,
            config.adapter_dir,
            device_map=config.device_map,
            local_files_only=True,
        )

    return model, tokenizer


def flush_cuda_cache():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def _resolve_max_new_tokens(config, base_tokens, extra_attr):
    if getattr(config, "soft_token_limit", False):
        return base_tokens + getattr(config, extra_attr, 0)
    return base_tokens


class HiddenReasoningFilter:
    def __init__(self):
        self.buffer = ""
        self.hidden = ""
        self.in_think = False

    def _has_incomplete_token_prefix(self, token: str) -> bool:
        max_prefix = min(len(self.buffer), len(token) - 1)
        for length in range(max_prefix, 0, -1):
            if self.buffer.endswith(token[:length]):
                return True
        return False

    def feed(self, chunk):
        self.buffer += chunk
        outputs = []

        while True:
            if self.in_think:
                end_index = self.buffer.find("</think>")
                if end_index == -1:
                    self.hidden += self.buffer
                    self.buffer = ""
                    break
                self.hidden += self.buffer[:end_index]
                self.buffer = self.buffer[end_index + len("</think>") :]
                self.in_think = False
                continue

            start_index = self.buffer.find("<think>")
            if start_index == -1:
                if self._has_incomplete_token_prefix("<think>"):
                    break
                if self.buffer:
                    outputs.append(self.buffer)
                    self.buffer = ""
                break

            outputs.append(self.buffer[:start_index])
            self.buffer = self.buffer[start_index + len("<think>") :]
            self.in_think = True

        return outputs

    def flush(self):
        if self.in_think:
            self.hidden += self.buffer
            self.buffer = ""
            return []

        if self.buffer:
            result = [self.buffer]
            self.buffer = ""
            return result

        return []

    def get_hidden(self):
        return self.hidden


def strip_hidden_reasoning(text):
    visible = []
    hidden = []
    cursor = 0

    while True:
        think_start = text.find("<think>", cursor)
        if think_start == -1:
            visible.append(text[cursor:])
            break

        visible.append(text[cursor:think_start])
        think_end = text.find("</think>", think_start + len("<think>"))
        if think_end == -1:
            hidden.append(text[think_start + len("<think>") :])
            break

        hidden.append(text[think_start + len("<think>") : think_end])
        cursor = think_end + len("</think>")

    return "".join(visible), "".join(hidden)


def generate_text_stream(model, tokenizer, prompt, config, generation_kwargs=None, seed=None):
    """Generate text using a background thread and a Hugging Face TextIteratorStreamer."""
    if seed is not None:
        set_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_input_tokens)
    if tokenizer.eos_token_id is None and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    elif tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    if generation_kwargs and "max_new_tokens" in generation_kwargs:
        max_new = generation_kwargs["max_new_tokens"]
    else:
        max_new = _resolve_max_new_tokens(config, getattr(config, "agent_max_new_tokens", config.max_new_tokens), "agent_extra_tokens")
    generation_params = {
        "max_new_tokens": max_new,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "repetition_penalty": config.repetition_penalty,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "streamer": streamer,
    }

    if getattr(config, "top_k", None) is not None and config.top_k > 0:
        generation_params["top_k"] = config.top_k
    if getattr(config, "no_repeat_ngram_size", None) is not None and config.no_repeat_ngram_size > 0:
        generation_params["no_repeat_ngram_size"] = config.no_repeat_ngram_size

    if generation_kwargs:
        generation_params.update(generation_kwargs)

    thread = Thread(target=model.generate, kwargs={**inputs, **generation_params})
    thread.start()
    return streamer, thread


def generate_text_batch(model, tokenizer, prompts, config, generation_kwargs=None, seed=None):
    """Generate text for a list of prompts in a single batched forward pass.

    Uses left-padding so all sequences are right-aligned, which is required
    for correct causal LM generation. The tokenizer's padding_side is
    temporarily set to 'left' and restored afterward.
    """
    if seed is not None:
        set_seed(seed)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_input_tokens,
        )
    finally:
        tokenizer.padding_side = original_padding_side

    input_length = inputs["input_ids"].shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print(f"Batch generating {len(prompts)} prompts, padded input length {input_length} tokens")

    if generation_kwargs and "max_new_tokens" in generation_kwargs:
        max_new = generation_kwargs["max_new_tokens"]
    else:
        max_new = _resolve_max_new_tokens(config, getattr(config, "agent_max_new_tokens", config.max_new_tokens), "agent_extra_tokens")
    generation_params = {
        "max_new_tokens": max_new,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "repetition_penalty": config.repetition_penalty,
        "num_beams": config.num_beams,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if getattr(config, "top_k", None) is not None and config.top_k > 0:
        generation_params["top_k"] = config.top_k
    if getattr(config, "no_repeat_ngram_size", None) is not None and config.no_repeat_ngram_size > 0:
        generation_params["no_repeat_ngram_size"] = config.no_repeat_ngram_size
    if generation_kwargs:
        generation_params.update(generation_kwargs)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_params)

    results = []
    for out in output_ids:
        new_tokens = out[input_length:]
        results.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    print(f"Batch generation complete: {len(results)} responses, up to {max_new} new tokens each")
    return results


def generate_text(model, tokenizer, prompt, config, generation_kwargs=None, seed=None):
    if seed is not None:
        set_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_input_tokens)
    if tokenizer.eos_token_id is None and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    elif tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print(f"Generating output for prompt length {inputs['input_ids'].shape[-1]} tokens")

    if generation_kwargs and "max_new_tokens" in generation_kwargs:
        max_new = generation_kwargs["max_new_tokens"]
    else:
        max_new = _resolve_max_new_tokens(config, config.max_new_tokens, "agent_extra_tokens")
    generation_params = {
        "max_new_tokens": max_new,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "repetition_penalty": config.repetition_penalty,
        "num_beams": config.num_beams,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    if getattr(config, "top_k", None) is not None and config.top_k > 0:
        generation_params["top_k"] = config.top_k
    if getattr(config, "no_repeat_ngram_size", None) is not None and config.no_repeat_ngram_size > 0:
        generation_params["no_repeat_ngram_size"] = config.no_repeat_ngram_size

    if generation_kwargs:
        generation_params.update(generation_kwargs)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_params)
    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"Generated {len(output_ids[0]) - inputs['input_ids'].shape[-1]} tokens")
    return generated.strip()
