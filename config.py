import os
from dataclasses import dataclass

DEFAULT_LOCAL_MODEL_PATH = os.path.join("models", "qwen-3.5-9b")

@dataclass
class LocalModelConfig:
    base_model: str = os.environ.get("LOCAL_MODEL_PATH", DEFAULT_LOCAL_MODEL_PATH)
    adapter_dir: str = "adapters/socialist-lora"
    device_map: str = "auto"
    load_in_4bit: bool = True
    use_torch_compile: bool = False
    # "sdpa" uses PyTorch 2.0+ built-in efficient attention (zero install). Set to "" to use the model default.
    attn_implementation: str = "sdpa"
    # Enable bfloat16 compute dtype; auto-detected from GPU. Ignored on CPU.
    compute_dtype_bfloat16: bool = False
    max_input_tokens: int = 2048
    # Maximum characters of cumulative discussion history passed to each agent per round.
    max_discussion_chars: int = 4000
    max_new_tokens: int = 300
    # Per-agent response cap used during batched round generation. Lower = faster.
    agent_max_new_tokens: int = 200
    agent_extra_tokens: int = 50
    summary_extra_tokens: int = 50
    soft_token_limit: bool = True
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    num_beams: int = 1
    no_repeat_ngram_size: int = 2
    debate_rounds: int = 2
    consensus_tolerance: float = 0.85
    summary_beams: int = 4
    summary_max_new_tokens: int = 128
    summary_temperature: float = 0.3
    summary_do_sample: bool = False
