# socialist-AI

**socialist-AI** is a local multi-agent inference framework for socialist analysis. It runs a shared local LLM through several role-based agents — a theoretician, a historian, a strategist, and a democratic socialist critic — then synthesizes their discussion into a concise consensus answer.

## Why this project
- Demonstrates multi-agent reasoning with a single local model.
- Emphasizes socialist theory, historical materialism, and democratic pluralism.
- Supports hidden agent reasoning while keeping visible output clean.
- Designed for portable local inference using 4-bit quantization.

## Features
- Multi-agent debate architecture with role-based prompts
- Batch generation for efficient round-based discussion
- Streaming and batch UI support via `gui.py`
- Hidden reasoning capture for developer inspection
- Optional LoRA adapter training with `train_adapter.py`
- Benchmark mode for measuring model performance

## Repository layout
- `agent_controller.py` — multi-agent debate orchestration and final-answer generation
- `gui.py` — desktop-style interface for running debates locally
- `prompts.py` — role prompts and conversation templates
- `model_utils.py` — model loading, generation helpers, and reasoning filtering
- `config.py` — default model settings and inference parameters
- `train_adapter.py` — adapter training script for domain specialization
- `validate_model.py` — lightweight model path validation
- `download_qwen_model.py` — helper for retrieving a local Qwen model
- `data/socialist_instructions.jsonl` — sample dataset placeholder for training
- `requirements.txt` — core Python dependencies

## Requirements
- Python 3.10+ (recommended)
- PyTorch with CUDA support for GPU inference
- `transformers`, `accelerate`, `bitsandbytes`, `peft`, and related packages

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Model setup
This repository does not include model weights.
A compatible local model directory is required, for example:

```text
models/qwen-3.5-9b
```

The default config points at `models/qwen-3.5-9b`, but you can override it with `LOCAL_MODEL_PATH`.

### Recommended model
- `Qwen/Qwen3.5-9B` (local directory with tokenizer and weight files)
- 4-bit quantization is used by default for lower VRAM usage

### Download example
```powershell
python -m pip install huggingface_hub
$env:HUGGINGFACE_HUB_TOKEN = "<your_token_here>"
python download_qwen_model.py --model-dir models/qwen-3.5-9b --repo-id Qwen/Qwen3.5-9B
```

## Run the CLI
Run a single debate question:

```bash
python agent_controller.py "What is the socialist critique of corporate power?"
```

Or explicitly set the model path:

```bash
LOCAL_MODEL_PATH=models/qwen-3.5-9b python agent_controller.py "Can socialism be democratic?"
```

## Run the GUI
Start the desktop interface:

```bash
python gui.py
```

The GUI provides a retro-styled debate terminal, sidebar controls, and a hidden reasoning window for internal debug output.

## Configurable options
The core configuration is in `config.py`, including:
- `load_in_4bit` — enable 4-bit quantization
- `use_torch_compile` — use `torch.compile()` for repeated inference
- `attn_implementation` — choose attention kernel (`sdpa` by default)
- `agent_max_new_tokens` — max response length per agent
- `debate_rounds` — number of discussion rounds
- `soft_token_limit` — allow buffer tokens for safe generation

## Benchmark mode
Measure inference performance with a simple benchmark:

```bash
python agent_controller.py --benchmark --benchmark-output benchmark_results.md --benchmark-seed 42 "What is the socialist critique of the state?"
```

Benchmark mode runs a small suite of cases and writes a Markdown report.

## Training adapters
If you want to specialize the system via LoRA, add examples to `data/socialist_instructions.jsonl` and run:

```bash
python train_adapter.py
```

## Notes for public use
- This repo is designed for local inference only.
- Do not commit large model files or private credentials.
- `models/` and `adapters/` are excluded from the public repository via `.gitignore`.
- The project is best used on a GPU with at least ~10GB VRAM.

## Contribution
Contributions are welcome. Suggested improvements:
- add more socialist role prompts
- refine hidden reasoning extraction
- add example discussion prompts and training data
- support additional lightweight LLM backends

## License
Include your preferred license when publishing this repository.
