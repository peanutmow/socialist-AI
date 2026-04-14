import argparse
import os
from pathlib import Path

REQUIRED_MODEL_FILES = [
    "config.json",
]

OPTIONAL_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "tokenizer_config.json",
]

MODEL_WEIGHT_FILES = [
    "pytorch_model.bin",
    "pytorch_model.safetensors",
]


def has_split_safetensors(existing_files):
    return any(name.startswith("model.safetensors-") and name.endswith(".safetensors") for name in existing_files)


def find_existing_files(model_path: Path):
    return {p.name for p in model_path.iterdir() if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description="Check a local model directory for required files.")
    parser.add_argument("--model", type=str, default="models/qwen-3.5-9b", help="Local model directory path")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    print(f"Checking local model path: {model_path}")

    if not model_path.exists():
        print("ERROR: model directory does not exist.")
        print("Create the directory by downloading or copying a compatible model to this path.")
        print("Example: models/qwen-3.5-9b")
        raise SystemExit(1)

    if not model_path.is_dir():
        print("ERROR: the provided path is not a directory.")
        raise SystemExit(1)

    existing = find_existing_files(model_path)
    print(f"Found {len(existing)} files in the model directory.")

    missing_required = [f for f in REQUIRED_MODEL_FILES if f not in existing]
    if missing_required:
        print("Missing required model files:")
        for f in missing_required:
            print(f"  - {f}")
        raise SystemExit(1)

    weight_present = any(f in existing for f in MODEL_WEIGHT_FILES) or has_split_safetensors(existing)
    if not weight_present:
        print("ERROR: no model weights found. Expected one of:")
        for f in MODEL_WEIGHT_FILES:
            print(f"  - {f}")
        print("  - model.safetensors-00001-of-00004.safetensors (split safetensors)")
        raise SystemExit(1)

    if not any(f in existing for f in OPTIONAL_TOKENIZER_FILES):
        print("ERROR: no tokenizer files found. Expected one of:")
        for f in OPTIONAL_TOKENIZER_FILES:
            print(f"  - {f}")
        raise SystemExit(1)

    print("Model directory exists and contains the expected files.")
    print("You can now run the agent with --model <path> or set LOCAL_MODEL_PATH.")


if __name__ == "__main__":
    main()
