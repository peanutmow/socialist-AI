import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "Qwen/Qwen3.5-9B"


def main():
    parser = argparse.ArgumentParser(description="Download the Qwen 3.5 9B model locally from Hugging Face.")
    parser.add_argument("--model-dir", type=str, default="models/qwen-3.5-9b", help="Local directory to save the model")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face model repo id")
    parser.add_argument("--revision", type=str, default="main", help="Model revision or branch")
    args = parser.parse_args()

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "Please set the HUGGINGFACE_HUB_TOKEN environment variable with your Hugging Face access token.\n"
            "You can create one at https://huggingface.co/settings/tokens."
        )

    model_dir = Path(args.model_dir).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} into {model_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        repo_type="model",
        token=token,
    )

    print("Download complete.")
    print("Validate the local model directory with:")
    print(f"  python validate_model.py --model {model_dir}")


if __name__ == "__main__":
    main()
