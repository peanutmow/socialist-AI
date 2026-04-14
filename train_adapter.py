import argparse
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import LocalModelConfig


@dataclass
class TrainingConfig:
    train_file: str = "data/socialist_instructions.jsonl"
    output_dir: str = "adapters/socialist-lora"
    epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class InstructionDataset(Dataset):
    def __init__(self, tokenizer, examples, max_length=1024):
        self.tokenizer = tokenizer
        self.examples = examples
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = self._build_prompt(example)
        tokenized = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length")
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        labels = [token_id if token_id != pad_id else -100 for token_id in tokenized["input_ids"]]
        tokenized["labels"] = labels
        return {k: torch.tensor(v) for k, v in tokenized.items()}

    def _build_prompt(self, example):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
        return prompt


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def _looks_like_local_path(path: str) -> bool:
    return any(sep in path for sep in (os.sep, "/")) or path.startswith(".") or path.startswith("~")


def train(config: LocalModelConfig, train_config: TrainingConfig):
    base_model_path = Path(config.base_model).expanduser()
    local_files_only = False
    if base_model_path.is_dir():
        config.base_model = str(base_model_path.resolve())
        local_files_only = True
    elif _looks_like_local_path(config.base_model):
        raise RuntimeError(
            f"Local model path does not exist: {config.base_model}\n"
            "Please download a compatible local model directory and point --model or LOCAL_MODEL_PATH to it."
        )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=False, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        load_in_4bit=config.load_in_4bit,
        device_map=config.device_map,
        torch_dtype=torch.float16,
        local_files_only=local_files_only,
    )

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=train_config.lora_r,
        lora_alpha=train_config.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=train_config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    examples = list(load_jsonl(train_config.train_file))
    dataset = InstructionDataset(tokenizer, examples)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        per_device_train_batch_size=train_config.batch_size,
        num_train_epochs=train_config.epochs,
        learning_rate=train_config.learning_rate,
        fp16=True,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(train_config.output_dir)


def main():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for socialist topic specialization.")
    parser.add_argument("--train-file", type=str, default="data/socialist_instructions.jsonl", help="Path to a JSONL training file.")
    parser.add_argument("--output-dir", type=str, default="adapters/socialist-lora", help="Directory to save the adapter.")
    parser.add_argument("--model", type=str, default=None, help="Optional local base model directory or Hugging Face ID.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    config = LocalModelConfig()
    if args.model:
        config.base_model = args.model

    train_config = TrainingConfig(
        train_file=args.train_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    os.makedirs(train_config.output_dir, exist_ok=True)
    train(config, train_config)


if __name__ == "__main__":
    main()
