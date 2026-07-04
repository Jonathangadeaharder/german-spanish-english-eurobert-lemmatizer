"""Fine-tune ByT5-small for Arabic lemmatization + UPOS tagging.

Multitask Seq2Seq: input is a prompted sentence with <target> markers,
output is "UPOS | lemma" generated autoregressively.

Byte-level tokenization (ByT5 is tokenizer-free) handles Arabic morphology
naturally — no edit-tree label explosion.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_NAME = "google/byt5-small"
DATA_DIR = Path("data/processed/ar_seq2seq")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "runs/byt5-ar-lemma"))
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 32


def load_jsonl(path: Path) -> Dataset:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return Dataset.from_list(rows)


def main():
    epochs = float(os.getenv("EPOCHS", "5"))
    lr = float(os.getenv("LR", "3e-4"))
    batch_size = int(os.getenv("BATCH_SIZE", "8"))
    grad_accum = int(os.getenv("GRAD_ACCUM", "4"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    print(f"Model: {MODEL_NAME}, params: {sum(p.numel() for p in model.parameters()):,}")

    def preprocess(examples):
        inputs = tokenizer(
            examples["input"],
            max_length=MAX_INPUT_LENGTH,
            padding="max_length",
            truncation=True,
        )
        targets = tokenizer(
            text_target=examples["target"],
            max_length=MAX_TARGET_LENGTH,
            padding="max_length",
            truncation=True,
        )
        inputs["labels"] = targets["input_ids"]
        inputs["labels"] = [
            [-100 if t == tokenizer.pad_token_id else t for t in label]
            for label in inputs["labels"]
        ]
        return inputs

    train_ds = load_jsonl(DATA_DIR / "train.jsonl").map(preprocess, batched=True, remove_columns=["input", "target"])
    val_ds = load_jsonl(DATA_DIR / "validation.jsonl").map(preprocess, batched=True, remove_columns=["input", "target"])

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        fp16=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save final model
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"Saved to {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
