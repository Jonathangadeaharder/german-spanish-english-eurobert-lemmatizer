"""Fine-tune bert-base-chinese for Chinese BIO-POS classification.

Architecture: pruned BERT (4 layers) + token classification head.
Input: raw Chinese characters (1 char = 1 token).
Output: BIO-POS label per character (35 classes: O + 17 B-X + 17 I-X).
Lemma = surface form (trivial copy in TypeScript post-processing).

Single ONNX graph, O(1) inference.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "bert-base-chinese"
DATA_DIR = Path("data/processed/zh_bio")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "runs/zh-bio-pos"))
MAX_LENGTH = 256
PRUNE_LAYERS = int(os.getenv("PRUNE_LAYERS", "8"))  # keep first N layers


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ChineseBioDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer, label2id: dict[str, int]):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        chars = item["chars"]
        labels = item["labels"]
        # Tokenize: [CLS] chars... [SEP]
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        # Labels: -100 for [CLS] and [SEP] and padding
        label_ids = [-100] * MAX_LENGTH
        for i, label_id in enumerate(labels[:MAX_LENGTH - 2]):
            label_ids[i + 1] = label_id  # offset by 1 for [CLS]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


def main():
    epochs = float(os.getenv("EPOCHS", "5"))
    lr = float(os.getenv("LR", "2e-5"))
    batch_size = int(os.getenv("BATCH_SIZE", "16"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=35,  # O + 17 B-X + 17 I-X
    )

    # Prune: keep only first PRUNE_LAYERS encoder layers
    if PRUNE_LAYERS < 12:
        model.bert.encoder.layer = model.bert.encoder.layer[:PRUNE_LAYERS]
        print(f"Pruned to {PRUNE_LAYERS} layers")

    print(f"Model: {MODEL_NAME} ({PRUNE_LAYERS} layers), params: {sum(p.numel() for p in model.parameters()):,}")

    label_meta = json.load(open(DATA_DIR / "labels.json"))
    label2id = label_meta["label2id"]

    train_data = load_jsonl(DATA_DIR / "train.jsonl")
    val_data = load_jsonl(DATA_DIR / "validation.jsonl")
    train_ds = ChineseBioDataset(train_data, tokenizer, label2id)
    val_ds = ChineseBioDataset(val_data, tokenizer, label2id)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"Saved to {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
