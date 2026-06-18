from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "EuroBERT/EuroBERT-210m"
DATASET_PATH = Path("data/processed/eurobert_vocab_classifier_dataset")
ARTIFACTS_DIR = Path("artifacts/vocab")
DEFAULT_OUTPUT_DIR = Path("runs/eurobert-vocab-classifier-210m")


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = logits.argmax(axis=-1)
    mask = labels != -100

    total = int(mask.sum())
    if total == 0:
        return {"accuracy": 0.0, "canonical_f1": 0.0}

    correct = int((predictions == labels)[mask].sum())
    pred_pos = (predictions == 1) & mask
    gold_pos = (labels == 1) & mask
    true_pos = int((pred_pos & gold_pos).sum())
    precision = true_pos / int(pred_pos.sum()) if int(pred_pos.sum()) else 0.0
    recall = true_pos / int(gold_pos.sum()) if int(gold_pos.sum()) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": round(correct / total, 4),
        "canonical_precision": round(precision, 4),
        "canonical_recall": round(recall, 4),
        "canonical_f1": round(f1, 4),
    }


def main():
    output_dir = Path(os.getenv("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    tokenizer_dir = Path(os.getenv("TOKENIZER_DIR", str(ARTIFACTS_DIR / "tokenizer")))
    dataset_path = Path(os.getenv("DATASET_PATH", str(DATASET_PATH)))
    label2id = load_json(ARTIFACTS_DIR / "label2id.json")
    id2label = {int(v): k for k, v in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    dataset = load_from_disk(str(dataset_path))
    eval_limit = env_int("TRAIN_EVAL_LIMIT", 0)
    if eval_limit > 0:
        dataset_any = cast(Any, dataset)
        validation = dataset_any["validation"]
        dataset_any["validation"] = validation.select(range(min(eval_limit, len(validation))))

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(os.getenv("TRAIN_FLOAT32_MATMUL_PRECISION", "high"))

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=env_float("TRAIN_LEARNING_RATE", 3e-5),
        warmup_ratio=env_float("TRAIN_WARMUP_RATIO", 0.06),
        per_device_train_batch_size=env_int("TRAIN_BATCH_SIZE", 8),
        per_device_eval_batch_size=env_int("TRAIN_EVAL_BATCH_SIZE", 8),
        gradient_accumulation_steps=env_int("TRAIN_GRADIENT_ACCUMULATION_STEPS", 1),
        num_train_epochs=env_float("TRAIN_EPOCHS", 3.0),
        max_steps=env_int("TRAIN_MAX_STEPS", 0) or -1,
        weight_decay=0.01,
        save_strategy="steps" if env_int("TRAIN_MAX_STEPS", 0) > 0 else "epoch",
        evaluation_strategy=(
            "no" if env_bool("TRAIN_EVAL_DURING_TRAINING", False) is False else "epoch"
        ),
        save_total_limit=env_int("TRAIN_SAVE_TOTAL_LIMIT", 2),
        report_to="none",
        bf16=env_bool("TRAIN_BF16", True),
        fp16=env_bool("TRAIN_FP16", False),
        group_by_length=env_bool("TRAIN_GROUP_BY_LENGTH", True),
        length_column_name="length",
        dataloader_num_workers=env_int("TRAIN_DATALOADER_NUM_WORKERS", 0),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=cast(Any, dataset["train"]),
        eval_dataset=cast(Any, dataset["validation"]),
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    resume_ckpt = os.getenv("RESUME_FROM_CHECKPOINT", "")
    if resume_ckpt and Path(resume_ckpt).exists():
        print(f"Resuming from checkpoint: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved vocab classifier to {output_dir}")


if __name__ == "__main__":
    main()
