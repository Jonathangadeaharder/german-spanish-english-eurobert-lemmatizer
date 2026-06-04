import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.2")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "1.0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from multitask_model import (
    EuroBertForUposLemma,
    EuroBertUposLemmaConfig,
    MultiTaskDataCollator,
    compute_multitask_metrics,
    set_lang_eval_indices,
)
from runtime_utils import MPSMemoryCleanupCallback, cleanup_torch_mps

MODEL_ID = "EuroBERT/EuroBERT-210m"
TOKENIZER_DIR = "artifacts/tokenizer"
PRETRAINED_MODEL_DIR = "models/eurobert-multilingual-lemma-210m-merged"
CHAR_DATASET_PATH = "data/processed/eurobert_char_lemma_dataset"
LABEL2ID_PATH = "artifacts/label2id.json"
UPOS_LABEL2ID_PATH = "artifacts/upos_label2id.json"
CHAR_VOCAB_PATH = "artifacts/char_vocab.json"
DEFAULT_OUTPUT_DIR = "runs/eurobert-multilingual-lemma-210m-stage2"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def env_str(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else value


def env_int(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def env_float(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def env_bool(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def warn_if_rosetta():
    if platform.machine() == "arm64":
        return
    print(
        "Warning: Python is running as x86_64. Use a native arm64 interpreter for the MPS path.",
        file=sys.stderr,
    )


def main():
    warn_if_rosetta()

    label2id = load_json(LABEL2ID_PATH)
    upos_label2id = load_json(UPOS_LABEL2ID_PATH)
    char_vocab = load_json(CHAR_VOCAB_PATH)
    output_dir = env_str("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    pretrained_dir = env_str("PRETRAINED_MODEL_DIR", PRETRAINED_MODEL_DIR)

    torch.set_float32_matmul_precision(env_str("TRAIN_FLOAT32_MATMUL_PRECISION", "high"))

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)

    config = EuroBertUposLemmaConfig(
        base_model_name_or_path=MODEL_ID,
        upos_label2id=upos_label2id,
        lemma_label2id=label2id,
        use_char_generator=True,
        char_vocab_size=char_vocab["vocab_size"],
        max_lemma_length=char_vocab["max_lemma_length"],
        vocab_size=len(tokenizer),
        route_pos_weight=env_float("TRAIN_ROUTE_POS_WEIGHT", 17.5),
    )

    model = EuroBertForUposLemma.from_pretrained(
        pretrained_dir,
        config=config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.upos_classifier.parameters():
        param.requires_grad = False
    for param in model.lemma_classifier.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Stage 2: {trainable:,} trainable / {total:,} total ({100 * trainable / total:.2f}%)")

    dataset = load_from_disk(CHAR_DATASET_PATH)
    eval_limit = int(os.getenv("TRAIN_EVAL_LIMIT", "0"))
    if eval_limit > 0:
        dataset = dataset.copy()
        dataset["validation"] = dataset["validation"].select(
            range(min(eval_limit, len(dataset["validation"])))
        )

    data_collator = MultiTaskDataCollator(tokenizer=tokenizer)

    max_steps = env_int("TRAIN_MAX_STEPS", 0)
    eval_steps = env_int("TRAIN_EVAL_STEPS", 200)
    save_steps = env_int("TRAIN_SAVE_STEPS", 200)
    num_train_epochs = env_float("TRAIN_EPOCHS", 3.0)
    learning_rate = env_float("TRAIN_LEARNING_RATE", 1e-4)
    warmup_ratio = env_float("TRAIN_WARMUP_RATIO", 0.1)
    train_batch_size = env_int("TRAIN_BATCH_SIZE", 4)
    eval_batch_size = env_int("TRAIN_EVAL_BATCH_SIZE", 4)
    save_total_limit = env_int("TRAIN_SAVE_TOTAL_LIMIT", 3)
    gradient_accumulation_steps = env_int("TRAIN_GRADIENT_ACCUMULATION_STEPS", 4)
    eval_accumulation_steps = env_int("TRAIN_EVAL_ACCUMULATION_STEPS", 8)
    dataloader_num_workers = env_int("TRAIN_DATALOADER_NUM_WORKERS", 0)
    gradient_checkpointing = env_bool("TRAIN_GRADIENT_CHECKPOINTING", False)
    eval_during_training = env_bool("TRAIN_EVAL_DURING_TRAINING", True)
    load_best = env_bool("TRAIN_LOAD_BEST_MODEL_AT_END", eval_during_training)
    bf16 = env_bool("TRAIN_BF16", True)
    fp16 = env_bool("TRAIN_FP16", False)
    torch_empty_cache_steps = env_int("TRAIN_TORCH_EMPTY_CACHE_STEPS", 0) or None
    max_grad_norm = env_float("TRAIN_MAX_GRAD_NORM", 1.0)
    early_stopping_patience = env_int("TRAIN_EARLY_STOPPING_PATIENCE", 3)

    if bf16 and fp16:
        raise ValueError("Set only one of TRAIN_BF16 or TRAIN_FP16")

    if gradient_checkpointing:
        if hasattr(model.model.config, "use_cache"):
            model.model.config.use_cache = False
        if hasattr(model.model, "gradient_checkpointing_enable"):
            model.model.gradient_checkpointing_enable()
        if hasattr(model.model, "enable_input_require_grads"):
            model.model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy=(
            ("steps" if max_steps > 0 else "epoch") if eval_during_training else "no"
        ),
        save_strategy="steps" if max_steps > 0 else "epoch",
        eval_steps=eval_steps if max_steps > 0 and eval_during_training else None,
        save_steps=save_steps if max_steps > 0 else None,
        save_total_limit=save_total_limit,
        eval_accumulation_steps=eval_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=False,
        group_by_length=True,
        length_column_name="length",
        gradient_checkpointing=gradient_checkpointing,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        label_names=["labels", "upos_labels", "lemma_route"],
        report_to="none",
        fp16=fp16,
        bf16=bf16,
        torch_empty_cache_steps=torch_empty_cache_steps,
        max_grad_norm=max_grad_norm,
    )

    callbacks = [MPSMemoryCleanupCallback()]

    if early_stopping_patience > 0 and eval_during_training:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )

    if "lang" in dataset["validation"].column_names:
        import numpy as _np

        lang_indices = {}
        langs = dataset["validation"]["lang"]
        for lang in sorted(set(langs)):
            lang_indices[lang] = _np.array([i for i, l in enumerate(langs) if l == lang])
        set_lang_eval_indices(lang_indices)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_multitask_metrics,
        callbacks=callbacks,
    )

    trainer.train()
    cleanup_torch_mps("post_train")
    trainer.save_model(output_dir)
    cleanup_torch_mps("post_save")
    tokenizer.save_pretrained(output_dir)
    print(f"Saved stage 2 model to {output_dir}")


if __name__ == "__main__":
    main()
