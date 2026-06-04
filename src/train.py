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
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer import TRAINING_ARGS_NAME

from multitask_model import (
    EuroBertForUposLemma,
    EuroBertUposLemmaConfig,
    MultiTaskDataCollator,
    compute_multitask_metrics,
)
from runtime_utils import MPSMemoryCleanupCallback, cleanup_torch_mps

MODEL_ID = "EuroBERT/EuroBERT-210m"
TOKENIZER_DIR = "artifacts/tokenizer"
DATASET_PATH = "data/processed/eurobert_multilingual_lemma_dataset"
CHAR_DATASET_PATH = "data/processed/eurobert_char_lemma_dataset"
LABEL2ID_PATH = "artifacts/label2id.json"
LABEL2ID_TOP300_PATH = "artifacts/label2id_top300.json"
UPOS_LABEL2ID_PATH = "artifacts/upos_label2id.json"
CHAR_VOCAB_PATH = "artifacts/char_vocab.json"
DEFAULT_OUTPUT_DIR = "runs/eurobert-multilingual-lemma-210m-lora"
LANGUAGE_TOKENS = ["[LANG_DE]", "[LANG_ES]", "[LANG_EN]"]


class NoEmbeddingSaveTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        model_to_save = self.accelerator.unwrap_model(self.model)

        if isinstance(model_to_save, PeftModel):
            os.makedirs(output_dir, exist_ok=True)
            model_to_save.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
                save_embedding_layers=False,
            )

            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)

            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
            return

        super()._save(output_dir=output_dir, state_dict=state_dict)


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


def find_latest_checkpoint(output_dir):
    root = Path(output_dir)
    checkpoints = []

    if not root.exists():
        return None

    for candidate in root.glob("checkpoint-*"):
        if not candidate.is_dir():
            continue

        suffix = candidate.name.removeprefix("checkpoint-")

        if not suffix.isdigit():
            continue

        checkpoints.append((int(suffix), candidate))

    if not checkpoints:
        return None

    return str(sorted(checkpoints, key=lambda item: item[0])[-1][1])


def resolve_resume_from(output_dir, resume_from):
    if not resume_from:
        return None

    if resume_from == "auto":
        latest = find_latest_checkpoint(output_dir)

        if latest is None:
            raise FileNotFoundError(
                f"TRAIN_RESUME_FROM=auto but no checkpoint-* folder exists under {output_dir}"
            )

        return latest

    return resume_from


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def trainable_language_token_indices(tokenizer):
    return [tokenizer.convert_tokens_to_ids(token) for token in LANGUAGE_TOKENS]


def configure_torch_runtime(torch_module):
    precision = env_str("TRAIN_FLOAT32_MATMUL_PRECISION", "high")

    if hasattr(torch_module, "set_float32_matmul_precision"):
        torch_module.set_float32_matmul_precision(precision)


def warn_if_rosetta():
    if platform.machine() == "arm64":
        return

    print(
        "Warning: Python is running as x86_64. Use a native arm64 interpreter "
        "(for example /opt/homebrew/bin/python3) for the MPS path.",
        file=sys.stderr,
    )


def main():
    warn_if_rosetta()

    use_char_gen = env_bool("TRAIN_USE_CHAR_GENERATOR", False)

    label2id_path = LABEL2ID_TOP300_PATH if use_char_gen else LABEL2ID_PATH
    label2id = load_json(label2id_path)
    upos_label2id = load_json(UPOS_LABEL2ID_PATH)
    output_dir = env_str("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    resume_from = resolve_resume_from(output_dir, env_str("TRAIN_RESUME_FROM", ""))

    configure_torch_runtime(torch)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        trust_remote_code=True,
    )

    config_kwargs = {
        "base_model_name_or_path": MODEL_ID,
        "upos_label2id": upos_label2id,
        "lemma_label2id": label2id,
    }

    if use_char_gen:
        char_vocab = load_json(CHAR_VOCAB_PATH)
        config_kwargs["use_char_generator"] = True
        config_kwargs["char_vocab_size"] = char_vocab["vocab_size"]
        config_kwargs["max_lemma_length"] = char_vocab["max_lemma_length"]

    config = EuroBertUposLemmaConfig(**config_kwargs)

    model = EuroBertForUposLemma.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
    )

    model.resize_token_embeddings(len(tokenizer))

    modules_to_save = ["upos_classifier", "lemma_classifier", "lemma_router"]
    if use_char_gen:
        modules_to_save.append("char_generator")

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=modules_to_save,
        trainable_token_indices=trainable_language_token_indices(tokenizer),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset_path = CHAR_DATASET_PATH if use_char_gen else DATASET_PATH
    dataset = load_from_disk(dataset_path)
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
    learning_rate = env_float("TRAIN_LEARNING_RATE", 2e-4)
    warmup_ratio = env_float("TRAIN_WARMUP_RATIO", 0.06)
    train_batch_size = env_int("TRAIN_BATCH_SIZE", 8)
    eval_batch_size = env_int("TRAIN_EVAL_BATCH_SIZE", 8)
    save_total_limit = env_int("TRAIN_SAVE_TOTAL_LIMIT", 3)
    gradient_accumulation_steps = env_int("TRAIN_GRADIENT_ACCUMULATION_STEPS", 1)
    eval_accumulation_steps = env_int("TRAIN_EVAL_ACCUMULATION_STEPS", 8)
    dataloader_num_workers = env_int("TRAIN_DATALOADER_NUM_WORKERS", 2)
    dataloader_pin_memory = env_bool("TRAIN_DATALOADER_PIN_MEMORY", False)
    group_by_length = env_bool("TRAIN_GROUP_BY_LENGTH", True)
    gradient_checkpointing = env_bool("TRAIN_GRADIENT_CHECKPOINTING", False)
    eval_during_training = env_bool("TRAIN_EVAL_DURING_TRAINING", True)
    load_best_model_at_end = env_bool("TRAIN_LOAD_BEST_MODEL_AT_END", eval_during_training)
    bf16 = env_bool("TRAIN_BF16", True)
    fp16 = env_bool("TRAIN_FP16", False)
    torch_empty_cache_steps = env_int("TRAIN_TORCH_EMPTY_CACHE_STEPS", 0) or None

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
        dataloader_pin_memory=dataloader_pin_memory,
        group_by_length=group_by_length,
        length_column_name="length",
        gradient_checkpointing=gradient_checkpointing,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="joint_accuracy",
        greater_is_better=True,
        label_names=["labels", "upos_labels"],
        report_to="none",
        fp16=fp16,
        bf16=bf16,
        torch_empty_cache_steps=torch_empty_cache_steps,
    )

    trainer = NoEmbeddingSaveTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_multitask_metrics,
        callbacks=[MPSMemoryCleanupCallback()],
    )

    trainer.train(resume_from_checkpoint=resume_from)
    cleanup_torch_mps("post_train")
    trainer.save_model(output_dir)
    cleanup_torch_mps("post_save_model")
    tokenizer.save_pretrained(output_dir)
    cleanup_torch_mps("post_save_tokenizer")

    print(f"Saved LoRA model to {output_dir}")


if __name__ == "__main__":
    main()
