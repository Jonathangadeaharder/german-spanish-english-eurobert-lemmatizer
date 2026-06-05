# Selective Forward + Two-Stage Training Implementation Plan

> **Status:** Partially implemented. The selective forward char_loss fix (Task 1) is already
> applied in `src/multitask_model.py` (lines 164-269). The char generator module is archived
> at `archive/char_generator.py` and disabled in the production build. `train_stage2.py` is
> archived at `archive/train_stage2.py`. The `configs/mps-stage2.toml` config has not been
> created. Stage 2 training has not been run.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the OOM crash by running the char generator only on route=1 tokens (~5% of batch), and train it in a second stage with a frozen backbone.

**Architecture:** Stage 1 (already done): backbone + edit tree classifier trained to 94-98% accuracy. Stage 2: freeze backbone + edit tree, train only the routing head + char generator on the residual ~5% of tokens that need character-level generation. Selective forward gathers only route=1 tokens before passing to the char generator, reducing memory from O(batch x seq_len) to O(batch x seq_len x 0.05).

**Current state:** The char generator is disabled. `use_char_generator=True` raises `ValueError` in both `multitask_model.py` and `train.py`. The routing head (`lemma_router`) and selective forward infrastructure exist but are unused.

**Tech Stack:** PyTorch, PEFT/LoRA, Transformers, MPS (Apple Silicon)

---

### Task 1: Fix selective forward in multitask_model.py ✅ DONE

**Files:**
- Modify: `src/multitask_model.py:168-280`

The selective forward is partially implemented but has a bug: the char_loss block references `char_outputs` which was renamed to `char_gen_result`. Also, the char_loss logic still assumes flat-batch shape. Fix both.

> **Already applied.** The current `src/multitask_model.py` uses `char_gen_result` (line 164),
> `sel_tc = None` (line 165), and the corrected char_loss block (lines 251-269).
> Note: `self.char_generator` raises `ValueError` on init, so this code path is unreachable.

- [ ] **Step 1: Replace the char_loss block to use `char_gen_result`**

The char generator now returns results only for route=1 tokens (shape `[N_selected, lemma_len, vocab_size]`), not for all tokens. The loss computation should use `char_gen_result` directly since it already contains only the selected tokens.

Replace lines 249-280 in `src/multitask_model.py`:

```python
        if (
            char_gen_result is not None
            and lemma_chars is not None
        ):
            char_logits = char_gen_result["char_logits"]
            if char_logits.shape[1] > 0:
                char_logits_shifted = char_logits[:, :-1, :]
                target_shifted = sel_tc[:, 1:]
                seq_len = min(
                    char_logits_shifted.shape[1], target_shifted.shape[1]
                )
                if seq_len > 0:
                    char_loss = nn.functional.cross_entropy(
                        char_logits_shifted[:, :seq_len, :].reshape(
                            -1, self.config.char_vocab_size
                        ),
                        target_shifted[:, :seq_len].reshape(-1),
                        ignore_index=0,
                    )
```

- [ ] **Step 2: Store `sel_tc` as an instance variable for loss computation**

The `sel_tc` tensor (selected target chars) is created inside the `if selected_hidden:` block but needs to be accessible in the loss computation. Add `sel_tc = None` before the `if selected_hidden:` block and reference it in the loss.

Before line 210 (`if selected_hidden:`), add:
```python
                sel_tc = None
```

- [ ] **Step 3: Verify the forward pass works**

Run:
```bash
uv run --active --no-sync python /var/folders/qw/gy0llydd2w73qk__cb0vr5h00000gn/T/opencode/test_lora_hybrid.py
```

Expected: `SUCCESS: full LoRA-wrapped hybrid model works` with a loss value.

- [ ] **Step 4: Lint**

Run: `uvx ruff check src/multitask_model.py`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/multitask_model.py
git commit -m "fix: selective forward char_loss uses char_gen_result"
```

---

### Task 2: Create stage-2 training script ⚠️ ARCHIVED

**Files:**
- Create: `src/train_stage2.py` → **archived at** `archive/train_stage2.py`

Stage 2 loads the existing trained model, freezes the backbone + edit tree classifier, and trains only the routing head + char generator.

> **Note:** `train_stage2.py` exists in `archive/`, not `src/`. It is not registered in
> `pyproject.toml` py-modules. The char generator is disabled in the production build.

- [ ] **Step 1: Write `src/train_stage2.py`**

```python
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
    Trainer,
    TrainingArguments,
)

from multitask_model import (
    EuroBertForUposLemma,
    EuroBertUposLemmaConfig,
    MultiTaskDataCollator,
    compute_multitask_metrics,
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
        "Warning: Python is running as x86_64. Use a native arm64 interpreter.",
        file=sys.stderr,
    )


def main():
    warn_if_rosetta()

    label2id = load_json(LABEL2ID_PATH)
    upos_label2id = load_json(UPOS_LABEL2ID_PATH)
    char_vocab = load_json(CHAR_VOCAB_PATH)
    output_dir = env_str("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    pretrained_dir = env_str("PRETRAINED_MODEL_DIR", PRETRAINED_MODEL_DIR)

    torch.set_float32_matmul_precision(
        env_str("TRAIN_FLOAT32_MATMUL_PRECISION", "high")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR, trust_remote_code=True
    )

    config = EuroBertUposLemmaConfig(
        base_model_name_or_path=MODEL_ID,
        upos_label2id=upos_label2id,
        lemma_label2id=label2id,
        use_char_generator=True,
        char_vocab_size=char_vocab["vocab_size"],
        max_lemma_length=char_vocab["max_lemma_length"],
    )

    model = EuroBertForUposLemma.from_pretrained(
        pretrained_dir,
        config=config,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.upos_classifier.parameters():
        param.requires_grad = False
    for param in model.lemma_classifier.parameters():
        param.requires_grad = False

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Stage 2: {trainable:,} trainable / {total:,} total "
        f"({100 * trainable / total:.2f}%)"
    )

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
    warmup_ratio = env_float("TRAIN_WARMUP_RATIO", 0.06)
    train_batch_size = env_int("TRAIN_BATCH_SIZE", 4)
    eval_batch_size = env_int("TRAIN_EVAL_BATCH_SIZE", 4)
    save_total_limit = env_int("TRAIN_SAVE_TOTAL_LIMIT", 3)
    gradient_accumulation_steps = env_int(
        "TRAIN_GRADIENT_ACCUMULATION_STEPS", 4
    )
    eval_accumulation_steps = env_int("TRAIN_EVAL_ACCUMULATION_STEPS", 8)
    dataloader_num_workers = env_int("TRAIN_DATALOADER_NUM_WORKERS", 0)
    gradient_checkpointing = env_bool("TRAIN_GRADIENT_CHECKPOINTING", False)
    eval_during_training = env_bool("TRAIN_EVAL_DURING_TRAINING", True)
    load_best = env_bool(
        "TRAIN_LOAD_BEST_MODEL_AT_END", eval_during_training
    )
    bf16 = env_bool("TRAIN_BF16", True)
    fp16 = env_bool("TRAIN_FP16", False)
    torch_empty_cache_steps = env_int(
        "TRAIN_TORCH_EMPTY_CACHE_STEPS", 0
    ) or None

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
            ("steps" if max_steps > 0 else "epoch")
            if eval_during_training
            else "no"
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
        metric_for_best_model="joint_accuracy",
        greater_is_better=True,
        label_names=["labels", "upos_labels"],
        report_to="none",
        fp16=fp16,
        bf16=bf16,
        torch_empty_cache_steps=torch_empty_cache_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_multitask_metrics,
        callbacks=[MPSMemoryCleanupCallback()],
    )

    trainer.train()
    cleanup_torch_mps("post_train")
    trainer.save_model(output_dir)
    cleanup_torch_mps("post_save")
    tokenizer.save_pretrained(output_dir)
    print(f"Saved stage 2 model to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register in pyproject.toml**

Add `"train_stage2"` to the `py-modules` list in `pyproject.toml`.

- [ ] **Step 3: Lint**

Run: `uvx ruff check src/train_stage2.py`
Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add src/train_stage2.py pyproject.toml
git commit -m "feat: add stage 2 training script (frozen backbone + char gen)"
```

---

### Task 3: Create stage-2 config ⚠️ NOT CREATED

**Files:**
- Create: `configs/mps-stage2.toml` → **does not exist**

> **Note:** The `configs/` directory contains `mps-full.toml` and `smoke.toml` only.
> No `mps-stage2.toml` has been created.

- [ ] **Step 1: Write the config**

```toml
[env]
OUTPUT_DIR = "runs/eurobert-multilingual-lemma-210m-stage2"
PRETRAINED_MODEL_DIR = "models/eurobert-multilingual-lemma-210m-merged"
TRAIN_USE_CHAR_GENERATOR = true
TRAIN_MAX_STEPS = 0
TRAIN_EPOCHS = 5
TRAIN_LEARNING_RATE = 0.0001
TRAIN_WARMUP_RATIO = 0.1
TRAIN_SAVE_TOTAL_LIMIT = 3
TRAIN_SAVE_STEPS = 500
TRAIN_EVAL_STEPS = 500
TRAIN_BATCH_SIZE = 4
TRAIN_EVAL_BATCH_SIZE = 4
TRAIN_GRADIENT_ACCUMULATION_STEPS = 4
TRAIN_EVAL_ACCUMULATION_STEPS = 1
TRAIN_DATALOADER_NUM_WORKERS = 0
TRAIN_BF16 = true
TRAIN_GROUP_BY_LENGTH = true
TRAIN_GRADIENT_CHECKPOINTING = false
TRAIN_EVAL_DURING_TRAINING = true
TRAIN_LOAD_BEST_MODEL_AT_END = true
TRAIN_TORCH_EMPTY_CACHE_STEPS = 1
PYTORCH_MPS_FAST_MATH = 1
EVAL_BATCH_SIZE = 4
```

Key differences from stage 1:
- **No LoRA** — backbone is frozen, only heads train
- **Higher batch size (4)** — no LoRA overhead means more memory for data
- **Lower learning rate (1e-4)** — only training small heads
- **More epochs (5)** — small heads need more iterations
- **Higher warmup (0.1)** — stabilize small head training
- **Eval during training** — monitor char gen quality

- [ ] **Step 2: Commit**

```bash
git add configs/mps-stage2.toml
git commit -m "feat: add stage 2 training config"
```

---

### Task 4: Smoke test stage 2 training

**Files:**
- None (verification only)

- [ ] **Step 1: Run a 10-step smoke test**

```bash
TRAIN_MAX_STEPS=10 TRAIN_EVAL_DURING_TRAINING=false \
  uv run --active --no-sync python -c "
import sys; sys.path.insert(0, 'src')
from config import apply_env, load_profile
apply_env(load_profile('mps-stage2'))
import os; os.environ['TRAIN_MAX_STEPS'] = '10'
os.environ['TRAIN_EVAL_DURING_TRAINING'] = 'false'
from train_stage2 import main
main()
"
```

Expected: Training completes 10 steps without OOM. Loss decreases. Memory stays stable.

- [ ] **Step 2: Check memory usage**

Verify from logs that `driver_allocated_memory` stays below 55 GB throughout the 10 steps.

- [ ] **Step 3: Commit (no changes expected)**

---

### Task 5: Run full stage 2 training

**Files:**
- None (execution only)

- [ ] **Step 1: Start training in background**

```bash
mkdir -p logs
nohup uv run --active --no-sync python -c "
import sys; sys.path.insert(0, 'src')
from config import apply_env, load_profile
apply_env(load_profile('mps-stage2'))
from train_stage2 import main
main()
" > logs/stage2-training.log 2>&1 &
echo "PID: $!"
```

- [ ] **Step 2: Monitor progress**

Check periodically:
```bash
tail -5 logs/stage2-training.log | strings
strings logs/stage2-training.log | grep "MPS-MEM" | tail -3
```

Expected: Training completes all epochs. Memory stays stable. Loss converges.

---

### Task 6: Update evaluate.py for stage 2 model ⚠️ NOT NEEDED YET

**Files:**
- Modify: `src/evaluate.py`

The evaluate script needs to load the stage 2 model (which has char_generator + router but uses the full label2id, not top-300).

> **Note:** The current `evaluate.py` loads the LoRA adapter and uses constrained label
> selection with lexicon fallback. It does not have `EVAL_USE_CHAR_GENERATOR` or
> `EVAL_STAGE2` support. Since the char generator is disabled, these additions are
> deferred until stage 2 is enabled.

- [ ] **Step 1: Update model loading to support stage 2**

The current evaluate.py already has `EVAL_USE_CHAR_GENERATOR` support. Update it to load from the stage 2 output directory and use the full `label2id.json` (not `label2id_top300.json`) when loading a stage 2 model.

In the `main()` function, change the label2id path logic:

```python
    use_char_gen = env_bool("EVAL_USE_CHAR_GENERATOR", False)
    stage2 = env_bool("EVAL_STAGE2", False)

    if stage2:
        label2id_path = LABEL2ID_PATH
        id2label_path = ID2LABEL_PATH
    elif use_char_gen:
        label2id_path = LABEL2ID_TOP300_PATH
        id2label_path = ID2LABEL_TOP300_PATH
    else:
        label2id_path = LABEL2ID_PATH
        id2label_path = ID2LABEL_PATH
```

- [ ] **Step 2: Lint**

Run: `uvx ruff check src/evaluate.py`
Expected: `All checks passed!`

- [ ] **Step 3: Commit**

```bash
git add src/evaluate.py
git commit -m "feat: evaluate.py supports stage 2 model loading"
```

---

## Summary

| Task | What | Risk |
|------|------|------|
| 1 | Fix selective forward char_loss | Low — small code fix |
| 2 | Create train_stage2.py | Low — new file, no LoRA |
| 3 | Create mps-stage2.toml | Low — config only |
| 4 | Smoke test (10 steps) | Medium — validates memory |
| 5 | Full training | Medium — long run |
| 6 | Update evaluate.py | Low — small change |

**Memory budget:** With backbone frozen (no LoRA), only ~6.4M params trainable. Selective forward processes ~25 tokens instead of 512. Expected peak: ~45-50 GB (well within 62 GB limit).
