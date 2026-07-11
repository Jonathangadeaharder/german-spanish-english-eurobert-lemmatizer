from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from transformers import AutoTokenizer

from lemmatizer.languages import LanguageSpec
from lemmatizer.train import TrainOptions


# openmed is an undeclared optional dep; lazy-import so the module loads
# without it. Fails at runtime only if ZH trainer is invoked.
def _load_openmed_model():
    from openmed.mlx.models import load_model

    return load_model


MAX_LENGTH = 256


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.weight = base.weight
        self.bias = getattr(base, "bias", None)
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = mx.random.normal((rank, base.weight.shape[1])) * 0.01
        self.lora_b = mx.zeros((base.weight.shape[0], rank))
        self.freeze(keys=["weight"])
        if self.bias is not None:
            self.freeze(keys=["bias"])

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y + ((x @ self.lora_a.T) @ self.lora_b.T) * self.scale


def attach_lora(model: nn.Module, rank: int, alpha: float) -> None:
    model.freeze()
    model.classifier.unfreeze()
    for layer in model.encoder.layers:
        layer.attention.query_proj = LoRALinear(layer.attention.query_proj, rank, alpha)
        layer.attention.key_proj = LoRALinear(layer.attention.key_proj, rank, alpha)
        layer.attention.value_proj = LoRALinear(layer.attention.value_proj, rank, alpha)
        layer.attention.out_proj = LoRALinear(layer.attention.out_proj, rank, alpha)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def make_dataset(data: list[dict], tokenizer, label2id: dict[str, int]) -> list[dict]:
    samples = []
    for item in data:
        chars = item["chars"]
        labels = item["labels"]
        encoding = tokenizer(
            chars,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Labels: -100 for [CLS] and [SEP] and padding
        label_ids = [-100] * len(input_ids)
        for i, label_id in enumerate(labels[: len(input_ids) - 2]):
            label_ids[i + 1] = label_id

        samples.append(
            {
                "input_ids": np.array(input_ids, dtype=np.int32),
                "attention_mask": np.array(attention_mask, dtype=np.int32),
                "labels": np.array(label_ids, dtype=np.int32),
            }
        )
    return samples


def pad_batch(batch_list: list[dict]) -> dict[str, mx.array]:
    max_len = max(b["input_ids"].shape[0] for b in batch_list)
    B = len(batch_list)

    input_ids = np.zeros((B, max_len), dtype=np.int32)
    attention_mask = np.zeros((B, max_len), dtype=np.int32)
    labels = np.full((B, max_len), -100, dtype=np.int32)

    for i, b in enumerate(batch_list):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = b["attention_mask"]
        labels[i, :n] = b["labels"]

    return {
        "input_ids": mx.array(input_ids),
        "attention_mask": mx.array(attention_mask),
        "labels": mx.array(labels),
    }


def loss_fn(model, batch):
    logits = model(batch["input_ids"])
    labels = batch["labels"]
    mask = (labels != -100).astype(mx.float32)
    safe_labels = mx.maximum(labels, 0)
    B, T, C = logits.shape
    flat_logits = logits.reshape(-1, C)
    flat_labels = safe_labels.reshape(-1)
    flat_mask = mask.reshape(-1)
    losses = nn.losses.cross_entropy(flat_logits, flat_labels, reduction="none")
    masked = losses * flat_mask
    return mx.sum(masked) / mx.maximum(mx.sum(flat_mask), 1)


def evaluate(model, rows: list[dict], batch_size: int, split: str = "") -> dict:
    model.eval()
    total = 0
    correct = 0
    loss_total = 0
    loss_batches = 0
    batches = math.ceil(len(rows) / batch_size)
    t0 = time.time()

    for batch_index, start in enumerate(range(0, len(rows), batch_size), start=1):
        batch_rows = rows[start : start + batch_size]
        batch = pad_batch(batch_rows)
        logits = model(batch["input_ids"])
        loss = loss_fn(model, batch)
        mx.eval(logits, loss)
        loss_total += float(loss)
        loss_batches += 1

        preds = np.array(mx.argmax(logits, axis=-1))
        gold = np.array(batch["labels"])
        mask = gold != -100
        correct += np.sum((preds == gold) & mask)
        total += np.sum(mask)

        if split and (batch_index % 100 == 0 or batch_index == batches):
            print(
                json.dumps(
                    {
                        "event": "eval_progress",
                        "split": split,
                        "batch": batch_index,
                        "batches": batches,
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                ),
                flush=True,
            )
        if batch_index % 10 == 0:
            mx.clear_cache()

    return {
        "loss": round(loss_total / max(loss_batches, 1), 4),
        "accuracy": round(correct / max(total, 1), 4),
        "total_chars": int(total),
    }


def find_struggles(model, rows: list[dict], batch_size: int) -> set[int]:
    model.eval()
    struggles = set()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = pad_batch(batch_rows)
        logits = model(batch["input_ids"])
        mx.eval(logits)
        preds = np.array(mx.argmax(logits, axis=-1))
        gold = np.array(batch["labels"])
        for b in range(len(batch_rows)):
            for t in range(gold.shape[1]):
                g = gold[b, t]
                if g != -100 and preds[b, t] != g:
                    struggles.add(int(g))
        mx.clear_cache()
    return struggles


def build_curriculum_datasets(train_data: list[dict], val_data: list[dict]):
    train_label_map = {}
    for idx, row in enumerate(train_data):
        for label in row["labels"]:
            if label != -100:
                train_label_map.setdefault(label, []).append(idx)

    val_label_map = {}
    for idx, row in enumerate(val_data):
        for label in row["labels"]:
            if label != -100:
                val_label_map.setdefault(label, []).append(idx)

    selected_val_indices = set()
    for label in val_label_map:
        selected_val_indices.add(val_label_map[label][0])
        if len(selected_val_indices) >= 700:
            break

    if len(selected_val_indices) < 700:
        for idx in range(len(val_data)):
            if idx not in selected_val_indices:
                selected_val_indices.add(idx)
            if len(selected_val_indices) == 700:
                break

    selected_train_indices = set()
    for label in train_label_map:
        selected_train_indices.add(train_label_map[label][0])
        if len(selected_train_indices) >= 7000:
            break

    if len(selected_train_indices) < 7000:
        for idx in range(len(train_data)):
            if idx not in selected_train_indices:
                selected_train_indices.add(idx)
            if len(selected_train_indices) == 7000:
                break

    final_train = [train_data[i] for i in selected_train_indices]
    final_val = [val_data[i] for i in selected_val_indices]
    return final_train, final_val


def train_epoch(
    model,
    rows: list[dict],
    batch_size: int,
    optimizer,
    epoch: int,
) -> float:
    model.train()
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    order = np.random.permutation(len(rows))
    total = n = 0
    batches = math.ceil(len(order) / batch_size)
    t0 = time.time()
    for batch_index, start in enumerate(range(0, len(order), batch_size), start=1):
        batch_rows = [rows[int(i)] for i in order[start : start + batch_size]]
        batch = pad_batch(batch_rows)
        loss, grads = loss_and_grad(model, batch)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(loss, model, optimizer)
        total += float(loss)
        n += 1
        if batch_index % 10 == 0 or batch_index == batches:
            print(
                json.dumps(
                    {
                        "event": "train_progress",
                        "epoch": epoch,
                        "batch": batch_index,
                        "batches": batches,
                        "loss": round(total / max(n, 1), 4),
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                ),
                flush=True,
            )
            mx.clear_cache()
    return total / max(n, 1)


def main():
    """argparse wrapper around `run()` for `python -m lemmatizer.train.zh_bio`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument(
        "--prune-layers",
        type=int,
        default=12,
        help="Number of layers to keep (0 or 12 to disable pruning)",
    )
    parser.add_argument("--output-dir", default="runs/mlx-zh-bio-pos")
    args = parser.parse_args()

    from lemmatizer.languages import spec

    opts = TrainOptions(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        curriculum=args.curriculum,
        extra={"prune_layers": args.prune_layers},
    )
    run(spec("zh"), opts)


def _load_bert_model(model_path: str):
    """Load a BERT model from model_path, tolerating extra MLM head weights.

    openmed's load_model is strict — it rejects pretrained checkpoints
    that carry MLM head keys (cls.*) not present in
    BertForTokenClassification. This wrapper builds the model from config
    and loads weights with strict=False to skip those unmatched keys.
    """
    import mlx.core as mx
    from openmed.mlx.artifact import load_artifact_config, resolve_weight_candidates
    from openmed.mlx.models import build_model, normalize_model_config

    model_path = Path(model_path)
    manifest, config = load_artifact_config(model_path)
    config = normalize_model_config(config, manifest=manifest)
    candidate_paths = resolve_weight_candidates(model_path, config=config, manifest=manifest)
    weights_path = next((p for p in candidate_paths if p.exists()), None)
    if weights_path is None:
        raise FileNotFoundError(
            f"No weights found in {model_path}. Expected weights.safetensors or weights.npz."
        )
    weights = dict(mx.load(str(weights_path)))
    model = build_model(config, manifest=manifest)
    # strict=False: skip cls.* (MLM head) and pooler keys not in the
    # token-classification model. The classifier head is replaced below.
    model.load_weights(list(weights.items()), strict=False)
    model.eval()
    mx.eval(model.parameters())
    return model


def run(spec: LanguageSpec, opts: TrainOptions) -> None:
    """Canonical entry: train the ZH BIO-POS model for `spec.lang` (zh)."""
    prune_layers = opts.extra.get("prune_layers", 12)

    tokenizer = AutoTokenizer.from_pretrained(opts.checkpoint)
    model = _load_bert_model(opts.checkpoint)

    # Prune model layers if requested
    if prune_layers > 0 and len(model.encoder.layers) > prune_layers:
        model.encoder.layers = model.encoder.layers[:prune_layers]
        print(f"Pruned model to {prune_layers} layers")

    data_dir = Path("data/processed/zh_bio")
    label_meta = json.load(open(data_dir / "labels.json"))
    label2id = label_meta["label2id"]

    # Attach classifier head sized to the loaded label set (was hardcoded 35).
    config = json.load(open(Path(opts.checkpoint) / "config.json"))
    hidden_size = config.get("hidden_size", 768)
    model.classifier = nn.Linear(hidden_size, len(label2id))

    print("Loading datasets...", flush=True)
    train_raw = load_jsonl(data_dir / "train.jsonl")
    val_raw = load_jsonl(data_dir / "validation.jsonl")

    train_data = make_dataset(train_raw, tokenizer, label2id)
    val_data = make_dataset(val_raw, tokenizer, label2id)
    print(f"Loaded train={len(train_data)}, val={len(val_data)}")

    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"event": "baseline_start"}), flush=True)
    results = {
        "baseline": {
            "train": evaluate(model, train_data, opts.batch_size, "baseline_train"),
            "validation": evaluate(model, val_data, opts.batch_size, "baseline_validation"),
        },
        "finetune": [],
    }
    print(json.dumps({"event": "baseline", **results["baseline"]}), flush=True)

    if opts.epochs > 0:
        if opts.lora_rank > 0:
            attach_lora(model, opts.lora_rank, opts.lora_alpha)
            print(
                json.dumps(
                    {"event": "lora_attached", "rank": opts.lora_rank, "alpha": opts.lora_alpha}
                ),
                flush=True,
            )
        else:
            model.unfreeze()
            print("Full fine-tuning (no LoRA)", flush=True)

        optimizer = optim.AdamW(learning_rate=opts.lr, weight_decay=0.01)

        if opts.curriculum:
            print(json.dumps({"event": "curriculum_pool_building"}), flush=True)
            train_pool, val_pool = build_curriculum_datasets(train_data, val_data)

            epochs = int(opts.epochs)
            current_train_indices = set(
                range(min(len(train_pool), max(1, len(train_pool) // epochs)))
            )
            current_val_indices = set(range(min(len(val_pool), max(1, len(val_pool) // epochs))))

            current_train = [train_pool[i] for i in current_train_indices]
            current_val = [val_pool[i] for i in current_val_indices]

            best_val_acc = -1.0

            for epoch in range(1, epochs + 1):
                t0 = time.time()
                train_loss = train_epoch(model, current_train, opts.batch_size, optimizer, epoch)
                metrics = {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train": evaluate(model, train_data, opts.batch_size, f"epoch_{epoch}_train"),
                    "validation": evaluate(
                        model, val_data, opts.batch_size, f"epoch_{epoch}_validation"
                    ),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                results["finetune"].append(metrics)
                print(json.dumps({"event": "epoch", **metrics}), flush=True)

                val_acc = metrics["validation"]["accuracy"]
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    model.save_weights(str(output_dir / "best.safetensors"))
                    print(f"  saved best model weights (val_acc={best_val_acc:.4f})", flush=True)

                model.save_weights(str(output_dir / f"epoch-{epoch}.safetensors"))

                if epoch < epochs:
                    struggles = find_struggles(model, current_val, opts.batch_size)
                    print(
                        json.dumps(
                            {
                                "event": f"struggles_identified_epoch_{epoch}",
                                "count": len(struggles),
                            }
                        ),
                        flush=True,
                    )

                    next_train_size = min(
                        int((epoch + 1) * len(train_pool) / epochs), len(train_pool)
                    )
                    next_val_size = min(int((epoch + 1) * len(val_pool) / epochs), len(val_pool))

                    remaining_train_indices = [
                        i for i in range(len(train_pool)) if i not in current_train_indices
                    ]
                    remaining_train_indices.sort(
                        key=lambda i: sum(1 for lbl in train_pool[i]["labels"] if lbl in struggles),
                        reverse=True,
                    )

                    remaining_val_indices = [
                        i for i in range(len(val_pool)) if i not in current_val_indices
                    ]
                    remaining_val_indices.sort(
                        key=lambda i: sum(1 for lbl in val_pool[i]["labels"] if lbl in struggles),
                        reverse=True,
                    )

                    added_train_indices = remaining_train_indices[
                        : (next_train_size - len(current_train_indices))
                    ]
                    added_val_indices = remaining_val_indices[
                        : (next_val_size - len(current_val_indices))
                    ]

                    current_train_indices.update(added_train_indices)
                    current_val_indices.update(added_val_indices)

                    current_train = [train_pool[i] for i in current_train_indices]
                    current_val = [val_pool[i] for i in current_val_indices]

        else:
            best_val_acc = -1.0
            for epoch in range(1, int(opts.epochs) + 1):
                t0 = time.time()
                train_loss = train_epoch(model, train_data, opts.batch_size, optimizer, epoch)
                metrics = {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train": evaluate(model, train_data, opts.batch_size, f"epoch_{epoch}_train"),
                    "validation": evaluate(
                        model, val_data, opts.batch_size, f"epoch_{epoch}_validation"
                    ),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                results["finetune"].append(metrics)
                print(json.dumps({"event": "epoch", **metrics}), flush=True)

                val_acc = metrics["validation"]["accuracy"]
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    model.save_weights(str(output_dir / "best.safetensors"))
                    print(
                        f"  saved best model weights (val_acc={best_val_acc:.4f})",
                        flush=True,
                    )
                model.save_weights(str(output_dir / f"epoch-{epoch}.safetensors"))

    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"event": "saved", "path": str(output_dir / "metrics.json")}), flush=True)


if __name__ == "__main__":
    main()
