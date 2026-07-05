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
from datasets import load_from_disk

from lemmatizer.train.byt5_lemma_model import ByT5EncoderLemmaClassifier, masked_cross_entropy

PAD_LABEL = -100


def collate_batch(rows: list[dict]) -> dict:
    max_len = 768
    max_words = 96

    B = len(rows)

    input_ids = np.zeros((B, max_len), dtype=np.int32)
    word_byte_spans = np.zeros((B, max_words, 2), dtype=np.int32)
    labels = np.full((B, max_words), -100, dtype=np.int32)

    for i, r in enumerate(rows):
        n_bytes = min(len(r["input_ids"]), max_len)
        n_words = min(len(r["word_byte_spans"]), max_words)
        input_ids[i, :n_bytes] = r["input_ids"][:n_bytes]
        spans = np.minimum(r["word_byte_spans"][:n_words], max_len)
        word_byte_spans[i, :n_words, :] = spans
        labels[i, :n_words] = r["labels"][:n_words]

    return {
        "input_ids": mx.array(input_ids),
        "word_byte_spans": mx.array(word_byte_spans),
        "labels": mx.array(labels),
    }


def evaluate(
    model: ByT5EncoderLemmaClassifier,
    rows: list[dict],
    batch_size: int,
    id2lemma: dict[int, str],
    lexicon: dict[str, str],
) -> dict:
    model.eval()
    stats = {"correct": 0, "total": 0, "propn": 0, "by_upos": {}}
    for i in range(0, len(rows), batch_size):
        batch = collate_batch(rows[i : i + batch_size])
        logits = model(batch["input_ids"], batch["word_byte_spans"])
        preds = mx.argmax(logits, axis=-1)
        mx.eval(preds)

        for b in range(len(rows[i : i + batch_size])):
            row = rows[i + b]
            for w, (word, lemma, upos) in enumerate(
                zip(row["words"], row["lemmas"], row["upos"], strict=True)
            ):
                if w >= len(preds[b]):
                    break
                if upos == "PROPN":
                    stats["propn"] += 1
                    continue
                if lemma in ("_", "-"):
                    continue
                pred_id = int(preds[b, w])
                pred_lemma = id2lemma.get(pred_id, "<UNK>")
                if pred_lemma == "<UNK>":
                    pred_lemma = lexicon.get(word, word)
                stats["total"] += 1
                if pred_lemma == lemma:
                    stats["correct"] += 1
                stats["by_upos"].setdefault(upos, {"correct": 0, "total": 0})
                stats["by_upos"][upos]["total"] += 1
                if pred_lemma == lemma:
                    stats["by_upos"][upos]["correct"] += 1
        if i % (batch_size * 10) == 0:
            mx.clear_cache()

    acc = stats["correct"] / max(stats["total"], 1)
    by_upos = {
        u: round(v["correct"] / max(v["total"], 1), 4)
        for u, v in stats["by_upos"].items()
    }
    return {"lemma_accuracy": acc, "by_upos": by_upos, "stats": stats}


def find_struggles(
    model: ByT5EncoderLemmaClassifier,
    rows: list[dict],
    batch_size: int,
    id2lemma: dict[int, str],
    lexicon: dict[str, str],
) -> set[int]:
    model.eval()
    struggles = set()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        batch = collate_batch(batch_rows)
        logits = model(batch["input_ids"], batch["word_byte_spans"])
        preds = np.array(mx.argmax(logits, axis=-1))
        
        for b, row in enumerate(batch_rows):
            for w, (word, lemma, upos) in enumerate(
                zip(row["words"], row["lemmas"], row["upos"], strict=True)
            ):
                if w >= len(preds[b]):
                    break
                if upos == "PROPN" or lemma in ("_", "-"):
                    continue
                pred_id = int(preds[b, w])
                pred_lemma = id2lemma.get(pred_id, "<UNK>")
                if pred_lemma == "<UNK>":
                    pred_lemma = lexicon.get(word, word)
                if pred_lemma != lemma:
                    # Mark label class as struggled
                    struggles.add(pred_id)
        mx.clear_cache()
    return struggles


def build_curriculum_datasets(train_data: list[dict], val_data: list[dict], max_train: int = 6075, max_val: int = 909):
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
        if len(selected_val_indices) >= max_val:
            break

    if len(selected_val_indices) < max_val:
        for idx in range(len(val_data)):
            if idx not in selected_val_indices:
                selected_val_indices.add(idx)
            if len(selected_val_indices) == max_val:
                break

    selected_train_indices = set()
    for label in train_label_map:
        selected_train_indices.add(train_label_map[label][0])
        if len(selected_train_indices) >= max_train:
            break

    if len(selected_train_indices) < max_train:
        for idx in range(len(train_data)):
            if idx not in selected_train_indices:
                selected_train_indices.add(idx)
            if len(selected_train_indices) == max_train:
                break

    final_train = [train_data[i] for i in selected_train_indices]
    final_val = [val_data[i] for i in selected_val_indices]
    return final_train, final_val


def train_epoch(
    model: ByT5EncoderLemmaClassifier,
    rows: list[dict],
    batch_size: int,
    grad_accum: int,
    optimizer,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()
    order = np.random.permutation(len(rows))
    batches = math.ceil(len(order) / batch_size)

    def loss_fn(model, batch):
        logits = model(batch["input_ids"], batch["word_byte_spans"])
        return masked_cross_entropy(logits, batch["labels"])

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def _tree_map_grads(grads, scale):
        if isinstance(grads, dict):
            return {k: _tree_map_grads(v, scale) for k, v in grads.items()}
        if isinstance(grads, list):
            return [_tree_map_grads(g, scale) for g in grads]
        return grads / scale if grads is not None else None

    for batch_index, start in enumerate(range(0, len(order), batch_size), start=1):
        batch_rows = [rows[int(i)] for i in order[start:start + batch_size]]
        batch = collate_batch(batch_rows)
        loss, grads = loss_and_grad(model, batch)
        grads = _tree_map_grads(grads, float(grad_accum))
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(loss, model, optimizer)
        total_loss += float(loss)
        n_batches += 1
        if batch_index % 50 == 0 or batch_index == batches:
            print(
                json.dumps({
                    "event": "train_progress",
                    "epoch": epoch,
                    "batch": batch_index,
                    "batches": batches,
                    "loss": round(total_loss / n_batches, 4),
                    "elapsed_s": round(time.time() - t0, 1),
                }),
                flush=True,
            )
            mx.clear_cache()
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=float, default=0.06)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--dataset-path", default="data/processed/ar_byt5_lemma")
    parser.add_argument("--artifacts-dir", default="artifacts/lemma_ar")
    parser.add_argument("--output-dir", default="runs/ar-byt5-mlx")
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== ByT5 Arabic lemma classifier (MLX) ===", flush=True)
    print(f"dataset={args.dataset_path} epochs={args.epochs} batch={args.batch_size} "
          f"grad_accum={args.grad_accum} lr={args.lr} dropout={args.dropout}", flush=True)

    lemma2id = json.loads((artifacts_dir / "lemma_label2id.json").read_text(encoding="utf-8"))
    id2lemma = {int(k): v for k, v in json.loads(
        (artifacts_dir / "lemma_id2label.json").read_text(encoding="utf-8")
    ).items()}
    print(f"Lemma vocab: {len(lemma2id)} classes", flush=True)

    lexicon_path = artifacts_dir / "lexicon.json"
    lexicon = json.loads(lexicon_path.read_text(encoding="utf-8")) if lexicon_path.exists() else {}
    print(f"Lexicon: {len(lexicon)} entries", flush=True)

    ds = load_from_disk(args.dataset_path)
    train_rows = [ds["train"][i] for i in range(len(ds["train"]))]
    val_rows = [ds["validation"][i] for i in range(len(ds["validation"]))]
    print(f"Loaded train={len(train_rows)}, val={len(val_rows)}", flush=True)

    model = ByT5EncoderLemmaClassifier(num_lemmas=len(lemma2id), dropout=args.dropout)
    if args.checkpoint:
        model.load_weights(args.checkpoint)
        print(f"Loaded weights from checkpoint: {args.checkpoint}", flush=True)
    else:
        model.load_pretrained()
        print("ByT5 encoder weights loaded", flush=True)

    print(json.dumps({"event": "baseline_start"}), flush=True)
    results = {
        "baseline": {
            "train": evaluate(model, train_rows[:1000], args.batch_size, id2lemma, lexicon),
            "validation": evaluate(model, val_rows, args.batch_size, id2lemma, lexicon),
        },
        "finetune": [],
    }
    print(json.dumps({"event": "baseline", **results["baseline"]}), flush=True)

    if args.epochs > 0:
        total_steps = len(train_rows) // (args.batch_size * args.grad_accum) * args.epochs
        warmup_steps = max(1, int(total_steps * args.warmup))
        decay_steps = max(1, total_steps - warmup_steps)
        print(f"Total optimizer steps: {total_steps}, warmup: {warmup_steps}", flush=True)

        lr_schedule = optim.join_schedules(
            [
                optim.linear_schedule(0.0, args.lr, warmup_steps),
                optim.cosine_decay(args.lr, decay_steps, end=0.0),
            ],
            [warmup_steps],
        )
        optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)

        best_val_loss = float("inf")
        best_val_acc = -1.0

        if args.curriculum:
            print(json.dumps({"event": "curriculum_pool_building"}), flush=True)
            train_pool, val_pool = build_curriculum_datasets(train_rows, val_rows, max_train=6075, max_val=909)

            epochs = int(args.epochs)
            current_train_indices = set(range(min(len(train_pool), max(1, len(train_pool) // epochs))))
            current_val_indices = set(range(min(len(val_pool), max(1, len(val_pool) // epochs))))

            current_train = [train_pool[i] for i in current_train_indices]
            current_val = [val_pool[i] for i in current_val_indices]

            best_val_acc = -1.0
            best_val_loss = float("inf")

            for epoch in range(1, epochs + 1):
                t0 = time.time()
                train_loss = train_epoch(model, current_train, args.batch_size, args.grad_accum, optimizer, epoch)
                metrics = {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train": evaluate(model, train_rows[:1000], args.batch_size, id2lemma, lexicon),
                    "validation": evaluate(model, val_rows, args.batch_size, id2lemma, lexicon),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                results["finetune"].append(metrics)
                print(json.dumps({"event": "epoch", **metrics}), flush=True)

                # Save best model by validation accuracy
                val_acc = metrics["validation"]["lemma_accuracy"]
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    model.save_weights(str(output_dir / "best.safetensors"))
                    print(f"  saved best model weights (val_acc={best_val_acc:.4f})", flush=True)
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    model.save_weights(str(output_dir / "best_loss.safetensors"))

                model.save_weights(str(output_dir / f"epoch-{epoch}.safetensors"))

                if epoch < epochs:
                    struggles = find_struggles(model, current_val, args.batch_size, id2lemma, lexicon)
                    print(json.dumps({"event": f"struggles_identified_epoch_{epoch}", "count": len(struggles)}), flush=True)

                    next_train_size = min(int((epoch + 1) * len(train_pool) / epochs), len(train_pool))
                    next_val_size = min(int((epoch + 1) * len(val_pool) / epochs), len(val_pool))

                    remaining_train_indices = [i for i in range(len(train_pool)) if i not in current_train_indices]
                    remaining_train_indices.sort(key=lambda i: sum(1 for lbl in train_pool[i]["labels"] if lbl in struggles), reverse=True)

                    remaining_val_indices = [i for i in range(len(val_pool)) if i not in current_val_indices]
                    remaining_val_indices.sort(key=lambda i: sum(1 for lbl in val_pool[i]["labels"] if lbl in struggles), reverse=True)

                    added_train_indices = remaining_train_indices[:(next_train_size - len(current_train_indices))]
                    added_val_indices = remaining_val_indices[:(next_val_size - len(current_val_indices))]

                    current_train_indices.update(added_train_indices)
                    current_val_indices.update(added_val_indices)

                    current_train = [train_pool[i] for i in current_train_indices]
                    current_val = [val_pool[i] for i in current_val_indices]

        else:
            for epoch in range(1, args.epochs + 1):
                t0 = time.time()
                train_loss = train_epoch(model, train_rows, args.batch_size, args.grad_accum, optimizer, epoch)
                metrics = {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train": evaluate(model, train_rows[:1000], args.batch_size, id2lemma, lexicon),
                    "validation": evaluate(model, val_rows, args.batch_size, id2lemma, lexicon),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                results["finetune"].append(metrics)
                print(json.dumps({"event": "epoch", **metrics}), flush=True)
                
                # Fix best model saving logic by validation accuracy
                if metrics["validation"]["lemma_accuracy"] >= best_val_acc:
                    best_val_acc = metrics["validation"]["lemma_accuracy"]
                    model.save_weights(str(output_dir / "best.safetensors"))
                    print(f"  saved best model weights (val_acc={best_val_acc:.4f})", flush=True)
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    model.save_weights(str(output_dir / "best_loss.safetensors"))

        model.save_weights(str(output_dir / "final.safetensors"))
        print(f"Done! → {output_dir}", flush=True)

    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
