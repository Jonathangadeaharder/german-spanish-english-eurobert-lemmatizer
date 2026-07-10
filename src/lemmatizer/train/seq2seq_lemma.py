"""Seq2Seq lemmatizer trainer (Stage 2 of the two-stage pipeline).

Trains a ByT5-small model (encoder + decoder) to produce lemma sequences
from UPOS-annotated input sentences. Uses MLX.

Training:
  Input:  "The [DET] fliegen [NOUN] are [AUX] . [PUNCT]"
  Output: "the fly be ."

The encoder reads the annotated input; the decoder autoregressively
generates the lemma sequence. Loss is cross-entropy on decoder tokens.

Usage:
    LEMMA_LANG=de uv run python -m lemmatizer.train.seq2seq_lemma --epochs 5
"""

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

from lemmatizer.data.byt5_dataset import MAX_SEQ_LEN
from lemmatizer.train.byt5_encoder import T5
from lemmatizer.train.byt5_lemma_model import (
    BYT5_SMALL_WEIGHTS,
    _resolve_byt5_path,
    load_byt5_config,
)
from lemmatizer.train.grad_utils import tree_add, tree_scale

BYT5_PAD = 0
BYT5_EOS = 1


def collate_batch(rows: list[dict]) -> dict:
    """Pad sequences to max length in batch (capped at MAX_SEQ_LEN).

    Returns an attention_mask so the encoder does not attend to BYT5_PAD
    padding positions (the T5 encoder combines it with the relative
    position bias).
    """
    max_input = min(max(len(r["input_ids"]) for r in rows), MAX_SEQ_LEN)
    max_label = min(max(len(r["labels"]) for r in rows), MAX_SEQ_LEN)

    B = len(rows)
    input_ids = np.zeros((B, max_input), dtype=np.int32)
    attention_mask = np.zeros((B, max_input), dtype=np.int32)
    labels = np.full((B, max_label), -100, dtype=np.int32)

    for i, r in enumerate(rows):
        inp = r["input_ids"][:max_input]
        lbl = r["labels"][:max_label]
        input_ids[i, : len(inp)] = inp
        attention_mask[i, : len(inp)] = 1
        labels[i, : len(lbl)] = lbl

    return {
        "input_ids": mx.array(input_ids),
        "attention_mask": mx.array(attention_mask),
        "labels": mx.array(labels),
    }


def _encoder_padding_mask(attention_mask: mx.array) -> mx.array:
    """Additive padding mask for the T5 encoder: 0 keep, -1e9 pad.

    Shape (B, 1, 1, T) to broadcast over heads and query positions.
    """
    am = attention_mask.astype(mx.float32)
    return (1.0 - am)[:, None, None, :] * -1e9


def loss_fn(model, batch):
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    padding_mask = _encoder_padding_mask(batch["attention_mask"])

    # Decoder input: shift labels right (prepend EOS as decoder start)
    B, T = labels.shape
    decoder_input = mx.full((B, T), BYT5_PAD, dtype=mx.int32)
    decoder_input[:, 0] = BYT5_EOS  # decoder start token
    if T > 1:
        decoder_input[:, 1:] = mx.maximum(labels[:, :-1], 0)

    # Forward: encoder + decoder
    logits = model(input_ids, decoder_input, padding_mask=padding_mask)

    # Cross-entropy loss on non-masked (-100) positions
    mask = (labels != -100).astype(mx.float32)
    # decoder_input = [EOS, l0, l1, ...] predicts [l0, l1, ...]:
    # logits[:, t] predicts labels[:, t].
    log_probs = nn.log_softmax(logits, axis=-1)

    # Gather the log-prob of the correct token at each position
    label_ids = mx.clip(labels, 0, logits.shape[-1] - 1)
    gathered = mx.take_along_axis(log_probs, label_ids[:, :, None], axis=-1).squeeze(-1)

    loss = -gathered * mask
    n_tokens = mask.sum()
    return loss.sum() / mx.maximum(n_tokens, mx.array(1.0))


def generate(
    model: T5,
    input_ids: mx.array,
    max_len: int = 256,
) -> list[list[int]]:
    """Autoregressive generation (slow, used only for final eval).

    Returns a list of token lists, one per input sequence. Each inner list
    is the predicted lemma byte-stream with the leading BYT5_EOS
    decoder-start token stripped and truncated at the first emitted
    BYT5_EOS (so callers receive clean predictions with no decoder-start
    or trailing EOS markers).
    """
    B = input_ids.shape[0]
    # Build a padding mask so the encoder ignores trailing PAD positions
    # (input_ids may be a padded batch).
    padding_mask = _encoder_padding_mask((input_ids != BYT5_PAD).astype(mx.int32))
    memory = model.encode(input_ids, padding_mask=padding_mask)

    decoder_input = mx.full((B, 1), BYT5_EOS, dtype=mx.int32)
    cache = None
    # Per-sequence completion: stop only when every sequence has emitted EOS,
    # rather than requiring the whole batch to hit EOS simultaneously (which
    # almost never happens for variable-length targets).
    finished = mx.zeros((B,), dtype=mx.bool_)

    for _ in range(max_len):
        logits, cache = model.decode(
            decoder_input[
                :,
                -1:,
            ],
            memory,
            cache=cache,
        )
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        decoder_input = mx.concatenate([decoder_input, next_token[:, None]], axis=1)
        # Materialize decoder_input each step: without this, the lazy MLX
        # graph grows linearly with max_len (each concatenate wraps the
        # previous array), bloating memory and slowing eval.
        mx.eval(decoder_input)
        finished = finished | (next_token == BYT5_EOS)
        if mx.all(finished):
            break

    # Post-process: drop the leading decoder-start BYT5_EOS (column 0) and
    # truncate each sequence at its first emitted BYT5_EOS so callers get
    # clean predicted streams with no sentinel markers.
    return _strip_decoder_start_and_truncate_at_eos(decoder_input)


def _strip_decoder_start_and_truncate_at_eos(
    decoder_input: mx.array,
) -> list[list[int]]:
    """Drop column 0 (decoder-start EOS) and cut each row at first EOS."""
    tokens_np = np.array(decoder_input)
    out: list[list[int]] = []
    for row in tokens_np:
        # row[0] is the decoder-start BYT5_EOS; predicted tokens start at 1.
        emitted = row[1:].tolist()
        truncated: list[int] = []
        for tok in emitted:
            if tok == BYT5_EOS:
                break
            truncated.append(int(tok))
        out.append(truncated)
    return out


def evaluate(
    model: T5,
    rows: list[dict],
    batch_size: int,
) -> dict:
    """Teacher-forcing evaluation (fast — one forward pass per batch).

    Measures token-level accuracy: for each position, does the model's
    argmax prediction match the gold token? This is an upper bound on
    autoregressive accuracy.
    """
    model.eval()
    total = 0
    correct = 0
    exact_match = 0
    n_sentences = 0

    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i : i + batch_size]
        batch = collate_batch(batch_rows)
        labels = batch["labels"]
        padding_mask = _encoder_padding_mask(batch["attention_mask"])

        # Build decoder input (shift right with EOS as start)
        B, T = labels.shape
        decoder_input = mx.full((B, T), BYT5_PAD, dtype=mx.int32)
        decoder_input[:, 0] = BYT5_EOS
        if T > 1:
            decoder_input[:, 1:] = mx.maximum(labels[:, :-1], 0)

        # Forward pass
        logits = model(batch["input_ids"], decoder_input, padding_mask=padding_mask)
        preds = mx.argmax(logits, axis=-1)
        mx.eval(preds)
        preds_np = np.array(preds)

        for b, row in enumerate(batch_rows):
            gold = [t for t in row["labels"] if t != -100]
            # Teacher-forcing: the decoder predicts at every position (input
            # is fixed), so pred_full has length T (padded). Comparing the
            # full prediction length against gold counts padding positions as
            # over-predictions, deflating token_accuracy. Only positions with
            # valid gold tokens contribute to total — over-prediction must be
            # measured via autoregressive generate(), not teacher-forcing.
            pred = preds_np[b][: len(gold)].tolist()

            n_sentences += 1
            if pred == gold:
                exact_match += 1

            for t in range(min(len(pred), len(gold))):
                total += 1
                if pred[t] == gold[t]:
                    correct += 1

        mx.clear_cache()

    token_acc = correct / max(total, 1)
    seq_acc = exact_match / max(n_sentences, 1)
    return {"token_accuracy": token_acc, "sequence_accuracy": seq_acc}


def train_epoch(
    model: T5,
    rows: list[dict],
    batch_size: int,
    optimizer,
    epoch: int,
    grad_accum: int = 8,
) -> float:
    """Train one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()
    order = np.random.permutation(len(rows))
    batches = math.ceil(len(order) / batch_size)
    accum_steps = max(1, int(grad_accum))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    accumulated = None
    pending = 0

    for batch_idx, start in enumerate(range(0, len(order), batch_size), 1):
        batch_rows = [rows[int(i)] for i in order[start : start + batch_size]]
        batch = collate_batch(batch_rows)
        loss, grads = loss_and_grad(model, batch)
        # Scale by 1/accum_steps so the accumulated gradient is the mean
        # over the accumulation window. Shared tree_scale/tree_add (with
        # None guards) live in grad_utils and are reused by train_byt5.
        grads = tree_scale(grads, 1.0 / accum_steps)

        # Accumulate
        if accumulated is None:
            accumulated = grads
        else:
            accumulated = tree_add(accumulated, grads)

        pending += 1
        total_loss += float(loss)
        n_batches += 1

        if pending >= accum_steps or batch_idx == batches:
            # The final window often has fewer than accum_steps batches.
            # Each batch was scaled by 1/accum_steps, so the accumulated
            # gradient is (pending/accum_steps) * mean_grad — under-weighted
            # at epoch boundaries. Rescale to a true mean by multiplying by
            # accum_steps/pending (no-op when pending == accum_steps).
            if pending < accum_steps:
                accumulated = tree_scale(accumulated, accum_steps / pending)
            accumulated, _ = optim.clip_grad_norm(accumulated, 1.0)
            optimizer.update(model, accumulated)
            mx.eval(optimizer)
            accumulated = None
            pending = 0
            # Clear only after the optimizer step evaluated the accumulated
            # gradient graph. Per-batch clearing would free intermediates
            # backing un-evaluated `accumulated`, forcing a full recompute.
            mx.clear_cache()

        if batch_idx % 50 == 0 or batch_idx == batches:
            print(
                json.dumps(
                    {
                        "event": "train_progress",
                        "epoch": epoch,
                        "batch": batch_idx,
                        "batches": batches,
                        "loss": round(total_loss / n_batches, 4),
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                ),
                flush=True,
            )
            mx.clear_cache()

    return total_loss / max(n_batches, 1)


def run(lang: str, epochs: int, batch_size: int, lr: float, output_dir: str, warmup: float = 0.06):
    """Train the seq2seq lemmatizer for one language."""
    print(f"=== Seq2Seq Lemmatizer ({lang}) ===", flush=True)

    dataset_path = f"data/processed/{lang}_seq2seq_lemma"
    ds = load_from_disk(dataset_path)
    train_rows = list(ds["train"])
    val_rows = list(ds["validation"])
    print(f"train={len(train_rows)}, val={len(val_rows)}", flush=True)

    # Build model
    config = load_byt5_config()
    model = T5(config)
    weights_path = _resolve_byt5_path(BYT5_SMALL_WEIGHTS, "model.safetensors")
    if not Path(weights_path).is_file():
        # Fallback: known snapshot path. This hash is fragile (HF may
        # purge/replace); warn loudly so a missing file plus strict=False
        # weight loading does not silently train a random-init model.
        import os

        weights_path = os.path.expanduser(
            "~/.cache/huggingface/hub/models--google--byt5-small/"
            "snapshots/6f07f879d308b7b762708b50c83d41b27e329992/model.safetensors"
        )
        print(
            "WARNING: resolved ByT5 weights path missing; falling back to a "
            f"hardcoded HF snapshot hash: {weights_path}. If this file does "
            "not exist, weights load with strict=False and the model trains "
            "from random initialization. Re-fetch google/byt5-small to fix.",
            flush=True,
        )
        if not Path(weights_path).is_file():
            raise FileNotFoundError(
                f"ByT5 weights not found at resolved path or fallback: "
                f"{weights_path}. Refusing to train with random weights."
            )
    weights = mx.load(weights_path)
    sanitized = T5.sanitize(weights)
    model.load_weights(list(sanitized.items()), strict=False)
    # strict=False silently skips unmatched keys, so a sanitize() rename
    # bug would leave backbone params at random init and produce garbage
    # with no error. Verify every model parameter was covered by the
    # sanitized weight set; this is a full encoder+decoder T5, so ALL
    # byt5-small weights should map after sanitization.
    param_keys = {k for k, _ in nn.utils.tree_flatten(model.parameters())}
    missing = sorted(param_keys - set(sanitized.keys()))
    if missing:
        raise RuntimeError(
            "ByT5 weight load failed: "
            f"{len(missing)} model params have no matching weight after "
            f"sanitize. First: {missing[:3]}. Refusing to train with "
            "random init for backbone parameters."
        )
    print("ByT5 weights loaded (encoder + decoder)", flush=True)

    # Baseline eval. The subset (val_rows[:100]) is intentional: a full
    # teacher-forcing pass over the dev split is slow, and the baseline
    # is only a sanity check, not model selection. CoNLL-U files are
    # ordered by sentence ID, so this is the file's first 100 sentences,
    # not a random sample — acceptable for a sanity baseline but not for
    # final accuracy reporting (use the full dev set for that).
    print(json.dumps({"event": "baseline_start"}), flush=True)
    baseline = evaluate(model, val_rows[:100], batch_size=4)
    print(json.dumps({"event": "baseline", **baseline}), flush=True)

    if epochs > 0:
        # grad_accum: optimizer steps happen every accum_steps batches. The
        # optimizer is reused across all epochs, so the schedule must span
        # every epoch's optimizer steps. Prior code froze later epochs.
        accum_steps = max(1, 8)
        batches_per_epoch = math.ceil(len(train_rows) / batch_size)
        optimizer_steps_per_epoch = math.ceil(batches_per_epoch / accum_steps)
        total_steps = max(1, optimizer_steps_per_epoch * int(epochs))
        warmup_frac = max(0.0, min(0.99, warmup))
        warmup_steps = min(
            max(1, int(total_steps * warmup_frac)),
            max(1, total_steps - 1),
        )
        decay_steps = max(1, total_steps - warmup_steps)
        lr_schedule = optim.join_schedules(
            [
                optim.linear_schedule(0.0, lr, warmup_steps),
                optim.cosine_decay(lr, decay_steps, end=0.0),
            ],
            [warmup_steps],
        )
        optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)

        # Select best checkpoint by validation sequence accuracy (not train
        # loss, which decreases monotonically and always picks the last epoch).
        # Per-epoch validation uses the first 200 dev rows to keep eval cost
        # bounded; re-evaluate on the full dev set before reporting final.
        best_val_acc = -1.0
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            try:
                train_loss = train_epoch(model, train_rows, batch_size, optimizer, epoch)
            except Exception as e:
                print(f"ERROR in train_epoch: {e}", flush=True)
                import traceback

                traceback.print_exc()
                # Re-raise after logging: a silently-swallowed training error
                # would leave run() printing {"event": "training_complete"}
                # and returning normally, risking deployment of a partially
                # trained model with no signal to the caller.
                raise
            val_metrics = evaluate(model, val_rows[:200], batch_size=4)
            metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "validation": val_metrics,
                "elapsed_s": round(time.time() - t0, 1),
            }
            print(json.dumps({"event": "epoch", **metrics}), flush=True)

            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            model.save_weights(str(out / f"epoch-{epoch}.safetensors"))
            val_acc = val_metrics.get("sequence_accuracy", 0.0)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                model.save_weights(str(out / "best.safetensors"))

    print(json.dumps({"event": "training_complete", "lang": lang}), flush=True)


def main():
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=os.getenv("LEMMA_LANG", "de"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=float, default=0.06)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    output_dir = args.output_dir or f"runs/{args.lang}-seq2seq-lemma"
    run(args.lang, args.epochs, args.batch_size, args.lr, output_dir, args.warmup)


if __name__ == "__main__":
    main()
