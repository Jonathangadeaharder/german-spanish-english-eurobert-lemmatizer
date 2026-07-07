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
from safetensors import safe_open
from transformers import AutoConfig

from lemmatizer.data.edit_trees import apply_edit_label
from lemmatizer.data.label_space import LabelSpace
from lemmatizer.languages import LanguageSpec, language_assets
from lemmatizer.train import TrainOptions


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_safetensors(path: Path) -> dict[str, mx.array]:
    with safe_open(str(path), framework="np") as f:
        return {key: mx.array(f.get_tensor(key)) for key in f.keys()}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight * (x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps))


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


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class EuroBertLayer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        h = cfg["hidden_size"]
        heads = cfg["num_attention_heads"]
        self.heads = heads
        self.head_dim = cfg.get("head_dim") or h // heads
        self.scale = self.head_dim ** -0.5
        self.input_layernorm = RMSNorm(h, cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(h, cfg["rms_norm_eps"])
        self.q_proj = nn.Linear(h, h, bias=cfg.get("attention_bias", False))
        self.k_proj = nn.Linear(h, h, bias=cfg.get("attention_bias", False))
        self.v_proj = nn.Linear(h, h, bias=cfg.get("attention_bias", False))
        self.o_proj = nn.Linear(h, h, bias=cfg.get("attention_bias", False))
        self.gate_proj = nn.Linear(h, cfg["intermediate_size"], bias=bool(cfg.get("mlp_bias", False)))
        self.up_proj = nn.Linear(h, cfg["intermediate_size"], bias=bool(cfg.get("mlp_bias", False)))
        self.down_proj = nn.Linear(cfg["intermediate_size"], h, bias=bool(cfg.get("mlp_bias", False)))

    def __call__(self, x: mx.array, mask: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        residual = x
        y = self.input_layernorm(x)
        b, t, _ = y.shape
        q = self.q_proj(y).reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(y).reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(y).reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale + mask
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        y = (attn @ v).transpose(0, 2, 1, 3).reshape(b, t, -1)
        x = residual + self.o_proj(y)

        residual = x
        y = self.post_attention_layernorm(x)
        y = self.down_proj(nn.silu(self.gate_proj(y)) * self.up_proj(y))
        return residual + y


class EuroBertMultitask(nn.Module):
    def __init__(self, cfg: dict, vocab_size: int, n_upos: int, n_lemma: int):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(vocab_size, cfg["hidden_size"])
        self.layers = [EuroBertLayer(cfg) for _ in range(cfg["num_hidden_layers"])]
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.upos_classifier = nn.Linear(cfg["hidden_size"], n_upos)
        self.lemma_classifier = nn.Linear(cfg["hidden_size"], n_lemma)
        self._inv_freq = self._build_inv_freq()

    def _build_inv_freq(self) -> mx.array:
        dim = self.cfg.get("head_dim") or self.cfg["hidden_size"] // self.cfg["num_attention_heads"]
        values = 1.0 / (self.cfg["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        return mx.array(values)

    def _rope(self, seq_len: int) -> tuple[mx.array, mx.array]:
        positions = mx.arange(seq_len, dtype=mx.float32)
        freqs = positions[:, None] * self._inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)[None, None, :, :]
        sin = mx.sin(emb)[None, None, :, :]
        return cos, sin

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> tuple[mx.array, mx.array]:
        x = self.embed_tokens(input_ids)
        mask = (1.0 - attention_mask.astype(mx.float32))[:, None, None, :] * -1e9
        cos, sin = self._rope(input_ids.shape[1])
        for layer in self.layers:
            x = layer(x, mask, cos, sin)
        x = self.norm(x)
        return self.upos_classifier(x), self.lemma_classifier(x)


class BertAttention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        h = cfg["hidden_size"]
        self.num_heads = cfg["num_attention_heads"]
        self.head_dim = h // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(h, h)
        self.key = nn.Linear(h, h)
        self.value = nn.Linear(h, h)
        self.dense = nn.Linear(h, h)
        self.LayerNorm = nn.LayerNorm(h, eps=cfg.get("layer_norm_eps", 1e-12))

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        b, t, _ = x.shape
        q = self.query(x).reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale + mask
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        y = (attn @ v).transpose(0, 2, 1, 3).reshape(b, t, -1)
        return self.LayerNorm(x + self.dense(y))


class BertLayer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        h = cfg["hidden_size"]
        self.attention = BertAttention(cfg)
        self.intermediate_dense = nn.Linear(h, cfg["intermediate_size"])
        self.output_dense = nn.Linear(cfg["intermediate_size"], h)
        self.output_LayerNorm = nn.LayerNorm(h, eps=cfg.get("layer_norm_eps", 1e-12))

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        attn_out = self.attention(x, mask)
        hidden = nn.gelu(self.intermediate_dense(attn_out))
        return self.output_LayerNorm(attn_out + self.output_dense(hidden))


class BertMultitask(nn.Module):
    def __init__(self, cfg: dict, vocab_size: int, n_upos: int, n_lemma: int):
        super().__init__()
        self.cfg = cfg
        self.word_embeddings = nn.Embedding(vocab_size, cfg["hidden_size"])
        self.position_embeddings = nn.Embedding(cfg["max_position_embeddings"], cfg["hidden_size"])
        self.token_type_embeddings = nn.Embedding(cfg.get("type_vocab_size", 2), cfg["hidden_size"])
        self.LayerNorm = nn.LayerNorm(cfg["hidden_size"], eps=cfg.get("layer_norm_eps", 1e-12))
        self.layers = [BertLayer(cfg) for _ in range(cfg["num_hidden_layers"])]
        self.upos_classifier = nn.Linear(cfg["hidden_size"], n_upos)
        self.lemma_classifier = nn.Linear(cfg["hidden_size"], n_lemma)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> tuple[mx.array, mx.array]:
        b, t = input_ids.shape
        words = self.word_embeddings(input_ids)
        positions = self.position_embeddings(mx.arange(t, dtype=mx.int32)[None, :])
        token_types = self.token_type_embeddings(mx.zeros((b, t), dtype=mx.int32))
        x = self.LayerNorm(words + positions + token_types)
        mask = (1.0 - attention_mask.astype(mx.float32))[:, None, None, :] * -1e9
        for layer in self.layers:
            x = layer(x, mask)
        return self.upos_classifier(x), self.lemma_classifier(x)


def assign(obj, name: str, value: mx.array) -> None:
    parts = name.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def load_eurobert_weights(model: EuroBertMultitask, weights: dict[str, mx.array]) -> None:
    assign(model, "embed_tokens.weight", weights["model.embed_tokens.weight"])
    assign(model, "norm.weight", weights["model.norm.weight"])
    assign(model, "upos_classifier.weight", weights["upos_classifier.weight"])
    assign(model, "upos_classifier.bias", weights["upos_classifier.bias"])
    assign(model, "lemma_classifier.weight", weights["lemma_classifier.weight"])
    assign(model, "lemma_classifier.bias", weights["lemma_classifier.bias"])
    for i, layer in enumerate(model.layers):
        prefix = f"model.layers.{i}"
        for local, remote in [
            ("input_layernorm.weight", "input_layernorm.weight"),
            ("post_attention_layernorm.weight", "post_attention_layernorm.weight"),
            ("q_proj.weight", "self_attn.q_proj.weight"),
            ("k_proj.weight", "self_attn.k_proj.weight"),
            ("v_proj.weight", "self_attn.v_proj.weight"),
            ("o_proj.weight", "self_attn.o_proj.weight"),
            ("gate_proj.weight", "mlp.gate_proj.weight"),
            ("up_proj.weight", "mlp.up_proj.weight"),
            ("down_proj.weight", "mlp.down_proj.weight"),
        ]:
            assign(layer, local, weights[f"{prefix}.{remote}"])


def load_bert_weights(model: BertMultitask, weights: dict[str, mx.array]) -> None:
    has_model_prefix = any(k.startswith("model.") for k in weights)
    
    def get_weight(key: str) -> mx.array:
        orig_key = key
        if not has_model_prefix and key.startswith("model."):
            key = key[6:]
        if key in weights:
            return weights[key]
        alternates = {
            "model.embeddings.LayerNorm.weight": "model.embeddings.norm.weight",
            "model.embeddings.LayerNorm.bias": "model.embeddings.norm.bias",
            "embeddings.LayerNorm.weight": "embeddings.norm.weight",
            "embeddings.LayerNorm.bias": "embeddings.norm.bias",
        }
        alt_key = alternates.get(key)
        if alt_key and alt_key in weights:
            return weights[alt_key]
        if alt_key and not has_model_prefix and alt_key.startswith("model."):
            alt_key_no_pref = alt_key[6:]
            if alt_key_no_pref in weights:
                return weights[alt_key_no_pref]
                
        # Check alternates for encoder layers
        if "model.encoder.layer." in orig_key:
            parts = orig_key.split(".")
            try:
                idx = parts[3]
                suffix = ".".join(parts[4:])
                mappings = {
                    "attention.self.query.weight": "attention.query_proj.weight",
                    "attention.self.query.bias": "attention.query_proj.bias",
                    "attention.self.key.weight": "attention.key_proj.weight",
                    "attention.self.key.bias": "attention.key_proj.bias",
                    "attention.self.value.weight": "attention.value_proj.weight",
                    "attention.self.value.bias": "attention.value_proj.bias",
                    "attention.output.dense.weight": "attention.out_proj.weight",
                    "attention.output.dense.bias": "attention.out_proj.bias",
                    "attention.output.LayerNorm.weight": "ln1.weight",
                    "attention.output.LayerNorm.bias": "ln1.bias",
                    "intermediate.dense.weight": "linear1.weight",
                    "intermediate.dense.bias": "linear1.bias",
                    "output.dense.weight": "linear2.weight",
                    "output.dense.bias": "linear2.bias",
                    "output.LayerNorm.weight": "ln2.weight",
                    "output.LayerNorm.bias": "ln2.bias",
                }
                mapped_suffix = mappings.get(suffix)
                if mapped_suffix:
                    alt_key = f"model.encoder.layers.{idx}.{mapped_suffix}"
                    if alt_key in weights:
                        return weights[alt_key]
                    alt_key_no_pref = f"encoder.layers.{idx}.{mapped_suffix}"
                    if alt_key_no_pref in weights:
                        return weights[alt_key_no_pref]
            except Exception:
                pass
                
        raise KeyError(f"Could not find key {key} in weights.")

    assign(model, "word_embeddings.weight", get_weight("model.embeddings.word_embeddings.weight"))
    assign(model, "position_embeddings.weight", get_weight("model.embeddings.position_embeddings.weight"))
    assign(model, "token_type_embeddings.weight", get_weight("model.embeddings.token_type_embeddings.weight"))
    assign(model, "LayerNorm.weight", get_weight("model.embeddings.LayerNorm.weight"))
    assign(model, "LayerNorm.bias", get_weight("model.embeddings.LayerNorm.bias"))
    
    if "upos_classifier.weight" in weights:
        assign(model, "upos_classifier.weight", weights["upos_classifier.weight"])
        assign(model, "upos_classifier.bias", weights["upos_classifier.bias"])
        assign(model, "lemma_classifier.weight", weights["lemma_classifier.weight"])
        assign(model, "lemma_classifier.bias", weights["lemma_classifier.bias"])
        
    for i, layer in enumerate(model.layers):
        prefix = f"model.encoder.layer.{i}"
        assign(layer, "attention.query.weight", get_weight(f"{prefix}.attention.self.query.weight"))
        assign(layer, "attention.query.bias", get_weight(f"{prefix}.attention.self.query.bias"))
        assign(layer, "attention.key.weight", get_weight(f"{prefix}.attention.self.key.weight"))
        assign(layer, "attention.key.bias", get_weight(f"{prefix}.attention.self.key.bias"))
        assign(layer, "attention.value.weight", get_weight(f"{prefix}.attention.self.value.weight"))
        assign(layer, "attention.value.bias", get_weight(f"{prefix}.attention.self.value.bias"))
        assign(layer, "attention.dense.weight", get_weight(f"{prefix}.attention.output.dense.weight"))
        assign(layer, "attention.dense.bias", get_weight(f"{prefix}.attention.output.dense.bias"))
        assign(layer, "attention.LayerNorm.weight", get_weight(f"{prefix}.attention.output.LayerNorm.weight"))
        assign(layer, "attention.LayerNorm.bias", get_weight(f"{prefix}.attention.output.LayerNorm.bias"))
        assign(layer, "intermediate_dense.weight", get_weight(f"{prefix}.intermediate.dense.weight"))
        assign(layer, "intermediate_dense.bias", get_weight(f"{prefix}.intermediate.dense.bias"))
        assign(layer, "output_dense.weight", get_weight(f"{prefix}.output.dense.weight"))
        assign(layer, "output_dense.bias", get_weight(f"{prefix}.output.dense.bias"))
        assign(layer, "output_LayerNorm.weight", get_weight(f"{prefix}.output.LayerNorm.weight"))
        assign(layer, "output_LayerNorm.bias", get_weight(f"{prefix}.output.LayerNorm.bias"))


def attach_lora(model: nn.Module, rank: int, alpha: float) -> None:
    model.freeze()
    model.upos_classifier.unfreeze()
    model.lemma_classifier.unfreeze()
    for layer in model.layers:
        if hasattr(layer, "q_proj"):
            layer.q_proj = LoRALinear(layer.q_proj, rank, alpha)
            layer.k_proj = LoRALinear(layer.k_proj, rank, alpha)
            layer.v_proj = LoRALinear(layer.v_proj, rank, alpha)
            layer.o_proj = LoRALinear(layer.o_proj, rank, alpha)
        else:
            layer.attention.query = LoRALinear(layer.attention.query, rank, alpha)
            layer.attention.key = LoRALinear(layer.attention.key, rank, alpha)
            layer.attention.value = LoRALinear(layer.attention.value, rank, alpha)
            layer.attention.dense = LoRALinear(layer.attention.dense, rank, alpha)


def pad_batch(rows: list[dict], label_remap: dict[int, int] | None = None) -> dict[str, mx.array]:
    max_len = max(len(row["input_ids"]) for row in rows)
    # Round to next multiple of 64 to avoid MLX dynamic shape recompilation
    max_len = ((max_len + 63) // 64) * 64

    batch = {
        "input_ids": np.zeros((len(rows), max_len), dtype=np.int32),
        "attention_mask": np.zeros((len(rows), max_len), dtype=np.int32),
        "labels": np.full((len(rows), max_len), -100, dtype=np.int32),
        "upos_labels": np.full((len(rows), max_len), -100, dtype=np.int32),
    }
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        batch["input_ids"][i, :n] = row["input_ids"]
        batch["attention_mask"][i, :n] = row["attention_mask"]
        labels = row["labels"]
        if label_remap is not None:
            labels = [label_remap.get(int(label), -100) if int(label) != -100 else -100 for label in labels]
        batch["labels"][i, :n] = labels
        batch["upos_labels"][i, :n] = row["upos_labels"]
    return {key: mx.array(value) for key, value in batch.items()}


def masked_ce(logits: mx.array, labels: mx.array) -> mx.array:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    mask = flat_labels != -100
    safe = mx.maximum(flat_labels, 0)
    losses = nn.losses.cross_entropy(flat_logits, safe, reduction="none")
    return mx.sum(losses * mask.astype(losses.dtype)) / mx.maximum(mx.sum(mask), 1)


def word_positions(row: dict) -> list[int]:
    return [i for i, label in enumerate(row["upos_labels"]) if label != -100]


def strip_lang(label: str, lang: str) -> str:
    prefix = f"{lang}::"
    return label[len(prefix):] if label.startswith(prefix) else label


def select_valid(logits: np.ndarray, ids: np.ndarray, id2label: dict[str, str], lang: str, word: str) -> str:
    values = logits[ids]
    if len(values) <= 12:
        offsets = np.argsort(values)[::-1]
    else:
        top = np.argpartition(values, -12)[-12:]
        offsets = top[np.argsort(values[top])[::-1]]
    for offset in offsets:
        label = id2label.get(str(int(ids[int(offset)])), "UNKNOWN")
        if label == "UNKNOWN":
            continue
        base = strip_lang(label, lang)
        if apply_edit_label(word, base) is not None:
            return base
    return "IDENTITY"


def resolve(word: str, predicted_upos: str, base_label: str | None, lexicon: dict) -> str | None:
    if predicted_upos == "PROPN":
        return None
    if base_label is not None:
        applied = apply_edit_label(word, base_label)
        if applied is not None:
            return applied
    entry = lexicon.get(word)
    if isinstance(entry, dict):
        return entry.get(predicted_upos, entry.get(next(iter(entry))))
    if isinstance(entry, str):
        return entry
    return word


def raw_to_contiguous_map(label2id: dict[str, str]) -> dict[int, int]:
    labels = LabelSpace(label2id)
    remapped = labels.remapped_label2id
    return {int(label2id[label]): int(remapped[label]) for label in label2id}


def evaluate(model, rows: list[dict], lang: str, assets, batch_size: int, split: str = "") -> dict:
    label2id = read_json(assets.label2id_path)
    upos_label2id = read_json(assets.upos_label2id_path)
    label_space = LabelSpace(label2id)
    label_remap = raw_to_contiguous_map(label2id)
    id2label = label_space.id2label
    upos_id2label = {str(v): k for k, v in upos_label2id.items()}
    ids = label_space.candidate_ids(lang)
    lexicon = read_json(assets.lexicon_path) if assets.lexicon_path.exists() else {}
    if not isinstance(lexicon, dict):
        lexicon = {}

    total = upos_correct = lemma_total = lemma_correct = 0
    loss_total = loss_batches = 0
    model.eval()
    batches = math.ceil(len(rows) / batch_size)
    t0 = time.time()
    for batch_index, start in enumerate(range(0, len(rows), batch_size), start=1):
        batch_rows = rows[start:start + batch_size]
        batch = pad_batch(batch_rows, label_remap)
        upos_logits, lemma_logits = model(batch["input_ids"], batch["attention_mask"])
        loss = masked_ce(upos_logits, batch["upos_labels"]) + masked_ce(lemma_logits, batch["labels"])
        mx.eval(upos_logits, lemma_logits, loss)
        loss_total += float(loss)
        loss_batches += 1
        upos_np = np.array(upos_logits)
        lemma_np = np.array(lemma_logits)
        for b, row in enumerate(batch_rows):
            positions = word_positions(row)
            for word_i, (word, gold_lemma, gold_pos) in enumerate(zip(row["words"], row["lemmas"], row["upos"], strict=True)):
                if word_i >= len(positions):
                    break
                token_i = positions[word_i]
                total += 1
                predicted_upos = upos_id2label.get(str(int(np.argmax(upos_np[b, token_i]))), "X")
                if predicted_upos == gold_pos:
                    upos_correct += 1
                if gold_pos == "PROPN":
                    continue
                lemma_total += 1
                base = None if predicted_upos == "PROPN" else select_valid(lemma_np[b, token_i], ids, id2label, lang, word)
                if resolve(word, predicted_upos, base, lexicon) == gold_lemma:
                    lemma_correct += 1
        if split and (batch_index % 100 == 0 or batch_index == batches):
            print(
                json.dumps(
                    {
                        "event": "eval_progress",
                        "lang": lang,
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
        "tokens": total,
        "lemma_total": lemma_total,
        "lemma_accuracy": round(lemma_correct / max(lemma_total, 1), 4),
        "upos_accuracy": round(upos_correct / max(total, 1), 4),
    }


def find_struggles(model, validation_rows: list[dict], lang: str, assets, batch_size: int) -> set[str]:
    label2id = read_json(assets.label2id_path)
    upos_label2id = read_json(assets.upos_label2id_path)
    label_space = LabelSpace(label2id)
    label_remap = raw_to_contiguous_map(label2id)
    id2label = label_space.id2label
    upos_id2label = {str(v): k for k, v in upos_label2id.items()}
    ids = label_space.candidate_ids(lang)
    lexicon = read_json(assets.lexicon_path) if assets.lexicon_path.exists() else {}
    if not isinstance(lexicon, dict):
        lexicon = {}

    struggles = set()
    model.eval()
    for start in range(0, len(validation_rows), batch_size):
        batch_rows = validation_rows[start:start + batch_size]
        batch = pad_batch(batch_rows, label_remap)
        upos_logits, lemma_logits = model(batch["input_ids"], batch["attention_mask"])
        mx.eval(upos_logits, lemma_logits)
        upos_np = np.array(upos_logits)
        lemma_np = np.array(lemma_logits)
        for b, row in enumerate(batch_rows):
            positions = word_positions(row)
            for word_i, (word, gold_lemma, gold_pos) in enumerate(zip(row["words"], row["lemmas"], row["upos"], strict=True)):
                if word_i >= len(positions):
                    break
                token_i = positions[word_i]
                predicted_upos = upos_id2label.get(str(int(np.argmax(upos_np[b, token_i]))), "X")
                if gold_pos == "PROPN":
                    continue
                base = None if predicted_upos == "PROPN" else select_valid(lemma_np[b, token_i], ids, id2label, lang, word)
                pred_lemma = resolve(word, predicted_upos, base, lexicon)
                if pred_lemma != gold_lemma:
                    struggles.add(gold_lemma)
        mx.clear_cache()
    return struggles


def build_curriculum_datasets(train_rows: list[dict], val_rows: list[dict], max_train: int = 7000, max_val: int = 700):
    train_lemma_map = {}
    for idx, row in enumerate(train_rows):
        for lemma in row["lemmas"]:
            train_lemma_map.setdefault(lemma, []).append(idx)

    val_lemma_map = {}
    for idx, row in enumerate(val_rows):
        for lemma in row["lemmas"]:
            val_lemma_map.setdefault(lemma, []).append(idx)

    all_lemmas = set(train_lemma_map.keys()) | set(val_lemma_map.keys())

    selected_val_indices = set()
    for lemma in val_lemma_map:
        selected_val_indices.add(val_lemma_map[lemma][0])
        if len(selected_val_indices) >= max_val:
            break

    if len(selected_val_indices) < max_val:
        for idx in range(len(val_rows)):
            if idx not in selected_val_indices:
                selected_val_indices.add(idx)
            if len(selected_val_indices) == max_val:
                break

    covered_in_val = set()
    for idx in selected_val_indices:
        covered_in_val.update(val_rows[idx]["lemmas"])

    uncovered_lemmas = all_lemmas - covered_in_val
    selected_train_indices = set()

    for lemma in uncovered_lemmas:
        if lemma in train_lemma_map:
            selected_train_indices.add(train_lemma_map[lemma][0])

    if len(selected_train_indices) < max_train:
        for idx in range(len(train_rows)):
            if idx not in selected_train_indices:
                selected_train_indices.add(idx)
            if len(selected_train_indices) == max_train:
                break

    final_train = [train_rows[i] for i in selected_train_indices]
    final_val = [val_rows[i] for i in selected_val_indices]
    return final_train, final_val


def build_model(lang: str, checkpoint: Path):
    assets = language_assets(lang)
    cfg = read_json(checkpoint.parent / "config.json")
    weights = load_safetensors(checkpoint)
    label_space = LabelSpace(read_json(assets.label2id_path))
    n_upos = len(read_json(assets.upos_label2id_path))
    n_lemma = len(label_space.id2label)

    base_model = cfg.get("base_model_name_or_path")
    if base_model:
        backbone_cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        for k, v in backbone_cfg.to_dict().items():
            if k not in cfg:
                cfg[k] = v

    has_model_prefix = any(k.startswith("model.") for k in weights)
    if "model.embed_tokens.weight" in weights or "embed_tokens.weight" in weights:
        key = "model.embed_tokens.weight" if "model.embed_tokens.weight" in weights else "embed_tokens.weight"
        vocab_size = weights[key].shape[0]
        model = EuroBertMultitask(cfg, vocab_size=vocab_size, n_upos=n_upos, n_lemma=n_lemma)
        load_eurobert_weights(model, weights)
    else:
        key = "model.embeddings.word_embeddings.weight" if has_model_prefix else "embeddings.word_embeddings.weight"
        vocab_size = weights[key].shape[0]
        model = BertMultitask(cfg, vocab_size=vocab_size, n_upos=n_upos, n_lemma=n_lemma)
        load_bert_weights(model, weights)

    mx.eval(model.parameters())
    return model


def train_epoch(
    model,
    rows: list[dict],
    batch_size: int,
    optimizer,
    lang: str,
    epoch: int,
    label_remap: dict[int, int],
) -> float:
    model.train()

    def loss_fn(model, batch):
        upos_logits, lemma_logits = model(batch["input_ids"], batch["attention_mask"])
        return masked_ce(upos_logits, batch["upos_labels"]) + masked_ce(lemma_logits, batch["labels"])

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    order = np.random.permutation(len(rows))
    total = n = 0
    batches = math.ceil(len(order) / batch_size)
    t0 = time.time()
    for batch_index, start in enumerate(range(0, len(order), batch_size), start=1):
        batch_rows = [rows[int(i)] for i in order[start:start + batch_size]]
        batch = pad_batch(batch_rows, label_remap)
        loss, grads = loss_and_grad(model, batch)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(loss, model, optimizer)
        total += float(loss)
        n += 1
        if batch_index % 100 == 0 or batch_index == batches:
            print(
                json.dumps(
                    {
                        "event": "train_progress",
                        "lang": lang,
                        "epoch": epoch,
                        "batch": batch_index,
                        "batches": batches,
                        "loss": round(total / max(n, 1), 4),
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                ),
                flush=True,
            )
        if batch_index % 10 == 0:
            mx.clear_cache()
    return total / max(n, 1)


def main() -> None:
    """argparse wrapper around `run()` for `python -m lemmatizer.train.mlx_multitask`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--epochs", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--finetune-rows", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    args = parser.parse_args()

    from lemmatizer.languages import spec

    opts = TrainOptions(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        curriculum=args.curriculum,
        unfreeze_encoder=args.unfreeze_encoder,
        unfreeze_last_n=args.unfreeze_last_n,
        extra={"finetune_rows": args.finetune_rows},
    )
    run(spec(args.lang), opts)


def run(spec: LanguageSpec, opts: TrainOptions) -> None:
    """Canonical entry: train the multilingual multitask model for `spec.lang`."""
    lang = spec.lang
    assets = language_assets(lang)
    label_remap = raw_to_contiguous_map(read_json(assets.label2id_path))
    model = build_model(lang, Path(opts.checkpoint))
    dataset = load_from_disk(str(assets.dataset_path))
    train_rows = list(dataset["train"])
    val_rows = list(dataset["validation"])
    if opts.max_train_rows > 0:
        train_rows = train_rows[:opts.max_train_rows]
    if opts.max_val_rows > 0:
        val_rows = val_rows[:opts.max_val_rows]

    output_dir = Path(opts.output_dir or f"runs/mlx-{lang}-multitask")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"event": "baseline_start", "lang": lang}), flush=True)
    results = {
        "lang": lang,
        "checkpoint": opts.checkpoint,
        "baseline": {
            "train": evaluate(model, train_rows[:5000], lang, assets, opts.batch_size, "baseline_train"),
            "validation": evaluate(model, val_rows, lang, assets, opts.batch_size, "baseline_validation"),
        },
        "finetune": [],
    }
    print(json.dumps({"event": "baseline", **results["baseline"]}), flush=True)

    if opts.epochs > 0:
        if opts.lora_rank > 0:
            attach_lora(model, opts.lora_rank, opts.lora_alpha)
            print(
                json.dumps(
                    {
                        "event": "lora_attached",
                        "lang": lang,
                        "rank": opts.lora_rank,
                        "alpha": opts.lora_alpha,
                    }
                ),
                flush=True,
            )
        else:
            print("No LoRA modules attached (full fine-tuning)", flush=True)

        if opts.unfreeze_encoder:
            for layer in model.layers:
                layer.unfreeze()
            print("Unfroze all encoder layers", flush=True)
        elif opts.unfreeze_last_n > 0:
            n_layers = len(model.layers)
            for idx in range(max(0, n_layers - opts.unfreeze_last_n), n_layers):
                model.layers[idx].unfreeze()
            print(f"Unfroze last {opts.unfreeze_last_n} encoder layers", flush=True)

        optimizer = optim.AdamW(learning_rate=opts.lr, weight_decay=0.01)

        if opts.curriculum:
            print(json.dumps({"event": "curriculum_pool_building"}), flush=True)
            max_train = 50000 if lang == "sv" else 7000
            max_val = 5000 if lang == "sv" else 700
            train_pool, val_pool = build_curriculum_datasets(train_rows, val_rows, max_train, max_val)

            epochs = int(opts.epochs)
            current_train_indices = set(range(min(len(train_pool), max(1, len(train_pool) // epochs))))
            current_val_indices = set(range(min(len(val_pool), max(1, len(val_pool) // epochs))))

            current_train = [train_pool[i] for i in current_train_indices]
            current_val = [val_pool[i] for i in current_val_indices]

            best_val_acc = -1.0

            for epoch in range(1, epochs + 1):
                t0 = time.time()
                train_loss = train_epoch(model, current_train, opts.batch_size, optimizer, lang, epoch, label_remap)
                metrics = {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train": evaluate(model, train_rows[:5000], lang, assets, opts.batch_size, f"epoch_{epoch}_train"),
                    "validation": evaluate(model, val_rows, lang, assets, opts.batch_size, f"epoch_{epoch}_validation"),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                results["finetune"].append(metrics)
                print(json.dumps({"event": "epoch", **metrics}), flush=True)

                val_acc = metrics["validation"]["lemma_accuracy"]
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    model.save_weights(str(output_dir / "best.safetensors"))
                    print(f"  saved best model weights (val_acc={best_val_acc:.4f})", flush=True)

                model.save_weights(str(output_dir / f"epoch-{epoch}.safetensors"))

                if epoch < epochs:
                    struggles = find_struggles(model, current_val, lang, assets, opts.batch_size)
                    print(json.dumps({"event": f"struggles_identified_epoch_{epoch}", "count": len(struggles)}), flush=True)

                    next_train_size = min(int((epoch + 1) * len(train_pool) / epochs), len(train_pool))
                    next_val_size = min(int((epoch + 1) * len(val_pool) / epochs), len(val_pool))

                    remaining_train_indices = [i for i in range(len(train_pool)) if i not in current_train_indices]
                    remaining_train_indices.sort(key=lambda i: sum(1 for lbl in train_pool[i]["lemmas"] if lbl in struggles), reverse=True)

                    remaining_val_indices = [i for i in range(len(val_pool)) if i not in current_val_indices]
                    remaining_val_indices.sort(key=lambda i: sum(1 for lbl in val_pool[i]["lemmas"] if lbl in struggles), reverse=True)

                    added_train_indices = remaining_train_indices[:(next_train_size - len(current_train_indices))]
                    added_val_indices = remaining_val_indices[:(next_val_size - len(current_val_indices))]

                    current_train_indices.update(added_train_indices)
                    current_val_indices.update(added_val_indices)

                    current_train = [train_pool[i] for i in current_train_indices]
                    current_val = [val_pool[i] for i in current_val_indices]

        else:
            finetune_rows_limit = opts.extra.get("finetune_rows", 0)
            finetune_rows = train_rows[:finetune_rows_limit] if finetune_rows_limit > 0 else train_rows
            for epoch in range(int(opts.epochs)):
                t0 = time.time()
                train_loss = train_epoch(
                    model,
                    finetune_rows,
                    opts.batch_size,
                    optimizer,
                    lang,
                    epoch + 1,
                    label_remap,
                )
                metrics = {
                    "epoch": epoch + 1,
                    "train_loss": round(train_loss, 4),
                    "train": evaluate(model, train_rows[:5000], lang, assets, opts.batch_size, f"epoch_{epoch + 1}_train"),
                    "validation": evaluate(model, val_rows, lang, assets, opts.batch_size, f"epoch_{epoch + 1}_validation"),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                results["finetune"].append(metrics)
                print(json.dumps({"event": "epoch", **metrics}), flush=True)
                model.save_weights(str(output_dir / f"epoch-{epoch + 1}.safetensors"))

    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"event": "saved", "path": str(output_dir / "metrics.json")}), flush=True)


if __name__ == "__main__":
    main()
