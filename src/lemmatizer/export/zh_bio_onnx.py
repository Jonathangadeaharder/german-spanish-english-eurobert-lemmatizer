"""Export MLX-trained zh_bio (bert-base-chinese) model to ONNX.

The zh model is a BERT-based char-level BIO-POS token classifier built via
openmed. This bridge loads HF BertForTokenClassification, syncs MLX weights
(merging LoRA adapters), and exports via torch.onnx.export.

ONNX contract: 2 inputs (input_ids, attention_mask), 1 output (logits).
Output shape: (B, T, 35) — 35 BIO-POS labels.

Usage:
    uv run python -m lemmatizer.export.zh_bio_onnx
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, BertForTokenClassification

BERT_PATH = "models/bert-base-chinese-mlx"
CHECKPOINT = "runs/mlx-zh-bio-pos/best.safetensors"
LABELS_PATH = "data/processed/zh_bio/labels.json"


def load_mlx_weights(path: str) -> dict[str, np.ndarray]:
    weights = {}
    with safe_open(path, framework="np") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def merge_lora(weights: dict[str, np.ndarray], rank: int, alpha: float) -> dict[str, np.ndarray]:
    """Merge LoRA adapters into BERT attention weights."""
    scale = alpha / rank
    lora_pairs: dict[str, dict[str, np.ndarray]] = {}
    for key in list(weights.keys()):
        m = re.match(
            r"encoder\.layers\.(\d+)\.attention\.(query_proj|key_proj|"
            r"value_proj|out_proj)\.lora_(a|b)",
            key,
        )
        if m:
            layer_idx, proj, ab = m.groups()
            base = f"encoder.layers.{layer_idx}.attention.{proj}"
            lora_pairs.setdefault(base, {})[ab] = weights.pop(key)

    for base, pair in lora_pairs.items():
        if "a" not in pair or "b" not in pair:
            continue
        w_key = f"{base}.weight"
        if w_key not in weights:
            continue
        a = pair["a"]
        b = pair["b"]
        delta = (b @ a) * scale
        weights[w_key] = weights[w_key] + delta
        # Also merge bias if present
        b_key = f"{base}.bias"
        if b_key in weights:
            pass

    return weights


def map_zh_mlx_to_hf(mlx_key: str) -> str | None:
    """Map MLX BERT (zh_bio) key to HF BERT state_dict key."""
    # Embeddings
    emb_map = {
        "embeddings.word_embeddings.weight": "bert.embeddings.word_embeddings.weight",
        "embeddings.position_embeddings.weight": "bert.embeddings.position_embeddings.weight",
        "embeddings.token_type_embeddings.weight": "bert.embeddings.token_type_embeddings.weight",
        "embeddings.norm.weight": "bert.embeddings.LayerNorm.weight",
        "embeddings.norm.bias": "bert.embeddings.LayerNorm.bias",
    }
    if mlx_key in emb_map:
        return emb_map[mlx_key]

    # Encoder layers
    if mlx_key.startswith("encoder.layers."):
        m = re.match(r"encoder\.layers\.(\d+)\.(.+)", mlx_key)
        if not m:
            return None
        idx, rest = m.groups()
        prefix = f"bert.encoder.layer.{idx}"
        mapping = {
            "attention.query_proj.weight": "attention.self.query.weight",
            "attention.query_proj.bias": "attention.self.query.bias",
            "attention.key_proj.weight": "attention.self.key.weight",
            "attention.key_proj.bias": "attention.self.key.bias",
            "attention.value_proj.weight": "attention.self.value.weight",
            "attention.value_proj.bias": "attention.self.value.bias",
            "attention.out_proj.weight": "attention.output.dense.weight",
            "attention.out_proj.bias": "attention.output.dense.bias",
            "linear1.weight": "intermediate.dense.weight",
            "linear1.bias": "intermediate.dense.bias",
            "linear2.weight": "output.dense.weight",
            "linear2.bias": "output.dense.bias",
            "ln1.weight": "attention.output.LayerNorm.weight",
            "ln1.bias": "attention.output.LayerNorm.bias",
            "ln2.weight": "output.LayerNorm.weight",
            "ln2.bias": "output.LayerNorm.bias",
        }
        hf_suffix = mapping.get(rest)
        if hf_suffix:
            return f"{prefix}.{hf_suffix}"

    # Pooler (not used but present in HF model)
    if mlx_key.startswith("pooler."):
        return None

    # Classifier head
    if mlx_key == "classifier.weight":
        return "classifier.weight"
    if mlx_key == "classifier.bias":
        return "classifier.bias"

    return None


def main() -> None:
    model_path = os.getenv("BERT_PATH", BERT_PATH)
    checkpoint = os.getenv("CHECKPOINT", CHECKPOINT)
    labels_path = os.getenv("LABELS_PATH", LABELS_PATH)
    onnx_dir = Path(os.getenv("ONNX_DIR", "onnx/eurobert-lemma-zh-210m"))
    quantize = os.getenv("QUANTIZE", "int8").lower()

    onnx_dir.mkdir(parents=True, exist_ok=True)

    label_meta = json.loads(Path(labels_path).read_text("utf-8"))
    n_labels = len(label_meta["label2id"])
    print(f"[zh] n_labels={n_labels}", flush=True)

    print(f"[zh] Loading MLX weights from {checkpoint}", flush=True)
    weights = load_mlx_weights(checkpoint)

    # Check for LoRA
    has_lora = any("lora" in k for k in weights)
    if has_lora:
        print("[zh] Found LoRA adapters, merging (rank=8, alpha=16)...", flush=True)
        weights = merge_lora(weights, rank=8, alpha=16.0)

    # Load HF BERT
    print(f"[zh] Loading HF BertForTokenClassification from {model_path}", flush=True)
    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = n_labels
    model = BertForTokenClassification.from_pretrained(
        model_path, config=config, ignore_mismatched_sizes=True
    )
    model.eval()

    # Sync weights
    hf_state = model.state_dict()
    mapped, skipped, missing = 0, [], []
    for mlx_key, tensor in weights.items():
        if "lora" in mlx_key:
            continue
        hf_key = map_zh_mlx_to_hf(mlx_key)
        if hf_key is None:
            skipped.append(mlx_key)
            continue
        if hf_key not in hf_state:
            missing.append((mlx_key, hf_key))
            continue
        target = hf_state[hf_key]
        if tuple(target.shape) != tuple(tensor.shape):
            missing.append(
                (
                    mlx_key,
                    f"{hf_key} (shape mismatch: {tuple(tensor.shape)} vs {tuple(target.shape)})",
                )
            )
            continue
        hf_state[hf_key] = torch.from_numpy(tensor).to(target.dtype)
        mapped += 1

    model.load_state_dict(hf_state, strict=False)
    print(f"  mapped={mapped} skipped={len(skipped)} missing={len(missing)}", flush=True)
    if skipped:
        print(f"  skipped: {skipped[:5]}", flush=True)
    if missing:
        print(f"  missing: {missing[:5]}", flush=True)

    # Export
    B, T = 2, 32
    sample_ids = torch.zeros((B, T), dtype=torch.long)
    sample_mask = torch.ones((B, T), dtype=torch.long)

    onnx_path = onnx_dir / "model.onnx"
    print(f"[zh] Exporting ONNX to {onnx_path}", flush=True)
    torch.onnx.export(
        model,
        (sample_ids, sample_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[zh] Saved {onnx_path}", flush=True)

    if quantize == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = onnx_dir / "model.int8.onnx"
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )
        print(f"[zh] Saved int8 quantized to {int8_path}", flush=True)


if __name__ == "__main__":
    main()
