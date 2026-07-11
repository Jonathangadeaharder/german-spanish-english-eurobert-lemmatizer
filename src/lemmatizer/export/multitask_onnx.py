"""Export MLX-trained multitask (EuroBERT/ScandiBERT) models to ONNX.

Handles de/en/es/fr/nl (EuroBERT + LoRA) and sv (ScandiBERT full finetune).
MLX has no native ONNX export — this bridge loads the HF backbone, syncs
MLX-trained weights (merging LoRA adapters into base weights), wraps with
the upos/lemma classifier heads, and exports via torch.onnx.export.

ONNX contract: 2 inputs (input_ids, attention_mask), 2 outputs
(upos_logits, lemma_logits). Matches OnnxBackend and web/demo.js.

Usage:
    LEMMA_LANG=de uv run python -m lemmatizer.export.multitask_onnx
    # Or with explicit paths:
    MODEL_DIR=runs/mlx-de-multitask-v4 ONNX_DIR=onnx/eurobert-lemma-de-210m \
      LEXICON_DIR=artifacts/lemma_de LEMMA_LANG=de \
      uv run python -m lemmatizer.export.multitask_onnx
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModel

RUN_DIRS = {
    "de": "runs/mlx-de-multitask-v4",
    "en": "runs/mlx-en-multitask-v2",
    "es": "runs/mlx-es-multitask-v2",
    "fr": "runs/mlx-fr-multitask-v2",
    "nl": "runs/mlx-nl-multitask-v2",
    "sv": "runs/mlx-sv-multitask-v2",
}

BASE_MODELS = {
    "de": "EuroBERT/EuroBERT-210m",
    "en": "EuroBERT/EuroBERT-210m",
    "es": "EuroBERT/EuroBERT-210m",
    "fr": "EuroBERT/EuroBERT-210m",
    "nl": "EuroBERT/EuroBERT-210m",
    "sv": "vesteinn/ScandiBERT",
}

EUROBERT_SNAPSHOT = (
    "~/.cache/huggingface/hub/models--EuroBERT--EuroBERT-210m/"
    "snapshots/39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e"
)
SCANDIBERT_SNAPSHOT = (
    "~/.cache/huggingface/hub/models--vesteinn--ScandiBERT/"
    "snapshots/e8339695d4bc4e61f1050b4c71853bed348a18b3"
)


class MultitaskONNXWrapper(nn.Module):
    """Wraps a HF backbone with upos + lemma classifier heads.

    forward(input_ids, attention_mask) -> (upos_logits, lemma_logits)
    """

    def __init__(self, backbone: nn.Module, hidden_size: int, n_upos: int, n_lemma: int):
        super().__init__()
        self.backbone = backbone
        self.upos_classifier = nn.Linear(hidden_size, n_upos)
        self.lemma_classifier = nn.Linear(hidden_size, n_lemma)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        return self.upos_classifier(hidden), self.lemma_classifier(hidden)


def load_mlx_weights(path: str) -> dict[str, np.ndarray]:
    """Load MLX safetensors checkpoint as numpy arrays."""
    weights = {}
    with safe_open(path, framework="np") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def merge_lora_weights(
    weights: dict[str, np.ndarray], lora_rank: int, lora_alpha: float
) -> dict[str, np.ndarray]:
    """Merge LoRA adapters into base weights: W += B@A * (alpha/rank)."""
    scale = lora_alpha / lora_rank
    # Collect LoRA pairs: layers.{i}.{q,k,v,o}_proj.lora_{a,b}
    lora_pairs: dict[str, dict[str, np.ndarray]] = {}
    for key in list(weights.keys()):
        m = re.match(r"layers\.(\d+)\.(q_proj|k_proj|v_proj|o_proj)\.lora_(a|b)", key)
        if m:
            layer_idx, proj, ab = m.groups()
            base = f"layers.{layer_idx}.{proj}"
            lora_pairs.setdefault(base, {})[ab] = weights.pop(key)

    for base, pair in lora_pairs.items():
        if "a" not in pair or "b" not in pair:
            continue
        w_key = f"{base}.weight"
        if w_key not in weights:
            continue
        a = pair["a"]  # (rank, in_features)
        b = pair["b"]  # (out_features, rank)
        delta = (b @ a) * scale
        weights[w_key] = weights[w_key] + delta

    return weights


def map_eurobert_mlx_to_hf(mlx_key: str) -> str | None:
    """Map MLX EuroBERT bare key to HF EuroBERT state_dict key.

    HF EuroBERT uses bare keys (no model. prefix): embed_tokens.weight,
    layers.{i}.self_attn.q_proj.weight, etc.
    """
    if mlx_key == "embed_tokens.weight":
        return "embed_tokens.weight"
    if mlx_key == "norm.weight":
        return "norm.weight"
    if mlx_key.startswith("layers."):
        m = re.match(r"layers\.(\d+)\.(.+)", mlx_key)
        if not m:
            return None
        idx, rest = m.groups()
        prefix = f"layers.{idx}"
        mapping = {
            "q_proj.weight": "self_attn.q_proj.weight",
            "k_proj.weight": "self_attn.k_proj.weight",
            "v_proj.weight": "self_attn.v_proj.weight",
            "o_proj.weight": "self_attn.o_proj.weight",
            "gate_proj.weight": "mlp.gate_proj.weight",
            "up_proj.weight": "mlp.up_proj.weight",
            "down_proj.weight": "mlp.down_proj.weight",
            "input_layernorm.weight": "input_layernorm.weight",
            "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        }
        hf_suffix = mapping.get(rest)
        if hf_suffix:
            return f"{prefix}.{hf_suffix}"
    return None


def map_bert_mlx_to_hf(mlx_key: str) -> str | None:
    """Map MLX BERT (ScandiBERT) key to HF RoBERTa state_dict key.

    HF RoBERTa (AutoModel) uses bare keys: embeddings.word_embeddings.weight,
    encoder.layer.{i}.attention.self.query.weight, etc.
    """
    embedding_map = {
        "word_embeddings.weight": "embeddings.word_embeddings.weight",
        "position_embeddings.weight": "embeddings.position_embeddings.weight",
        "token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
        "LayerNorm.weight": "embeddings.LayerNorm.weight",
        "LayerNorm.bias": "embeddings.LayerNorm.bias",
    }
    if mlx_key in embedding_map:
        return embedding_map[mlx_key]
    if mlx_key.startswith("layers."):
        m = re.match(r"layers\.(\d+)\.(.+)", mlx_key)
        if not m:
            return None
        idx, rest = m.groups()
        prefix = f"encoder.layer.{idx}"
        mapping = {
            "attention.query.weight": "attention.self.query.weight",
            "attention.query.bias": "attention.self.query.bias",
            "attention.key.weight": "attention.self.key.weight",
            "attention.key.bias": "attention.self.key.bias",
            "attention.value.weight": "attention.self.value.weight",
            "attention.value.bias": "attention.self.value.bias",
            "attention.dense.weight": "attention.output.dense.weight",
            "attention.dense.bias": "attention.output.dense.bias",
            "attention.LayerNorm.weight": "attention.output.LayerNorm.weight",
            "attention.LayerNorm.bias": "attention.output.LayerNorm.bias",
            "intermediate_dense.weight": "intermediate.dense.weight",
            "intermediate_dense.bias": "intermediate.dense.bias",
            "output_dense.weight": "output.dense.weight",
            "output_dense.bias": "output.dense.bias",
            "output_LayerNorm.weight": "output.LayerNorm.weight",
            "output_LayerNorm.bias": "output.LayerNorm.bias",
        }
        hf_suffix = mapping.get(rest)
        if hf_suffix:
            return f"{prefix}.{hf_suffix}"
    return None


def sync_weights(
    weights: dict[str, np.ndarray],
    backbone: nn.Module,
    wrapper: MultitaskONNXWrapper,
    is_eurobert: bool,
) -> dict:
    """Sync MLX weights into HF backbone + classifier heads."""
    hf_state = backbone.state_dict()
    mapper = map_eurobert_mlx_to_hf if is_eurobert else map_bert_mlx_to_hf

    mapped, skipped, missing = 0, [], []
    for mlx_key, tensor in weights.items():
        if mlx_key in (
            "upos_classifier.weight",
            "upos_classifier.bias",
            "lemma_classifier.weight",
            "lemma_classifier.bias",
        ):
            continue
        if mlx_key.startswith("layers.") and "lora" in mlx_key:
            continue
        hf_key = mapper(mlx_key)
        if hf_key is None:
            skipped.append(mlx_key)
            continue
        if hf_key not in hf_state:
            missing.append((mlx_key, hf_key))
            continue
        target = hf_state[hf_key]
        if tuple(target.shape) != tuple(tensor.shape):
            missing.append(
                (mlx_key, f"{hf_key} (shape: {tuple(tensor.shape)} vs {tuple(target.shape)})")
            )
            continue
        hf_state[hf_key] = torch.from_numpy(tensor).to(target.dtype)
        mapped += 1

    backbone.load_state_dict(hf_state, strict=False)

    # Classifier heads
    cls_state = {}
    for name in ("upos_classifier", "lemma_classifier"):
        for suffix in ("weight", "bias"):
            key = f"{name}.{suffix}"
            if key in weights:
                param = getattr(wrapper, name)
                target = getattr(param, suffix)
                cls_state[key] = torch.from_numpy(weights[key]).to(target.dtype)
    wrapper.load_state_dict(cls_state, strict=False)

    # Verify all 4 classifier params were loaded
    missing_cls = [
        k
        for k in (
            "upos_classifier.weight",
            "upos_classifier.bias",
            "lemma_classifier.weight",
            "lemma_classifier.bias",
        )
        if k not in weights
    ]
    if missing_cls:
        raise RuntimeError(f"Missing classifier weights: {missing_cls}")

    return {
        "mapped": mapped,
        "skipped": skipped[:10],
        "missing": missing,
        "n_skipped": len(skipped),
        "n_missing": len(missing),
    }


def export_lang(lang: str) -> None:
    if lang not in RUN_DIRS:
        raise ValueError(f"Unknown language '{lang}'. Valid: {sorted(RUN_DIRS.keys())}")
    model_dir = Path(os.getenv("MODEL_DIR", RUN_DIRS[lang]))
    onnx_dir = Path(os.getenv("ONNX_DIR", f"onnx/eurobert-lemma-{lang}-210m"))
    lexicon_dir = Path(os.getenv("LEXICON_DIR", f"artifacts/lemma_{lang}"))
    export_dtype = os.getenv("EXPORT_DTYPE", "fp32").lower()
    quantize = os.getenv("QUANTIZE", "int8").lower()

    mlx_path = model_dir / "best.safetensors"
    if not mlx_path.exists():
        raise FileNotFoundError(f"No checkpoint at {mlx_path}")

    onnx_dir.mkdir(parents=True, exist_ok=True)

    # Load label maps — use label2id for classifier size (it may have one
    # more entry than id2label due to the UNKNOWN token).
    upos_label2id = json.loads((lexicon_dir / "upos_label2id.json").read_text("utf-8"))
    label2id = json.loads((lexicon_dir / "label2id.json").read_text("utf-8"))
    n_upos = len(upos_label2id)
    n_lemma = len(label2id)
    print(f"[{lang}] n_upos={n_upos}, n_lemma={n_lemma}", flush=True)

    # Load MLX weights
    print(f"[{lang}] Loading MLX weights from {mlx_path}", flush=True)
    weights = load_mlx_weights(str(mlx_path))

    # Check for LoRA keys
    has_lora = any("lora" in k for k in weights)
    if has_lora:
        lora_keys = [k for k in weights if "lora" in k]
        # Infer rank from lora_a tensor shape: (rank, in_features)
        first_a = next((k for k in lora_keys if "lora_a" in k), None)
        if first_a is not None:
            lora_rank = weights[first_a].shape[0]
        else:
            lora_rank = 32 if lang == "de" else 16
        # Alpha is a training hyperparameter not stored in the checkpoint.
        # Override via LORA_ALPHA env var if training used a non-standard value.
        default_alpha = 64.0 if lang == "de" else 32.0
        lora_alpha = float(os.getenv("LORA_ALPHA", str(default_alpha)))
        print(
            f"[{lang}] Found LoRA adapters ({len(lora_keys)} keys), "
            f"merging (rank={lora_rank}, alpha={lora_alpha})...",
            flush=True,
        )
        weights = merge_lora_weights(weights, lora_rank, lora_alpha)

    # Load HF backbone
    base_model = BASE_MODELS[lang]
    is_eurobert = "EuroBERT" in base_model
    if is_eurobert:
        model_path = os.path.expanduser(EUROBERT_SNAPSHOT)
    else:
        model_path = os.path.expanduser(SCANDIBERT_SNAPSHOT)

    print(f"[{lang}] Loading HF backbone from {model_path}", flush=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    backbone.eval()

    hidden_size = config.hidden_size
    wrapper = MultitaskONNXWrapper(backbone, hidden_size, n_upos, n_lemma)

    # Sync weights
    print(f"[{lang}] Syncing weights...", flush=True)
    report = sync_weights(weights, backbone, wrapper, is_eurobert)
    print(
        f"  mapped={report['mapped']} skipped={report['n_skipped']} missing={report['n_missing']}",
        flush=True,
    )
    if report["skipped"]:
        print(f"  skipped: {report['skipped'][:5]}", flush=True)
    if report["missing"]:
        critical = [
            m
            for m in report["missing"]
            if any(c in m[0] for c in ("embed", "norm", "q_proj", "k_proj"))
        ]
        if critical:
            raise RuntimeError(
                f"[{lang}] Critical backbone weights not synced: {critical[:3]}. "
                "Refusing to export with pretrained/random weights."
            )
        print(f"  missing: {report['missing'][:5]}", flush=True)

    wrapper.eval()
    if export_dtype == "fp16":
        wrapper = wrapper.half()
    elif export_dtype == "bf16":
        wrapper = wrapper.to(torch.bfloat16)

    # Export
    B, T = 2, 32
    sample_ids = torch.zeros((B, T), dtype=torch.long)
    sample_mask = torch.ones((B, T), dtype=torch.long)

    onnx_path = onnx_dir / "model.onnx"
    print(f"[{lang}] Exporting ONNX to {onnx_path}", flush=True)
    torch.onnx.export(
        wrapper,
        (sample_ids, sample_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["upos_logits", "lemma_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "upos_logits": {0: "batch", 1: "sequence"},
            "lemma_logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[{lang}] Saved {onnx_path}", flush=True)

    if quantize == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = onnx_dir / "model.int8.onnx"
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )
        print(f"[{lang}] Saved int8 quantized to {int8_path}", flush=True)


def main() -> None:
    lang = os.getenv("LEMMA_LANG", "de")
    if lang == "all":
        for lang_code in RUN_DIRS:
            export_lang(lang_code)
    else:
        export_lang(lang)


if __name__ == "__main__":
    main()
