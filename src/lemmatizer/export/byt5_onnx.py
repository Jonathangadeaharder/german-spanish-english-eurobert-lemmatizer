"""Export the MLX-trained ByT5 Arabic lemma classifier to ONNX.

The web runtime is ONNX-only, but MLX has no native ONNX export. This bridge
syncs MLX-trained weights into a PyTorch `T5EncoderModel` reference (loaded
from `google/byt5-small`), wraps it with the per-word byte-pooling head and
the linear lemma classifier, then exports via `torch.onnx.export`.

Weight name mapping (MLX t5.py → HF T5):
  wte.weight                                    → shared.weight
  encoder.relative_attention_bias.embeddings.weight
      → encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight
  encoder.layers[i].attention.query_proj.weight → encoder.block.{i}.layer.0.SelfAttention.q.weight
  encoder.layers[i].attention.key_proj.weight   → encoder.block.{i}.layer.0.SelfAttention.k.weight
  encoder.layers[i].attention.value_proj.weight → encoder.block.{i}.layer.0.SelfAttention.v.weight
  encoder.layers[i].attention.out_proj.weight   → encoder.block.{i}.layer.0.SelfAttention.o.weight
  encoder.layers[i].ln1.weight                  → encoder.block.{i}.layer.0.layer_norm.weight
  encoder.layers[i].dense.wi_0.weight            → encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight
  encoder.layers[i].dense.wi_1.weight            → encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight
  encoder.layers[i].dense.wo.weight              → encoder.block.{i}.layer.1.DenseReluDense.wo.weight
  encoder.layers[i].ln2.weight                  → encoder.block.{i}.layer.1.layer_norm.weight
  encoder.ln.weight                              → encoder.final_layer_norm.weight

The classifier head (`classifier.weight`, `classifier.bias`) maps directly.

Usage:
  MODEL_DIR=runs/ar-byt5-mlx ONNX_DIR=onnx/lemma_ar_byt5 \\
    LEXICON_DIR=artifacts/lemma_ar uv run python src/export_byt5_onnx.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors
from transformers import T5Config, T5EncoderModel

BYT5_MODEL_ID = "google/byt5-small"


def _map_mlx_to_hf(mlx_key: str) -> str | None:
    """Translate an MLX t5.py parameter name to the HF T5 state_dict key.

    Returns None for keys we don't map (decoder weights, classifier — the
    classifier is handled separately as it lives on the wrapper, not T5).
    """
    if mlx_key == "wte.weight":
        return "shared.weight"
    if mlx_key == "encoder.relative_attention_bias.embeddings.weight":
        return "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    if mlx_key == "encoder.ln.weight":
        return "encoder.final_layer_norm.weight"
    if mlx_key.startswith("encoder.layers."):
        rest = mlx_key[len("encoder.layers.") :]
        idx_str, tail = rest.split(".", 1)
        idx = int(idx_str)
        block = f"encoder.block.{idx}"
        if tail == "attention.query_proj.weight":
            return f"{block}.layer.0.SelfAttention.q.weight"
        if tail == "attention.key_proj.weight":
            return f"{block}.layer.0.SelfAttention.k.weight"
        if tail == "attention.value_proj.weight":
            return f"{block}.layer.0.SelfAttention.v.weight"
        if tail == "attention.out_proj.weight":
            return f"{block}.layer.0.SelfAttention.o.weight"
        if tail == "ln1.weight":
            return f"{block}.layer.0.layer_norm.weight"
        if tail == "dense.wi_0.weight":
            return f"{block}.layer.1.DenseReluDense.wi_0.weight"
        if tail == "dense.wi_1.weight":
            return f"{block}.layer.1.DenseReluDense.wi_1.weight"
        if tail == "dense.wo.weight":
            return f"{block}.layer.1.DenseReluDense.wo.weight"
        if tail == "ln2.weight":
            return f"{block}.layer.1.layer_norm.weight"
    return None


class ByT5LemmaONNXWrapper(nn.Module):
    """PyTorch mirror of ByT5EncoderLemmaClassifier's forward path.

    Wraps a HF T5EncoderModel (the encoder) plus the per-word byte-pooling
    and the linear lemma classifier. The pooling takes `word_byte_spans`
    (B, N_words, 2) and mean-pools each word's byte representations.
    """

    def __init__(self, encoder: T5EncoderModel, hidden_size: int, num_lemmas: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_lemmas)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        word_byte_spans: torch.LongTensor,
    ) -> torch.Tensor:
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state  # (B, T, D)

        B, T, D = enc_out.shape
        byte_idx = torch.arange(T, device=enc_out.device)  # (T,)
        starts = word_byte_spans[:, :, 0:1]  # (B, N_words, 1)
        ends = word_byte_spans[:, :, 1:2]
        in_span = ((byte_idx >= starts) & (byte_idx < ends)).to(enc_out.dtype)  # (B, N_words, T)
        pooled = torch.einsum("bnt,btd->bnd", in_span, enc_out)  # (B, N_words, D)
        word_lens = in_span.sum(dim=2, keepdim=True).clamp(min=1.0)  # (B, N_words, 1)
        pooled = pooled / word_lens
        empty = (word_lens == 0).to(pooled.dtype)
        pooled = pooled * (1.0 - empty)
        return self.classifier(pooled)  # (B, N_words, num_lemmas)


def load_mlx_weights_into_t5(mlx_weights_path: str, encoder: T5EncoderModel) -> dict:
    """Copy MLX-trained T5 encoder weights into the HF T5EncoderModel.

    Returns a report dict: {mapped, skipped, missing_in_hf}.
    """
    mlx_state = load_safetensors(mlx_weights_path)
    hf_state = encoder.state_dict()

    mapped, skipped, missing = 0, [], []
    for mlx_key, tensor in mlx_state.items():
        hf_key = _map_mlx_to_hf(mlx_key)
        if hf_key is None:
            if not mlx_key.startswith("decoder."):
                skipped.append(mlx_key)
            continue
        if hf_key not in hf_state:
            missing.append((mlx_key, hf_key))
            continue
        target = hf_state[hf_key]
        if tuple(target.shape) != tuple(tensor.shape):
            raise RuntimeError(
                f"Shape mismatch for {mlx_key}→{hf_key}: "
                f"mlx {tuple(tensor.shape)} vs hf {tuple(target.shape)}"
            )
        hf_state[hf_key] = tensor.to(target.dtype)
        mapped += 1

    encoder.load_state_dict(hf_state, strict=False)
    return {"mapped": mapped, "skipped": skipped, "missing_in_hf": missing}


def load_classifier_weights(mlx_weights_path: str, wrapper: ByT5LemmaONNXWrapper) -> None:
    """Copy the MLX-trained classifier Linear into the wrapper."""
    mlx_state = load_safetensors(mlx_weights_path)
    cls_w = mlx_state.get("classifier.weight")
    cls_b = mlx_state.get("classifier.bias")
    if cls_w is None:
        raise RuntimeError("classifier.weight not found in MLX weights")
    with torch.no_grad():
        wrapper.classifier.weight.copy_(cls_w.to(wrapper.classifier.weight.dtype))
        if cls_b is not None:
            wrapper.classifier.bias.copy_(cls_b.to(wrapper.classifier.bias.dtype))


def main() -> None:
    model_dir = Path(os.getenv("MODEL_DIR", "runs/ar-byt5-mlx"))
    onnx_dir = Path(os.getenv("ONNX_DIR", "onnx/lemma_ar_byt5"))
    lexicon_dir = Path(os.getenv("LEXICON_DIR", "artifacts/lemma_ar"))
    export_dtype = os.getenv("EXPORT_DTYPE", "fp32").lower()
    quantize = os.getenv("QUANTIZE", "none").lower()

    mlx_weights = model_dir / "final.safetensors"
    if not mlx_weights.exists():
        # Fall back to best.safetensors if final not saved yet.
        mlx_weights = model_dir / "best.safetensors"
    if not mlx_weights.exists():
        raise FileNotFoundError(
            f"No MLX weights found in {model_dir} (looked for final.safetensors, best.safetensors)"
        )

    onnx_dir.mkdir(parents=True, exist_ok=True)

    num_lemmas = len(json.loads((lexicon_dir / "lemma_label2id.json").read_text(encoding="utf-8")))
    print(f"num_lemmas={num_lemmas}", flush=True)

    print(f"Loading HF T5EncoderModel from {BYT5_MODEL_ID}", flush=True)
    config = T5Config.from_pretrained(BYT5_MODEL_ID)
    encoder = T5EncoderModel.from_pretrained(BYT5_MODEL_ID)
    encoder.eval()

    print(f"Syncing MLX weights from {mlx_weights}", flush=True)
    report = load_mlx_weights_into_t5(str(mlx_weights), encoder)
    print(
        f"  mapped={report['mapped']} skipped={len(report['skipped'])} "
        f"missing_in_hf={len(report['missing_in_hf'])}",
        flush=True,
    )
    if report["skipped"]:
        print(f"  skipped (first 5): {report['skipped'][:5]}", flush=True)

    wrapper = ByT5LemmaONNXWrapper(
        encoder=encoder, hidden_size=config.d_model, num_lemmas=num_lemmas
    )
    load_classifier_weights(str(mlx_weights), wrapper)
    wrapper.eval()

    if export_dtype == "fp16":
        wrapper = wrapper.half()
    elif export_dtype == "bf16":
        wrapper = wrapper.to(torch.bfloat16)

    B, T, _N = 2, 32, 4
    sample_input_ids = torch.zeros((B, T), dtype=torch.long)
    sample_attn = torch.ones((B, T), dtype=torch.long)
    sample_spans = torch.tensor([[[0, 4], [5, 9], [10, 14], [15, 19]]] * B, dtype=torch.long)

    onnx_path = onnx_dir / "model.onnx"
    print(f"Exporting ONNX to {onnx_path}", flush=True)
    torch.onnx.export(
        wrapper,
        (sample_input_ids, sample_attn, sample_spans),
        onnx_path,
        input_names=["input_ids", "attention_mask", "word_byte_spans"],
        output_names=["lemma_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "word_byte_spans": {0: "batch", 1: "num_words"},
            "lemma_logits": {0: "batch", 1: "num_words"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Saved ONNX model to {onnx_path}", flush=True)

    if quantize == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = onnx_dir / "model.int8.onnx"
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )
        print(f"Saved int8-quantized ONNX model to {int8_path}", flush=True)


if __name__ == "__main__":
    main()
