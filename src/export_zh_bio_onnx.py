"""Export ZH BIO-POS model to ONNX (single graph, token classification)."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, BertForTokenClassification

MODEL_DIR = os.getenv("MODEL_DIR", "runs/zh-bio-pos/final")
ONNX_DIR = os.getenv("ONNX_DIR", "onnx/zh-bio-pos")


class BioPosWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits


def main():
    Path(ONNX_DIR).mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = BertForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    sample = tokenizer("我在图书馆看书", return_tensors="pt")
    wrapper = BioPosWrapper(model)

    onnx_path = Path(ONNX_DIR) / "model.onnx"
    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Saved ONNX to {onnx_path}")

    # INT8 quantize
    from onnxruntime.quantization import QuantType, quantize_dynamic
    int8_path = Path(ONNX_DIR) / "model.int8.onnx"
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    print(f"Saved int8 ONNX to {int8_path}")


if __name__ == "__main__":
    main()
