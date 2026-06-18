from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_DIR = Path("runs/eurobert-vocab-classifier-210m")
ONNX_DIR = Path("onnx/eurobert-vocab-classifier-210m")


class TokenClassificationExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main():
    model_dir = Path(os.getenv("MODEL_DIR", str(MODEL_DIR)))
    onnx_dir = Path(os.getenv("ONNX_DIR", str(ONNX_DIR)))
    onnx_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_dir), trust_remote_code=True, attn_implementation="eager"
    )
    model = model.half()
    model.eval()
    sample = tokenizer(["[LANG_DE]", "Haus"], is_split_into_words=True, return_tensors="pt")
    wrapper = TokenClassificationExportWrapper(model)
    onnx_path = onnx_dir / "model.onnx"

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
    print(f"Saved vocab classifier ONNX model to {onnx_path}")


if __name__ == "__main__":
    main()
