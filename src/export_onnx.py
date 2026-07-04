import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from multitask_model import EuroBertForUposLemma, _patch_rope_default

_patch_rope_default()

MERGED_DIR = "models/eurobert-multilingual-lemma-210m-merged"
ONNX_DIR = "onnx/eurobert-multilingual-lemma-210m"


class MultiTaskExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits[0], outputs.logits[1]


def main():
    merged_dir = os.getenv("MERGED_DIR", MERGED_DIR)
    onnx_dir = os.getenv("ONNX_DIR", ONNX_DIR)

    Path(onnx_dir).mkdir(parents=True, exist_ok=True)

    export_dtype = os.getenv("EXPORT_DTYPE", "fp16").lower()
    quantize = os.getenv("QUANTIZE", "none").lower()

    tokenizer = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)
    model = EuroBertForUposLemma.from_pretrained(
        merged_dir,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    if export_dtype == "fp16":
        model = model.half()
    model.eval()

    sample = tokenizer("Dies ist ein Test.", return_tensors="pt")
    wrapper = MultiTaskExportWrapper(model)

    onnx_path = Path(onnx_dir) / "model.onnx"

    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["upos_logits", "lemma_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "upos_logits": {0: "batch", 1: "sequence"},
            "lemma_logits": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"Saved ONNX model to {onnx_path}")

    if quantize == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = Path(onnx_dir) / "model.int8.onnx"
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )
        print(f"Saved int8-quantized ONNX model to {int8_path}")


if __name__ == "__main__":
    main()
