from __future__ import annotations

import os
import shutil
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts/vocab")
MODEL_DIR = Path("runs/eurobert-vocab-classifier-210m")
ONNX_DIR = Path("onnx/eurobert-vocab-classifier-210m")
WEB_MODEL_DIR = Path("web/model/vocab")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", str(ARTIFACTS_DIR)))
    model_dir = Path(os.getenv("MODEL_DIR", str(MODEL_DIR)))
    onnx_dir = Path(os.getenv("ONNX_DIR", str(ONNX_DIR)))
    web_model_dir = Path(os.getenv("WEB_MODEL_DIR", str(WEB_MODEL_DIR)))

    required_files = [
        (onnx_dir / "model.onnx", web_model_dir / "model.onnx"),
        (model_dir / "config.json", web_model_dir / "config.json"),
        (model_dir / "tokenizer.json", web_model_dir / "tokenizer.json"),
        (model_dir / "tokenizer_config.json", web_model_dir / "tokenizer_config.json"),
        (model_dir / "special_tokens_map.json", web_model_dir / "special_tokens_map.json"),
        (artifacts_dir / "id2label.json", web_model_dir / "id2label.json"),
        (artifacts_dir / "label2id.json", web_model_dir / "label2id.json"),
        (artifacts_dir / "canonical_inventory.json", web_model_dir / "canonical_inventory.json"),
    ]

    for src, dst in required_files:
        copy_file(src, dst)

    for optional_name in ["added_tokens.json"]:
        src = model_dir / optional_name
        if src.exists():
            copy_file(src, web_model_dir / optional_name)

    print(f"Packaged vocab classifier into {web_model_dir}")


if __name__ == "__main__":
    main()
