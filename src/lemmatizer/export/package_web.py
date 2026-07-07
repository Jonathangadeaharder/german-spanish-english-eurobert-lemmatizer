import os
import shutil
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
MERGED_DIR = Path("models/eurobert-multilingual-lemma-210m-merged")
ONNX_DIR = Path("onnx/eurobert-multilingual-lemma-210m")
WEB_MODEL_DIR = Path("web/model")


def copy_file(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"Missing required file: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", str(ARTIFACTS_DIR)))
    merged_dir = Path(os.getenv("MERGED_DIR", str(MERGED_DIR)))
    onnx_dir = Path(os.getenv("ONNX_DIR", str(ONNX_DIR)))
    web_model_dir = Path(os.getenv("WEB_MODEL_DIR", str(WEB_MODEL_DIR)))

    web_model_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        (onnx_dir / "model.onnx", web_model_dir / "model.onnx"),
        (merged_dir / "config.json", web_model_dir / "config.json"),
        (merged_dir / "tokenizer.json", web_model_dir / "tokenizer.json"),
        (merged_dir / "tokenizer_config.json", web_model_dir / "tokenizer_config.json"),
        (merged_dir / "special_tokens_map.json", web_model_dir / "special_tokens_map.json"),
        (artifacts_dir / "id2label.json", web_model_dir / "id2label.json"),
        (artifacts_dir / "upos_label2id.json", web_model_dir / "upos_label2id.json"),
        (artifacts_dir / "upos_id2label.json", web_model_dir / "upos_id2label.json"),
        (artifacts_dir / "edit_trees.json", web_model_dir / "edit_trees.json"),
        (artifacts_dir / "lexicon.json", web_model_dir / "lexicon.json"),
    ]

    for src, dst in required_files:
        copy_file(src, dst)

    for optional_name in ["added_tokens.json"]:
        src = merged_dir / optional_name
        if src.exists():
            copy_file(src, web_model_dir / optional_name)

    print(f"Packaged browser model into {web_model_dir}")


if __name__ == "__main__":
    main()
