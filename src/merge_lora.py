import json
import os
from pathlib import Path

from peft import PeftModel
from transformers import AutoTokenizer

from multitask_model import EuroBertForUposLemma, EuroBertUposLemmaConfig

BASE_MODEL = "EuroBERT/EuroBERT-210m"
ADAPTER_DIR = "runs/eurobert-multilingual-lemma-210m-lora"
MERGED_DIR = "models/eurobert-multilingual-lemma-210m-merged"
LABEL2ID_PATH = "artifacts/label2id.json"
UPOS_LABEL2ID_PATH = "artifacts/upos_label2id.json"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    adapter_dir = os.getenv("ADAPTER_DIR", ADAPTER_DIR)
    merged_dir = os.getenv("MERGED_DIR", MERGED_DIR)

    Path(merged_dir).mkdir(parents=True, exist_ok=True)

    label2id = load_json(LABEL2ID_PATH)
    upos_label2id = load_json(UPOS_LABEL2ID_PATH)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    config = EuroBertUposLemmaConfig(
        base_model_name_or_path=BASE_MODEL,
        upos_label2id=upos_label2id,
        lemma_label2id=label2id,
    )
    base_model = EuroBertForUposLemma.from_pretrained(
        BASE_MODEL,
        config=config,
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()

    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print(f"Saved merged model to {merged_dir}")


if __name__ == "__main__":
    main()
