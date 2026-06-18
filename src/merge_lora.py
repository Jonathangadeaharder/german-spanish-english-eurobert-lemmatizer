import json
import os
from pathlib import Path

from peft import PeftModel
from transformers import AutoTokenizer

from language_assets import language_assets
from multitask_model import EuroBertForUposLemma, EuroBertUposLemmaConfig

BASE_MODEL = "EuroBERT/EuroBERT-210m"
MULTILINGUAL_TOKENIZER_DIR = "artifacts/tokenizer"
WARM_START_DIR = "models/eurobert-multilingual-lemma-210m-merged"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    assets = language_assets()
    lang = assets.lang
    print(f"Merging LoRA for language: {lang}")

    adapter_dir = os.getenv("ADAPTER_DIR", str(assets.output_dir))
    merged_dir = os.getenv("MERGED_DIR", str(assets.merged_dir))

    Path(merged_dir).mkdir(parents=True, exist_ok=True)

    label2id = load_json(str(assets.label2id_path))
    upos_label2id = load_json(str(assets.upos_label2id_path))
    tokenizer = AutoTokenizer.from_pretrained(MULTILINGUAL_TOKENIZER_DIR, trust_remote_code=True)
    config = EuroBertUposLemmaConfig(
        base_model_name_or_path=BASE_MODEL,
        upos_label2id=upos_label2id,
        lemma_label2id=label2id,
    )
    warm_start = os.getenv("TRAIN_WARM_START", WARM_START_DIR)
    base_model = EuroBertForUposLemma.from_pretrained(
        warm_start,
        config=config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()

    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print(f"Saved merged model to {merged_dir}")


if __name__ == "__main__":
    main()
