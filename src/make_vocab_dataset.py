from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from language_assets import LANGS, SPACY_MODELS, normalize_lang
from vocab_inventory import canonical_terms, load_inventory, load_spacy, save_inventory

MODEL_ID = "EuroBERT/EuroBERT-210m"
MAX_LENGTH = 128
VOCAB_ARTIFACTS_DIR = Path("artifacts/vocab")
VOCAB_DATASET_PATH = Path("data/processed/eurobert_vocab_classifier_dataset")
LABEL2ID = {"not_vocab": 0, "canonical_vocab": 1}
LANG_TOKEN = {
    "de": "[LANG_DE]",
    "es": "[LANG_ES]",
    "en": "[LANG_EN]",
}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def silver_paths() -> list[Path]:
    configured = os.getenv("SILVER_PATH")
    if configured:
        return [Path(configured)]
    paths: list[Path] = []
    silver_dir = Path(os.getenv("SILVER_OUTPUT_DIR", "data/silver"))
    for lang in LANGS:
        paths.append(silver_dir / f"lemma_{lang}_raw.jsonl")
        paths.append(silver_dir / f"{lang}/{lang}_lemma_silver.jsonl")
    targeted_dir = Path(os.getenv("TARGETED_OUTPUT_DIR", "data/silver/targeted"))
    for lang in LANGS:
        paths.append(targeted_dir / f"targeted_{lang}_raw.jsonl")
    annotated_dir = Path(os.getenv("ANNOTATED_SILVER_DIR", "data/silver/annotated"))
    for lang in LANGS:
        paths.append(annotated_dir / f"{lang}_silver_annotated.jsonl")
    return paths


def load_silver_sentences(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if "sentences" in row:
                    lang = normalize_lang(row.get("lang", "de"))
                    for sentence in row.get("sentences", []):
                        sentence = str(sentence).strip()
                        if sentence:
                            rows.append({"lang": lang, "text": sentence})
                elif "text" in row:
                    lang = normalize_lang(row.get("lang", path.stem.split("_")[0]))
                    text = str(row["text"]).strip()
                    if text:
                        rows.append({"lang": lang, "text": text})
    return rows


def split_rows(rows: list[dict[str, str]]) -> DatasetDict:
    rows = list(rows)
    if not rows:
        raise RuntimeError("No silver sentences found. Run generate-silver first.")

    validation_size = max(1, len(rows) // 10)
    test_size = max(1, len(rows) // 10)
    train_end = max(1, len(rows) - validation_size - test_size)
    validation_end = train_end + validation_size

    return DatasetDict(
        {
            "train": Dataset.from_list(rows[:train_end]),
            "validation": Dataset.from_list(rows[train_end:validation_end]),
            "test": Dataset.from_list(rows[validation_end:]),
        }
    )


def tokenize_and_label(row, tokenizer, nlp_by_lang, canonical_by_lang):
    lang = normalize_lang(row["lang"])
    doc = nlp_by_lang[lang](row["text"])
    original_words = [token.text for token in doc]
    labels_by_word = []
    canonical = canonical_by_lang[lang]

    for token in doc:
        text_key = token.text.lower()
        lemma_key = token.lemma_.lower()
        labels_by_word.append(1 if text_key in canonical or lemma_key in canonical else 0)

    words = [LANG_TOKEN[lang], *original_words]
    labels = [-100, *labels_by_word]
    encoded = tokenizer(words, is_split_into_words=True, truncation=True, max_length=MAX_LENGTH)
    token_labels = []
    previous_word_id = None

    for word_id in encoded.word_ids():
        if word_id is None:
            token_labels.append(-100)
        elif word_id != previous_word_id:
            token_labels.append(labels[word_id])
        else:
            token_labels.append(-100)
        previous_word_id = word_id

    encoded["labels"] = token_labels
    encoded["lang"] = lang
    encoded["words"] = original_words
    encoded["vocab_labels"] = labels_by_word
    encoded["length"] = len(encoded["input_ids"])
    return encoded


def main():
    VOCAB_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    inventory_path = VOCAB_ARTIFACTS_DIR / "canonical_inventory.json"
    if not inventory_path.exists():
        save_inventory(inventory_path)
    inventory = load_inventory(inventory_path)
    canonical_by_lang = {lang: canonical_terms(inventory, lang) for lang in LANGS}
    nlp_by_lang = {lang: load_spacy(lang) for lang in LANGS}

    rows = load_silver_sentences(silver_paths())
    limit = env_int("VOCAB_DATASET_LIMIT", 0)
    if limit > 0:
        rows = rows[:limit]

    raw_dataset = split_rows(rows)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": list(LANG_TOKEN.values())})

    dataset = DatasetDict(
        {
            split: raw_dataset[split].map(
                lambda row: tokenize_and_label(row, tokenizer, nlp_by_lang, canonical_by_lang),
                remove_columns=raw_dataset[split].column_names,
            )
            for split in raw_dataset
        }
    )

    VOCAB_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(VOCAB_DATASET_PATH))
    tokenizer.save_pretrained(str(VOCAB_ARTIFACTS_DIR / "tokenizer"))
    (VOCAB_ARTIFACTS_DIR / "label2id.json").write_text(
        json.dumps(LABEL2ID, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (VOCAB_ARTIFACTS_DIR / "id2label.json").write_text(
        json.dumps({str(v): k for k, v in LABEL2ID.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (VOCAB_ARTIFACTS_DIR / "spacy_models.json").write_text(
        json.dumps(SPACY_MODELS, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(dataset)
    print(f"Saved vocab classifier dataset to {VOCAB_DATASET_PATH}")


if __name__ == "__main__":
    main()
