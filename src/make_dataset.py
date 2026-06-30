import json
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

from conllu_reader import read_conllu
from edit_trees import make_edit_label

MODEL_ID = "EuroBERT/EuroBERT-210m"
MAX_LENGTH = 256

LANG_TOKEN = {
    "de": "[LANG_DE]",
    "es": "[LANG_ES]",
    "en": "[LANG_EN]",
    "fr": "[LANG_FR]",
}

DATA_FILES = {
    "train": {
        "de": "data/gold/de/train.conllu",
        "es": "data/gold/es/train.conllu",
        "en": "data/gold/en/train.conllu",
        "fr": "data/gold/fr/train.conllu",
    },
    "validation": {
        "de": "data/gold/de/dev.conllu",
        "es": "data/gold/es/dev.conllu",
        "en": "data/gold/en/dev.conllu",
        "fr": "data/gold/fr/dev.conllu",
    },
    "test": {
        "de": "data/gold/de/test.conllu",
        "es": "data/gold/es/test.conllu",
        "en": "data/gold/en/test.conllu",
        "fr": "data/gold/fr/test.conllu",
    },
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def convert_file(path, lang, tokenizer, lemma_label2id, upos_label2id):
    sentences = read_conllu(path, lang=lang)
    rows = []

    for sent in sentences:
        original_words = sent["words"]
        lemmas = sent["lemmas"]
        upos_tags = sent["upos"]

        words = [LANG_TOKEN[lang]] + original_words
        lemma_labels = [-100]
        upos_labels = [-100]

        for word, lemma, upos in zip(original_words, lemmas, upos_tags, strict=True):
            base_label = make_edit_label(word, lemma)
            full_label = f"{lang}::{base_label}"
            lemma_label_id = lemma_label2id.get(full_label, lemma_label2id["UNKNOWN"])
            upos_label_id = upos_label2id.get(upos, -100)

            if upos == "PROPN":
                lemma_label_id = -100

            lemma_labels.append(lemma_label_id)
            upos_labels.append(upos_label_id)

        enc = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        word_ids = enc.word_ids()

        labels = []
        upos_batch_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
                upos_batch_labels.append(-100)
            elif word_id != previous_word_id:
                labels.append(lemma_labels[word_id])
                upos_batch_labels.append(upos_labels[word_id])
            else:
                labels.append(-100)
                upos_batch_labels.append(-100)

            previous_word_id = word_id

        enc["labels"] = labels
        enc["upos_labels"] = upos_batch_labels
        enc["lang"] = lang
        enc["words"] = original_words
        enc["lemmas"] = lemmas
        enc["upos"] = sent["upos"]
        enc["length"] = len(enc["input_ids"])

        rows.append(enc)

    return Dataset.from_list(rows)


def convert_split(split_name, tokenizer, lemma_label2id, upos_label2id):
    datasets = []

    for lang, path in DATA_FILES[split_name].items():
        ds = convert_file(path, lang, tokenizer, lemma_label2id, upos_label2id)
        datasets.append(ds)

    return concatenate_datasets(datasets)


def main():
    lemma_label2id = load_json("artifacts/label2id.json")
    upos_label2id = load_json("artifacts/upos_label2id.json")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    tokenizer.add_special_tokens({"additional_special_tokens": list(LANG_TOKEN.values())})

    dataset = DatasetDict(
        {
            "train": convert_split("train", tokenizer, lemma_label2id, upos_label2id),
            "validation": convert_split("validation", tokenizer, lemma_label2id, upos_label2id),
            "test": convert_split("test", tokenizer, lemma_label2id, upos_label2id),
        }
    )

    out_path = "data/processed/eurobert_multilingual_lemma_dataset"
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(out_path)

    tokenizer.save_pretrained("artifacts/tokenizer")

    print(dataset)
    print(f"Saved dataset to {out_path}")
    print("Saved tokenizer with language tokens to artifacts/tokenizer")


if __name__ == "__main__":
    main()
