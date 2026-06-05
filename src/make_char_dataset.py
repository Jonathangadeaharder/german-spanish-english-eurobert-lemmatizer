from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

from conllu_reader import read_conllu
from edit_trees import make_edit_label

MODEL_ID = "EuroBERT/EuroBERT-210m"
MAX_LENGTH = 256
MAX_LEMMA_LENGTH = 32
MAX_WORD_LENGTH = 64

LANG_TOKEN = {
    "de": "[LANG_DE]",
    "es": "[LANG_ES]",
    "en": "[LANG_EN]",
}

DATA_FILES = {
    "train": {
        "de": "data/gold/de/train.conllu",
        "es": "data/gold/es/train.conllu",
        "en": "data/gold/en/train.conllu",
    },
    "validation": {
        "de": "data/gold/de/dev.conllu",
        "es": "data/gold/es/dev.conllu",
        "en": "data/gold/en/dev.conllu",
    },
    "test": {
        "de": "data/gold/de/test.conllu",
        "es": "data/gold/es/test.conllu",
        "en": "data/gold/en/test.conllu",
    },
}

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode_chars(text: str, char2id: dict[str, int]) -> list[int]:
    return [char2id.get(ch, UNK_ID) for ch in text]


def convert_file(path, lang, tokenizer, lemma_label2id, upos_label2id, char2id):
    sentences = read_conllu(path, lang=lang)
    top_labels = set(lemma_label2id.keys()) - {"UNKNOWN"}
    rows = []

    for sent in sentences:
        original_words = sent["words"]
        lemmas = sent["lemmas"]
        upos_tags = sent["upos"]

        words = [LANG_TOKEN[lang]] + original_words
        lemma_tree_labels = [-100]
        upos_labels = [-100]
        lemma_routes = [-100]
        lemma_char_ids = [[PAD_ID] * MAX_LEMMA_LENGTH]
        word_char_ids = [[PAD_ID] * MAX_WORD_LENGTH]

        for word, lemma, upos in zip(original_words, lemmas, upos_tags, strict=True):
            base_label = make_edit_label(word, lemma)
            full_label = f"{lang}::{base_label}"

            is_propn = upos == "PROPN"

            if is_propn:
                lemma_tree_labels.append(-100)
                lemma_routes.append(-100)
                lemma_char_ids.append([PAD_ID] * MAX_LEMMA_LENGTH)
            else:
                in_top = full_label in top_labels
                lemma_routes.append(0 if in_top else 1)

                if in_top:
                    tree_id = lemma_label2id.get(full_label, lemma_label2id["UNKNOWN"])
                    lemma_tree_labels.append(tree_id)
                    lemma_char_ids.append([PAD_ID] * MAX_LEMMA_LENGTH)
                else:
                    lemma_tree_labels.append(lemma_label2id["UNKNOWN"])
                    char_seq = (
                        [BOS_ID] + encode_chars(lemma, char2id)[: MAX_LEMMA_LENGTH - 2] + [EOS_ID]
                    )
                    padded = char_seq + [PAD_ID] * (MAX_LEMMA_LENGTH - len(char_seq))
                    lemma_char_ids.append(padded)

            upos_labels.append(upos_label2id.get(upos, -100))

            wchars = encode_chars(word, char2id)[:MAX_WORD_LENGTH]
            wchars_padded = wchars + [PAD_ID] * (MAX_WORD_LENGTH - len(wchars))
            word_char_ids.append(wchars_padded)

        enc = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        word_ids = enc.word_ids()

        labels = []
        upos_batch_labels = []
        route_labels = []
        lemma_char_batch = []
        word_char_batch = []
        previous_word_id = None

        for wid in word_ids:
            if wid is None:
                labels.append(-100)
                upos_batch_labels.append(-100)
                route_labels.append(-100)
                lemma_char_batch.append([PAD_ID] * MAX_LEMMA_LENGTH)
                word_char_batch.append([PAD_ID] * MAX_WORD_LENGTH)
            elif wid != previous_word_id:
                labels.append(lemma_tree_labels[wid])
                upos_batch_labels.append(upos_labels[wid])
                route_labels.append(lemma_routes[wid])
                lemma_char_batch.append(lemma_char_ids[wid])
                word_char_batch.append(word_char_ids[wid])
            else:
                labels.append(-100)
                upos_batch_labels.append(-100)
                route_labels.append(-100)
                lemma_char_batch.append([PAD_ID] * MAX_LEMMA_LENGTH)
                word_char_batch.append([PAD_ID] * MAX_WORD_LENGTH)

            previous_word_id = wid

        enc["labels"] = labels
        enc["upos_labels"] = upos_batch_labels
        enc["lemma_route"] = route_labels
        enc["lemma_chars"] = lemma_char_batch
        enc["word_chars"] = word_char_batch
        enc["lang"] = lang
        enc["words"] = original_words
        enc["lemmas"] = lemmas
        enc["upos"] = sent["upos"]
        enc["length"] = len(enc["input_ids"])

        rows.append(enc)

    return Dataset.from_list(rows)


def convert_split(split_name, tokenizer, lemma_label2id, upos_label2id, char2id):
    datasets = []

    for lang, path in DATA_FILES[split_name].items():
        ds = convert_file(path, lang, tokenizer, lemma_label2id, upos_label2id, char2id)
        datasets.append(ds)

    return concatenate_datasets(datasets)


def main():
    lemma_label2id = load_json("artifacts/label2id_top300.json")
    upos_label2id = load_json("artifacts/upos_label2id.json")
    char_vocab = load_json("artifacts/char_vocab.json")
    char2id = char_vocab["char2id"]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "[LANG_DE]",
                "[LANG_ES]",
                "[LANG_EN]",
            ]
        }
    )

    dataset = DatasetDict(
        {
            "train": convert_split("train", tokenizer, lemma_label2id, upos_label2id, char2id),
            "validation": convert_split(
                "validation", tokenizer, lemma_label2id, upos_label2id, char2id
            ),
            "test": convert_split("test", tokenizer, lemma_label2id, upos_label2id, char2id),
        }
    )

    out_path = "data/processed/eurobert_char_lemma_dataset"
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(out_path)

    print(dataset)
    print(f"Saved character-level dataset to {out_path}")

    route_counts = {"edit_tree": 0, "char_gen": 0, "skipped": 0}
    for row in dataset["train"]:
        for r in row["lemma_route"]:
            if r == -100:
                route_counts["skipped"] += 1
            elif r == 0:
                route_counts["edit_tree"] += 1
            else:
                route_counts["char_gen"] += 1

    print(f"Training route distribution: {route_counts}")


if __name__ == "__main__":
    main()
