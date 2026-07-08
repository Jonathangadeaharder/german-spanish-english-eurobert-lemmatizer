import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value, concatenate_datasets
from transformers import AutoTokenizer

from lemmatizer.data.conllu import read_conllu
from lemmatizer.data.edit_trees import make_edit_label
from lemmatizer.languages import LANGUAGES, UD_FILES, language_assets

MODEL_ID = "EuroBERT/EuroBERT-210m"
MAX_LENGTH = 256


def build_row_features(sample_enc: dict) -> Features:
    """Build a pinned Features schema from the tokenizer's output keys.

    pyarrow infers list<str> columns as list<int64> when the first row happens
    to contain all-numeric strings (e.g. a sentence whose lemmas are
    ["2", "3"]). Pinning types prevents the mixed-type crash on single-
    language builds. Tokenizer output keys (e.g. token_type_ids is present
    for BERT but not for EuroBERT) drive which int64 sequence columns are
    included.
    """
    features: dict[str, object] = {
        "labels": Sequence(Value("int64")),
        "upos_labels": Sequence(Value("int64")),
        "lang": Value("string"),
        "words": Sequence(Value("string")),
        "lemmas": Sequence(Value("string")),
        "upos": Sequence(Value("string")),
        "length": Value("int64"),
    }
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        if key in sample_enc:
            features[key] = Sequence(Value("int64"))
    return Features(features)


LANG_TOKEN = {s.lang: s.lang_token for s in LANGUAGES}

# Gold split paths per lang, derived from the registry. Use split_files_for_lang
# rather than indexing this dict directly in new code.
DATA_FILES = UD_FILES


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def coerce_label2id(label2id: dict) -> dict:
    """JSON label maps store int ids as strings. Coerce to int for tensor use."""
    return {k: int(v) for k, v in label2id.items()}


def convert_file(path, lang, tokenizer, lemma_label2id, upos_label2id):
    sentences = read_conllu(path, lang=lang)
    rows = []
    features: Features | None = None

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
        # Truncate words/lemmas/upos to the set of words that have a
        # non-masked UPOS position (first sub-token within MAX_LENGTH).
        # This keeps words/lemmas/upos aligned with upos_batch_labels.
        words_with_positions = set()
        for idx_t, wid in enumerate(word_ids):
            if wid is not None and upos_batch_labels[idx_t] != -100:
                words_with_positions.add(wid)
        n_kept = max(words_with_positions) + 1 if words_with_positions else 0
        enc["words"] = original_words[:n_kept]
        enc["lemmas"] = lemmas[:n_kept]
        enc["upos"] = sent["upos"][:n_kept]
        enc["length"] = len(enc["input_ids"])

        rows.append(enc)

        if features is None:
            features = build_row_features(enc)

    return Dataset.from_list(rows, features=features)


def convert_split(split_name, tokenizer, lemma_label2id, upos_label2id, langs=None):
    datasets = []

    target_langs = langs if langs is not None else list(DATA_FILES[split_name].keys())
    for lang in target_langs:
        path = DATA_FILES[split_name][lang]
        ds = convert_file(path, lang, tokenizer, lemma_label2id, upos_label2id)
        datasets.append(ds)

    return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]


def main():
    # LEMMA_LANG: build a single-language dataset using that language's
    # backbone tokenizer (e.g. KB/bert-base-swedish-cased for sv). When unset,
    # build the shared multilingual dataset on EuroBERT.
    lang = os.getenv("LEMMA_LANG")
    assets = language_assets(lang) if lang else None

    if assets is not None:
        # Per-language build: load labels from the per-lang artifacts dir.
        lemma_label2id = coerce_label2id(load_json(str(assets.label2id_path)))
        upos_label2id = coerce_label2id(load_json(str(assets.upos_label2id_path)))
        tokenizer_name = assets.tokenizer_name
        out_path = str(assets.dataset_path)
        tokenizer_dir = str(assets.tokenizer_dir)
        target_lang = assets.lang
        lang_tokens = [LANG_TOKEN[target_lang]]
    else:
        lemma_label2id = coerce_label2id(load_json("artifacts/label2id.json"))
        upos_label2id = coerce_label2id(load_json("artifacts/upos_label2id.json"))
        tokenizer_name = MODEL_ID
        out_path = "data/processed/eurobert_multilingual_lemma_dataset"
        tokenizer_dir = "artifacts/tokenizer"
        target_lang = None
        lang_tokens = list(LANG_TOKEN.values())

    # trust_remote_code is only safe for the known EuroBERT backbone id; an
    # env-supplied TOKENIZER_NAME could point at an arbitrary repo, so we
    # disable remote code execution for env-resolved tokenizer ids.
    trust_remote = tokenizer_name == MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": lang_tokens})

    langs_for_split = [target_lang] if target_lang is not None else None
    dataset = DatasetDict(
        {
            "train": convert_split(
                "train", tokenizer, lemma_label2id, upos_label2id, langs_for_split
            ),
            "validation": convert_split(
                "validation", tokenizer, lemma_label2id, upos_label2id, langs_for_split
            ),
            "test": convert_split(
                "test", tokenizer, lemma_label2id, upos_label2id, langs_for_split
            ),
        }
    )

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(out_path)

    Path(tokenizer_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)

    print(dataset)
    print(f"Saved dataset to {out_path}")
    print(f"Saved tokenizer to {tokenizer_dir}")


if __name__ == "__main__":
    main()
