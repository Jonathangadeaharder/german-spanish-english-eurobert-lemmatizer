"""Augment existing HF datasets with CEFR words WITHOUT rebuilding labels.

The key insight: the original label space already has IDENTITY and LOWERCASE
labels. CEFR words that are already lemmas will map to these existing labels.
We just add new rows to the existing dataset.

This preserves the original label2id mapping, so existing trained models
can warm-start (same classifier dimensions).

Usage:
    uv run python augment_dataset.py --lang de
    uv run python augment_dataset.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from datasets import Dataset, load_from_disk

from lemmatizer.languages import vocab_levels_root

ROOT = Path(__file__).parent
VOCAB_ROOT = vocab_levels_root()

LANGUAGES = {
    "de": ("german", "German_Lemma"),
    "en": ("english", "English_Lemma"),
    "es": ("spanish", "Spanish_Lemma"),
    "fr": ("french", "French_Lemma"),
    "sv": ("swedish", "Swedish_Lemma"),
    "ar": ("arabic", "Arabic_Lemma"),
    "zh": ("chinese", "Chinese_Lemma"),
    "nl": ("dutch", "Dutch_Lemma"),
}

LEVELS = ("A1", "A2", "B1", "B2", "C1")
MAX_LENGTH = 256


def read_cefr_words(lang_name: str, lemma_col: str) -> list[tuple[str, str]]:
    """Read CEFR words. Returns (word, upos) tuples."""
    words: list[tuple[str, str]] = []
    seen: set[str] = set()
    for level in LEVELS:
        csv_path = VOCAB_ROOT / lang_name / f"{level}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row.get(lemma_col, "").strip()
                upos = row.get("POS", "").strip() or "X"
                if not word or word.lower() in seen:
                    continue
                seen.add(word.lower())
                words.append((word, upos))
    return words


def read_treebank_tokens(train_path: Path, dev_path: Path) -> set[str]:
    """Read all FORM+LEMMA from CoNLL-U (lowercased)."""
    tokens: set[str] = set()
    for p in (train_path, dev_path):
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                cols = line.split("\t")
                if len(cols) < 4 or "-" in cols[0] or "." in cols[0]:
                    continue
                tokens.add(cols[1].strip().lower())
                tokens.add(cols[2].strip().lower())
    return tokens


def augment_multitask(lang_code: str) -> int:
    """Augment a multitask (EuroBERT) dataset with CEFR words."""
    lang_name, lemma_col = LANGUAGES[lang_code]
    assets_dir = ROOT / f"artifacts/lemma_{lang_code}"
    dataset_path = ROOT / f"data/processed/eurobert_lemma_{lang_code}_dataset"
    gold_dir = ROOT / f"data/gold/{lang_code}"

    # Load existing dataset (with ORIGINAL label space)
    ds = load_from_disk(str(dataset_path))
    train_rows = list(ds["train"])

    # Load existing label2id + upos_label2id
    label2id = json.loads((assets_dir / "label2id.json").read_text())
    upos_label2id = json.loads((assets_dir / "upos_label2id.json").read_text())

    # Find missing CEFR words
    train_src = gold_dir / "train_original.conllu"
    if not train_src.exists():
        train_src = gold_dir / "train.conllu"
    existing = read_treebank_tokens(
        train_src,
        gold_dir / "dev.conllu",
    )
    cefr_words = read_cefr_words(lang_name, lemma_col)
    missing = [(w, u) for w, u in cefr_words if w.lower() not in existing]

    print(f"  {lang_code}: {len(missing)} CEFR words missing from treebank", flush=True)

    if not missing:
        return 0

    # Load tokenizer (already saved)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(assets_dir / "tokenizer"),
        legacy=False,
    )

    # Create rows for missing CEFR words
    # Each word becomes a single-token sentence: "[LANG_XX] word"
    lang_token = f"[LANG_{lang_code.upper()}]"
    new_rows = []

    for word, upos in missing:
        # Tokenize: [LANG_XX] word
        words = [word]
        enc = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        word_ids = enc.word_ids()

        # Build labels. CEFR words are lemmas → IDENTITY or LOWERCASE.
        # Check both prefixed and unprefixed forms in label2id.
        identity_label = f"{lang_code}::IDENTITY"
        lowercase_label = f"{lang_code}::LOWERCASE"
        # Fallback to unprefixed if prefixed not in label2id
        if identity_label not in label2id:
            identity_label = "IDENTITY"
        if lowercase_label not in label2id:
            lowercase_label = "LOWERCASE"

        # Use IDENTITY by default, LOWERCASE if word is capitalized
        if word[0].isupper() and word.lower() != word:
            label_name = lowercase_label
        else:
            label_name = identity_label

        fallback = label2id.get(identity_label, label2id.get("UNKNOWN", 0))
        lemma_label_id = label2id.get(label_name, fallback)
        upos_label_id = upos_label2id.get(upos, upos_label2id.get("X", -1))

        if upos == "PROPN":
            lemma_label_id = -100

        labels = []
        upos_labels = []
        prev_wid = None
        for wid in word_ids:
            if wid is None or wid == prev_wid:
                labels.append(-100)
                upos_labels.append(-100)
            else:
                labels.append(lemma_label_id)
                upos_labels.append(upos_label_id)
            prev_wid = wid

        enc["labels"] = [int(x) for x in labels]
        enc["upos_labels"] = [int(x) for x in upos_labels]
        # BERT tokenizers include token_type_ids; add if present in existing data
        if "token_type_ids" not in enc and "token_type_ids" in ds["train"].column_names:
            enc["token_type_ids"] = [0] * len(enc["input_ids"])
        enc["lang"] = lang_code
        enc["words"] = list(words)
        enc["lemmas"] = list([word])
        enc["upos"] = list([upos])
        enc["length"] = int(len(enc["input_ids"]))
        new_rows.append(enc)

    # Prepend the language token to new rows when the original dataset
    # starts with it. Resolve the lang-token id from the tokenizer once and
    # check its presence at index 0 of the first original row.
    lang_token_ids = tokenizer.encode(lang_token, add_special_tokens=False)
    orig_has_lang = bool(
        train_rows and lang_token_ids and train_rows[0]["input_ids"][0] == lang_token_ids[0]
    )
    new_has_lang = bool(
        lang_token_ids and new_rows and new_rows[0]["input_ids"][0] == lang_token_ids[0]
    )
    if orig_has_lang and not new_has_lang and lang_token_ids:
        lang_id = lang_token_ids[0]
        has_token_type = "token_type_ids" in new_rows[0]
        for row in new_rows:
            row["input_ids"] = [lang_id] + row["input_ids"]
            row["attention_mask"] = [1] + row["attention_mask"]
            row["labels"] = [-100] + row["labels"]
            row["upos_labels"] = [-100] + row["upos_labels"]
            # Keep token_type_ids aligned with input_ids in length so the
            # model does not crash on the +1 length mismatch.
            if has_token_type:
                row["token_type_ids"] = [0] + row["token_type_ids"]
            row["length"] = len(row["input_ids"])

    # Combine
    all_train = train_rows + new_rows
    features = ds["train"].features
    new_train = Dataset.from_list(all_train, features=features)

    # Save to new path (can't overwrite in place)
    aug_path = dataset_path.parent / f"{dataset_path.name}_cefr_augmented"
    ds["train"] = new_train
    ds.save_to_disk(str(aug_path))
    print(f"  {lang_code}: added {len(new_rows)} rows (total train: {len(all_train)})", flush=True)
    print(f"  Saved to {aug_path}", flush=True)
    return len(new_rows)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv[1:])

    if args.all:
        total = 0
        failed: list[str] = []
        for code in LANGUAGES:
            try:
                total += augment_multitask(code)
            except Exception as exc:  # noqa: BLE001 — report, don't abort siblings
                failed.append(code)
                print(f"  {code}: FAILED — {exc}", flush=True)
        print(f"\nTotal: {total} CEFR rows added across all languages")
        if failed:
            print(f"Failed languages: {', '.join(failed)}", flush=True)
            return 1
    elif args.lang:
        augment_multitask(args.lang)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
