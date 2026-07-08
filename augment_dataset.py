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
import traceback
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk

from lemmatizer.languages import LANG_TOKENS, LANGUAGES, vocab_levels_root

ROOT = Path(__file__).parent
VOCAB_ROOT = vocab_levels_root()

# Single source of truth: build (lang_name, lemma_col) from the registry
# in lemmatizer.languages instead of duplicating the mapping here.
_LANGUAGE_TABLE = {
    s.lang: (s.name, s.vocab_lemma_column) for s in LANGUAGES if s.vocab_lemma_column
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
                if not word or word in seen:
                    continue
                seen.add(word)
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
    lang_name, lemma_col = _LANGUAGE_TABLE[lang_code]
    assets_dir = ROOT / f"artifacts/lemma_{lang_code}"
    dataset_path = ROOT / f"data/processed/eurobert_lemma_{lang_code}_dataset"
    gold_dir = ROOT / f"data/gold/{lang_code}"

    # Load existing dataset (with ORIGINAL label space)
    ds = load_from_disk(str(dataset_path))
    n_existing_train = len(ds["train"])

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

    # Create rows for missing CEFR words. The original dataset inserts the
    # language token BEFORE tokenization (dataset.py: words=[LANG_TOKEN]+...),
    # so rows are [BOS, lang_token, word..., EOS]; mirror that here.
    lang_token = LANG_TOKENS[lang_code]
    new_rows = []

    for word, upos in missing:
        # Tokenize: [LANG_XX] word — lang token first, mirroring dataset.py.
        words = [lang_token, word]
        enc = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        word_ids = enc.word_ids()

        # Build labels. CEFR words are already lemmas (per the module
        # docstring), so the correct edit label is always IDENTITY — the
        # surface form equals the lemma. LOWERCASE would train the model to
        # output word.lower() (e.g. German "Haus" → "haus"), which is wrong
        # for languages where capitalized nouns ARE the lemma.
        identity_label = f"{lang_code}::IDENTITY"
        if identity_label not in label2id:
            identity_label = "IDENTITY"
        label_name = identity_label

        fallback = label2id.get(identity_label, label2id.get("UNKNOWN", 0))
        if label_name not in label2id:
            # The IDENTITY label is absent from label2id AND the fallback
            # resolved to UNKNOWN (id 0) — the CEFR word would silently get
            # the UNKNOWN class, corrupting training. Warn so this surfaces.
            print(
                f"  WARNING: label '{label_name}' not in label2id for "
                f"{lang_code}; falling back to UNKNOWN (id={fallback}).",
                flush=True,
            )
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
            elif wid == 0:
                # word 0 is the language token; mask it (parity with
                # dataset.py, which sets lemma_labels[0] = -100).
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
        # words/lemmas/upos hold only the real CEFR word (not the lang token),
        # matching dataset.py which stores original_words (sans LANG_TOKEN).
        enc["words"] = [word]
        enc["lemmas"] = [word]
        enc["upos"] = [upos]
        enc["length"] = int(len(enc["input_ids"]))
        new_rows.append(enc)

    # The language token is baked into each augmented row via the tokenizer
    # (words=[lang_token, word]); the prior logic checked input_ids[0]
    # against the lang token id, but index 0 is BOS, so it never matched.

    # Combine. concatenate_datasets appends the new rows in Arrow directly,
    # avoiding the doubling of memory from list(ds["train"]) -> from_list.
    # all_train count is kept only for the summary log.
    all_train = n_existing_train + len(new_rows)
    features = ds["train"].features
    new_ds = Dataset.from_list(new_rows, features=features)
    new_train = concatenate_datasets([ds["train"], new_ds])

    # Save to new path (can't overwrite in place)
    aug_path = dataset_path.parent / f"{dataset_path.name}_cefr_augmented"
    ds["train"] = new_train
    ds.save_to_disk(str(aug_path))
    print(f"  {lang_code}: added {len(new_rows)} rows (total train: {all_train})", flush=True)
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
        for code in _LANGUAGE_TABLE:
            try:
                total += augment_multitask(code)
            except Exception as exc:  # noqa: BLE001 — report, don't abort siblings
                failed.append(code)
                print(
                    f"  {code}: FAILED — {exc}\n{traceback.format_exc()}",
                    flush=True,
                )
        print(f"\nTotal: {total} CEFR rows added across all languages")
        if failed:
            print(f"Failed languages: {', '.join(failed)}", flush=True)
            return 1
    elif args.lang:
        if args.lang not in _LANGUAGE_TABLE:
            available = ", ".join(sorted(_LANGUAGE_TABLE))
            parser.error(f"unsupported --lang {args.lang!r}; expected one of: {available}")
        augment_multitask(args.lang)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
