"""Augment UD treebank training data with synthetic sentences for CEFR words.

For each language, finds CEFR words (from VocabLevels CSVs) that are NOT
present in the UD treebank train+dev, and creates single-word CoNLL-U
sentences for them. These are appended to a new augmented train.conllu.

The model then sees every CEFR word during training, learning IDENTITY
edit trees for them (since CEFR words are already lemmas).

Usage:
    uv run python augment_training_with_cefr.py --lang nl
    uv run python augment_training_with_cefr.py --all
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

LEM_REPO = Path(__file__).parent
VOCAB_REPO = LEM_REPO.parent / "VocabLevels"

LANGUAGES = {
    "de": "german",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "sv": "swedish",
    "ar": "arabic",
    "zh": "chinese",
    "nl": "dutch",
}

LEMMAS = {
    "de": "German_Lemma",
    "en": "English_Lemma",
    "es": "Spanish_Lemma",
    "fr": "French_Lemma",
    "sv": "Swedish_Lemma",
    "ar": "Arabic_Lemma",
    "zh": "Chinese_Lemma",
    "nl": "Dutch_Lemma",
}

CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1")


def read_treebank_tokens(conllu_path: Path) -> set[str]:
    """Read all FORM and LEMMA tokens from a CoNLL-U file (lowercased)."""
    tokens: set[str] = set()
    if not conllu_path.exists():
        return tokens
    with conllu_path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            token_id = cols[0]
            if "-" in token_id or "." in token_id:
                continue
            form = cols[1].strip().lower()
            lemma = cols[2].strip().lower()
            if form:
                tokens.add(form)
            if lemma:
                tokens.add(lemma)
    return tokens


def read_cefr_words(lang_dir: str, lemma_col: str) -> list[tuple[str, str]]:
    """Read all CEFR words from A1-C1 CSVs. Returns (word, upos) tuples."""
    words: list[tuple[str, str]] = []
    seen: set[str] = set()
    for level in CEFR_LEVELS:
        csv_path = VOCAB_REPO / lang_dir / f"{level}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row.get(lemma_col, "").strip()
                upos = row.get("POS", "").strip()
                if not word or word.lower() in seen:
                    continue
                seen.add(word.lower())
                if upos not in (
                    "NOUN", "VERB", "ADJ", "ADV", "PROPN", "ADP",
                    "PRON", "DET", "PART", "NUM", "CCONJ", "SCONJ",
                    "INTJ", "AUX", "SYM", "X",
                ):
                    upos = "X"
                words.append((word, upos))
    return words


def make_conllu_sentence(
    sent_id: str, word: str, lemma: str, upos: str
) -> str:
    """Create a single-word CoNLL-U sentence with blank-line separator."""
    return (
        f"# sent_id = {sent_id}\n"
        f"# text = {word}\n"
        f"1\t{word}\t{lemma}\t{upos}\t_\t_\t0\troot\t_\t_\n"
        f"\n"
    )


def augment_language(lang_code: str) -> int:
    """Augment training data for one language. Returns count of added words."""
    lang_name = LANGUAGES[lang_code]
    lemma_col = LEMMAS[lang_code]
    gold_dir = LEM_REPO / "data" / "gold" / lang_code

    train_path = gold_dir / "train.conllu"
    dev_path = gold_dir / "dev.conllu"
    aug_path = gold_dir / "train_cefr_augmented.conllu"

    print(f"=== {lang_name} ({lang_code}) ===", flush=True)

    # Read existing treebank tokens
    existing = read_treebank_tokens(train_path) | read_treebank_tokens(dev_path)
    print(f"  Treebank tokens: {len(existing)}", flush=True)

    # Read CEFR words
    cefr_words = read_cefr_words(lang_name, lemma_col)
    print(f"  CEFR words (unique): {len(cefr_words)}", flush=True)

    # Find missing
    missing = [
        (w, u) for w, u in cefr_words
        if w.lower() not in existing
    ]
    print(f"  Missing from treebank: {len(missing)}", flush=True)

    if not missing:
        print("  Nothing to augment.", flush=True)
        return 0

    # Copy original train.conllu and append synthetic sentences
    with train_path.open(encoding="utf-8") as f:
        original = f.read()

    augmented = original
    for i, (word, upos) in enumerate(missing, start=1):
        sent_id = f"cefr-augmented-{lang_code}-{i:05d}"
        # CEFR words are already lemmas, so lemma = word
        augmented += make_conllu_sentence(sent_id, word, word, upos)

    aug_path.write_text(augmented, encoding="utf-8")
    print(f"  Written {len(missing)} synthetic sentences to {aug_path}", flush=True)

    # Also create augmented version that REPLACES train.conllu
    # (for the dataset builder to pick up)
    backup = gold_dir / "train_original.conllu"
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
        print(f"  Backed up original to {backup}", flush=True)

    train_path.write_text(augmented, encoding="utf-8")
    print(f"  Replaced {train_path} with augmented version", flush=True)

    return len(missing)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Augment UD training data with CEFR words."
    )
    parser.add_argument(
        "--lang",
        help="Language code (de/en/es/fr/sv/ar/zh/nl)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Augment all languages",
    )
    args = parser.parse_args(argv[1:])

    if args.all:
        total = 0
        for code in LANGUAGES:
            total += augment_language(code)
        print(f"\nTotal augmented: {total} words across all languages")
    elif args.lang:
        augment_language(args.lang)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
