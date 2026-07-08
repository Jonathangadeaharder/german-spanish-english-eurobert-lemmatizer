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
import os
import sys
import tempfile
from pathlib import Path

from lemmatizer.languages import LANGUAGES, vocab_levels_root

LEM_REPO = Path(__file__).parent
VOCAB_REPO = vocab_levels_root()

# Single source of truth: derive (lang_name, lemma_col) from the registry in
# lemmatizer.languages instead of duplicating the mapping here. Adding or
# renaming a language/column in LanguageSpec propagates automatically.
_LANG_TABLE = {s.lang: (s.name, s.vocab_lemma_column) for s in LANGUAGES if s.vocab_lemma_column}

CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1")


def _tokens_from_conllu_text(text: str) -> set[str]:
    """Read all FORM and LEMMA tokens from CoNLL-U text (lowercased)."""
    tokens: set[str] = set()
    for line in text.splitlines():
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


def read_treebank_tokens(conllu_path: Path) -> set[str]:
    """Read all FORM and LEMMA tokens from a CoNLL-U file (lowercased)."""
    if not conllu_path.exists():
        return set()
    return _tokens_from_conllu_text(conllu_path.read_text(encoding="utf-8"))


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
                # Sanitize: tabs/newlines in a CEFR word would corrupt the
                # CoNLL-U (broken columns or spurious sentence boundaries).
                word = word.replace("\t", " ").replace("\r", " ").replace("\n", " ")
                if not word or word.lower() in seen:
                    continue
                seen.add(word.lower())
                if upos not in (
                    "NOUN",
                    "VERB",
                    "ADJ",
                    "ADV",
                    "PROPN",
                    "ADP",
                    "PRON",
                    "DET",
                    "PART",
                    "NUM",
                    "CCONJ",
                    "SCONJ",
                    "INTJ",
                    "AUX",
                    "SYM",
                    "X",
                ):
                    upos = "X"
                words.append((word, upos))
    return words


def make_conllu_sentence(sent_id: str, word: str, lemma: str, upos: str) -> str:
    """Create a single-word CoNLL-U sentence with blank-line separator."""
    return (
        f"# sent_id = {sent_id}\n"
        f"# text = {word}\n"
        f"1\t{word}\t{lemma}\t{upos}\t_\t_\t0\troot\t_\t_\n"
        f"\n"
    )


def augment_language(lang_code: str) -> int:
    """Augment training data for one language. Returns count of added words."""
    lang_name, lemma_col = _LANG_TABLE[lang_code]
    gold_dir = LEM_REPO / "data" / "gold" / lang_code

    train_path = gold_dir / "train.conllu"
    dev_path = gold_dir / "dev.conllu"
    aug_path = gold_dir / "train_cefr_augmented.conllu"

    print(f"=== {lang_name} ({lang_code}) ===", flush=True)

    # Fail fast if the VocabLevels repository is missing or the language
    # directory has no level CSVs — otherwise read_cefr_words silently
    # returns an empty list and "Nothing to augment" misleads the user.
    lang_vocab_dir = VOCAB_REPO / lang_name
    if not VOCAB_REPO.exists() or not lang_vocab_dir.exists():
        print(
            f"  ERROR: CEFR vocab directory not found: {lang_vocab_dir}\n"
            "         Set VOCAB_LEVELS_DIR or clone the VocabLevels repo.",
            flush=True,
        )
        return 0

    # Read existing treebank tokens
    existing = read_treebank_tokens(train_path) | read_treebank_tokens(dev_path)
    print(f"  Treebank tokens: {len(existing)}", flush=True)

    # Read CEFR words
    cefr_words = read_cefr_words(lang_name, lemma_col)
    print(f"  CEFR words (unique): {len(cefr_words)}", flush=True)

    # Find missing
    missing = [(w, u) for w, u in cefr_words if w.lower() not in existing]
    print(f"  Missing from treebank: {len(missing)}", flush=True)

    if not missing:
        print("  Nothing to augment.", flush=True)
        return 0

    if not train_path.exists():
        print(f"Error: {train_path} not found. Run fetch-ud first.", flush=True)
        return 0

    with train_path.open(encoding="utf-8") as f:
        original = f.read()

    backup = gold_dir / "train_original.conllu"
    if not backup.exists():
        # First run: persist the pristine treebank before augmentation so
        # later re-runs can restore from it.
        backup.write_text(original, encoding="utf-8")
        print(f"  Backed up original to {backup}", flush=True)
    elif "cefr-augmented" in original:
        # Already augmented on a previous run: restore the pristine backup
        # BEFORE re-augmenting, then recompute missing against the pristine
        # content so previously-augmented CEFR words are not dropped.
        print(
            "  WARNING: train.conllu appears already augmented; "
            "restoring from backup before re-augmenting...",
            flush=True,
        )
        original = backup.read_text(encoding="utf-8")
        pristine_existing = _tokens_from_conllu_text(original) | read_treebank_tokens(dev_path)
        missing = [(w, u) for w, u in cefr_words if w.lower() not in pristine_existing]
        print(f"  Missing from pristine treebank: {len(missing)}", flush=True)
        if not missing:
            print("  Nothing to augment after restore.", flush=True)
            return 0

    parts = [original]
    if original and not original.endswith("\n"):
        parts.append("\n")
    for i, (word, upos) in enumerate(missing, start=1):
        sent_id = f"cefr-augmented-{lang_code}-{i:05d}"
        parts.append(make_conllu_sentence(sent_id, word, word, upos))
    augmented = "".join(parts)

    # Write the augmented snapshot (aug_path) AFTER the restore/recompute
    # block so it always reflects the same final content as train_path.
    # Crash-window note: if a crash occurs between this write and the
    # os.replace below, aug_path may briefly differ from train_path on
    # disk. This is self-healing — the next run detects 'cefr-augmented'
    # in train.conllu and restores from the backup before re-augmenting.
    aug_path.write_text(augmented, encoding="utf-8")
    print(f"  Written {len(missing)} synthetic sentences to {aug_path}", flush=True)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=gold_dir, suffix=".tmp")
    try:
        try:
            tmp_f = os.fdopen(tmp_fd, "w", encoding="utf-8")
        except Exception:
            # os.fdopen takes ownership of tmp_fd; if it raises, mkstemp
            # already opened the FD — close it explicitly to avoid a leak.
            os.close(tmp_fd)
            raise
        with tmp_f:
            tmp_f.write(augmented)
        os.replace(tmp_path, train_path)
    except Exception:
        if Path(tmp_path).exists():
            os.unlink(tmp_path)
        raise
    print(f"  Replaced {train_path} with augmented version", flush=True)

    return len(missing)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Augment UD training data with CEFR words.")
    parser.add_argument(
        "--lang",
        choices=tuple(_LANG_TABLE.keys()),
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
        failed: list[str] = []
        for code in _LANG_TABLE:
            # Isolate per-language failures so one language raising mid-way
            # does not abort the batch (previously-augmented languages already
            # had train.conllu replaced; remaining ones would be skipped).
            try:
                total += augment_language(code)
            except Exception as exc:  # noqa: BLE001 — report, don't abort siblings
                failed.append(code)
                print(f"  {code}: FAILED — {exc}", flush=True)
        print(f"\nTotal augmented: {total} words across all languages")
        if failed:
            print(f"Failed languages: {', '.join(failed)}", flush=True)
            return 1
    elif args.lang:
        augment_language(args.lang)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
