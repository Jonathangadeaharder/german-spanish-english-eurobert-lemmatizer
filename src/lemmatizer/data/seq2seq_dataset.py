"""Build seq2seq training data for the two-stage lemmatizer pipeline.

Stage 2 takes UPOS-annotated sentences and produces lemma sequences:
  Input:  "The [DET] fliegen [NOUN] are [AUX] annoying [ADJ] . [PUNCT]"
  Output: "the fly be annoying ."

This module reads CoNLL-U treebank files + CEFR contextual sentences,
formats them as input/output pairs, and builds byte-level ByT5 datasets.

Noise injection: 10% of UPOS tags are randomly corrupted during training
to mitigate exposure bias (Model 1 will make mistakes at inference time).

Usage:
    LEMMA_LANG=de uv run python -m lemmatizer.data.seq2seq_dataset
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets

from lemmatizer.data.byt5_dataset import BYT5_EOS, BYTE_ID_OFFSET, MAX_SEQ_LEN
from lemmatizer.data.conllu import read_conllu
from lemmatizer.data.script_guard import assert_language_plausible

# All valid UPOS tags for noise injection
UPOS_TAGS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]

NOISE_RATE = 0.10

# Imported from byt5_dataset so the dataset builder and the trainer
# (seq2seq_lemma.MAX_SEQ_LEN) share one definition; a drift between the
# two would let collate_batch silently truncate the trailing EOS and
# corrupt the training signal. See byt5_dataset.MAX_SEQ_LEN.

# Lemma delimiter. A bare space is ambiguous when a lemma itself contains a
# space (multi-word expressions like "à la"); the model cannot tell word
# boundaries from intra-lemma spaces. Use " | " so multi-word lemmas stay
# unambiguous and decoders can split on it cleanly.
LEMMA_DELIM = " | "


def encode_bytes(text: str) -> list[int]:
    """Encode text as ByT5 byte-level token IDs."""
    byte_ids = [BYT5_EOS]
    for b in text.encode("utf-8"):
        byte_ids.append(b + BYTE_ID_OFFSET)
    byte_ids.append(BYT5_EOS)
    return byte_ids


def format_input(
    words: list[str],
    upos_tags: list[str],
    noise: bool = False,
    rng: random.Random | None = None,
) -> str:
    """Format a sentence as 'word [UPOS] word [UPOS] ...' for Model 2 input.

    Unknown UPOS tags (e.g. "_") are normalized to the "X" sentinel before
    formatting so they appear verbatim on dev/test (noise=False) and so noise
    injection only swaps among the 17 valid tags rather than "fixing" invalid
    tags 10% of the time.
    """
    r = rng if rng is not None else random
    parts = []
    for word, upos in zip(words, upos_tags, strict=True):
        if upos not in UPOS_TAGS:
            upos = "X"
        if noise and r.random() < NOISE_RATE:
            # Corrupt the UPOS tag with a random wrong one
            wrong_choices = [t for t in UPOS_TAGS if t != upos]
            upos = r.choice(wrong_choices)
        parts.append(f"{word} [{upos}]")
    return " ".join(parts)


def format_output(lemmas: list[str]) -> str:
    """Format lemmas as a delimiter-joined string for Model 2 output.

    Uses LEMMA_DELIM (" | ") rather than a bare space so multi-word lemmas
    (e.g. "à la") stay unambiguous — the decoder can split on the sentinel
    without conflating word boundaries with intra-lemma spaces.

    Assumption: no input lemma contains the delimiter string itself.
    """
    for lemma in lemmas:
        if LEMMA_DELIM in lemma:
            raise ValueError(f"Lemma contains delimiter: {lemma!r}")
    return LEMMA_DELIM.join(lemmas)


def build_split(
    conllu_path: str,
    lang: str,
    noise: bool = False,
    seed: int = 42,
) -> Dataset:
    """Build a seq2seq dataset split from a CoNLL-U file."""
    # Guard against a missing treebank file so callers get a clear error
    # instead of an unhandled FileNotFoundError deep inside read_conllu.
    if not Path(conllu_path).exists():
        raise FileNotFoundError(
            f"CoNLL-U file not found: {conllu_path}. Run fetch-ud/prepare "
            "first or check LEMMA_LANG."
        )
    # Local RNG isolates noise-injection state from the global `random`
    # module so callers relying on the global stream aren't disturbed.
    rng = random.Random(seed)
    rows = []
    all_raw_words: list[str] = []
    skipped = 0
    for sent in read_conllu(conllu_path, lang=lang):
        words = sent["words"]
        lemmas = sent["lemmas"]
        upos_tags = sent["upos"]

        if not words or len(words) != len(lemmas) or len(words) != len(upos_tags):
            skipped += 1
            continue

        input_text = format_input(words, upos_tags, noise=noise, rng=rng)
        output_text = format_output(lemmas)

        input_ids = encode_bytes(input_text)
        labels = encode_bytes(output_text)

        # Filter over-length sequences: a long sentence can produce byte
        # sequences exceeding ByT5-small's context window, causing OOM or
        # silent truncation at collation. The `length` field is now enforced
        # rather than computed-but-unused.
        if len(input_ids) > MAX_SEQ_LEN or len(labels) > MAX_SEQ_LEN:
            skipped += 1
            continue

        rows.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "input_text": input_text,
                "output_text": output_text,
            }
        )
        all_raw_words.extend(words)

    if skipped:
        print(f"  Skipped {skipped} malformed sentences in {conllu_path}")

    if all_raw_words:
        assert_language_plausible(lang, all_raw_words)

    return Dataset.from_list(rows)


def build_cefr_split(
    cefr_path: str,
    lang: str,
    noise: bool = False,
    seed: int = 42,
) -> Dataset:
    """Build a seq2seq dataset split from CEFR contextual sentences."""
    if not Path(cefr_path).exists():
        # CEFR sentences are high-value vocab items; missing them silently
        # degrades training quality. Write to stderr (not stdout) so the
        # warning stays visible in CI logs that separate the two streams.
        print(
            f"  WARNING: CEFR sentences file not found at {cefr_path}; "
            "training will proceed without CEFR data.",
            file=sys.stderr,
            flush=True,
        )
        return Dataset.from_list([])
    return build_split(cefr_path, lang, noise=noise, seed=seed)


def main():
    lang = os.getenv("LEMMA_LANG", "de")
    gold_dir = Path(f"data/gold/{lang}")
    out_dir = Path(os.getenv("DATASET_PATH", f"data/processed/{lang}_seq2seq_lemma"))

    train_path = str(gold_dir / "train.conllu")
    dev_path = str(gold_dir / "dev.conllu")
    test_path = str(gold_dir / "test.conllu")
    cefr_path = str(gold_dir / "cefr_sentences.conllu")

    print(f"Building seq2seq dataset for {lang}...")

    # Train with noise injection for exposure bias mitigation
    train_ds = build_split(train_path, lang, noise=True, seed=42)
    print(f"  train (treebank): {len(train_ds)} sentences")

    # CEFR contextual sentences are gold identity pairs (lemma == word);
    # noise is off so the model learns clean identity mappings for these
    # high-value vocab items. Treebank rows above carry the 10% noise.
    cefr_ds = build_cefr_split(cefr_path, lang, noise=False, seed=43)
    if len(cefr_ds) > 0:
        train_ds = concatenate_datasets([train_ds, cefr_ds])
        print(f"  train (+CEFR): {len(train_ds)} sentences")

    dev_ds = build_split(dev_path, lang, noise=False, seed=44)
    test_ds = build_split(test_path, lang, noise=False, seed=45)
    print(f"  dev: {len(dev_ds)} sentences")
    print(f"  test: {len(test_ds)} sentences")

    dataset = DatasetDict({"train": train_ds, "validation": dev_ds, "test": test_ds})

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out_dir))
    print(f"Saved to {out_dir}")

    # Show sample
    if len(train_ds) > 0:
        sample = train_ds[0]
        print(f"\n  Sample input:  {sample['input_text'][:100]}...")
        print(f"  Sample output: {sample['output_text'][:100]}...")


if __name__ == "__main__":
    main()
