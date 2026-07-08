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
from pathlib import Path

from datasets import Dataset, DatasetDict

from lemmatizer.data.byt5_dataset import BYT5_EOS, BYTE_ID_OFFSET
from lemmatizer.data.conllu import read_conllu

# All valid UPOS tags for noise injection
UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]

NOISE_RATE = 0.10


def encode_bytes(text: str) -> list[int]:
    """Encode text as ByT5 byte-level token IDs."""
    byte_ids = [BYT5_EOS]
    for b in text.encode("utf-8"):
        byte_ids.append(b + BYTE_ID_OFFSET)
    byte_ids.append(BYT5_EOS)
    return byte_ids


def format_input(words: list[str], upos_tags: list[str], noise: bool = False) -> str:
    """Format a sentence as 'word [UPOS] word [UPOS] ...' for Model 2 input."""
    parts = []
    for word, upos in zip(words, upos_tags, strict=True):
        if noise and random.random() < NOISE_RATE:
            # Corrupt the UPOS tag with a random wrong one
            wrong_choices = [t for t in UPOS_TAGS if t != upos]
            upos = random.choice(wrong_choices)
        parts.append(f"{word} [{upos}]")
    return " ".join(parts)


def format_output(lemmas: list[str]) -> str:
    """Format lemmas as space-separated string for Model 2 output."""
    return " ".join(lemmas)


def build_split(
    conllu_path: str,
    lang: str,
    noise: bool = False,
    seed: int = 42,
) -> Dataset:
    """Build a seq2seq dataset split from a CoNLL-U file."""
    random.seed(seed)
    rows = []
    for sent in read_conllu(conllu_path, lang=lang):
        words = sent["words"]
        lemmas = sent["lemmas"]
        upos_tags = sent["upos"]

        if not words or len(words) != len(lemmas) or len(words) != len(upos_tags):
            continue

        input_text = format_input(words, upos_tags, noise=noise)
        output_text = format_output(lemmas)

        input_ids = encode_bytes(input_text)
        labels = encode_bytes(output_text)

        rows.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "input_text": input_text,
                "output_text": output_text,
                "length": len(input_ids),
            }
        )

    return Dataset.from_list(rows)


def build_cefr_split(
    cefr_path: str,
    lang: str,
    noise: bool = False,
    seed: int = 42,
) -> Dataset:
    """Build a seq2seq dataset split from CEFR contextual sentences."""
    if not Path(cefr_path).exists():
        return Dataset.from_list([])
    return build_split(cefr_path, lang, noise=noise, seed=seed)


def main():
    lang = os.getenv("LEMMA_LANG", "de")
    gold_dir = Path(f"data/gold/{lang}")
    out_dir = Path(
        os.getenv("DATASET_PATH", f"data/processed/{lang}_seq2seq_lemma")
    )

    train_path = str(gold_dir / "train.conllu")
    dev_path = str(gold_dir / "dev.conllu")
    test_path = str(gold_dir / "test.conllu")
    cefr_path = str(gold_dir / "cefr_sentences.conllu")

    print(f"Building seq2seq dataset for {lang}...")

    # Train with noise injection for exposure bias mitigation
    train_ds = build_split(train_path, lang, noise=True, seed=42)
    print(f"  train (treebank): {len(train_ds)} sentences")

    # Add CEFR contextual sentences (no noise — these are gold)
    cefr_ds = build_cefr_split(cefr_path, lang, noise=False, seed=43)
    if len(cefr_ds) > 0:
        from datasets import concatenate_datasets

        train_ds = concatenate_datasets([train_ds, cefr_ds])
        print(f"  train (+CEFR): {len(train_ds)} sentences")

    dev_ds = build_split(dev_path, lang, noise=False, seed=44)
    test_ds = build_split(test_path, lang, noise=False, seed=45)
    print(f"  dev: {len(dev_ds)} sentences")
    print(f"  test: {len(test_ds)} sentences")

    dataset = DatasetDict(
        {"train": train_ds, "validation": dev_ds, "test": test_ds}
    )

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
