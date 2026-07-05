"""Dataset builder for the ByT5 Arabic lemma classifier.

Produces a token-level, sentence-context dataset:
- Input: full sentence encoded as UTF-8 bytes (ByT5 is tokenizer-free).
- Labels: one lemma class per word.
- word_byte_spans: (N_words, 2) start/end byte indices per word, used by the
  model to mean-pool each word's byte representations.

PROPN tokens are masked to PAD_LABEL (-100) (parity with the EuroBERT path).
Lemma vocabulary is built from the train split, excluding PROPN and "_" lemmas.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, Value

from lemmatizer.data.conllu import read_conllu

PAD_LABEL = -100

# ByT5 byte vocabulary: 256 bytes + PAD(0)... actually ByT5 uses 384-token
# vocab where 0-255 are bytes and 256 is EOS. We use raw byte ids here.
# ByT5 pad_token_id = 0, eos_token_id = 1. Bytes 2-255 are literal bytes.
# We prepend EOS at sentence start and append EOS at the end (ByT5 convention).
BYT5_EOS = 1
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<IDENTITY>"]
SPECIAL_TOKEN_IDS = {"<PAD>": 0, "<UNK>": 1, "<IDENTITY>": 2}


def build_lemma_vocab(
    conllu_paths: list[str],
    min_count: int = 1,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build lemma vocabulary from CoNLL-U files.

    Excludes PROPN tokens and "_" / "-" lemmas. Lemmas are sorted by frequency
    (descending) so common lemmas get low ids. Special tokens are prepended.

    Returns:
        lemma2id: {lemma_str: int}
        id2lemma: {int: lemma_str}  (inverse)
    """
    counter: Counter = Counter()
    for path in conllu_paths:
        for sent in read_conllu(path, lang="ar"):
            for lemma, upos in zip(sent["lemmas"], sent["upos"], strict=True):
                if upos == "PROPN":
                    continue
                if lemma in ("_", "-", ""):
                    continue
                counter[lemma] += 1

    lemma2id: dict[str, int] = {}
    for i, token in enumerate(SPECIAL_TOKENS):
        lemma2id[token] = i

    next_id = len(SPECIAL_TOKENS)
    for lemma, count in counter.most_common():
        if count < min_count:
            continue
        if lemma in lemma2id:
            continue
        lemma2id[lemma] = next_id
        next_id += 1

    id2lemma = {idx: lemma for lemma, idx in lemma2id.items()}
    return lemma2id, id2lemma


def encode_sentence(
    words: list[str],
    lemmas: list[str],
    upos_tags: list[str],
    lemma2id: dict[str, int],
) -> dict:
    """Encode one sentence to byte-level input_ids + per-word spans + labels.

    Words are joined with a single space byte, then the whole sentence is
    UTF-8 encoded. Each word's byte span is recorded for downstream pooling.

    PROPN words are masked to PAD_LABEL. Unknown lemmas get <UNK>.

    Returns a dict with mlx arrays:
        input_ids: (T,) int32 byte ids, EOS-framed.
        word_byte_spans: (N_words, 2) int32 start/end byte indices.
        labels: (N_words,) int32 lemma ids (or PAD_LABEL).
    """
    # Build the byte sequence: EOS + word1 + space + word2 + space + ... + EOS
    # Track each word's byte span as we go.
    byte_ids: list[int] = [BYT5_EOS]
    spans: list[tuple[int, int]] = []
    labels: list[int] = []

    unk_id = lemma2id.get("<UNK>", 1)

    for word, lemma, upos in zip(words, lemmas, upos_tags, strict=True):
        start = len(byte_ids)
        word_bytes = word.encode("utf-8")
        for b in word_bytes:
            # ByT5 byte ids: byte value b in [0,255] maps directly. Bytes 0/1
            # are PAD/EOS in ByT5; we remap literal 0/1 bytes to 2/3 to avoid
            # collision. (ByT5's vocab reserves 256-383 for special tokens.)
            byte_ids.append(b if b > 1 else b + 256)
        end = len(byte_ids)
        spans.append((start, end))

        # Separator space byte between words.
        byte_ids.append(ord(" "))

        if upos == "PROPN" or lemma in ("_", "-"):
            labels.append(PAD_LABEL)
        else:
            labels.append(lemma2id.get(lemma, unk_id))

    byte_ids.append(BYT5_EOS)

    return {
        "input_ids": mx.array(byte_ids, dtype=mx.int32),
        "word_byte_spans": mx.array(spans, dtype=mx.int32),
        "labels": mx.array(labels, dtype=mx.int32),
    }


def _encoded_to_row(encoded: dict) -> dict:
    """Convert mlx arrays to lists for HF Dataset compatibility."""
    return {
        "input_ids": np.array(encoded["input_ids"]).tolist(),
        "word_byte_spans": np.array(encoded["word_byte_spans"]).tolist(),
        "labels": np.array(encoded["labels"]).tolist(),
        "length": len(encoded["input_ids"]),
    }


def build_split(
    conllu_path: str,
    lemma2id: dict[str, int],
) -> Dataset:
    """Build a Dataset for one split (train/dev/test)."""
    rows = []
    for sent in read_conllu(conllu_path, lang="ar"):
        encoded = encode_sentence(
            sent["words"], sent["lemmas"], sent["upos"], lemma2id
        )
        row = _encoded_to_row(encoded)
        row["words"] = sent["words"]
        row["lemmas"] = sent["lemmas"]
        row["upos"] = sent["upos"]
        rows.append(row)

    features = Features(
        {
            "input_ids": Sequence(Value("int64")),
            "word_byte_spans": Sequence(Sequence(Value("int64"))),
            "labels": Sequence(Value("int64")),
            "length": Value("int64"),
            "words": Sequence(Value("string")),
            "lemmas": Sequence(Value("string")),
            "upos": Sequence(Value("string")),
        }
    )
    return Dataset.from_list(rows, features=features)


def main():
    import os

    train_path = os.getenv("TRAIN_CONLLU", "data/gold/ar/train.conllu")
    dev_path = os.getenv("DEV_CONLLU", "data/gold/ar/dev.conllu")
    test_path = os.getenv("TEST_CONLLU", "data/gold/ar/test.conllu")
    out_dir = Path(os.getenv("DATASET_PATH", "data/processed/ar_byt5_lemma"))
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts/lemma_ar"))

    print(f"Building lemma vocab from {train_path} (+{dev_path})...")
    lemma2id, id2lemma = build_lemma_vocab([train_path, dev_path])
    print(f"Lemma vocab: {len(lemma2id)} classes")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "lemma_label2id.json").write_text(
        json.dumps(lemma2id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifacts_dir / "lemma_id2label.json").write_text(
        json.dumps(id2lemma, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved lemma vocab to {artifacts_dir}")

    print(f"Building train split from {train_path}...")
    train_ds = build_split(train_path, lemma2id)
    print(f"  train: {len(train_ds)} sentences")

    print(f"Building dev split from {dev_path}...")
    dev_ds = build_split(dev_path, lemma2id)
    print(f"  dev: {len(dev_ds)} sentences")

    print(f"Building test split from {test_path}...")
    test_ds = build_split(test_path, lemma2id)
    print(f"  test: {len(test_ds)} sentences")

    dataset = DatasetDict(
        {"train": train_ds, "validation": dev_ds, "test": test_ds}
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out_dir))
    print(f"Saved dataset to {out_dir}")
    print(dataset)


if __name__ == "__main__":
    main()
