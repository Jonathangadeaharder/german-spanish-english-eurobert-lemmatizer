"""Build BIO-POS training data from UD Chinese CoNLL-U.

Chinese is isolating: lemma = surface form. The task is word segmentation
+ POS tagging. We formulate it as per-character BIO-POS classification:
  B-NOUN, I-NOUN, B-VERB, I-VERB, etc.

Each character in the sentence gets a label:
  B-X = first character of a word with UPOS X
  I-X = continuation character of a word with UPOS X

Lemma = surface form (trivial copy in post-processing).
"""

from __future__ import annotations

import json
from pathlib import Path

from lemmatizer.data.conllu import read_conllu

ZH_TRAIN = Path("data/gold/zh/train.conllu")
ZH_DEV = Path("data/gold/zh/dev.conllu")
ZH_TEST = Path("data/gold/zh/test.conllu")
OUTPUT_DIR = Path("data/processed/zh_bio")
MAX_LENGTH = 256

# 17 UPOS tags × 2 (B/I) + O = 35 labels
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
LABELS = ["O"] + [f"B-{t}" for t in UPOS_TAGS] + [f"I-{t}" for t in UPOS_TAGS]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def build_pairs(conllu_path: Path) -> list[dict]:
    pairs = []
    for sent in read_conllu(str(conllu_path), "zh"):
        words = sent["words"]
        upos_tags = sent["upos"]
        # Flatten words into characters with BIO labels
        chars = []
        labels = []
        for word, upos in zip(words, upos_tags, strict=True):
            upos = upos if upos in UPOS_TAGS else "X"
            for i, char in enumerate(word):
                chars.append(char)
                labels.append(f"B-{upos}" if i == 0 else f"I-{upos}")

        if len(chars) > MAX_LENGTH - 2:  # reserve [CLS] and [SEP]
            chars = chars[: MAX_LENGTH - 2]
            labels = labels[: MAX_LENGTH - 2]

        label_ids = [LABEL2ID[lbl] for lbl in labels]
        pairs.append(
            {
                "chars": chars,
                "labels": label_ids,
                "length": len(chars),
            }
        )
    return pairs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Save label mapping
    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump({"labels": LABELS, "label2id": LABEL2ID}, f, ensure_ascii=False, indent=2)
    print(f"Labels: {len(LABELS)} ({LABELS[:5]}...{LABELS[-5:]})")

    for split, path in [("train", ZH_TRAIN), ("validation", ZH_DEV), ("test", ZH_TEST)]:
        pairs = build_pairs(path)
        out = OUTPUT_DIR / f"{split}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"{split}: {len(pairs)} sentences -> {out}")
        if pairs:
            chars = pairs[0]["chars"][:10]
            labels = [LABELS[lbl] for lbl in pairs[0]["labels"][:10]]
            print(f"  sample: {''.join(chars)}")
            print(f"  labels: {labels}")


if __name__ == "__main__":
    main()
