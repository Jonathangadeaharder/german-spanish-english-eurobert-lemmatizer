"""Build multitask Seq2Seq training data from UD Arabic CoNLL-U.

Input format:  "Lemmatize and POS: <left_context> <target>WORD</target> <right_context>"
Target format: "UPOS | lemma"

Example:
  Input:  "Lemmatize and POS: قال <target>بالكتاب</target> إنه"
  Target: "ADP | كتاب"
"""
from __future__ import annotations

import json
from pathlib import Path

from conllu_reader import read_conllu

MAX_CONTEXT = 5  # words left/right of target
ARABIC_TRAIN = Path("data/gold/ar/train.conllu")
ARABIC_DEV = Path("data/gold/ar/dev.conllu")
ARABIC_TEST = Path("data/gold/ar/test.conllu")
OUTPUT_DIR = Path("data/processed/ar_seq2seq")


def build_pairs(conllu_path: Path) -> list[dict[str, str]]:
    pairs = []
    for sent in read_conllu(str(conllu_path), "ar"):
        words = sent["words"]
        lemmas = sent["lemmas"]
        upos_tags = sent["upos"]
        for i, (word, lemma, upos) in enumerate(zip(words, lemmas, upos_tags)):
            if upos == "PUNCT" or lemma == "_":
                continue
            left = words[max(0, i - MAX_CONTEXT):i]
            right = words[i + 1:i + 1 + MAX_CONTEXT]
            left_str = " ".join(left) if left else ""
            right_str = " ".join(right) if right else ""
            input_text = f"Lemmatize and POS: {left_str} <target>{word}</target> {right_str}".strip()
            target_text = f"{upos} | {lemma}"
            pairs.append({"input": input_text, "target": target_text})
    return pairs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, path in [("train", ARABIC_TRAIN), ("validation", ARABIC_DEV), ("test", ARABIC_TEST)]:
        pairs = build_pairs(path)
        out = OUTPUT_DIR / f"{split}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"{split}: {len(pairs)} pairs -> {out}")
        if pairs:
            print(f"  sample input:  {pairs[0]['input'][:80]}")
            print(f"  sample target: {pairs[0]['target']}")


if __name__ == "__main__":
    main()
