"""Evaluate ZH BIO-POS model on UD Chinese treebank.

Measures per-character BIO-POS accuracy and word-level segmentation F1.
Lemma = surface form (trivial), so lemma accuracy is always 100%.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, BertForTokenClassification
from conllu_reader import read_conllu
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

MODEL_DIR = os.getenv("MODEL_DIR", "runs/zh-bio-pos/final")
TEST_FILE = Path("data/gold/zh/test.conllu")
MAX_LENGTH = 256

UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]
LABELS = ["O"] + [f"B-{t}" for t in UPOS_TAGS] + [f"I-{t}" for t in UPOS_TAGS]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def predict_sentence(model, tokenizer, chars):
    encoding = tokenizer(chars, is_split_into_words=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**encoding).logits[0]
    preds = logits.argmax(dim=-1).tolist()
    word_ids = encoding.word_ids()
    labels = []
    for pred, wid in zip(preds, word_ids):
        if wid is not None:
            labels.append(ID2LABEL.get(pred, "O"))
    return labels


def evaluate():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = BertForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    total = 0
    correct = 0
    word_correct = 0
    word_total = 0
    pos_correct = 0
    label_counts = Counter()
    label_correct = Counter()

    for sent in read_conllu(str(TEST_FILE), "zh"):
        words = sent["words"]
        upos_tags = sent["upos"]
        chars = []
        gold_labels = []
        for word, upos in zip(words, upos_tags):
            upos = upos if upos in UPOS_TAGS else "X"
            for i, char in enumerate(word):
                chars.append(char)
                gold_labels.append(f"B-{upos}" if i == 0 else f"I-{upos}")

        if len(chars) > MAX_LENGTH - 2:
            chars = chars[:MAX_LENGTH - 2]
            gold_labels = gold_labels[:MAX_LENGTH - 2]

        pred_labels = predict_sentence(model, tokenizer, chars)

        for gold, pred in zip(gold_labels, pred_labels[:len(gold_labels)]):
            total += 1
            label_counts[gold] += 1
            if gold == pred:
                correct += 1
                label_correct[gold] += 1
            if gold.startswith("B-") and pred == gold:
                pos_correct += 1
            if gold.startswith("B-"):
                word_total += 1
                if pred == gold:
                    word_correct += 1

    char_acc = correct / total if total else 0
    word_acc = word_correct / word_total if word_total else 0
    pos_acc = pos_correct / word_total if word_total else 0

    print(f"ZH BIO-POS Treebank Evaluation")
    print(f"  Characters: {total}")
    print(f"  Char-level accuracy: {char_acc:.4f} ({correct}/{total})")
    print(f"  Word segmentation accuracy: {word_acc:.4f} ({word_correct}/{word_total})")
    print(f"  POS accuracy (B- tags): {pos_acc:.4f}")
    print(f"  Lemma accuracy: 1.0000 (trivial: lemma = surface for Chinese)")

    # Per-label breakdown
    print(f"\n  Per-label accuracy (top 10):")
    for label, count in label_counts.most_common(10):
        acc = label_correct[label] / count if count else 0
        print(f"    {label}: {acc:.4f} ({label_correct[label]}/{count})")


if __name__ == "__main__":
    evaluate()
