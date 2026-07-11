"""Independent manual evaluation for Chinese (zh) lemmatizer + UPOS tagger.

Evaluates against data/gold/zh/test.conllu (gold held-out, 12010 tokens).
Does NOT trust training-loop metrics — loads the model from disk and
scores word-level accuracy directly.

Lemma: identity + exception lexicon (exceptions.get(word, word)).
UPOS: bert-base-chinese zh_bio model, char-level BIO → word-level decode.
Uses gold word boundaries (known from CoNLL-U) to extract the B- label at
each word's first char position — avoids BIO-decode alignment drift.

Usage:
    uv run python -m lemmatizer.eval.zh_manual_eval
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from lemmatizer.data.conllu import read_conllu

GOLD_TEST = "data/gold/zh/test.conllu"
BERT_PATH = "models/bert-base-chinese-mlx"
ZH_BIO_CHECKPOINT = "runs/mlx-zh-bio-pos/best.safetensors"
EXCEPTIONS_PATH = "artifacts/lemma_zh/exceptions.json"
BIO_LABELS_PATH = "data/processed/zh_bio/labels.json"
MAX_LENGTH = 256

# UPOS tags skipped from lemma scoring (lemma == surface form).
IDENTITY_UPOS = {"PROPN", "PUNCT", "SYM", "X", "NUM"}


def load_label_maps() -> tuple[dict[str, int], dict[int, str]]:
    meta = json.loads(Path(BIO_LABELS_PATH).read_text(encoding="utf-8"))
    label2id = meta["label2id"]
    id2label = {int(v): k for k, v in label2id.items()}
    return label2id, id2label


def label_to_upos(label: str) -> str:
    """Convert a BIO label string to its UPOS tag."""
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]
    return "X"


def run() -> None:
    exceptions = json.loads(Path(EXCEPTIONS_PATH).read_text(encoding="utf-8"))
    print(f"Loaded {len(exceptions)} exception entries")

    sentences = read_conllu(GOLD_TEST, lang="zh")
    print(f"Gold test: {len(sentences)} sentences")

    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    _, id2label = load_label_maps()
    print(f"Label space: {len(id2label)} labels")

    from lemmatizer.train.zh_bio import _load_bert_model

    model = _load_bert_model(BERT_PATH)
    # Full finetune checkpoint — no LoRA structure to attach.
    model.load_weights(ZH_BIO_CHECKPOINT, strict=False)
    print(f"Model loaded from {ZH_BIO_CHECKPOINT}")
    model.eval()

    lemma_total = 0
    lemma_correct = 0
    upos_total = 0
    upos_correct = 0
    total_tokens = 0
    samples: list[dict] = []

    for sent_idx, sent in enumerate(sentences):
        gold_words = sent["words"]
        gold_lemmas = sent["lemmas"]
        gold_upos = sent["upos"]

        # Build flat char list and track word start offsets.
        chars: list[str] = []
        word_start_offsets: list[int] = []
        for word in gold_words:
            word_start_offsets.append(len(chars))
            chars.extend(list(word))

        # Truncate to model max length.
        n_chars = min(len(chars), MAX_LENGTH - 2)

        # Tokenize char list.
        encoding = tokenizer(
            chars[:n_chars],
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        # Run model.
        input_ids = mx.array(np.array([encoding["input_ids"]]))
        logits = model(input_ids)
        mx.eval(logits)
        preds = np.array(mx.argmax(logits, axis=-1))[0]

        # Map char positions → predicted label via tokenizer word_ids.
        # word_ids[b][i] gives the original char index for token i.
        word_ids = encoding.word_ids()
        char_label: list[int | None] = [None] * n_chars
        prev_wid = None
        for token_idx, wid in enumerate(word_ids):
            if wid is None or wid == prev_wid:
                continue
            prev_wid = wid
            if wid < n_chars and token_idx < len(preds):
                char_label[wid] = int(preds[token_idx])

        # For each gold word, extract UPOS from its first char's B- label.
        for i, (word, gold_lemma, gold_pos) in enumerate(
            zip(gold_words, gold_lemmas, gold_upos, strict=True)
        ):
            total_tokens += 1

            # UPOS: use the first char of this word.
            offset = word_start_offsets[i] if i < len(word_start_offsets) else 0
            if offset < n_chars and char_label[offset] is not None:
                raw_label = id2label.get(char_label[offset], "O")
                pred_pos = label_to_upos(raw_label)
            else:
                pred_pos = "X"

            upos_total += 1
            if pred_pos == gold_pos:
                upos_correct += 1

            # Lemma: identity + exceptions.
            pred_lemma = exceptions.get(word, word)
            if gold_pos not in IDENTITY_UPOS:
                lemma_total += 1
                if pred_lemma == gold_lemma:
                    lemma_correct += 1

            if sent_idx < 3 and i < 5:
                samples.append(
                    {
                        "word": word,
                        "gold_lemma": gold_lemma,
                        "pred_lemma": pred_lemma,
                        "gold_upos": gold_pos,
                        "pred_upos": pred_pos,
                    }
                )

    print(f"\n{'=' * 60}")
    print("Chinese Manual Evaluation (gold test)")
    print(f"{'=' * 60}")
    print(f"Total tokens: {total_tokens}")
    print(f"Lemma scored: {lemma_total}")
    print(
        f"Lemma accuracy: {lemma_correct}/{lemma_total} = {lemma_correct / max(lemma_total, 1):.4f}"
    )
    print(f"UPOS accuracy: {upos_correct}/{upos_total} = {upos_correct / max(upos_total, 1):.4f}")
    print("\nSample predictions:")
    for s in samples:
        match_l = "✓" if s["pred_lemma"] == s["gold_lemma"] else "✗"
        match_u = "✓" if s["pred_upos"] == s["gold_upos"] else "✗"
        print(
            f"  {s['word']:8s} | gold_lemma={s['gold_lemma']:8s} "
            f"pred={s['pred_lemma']:8s} {match_l} | "
            f"gold_upos={s['gold_upos']:6s} pred={s['pred_upos']:6s} {match_u}"
        )


if __name__ == "__main__":
    run()
