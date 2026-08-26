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

from transformers import AutoTokenizer

from lemmatizer.data.conllu import read_conllu
from lemmatizer.eval.zh_char_utils import (
    MAX_LENGTH,
    _predict_sentence_chars,
    label_to_upos,
)

GOLD_TEST = "data/gold/zh/test.conllu"
BERT_PATH = "models/bert-base-chinese-mlx"
ZH_BIO_CHECKPOINT = "runs/mlx-zh-bio-pos/best.safetensors"
EXCEPTIONS_PATH = "artifacts/lemma_zh/exceptions.json"
BIO_LABELS_PATH = "data/processed/zh_bio/labels.json"

# UPOS tags skipped from lemma scoring (lemma == surface form).
IDENTITY_UPOS = {"PROPN", "PUNCT", "SYM", "X", "NUM"}


def load_label_maps() -> tuple[dict[str, int], dict[int, str]]:
    meta = json.loads(Path(BIO_LABELS_PATH).read_text(encoding="utf-8"))
    label2id = meta["label2id"]
    id2label = {int(v): k for k, v in label2id.items()}
    return label2id, id2label


def _maybe_sample(samples, sent_idx, i, word, gold_lemma, pred_lemma, gold_pos, pred_pos):
    if sent_idx < 3 and i < 5:
        samples.append({
            "word": word, "gold_lemma": gold_lemma,
            "pred_lemma": pred_lemma, "gold_upos": gold_pos,
            "pred_upos": pred_pos,
        })


def _score_sentence(
    gold_words: list[str],
    gold_lemmas: list[str],
    gold_upos: list[str],
    word_start_offsets: list[int],
    n_chars: int,
    char_label: list[int | None],
    id2label: dict[int, str],
    exceptions: dict,
    sent_idx: int,
    counters: dict,
    samples: list[dict],
) -> None:
    """Score lemma and UPOS predictions for one sentence."""
    for i, (word, gold_lemma, gold_pos) in enumerate(
        zip(gold_words, gold_lemmas, gold_upos, strict=True)
    ):
        offset = word_start_offsets[i] if i < len(word_start_offsets) else 0

        # Skip words beyond the truncation boundary — they can't be
        # scored because the model never saw their chars.
        if offset >= n_chars:
            continue

        counters["total_tokens"] += 1

        # UPOS: use the first char of this word.
        if char_label[offset] is not None:
            raw_label = id2label.get(char_label[offset], "O")
            pred_pos = label_to_upos(raw_label)
        else:
            pred_pos = "X"

        counters["upos_total"] += 1
        if pred_pos == gold_pos:
            counters["upos_correct"] += 1

        # Lemma: identity + exceptions.
        pred_lemma = exceptions.get(word, word)
        if gold_pos not in IDENTITY_UPOS:
            counters["lemma_total"] += 1
            if pred_lemma == gold_lemma:
                counters["lemma_correct"] += 1

        _maybe_sample(samples, sent_idx, i, word, gold_lemma, pred_lemma, gold_pos, pred_pos)


def _process_sentence(
    sent_idx: int,
    sent: dict,
    tokenizer,
    model,
    id2label: dict[int, str],
    exceptions: dict,
    counters: dict,
    samples: list[dict],
) -> None:
    """Tokenize, predict, and score one sentence."""
    gold_words = sent["words"]
    gold_lemmas = sent["lemmas"]
    gold_upos = sent["upos"]

    char_label, word_start_offsets = _predict_sentence_chars(
        tokenizer, model, gold_words, sent_idx
    )
    n_chars = min(sum(len(w) for w in gold_words), MAX_LENGTH - 2)

    _score_sentence(
        gold_words, gold_lemmas, gold_upos, word_start_offsets,
        n_chars, char_label, id2label, exceptions, sent_idx,
        counters, samples,
    )


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

    counters: dict[str, int] = {
        "lemma_total": 0,
        "lemma_correct": 0,
        "upos_total": 0,
        "upos_correct": 0,
        "total_tokens": 0,
    }
    samples: list[dict] = []

    for sent_idx, sent in enumerate(sentences):
        _process_sentence(
            sent_idx, sent, tokenizer, model, id2label,
            exceptions, counters, samples,
        )

    print(f"\n{'=' * 60}")
    print("Chinese Manual Evaluation (gold test)")
    print(f"{'=' * 60}")
    print(f"Total tokens: {counters['total_tokens']}")
    print(f"Lemma scored: {counters['lemma_total']}")
    print(
        f"Lemma accuracy: {counters['lemma_correct']}/{counters['lemma_total']} = "
        f"{counters['lemma_correct'] / max(counters['lemma_total'], 1):.4f}"
    )
    print(
        f"UPOS accuracy: {counters['upos_correct']}/{counters['upos_total']} = "
        f"{counters['upos_correct'] / max(counters['upos_total'], 1):.4f}"
    )
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
