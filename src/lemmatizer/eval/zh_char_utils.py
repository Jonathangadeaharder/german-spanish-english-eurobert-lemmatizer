"""Shared char-level helpers for zh UPOS evaluation and error analysis.

Both ``zh_manual_eval`` and ``zh_error_analysis`` flatten gold words into
a char list, run the bert-base-chinese BIO model, and decode per-char
predicted labels. The logic is identical, so it lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

MAX_LENGTH = 256


def label_to_upos(label: str) -> str:
    """Convert a BIO label string to its UPOS tag."""
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]
    return "X"


def build_char_layout(
    gold_words: list[str],
) -> tuple[list[str], list[int]]:
    """Flatten gold words into a char list, tracking word start offsets."""
    chars: list[str] = []
    word_start_offsets: list[int] = []
    for word in gold_words:
        word_start_offsets.append(len(chars))
        chars.extend(list(word))
    return chars, word_start_offsets


def decode_char_labels(
    encoding, preds: np.ndarray, n_chars: int
) -> list[int | None]:
    """Map tokenizer word_ids to per-char predicted label indices."""
    word_ids = encoding.word_ids()
    char_label: list[int | None] = [None] * n_chars
    prev_wid = None
    for token_idx, wid in enumerate(word_ids):
        if wid is None or wid == prev_wid:
            continue
        prev_wid = wid
        if wid < n_chars and token_idx < len(preds):
            char_label[wid] = int(preds[token_idx])
    return char_label


def predict_sentence_chars(
    tokenizer: PreTrainedTokenizer,
    model,
    gold_words: list[str],
    sent_idx: int,
) -> tuple[list[int | None], list[int]]:
    """Tokenize chars, run model, decode per-char labels.

    Returns (char_label, word_start_offsets). Clears MLX cache every
    50 sentences to bound memory during long evaluation runs.
    """
    chars, word_start_offsets = build_char_layout(gold_words)
    n_chars = min(len(chars), MAX_LENGTH - 2)
    encoding = tokenizer(
        chars[:n_chars],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    input_ids = mx.array(np.array([encoding["input_ids"]]))
    logits = model(input_ids)
    mx.eval(logits)
    preds = np.array(mx.argmax(logits, axis=-1))[0]
    if sent_idx % 50 == 0:
        mx.clear_cache()
    char_label = decode_char_labels(encoding, preds, n_chars)
    return char_label, word_start_offsets
