"""Evaluate ONNX models on handcrafted test sentences.

Loads handcrafted CoNLL-U files from data/handcraft/{lang}_test.conllu,
runs the ONNX model for each language, and reports lemma + UPOS accuracy.

Usage:
    uv run python -m lemmatizer.eval.handcraft_eval
    uv run python -m lemmatizer.eval.handcraft_eval --lang de
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from lemmatizer.data.conllu import read_conllu
from lemmatizer.data.edit_trees import apply_edit_label
from lemmatizer.eval.zh_char_utils import (
    MAX_LENGTH,
    build_char_layout,
    decode_char_labels,
    label_to_upos,
)

EUROBERT_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--EuroBERT--EuroBERT-210m/"
    "snapshots/39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e"
)
SCANDIBERT_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--vesteinn--ScandiBERT/"
    "snapshots/e8339695d4bc4e61f1050b4c71853bed348a18b3"
)
BERT_ZH_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--bert-base-chinese/"
    "snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea"
)

LANG_CONFIGS = {
    "de": {
        "onnx": "onnx/eurobert-lemma-de-210m/model.int8.onnx",
        "tokenizer": "artifacts/lemma_de/tokenizer",
        "type": "multitask",
    },
    "en": {
        "onnx": "onnx/eurobert-lemma-en-210m/model.int8.onnx",
        "tokenizer": "artifacts/lemma_en/tokenizer",
        "type": "multitask",
    },
    "es": {
        "onnx": "onnx/eurobert-lemma-es-210m/model.int8.onnx",
        "tokenizer": "artifacts/lemma_es/tokenizer",
        "type": "multitask",
    },
    "fr": {
        "onnx": "onnx/eurobert-lemma-fr-210m/model.int8.onnx",
        "tokenizer": "artifacts/lemma_fr/tokenizer",
        "type": "multitask",
    },
    "nl": {
        "onnx": "onnx/eurobert-lemma-nl-210m/model.int8.onnx",
        "tokenizer": "artifacts/lemma_nl/tokenizer",
        "type": "multitask",
    },
    "sv": {
        "onnx": "onnx/eurobert-lemma-sv-210m/model.int8.onnx",
        "tokenizer": SCANDIBERT_PATH,
        "type": "multitask",
    },
    "zh": {
        "onnx": "onnx/eurobert-lemma-zh-210m/model.int8.onnx",
        "tokenizer": BERT_ZH_PATH,
        "type": "zh_bio",
    },
}

IDENTITY_UPOS = {"PROPN", "PUNCT", "SYM", "X", "NUM"}


def load_label_maps(lang: str) -> tuple[dict, dict, dict, dict]:
    artifacts = Path(f"artifacts/lemma_{lang}")
    label2id = json.loads((artifacts / "label2id.json").read_text("utf-8"))
    id2label = {int(v): k for k, v in label2id.items()}
    upos_label2id = json.loads((artifacts / "upos_label2id.json").read_text("utf-8"))
    upos_id2label = {int(v): k for k, v in upos_label2id.items()}
    lexicon = {}
    lex_path = artifacts / "lexicon.json"
    if lex_path.exists():
        lexicon = json.loads(lex_path.read_text("utf-8"))
    return label2id, id2label, upos_id2label, lexicon


# --- eval_multitask helpers ---


def _find_token_index(word_ids: list, word_id: int) -> int | None:
    """Return first token index whose word_id matches, or None."""
    prev_wid = None
    for ti, wid in enumerate(word_ids):
        if wid is not None and wid == word_id and wid != prev_wid:
            return ti
        prev_wid = wid
    return None


def _encode_multitask_sentence(tokenizer, words: list, lang: str):
    """Tokenize words with optional lang-token prefix.

    Return (encoding, input_ids, attention_mask, prepend).
    """
    lang_token = f"[LANG_{lang.upper()}]"
    vocab = tokenizer.get_vocab()
    prepend = lang_token in vocab
    input_words = ([lang_token] + words) if prepend else words
    encoding = tokenizer(
        input_words,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="np",
    )
    input_ids = np.array(encoding["input_ids"], dtype=np.int64)
    attention_mask = np.array(encoding["attention_mask"], dtype=np.int64)
    return encoding, input_ids, attention_mask, prepend


def _predict_upos(upos_logits: np.ndarray, token_idx: int | None, upos_id2label: dict) -> str:
    """Predict UPOS from logits at token_idx, or 'X' if invalid."""
    if token_idx is not None and token_idx < upos_logits.shape[1]:
        pred_upos_id = int(np.argmax(upos_logits[0][token_idx]))
        return upos_id2label.get(pred_upos_id, "X")
    return "X"


def _apply_edit_tree_prediction(
    word: str,
    token_idx: int | None,
    lemma_logits: np.ndarray,
    candidate_ids: np.ndarray,
    id2label: dict,
) -> str | None:
    """Try edit-tree labels to predict a lemma. Return applied lemma or None."""
    if token_idx is None or token_idx >= lemma_logits.shape[1]:
        return None

    row = lemma_logits[0][token_idx]
    valid = row[candidate_ids]
    order = np.argsort(valid)[::-1][:12]
    for offset in order:
        label_id = int(candidate_ids[offset])
        label = id2label.get(label_id, "UNKNOWN")
        if label == "UNKNOWN":
            continue
        base_label = label.split("::", 1)[-1] if "::" in label else label
        applied = apply_edit_label(word, base_label)
        if applied is not None:
            return applied
    return None


def _resolve_lemma_multitask(
    word: str,
    token_idx: int | None,
    lemma_logits: np.ndarray,
    candidate_ids: np.ndarray,
    id2label: dict,
    lexicon: dict,
    gold_pos: str,
) -> str:
    """Predict lemma via edit trees with lexicon fallback."""
    edit_result = _apply_edit_tree_prediction(
        word, token_idx, lemma_logits, candidate_ids, id2label
    )
    if edit_result is not None and edit_result != word:
        return edit_result

    # Edit tree gave identity or nothing → lexicon fallback
    if word not in lexicon:
        return edit_result if edit_result is not None else word
    entry = lexicon[word]
    if isinstance(entry, dict):
        if token_idx is not None and token_idx < lemma_logits.shape[1]:
            return entry.get(gold_pos, next(iter(entry.values())))
        return entry.get(gold_pos, word)
    return entry


def _process_multitask_word(
    word: str,
    gold_lemma: str,
    gold_pos: str,
    word_id: int,
    word_ids: list,
    upos_logits: np.ndarray,
    lemma_logits: np.ndarray,
    upos_id2label: dict,
    id2label: dict,
    candidate_ids: np.ndarray,
    lexicon: dict,
) -> tuple[int, int, int, str, str]:
    """Process one word.

    Return (upos_correct, lemma_total, lemma_correct, pred_lemma, pred_upos).
    """
    token_idx = _find_token_index(word_ids, word_id)
    pred_upos = _predict_upos(upos_logits, token_idx, upos_id2label)
    upos_correct = 1 if pred_upos == gold_pos else 0

    if gold_pos in IDENTITY_UPOS:
        return upos_correct, 0, 0, word, pred_upos

    pred_lemma = _resolve_lemma_multitask(
        word, token_idx, lemma_logits, candidate_ids, id2label, lexicon, gold_pos
    )
    lemma_correct = 1 if pred_lemma == gold_lemma else 0
    return upos_correct, 1, lemma_correct, pred_lemma, pred_upos


def _make_sample(
    word: str, gold_lemma: str, pred_lemma: str, gold_pos: str, pred_upos: str
) -> dict:
    return {
        "word": word,
        "gold_lemma": gold_lemma,
        "pred_lemma": pred_lemma,
        "gold_upos": gold_pos,
        "pred_upos": pred_upos,
    }


def eval_multitask(lang: str, sentences: list) -> dict:
    cfg = LANG_CONFIGS[lang]
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"], trust_remote_code=True)
    sess = ort.InferenceSession(cfg["onnx"], providers=["CPUExecutionProvider"])

    label2id, id2label, upos_id2label, lexicon = load_label_maps(lang)

    # Build candidate label IDs for this language
    lang_prefixes = ("de::", "es::", "en::", "fr::", "nl::", "sv::", "ar::", "zh::")
    candidate_ids = np.array(
        [
            int(v)
            for k, v in label2id.items()
            if k.startswith(f"{lang}::") or not k.startswith(lang_prefixes)
        ],
        dtype=np.int64,
    )
    if len(candidate_ids) == 0:
        candidate_ids = np.array(list(id2label.keys()), dtype=np.int64)

    total = upos_total = upos_correct = 0
    lemma_total = lemma_correct = 0
    samples = []

    for sent_idx, sent in enumerate(sentences):
        words = sent["words"]
        gold_lemmas = sent["lemmas"]
        gold_upos = sent["upos"]

        encoding, input_ids, attention_mask, prepend = _encode_multitask_sentence(
            tokenizer, words, lang
        )
        outputs = sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        upos_logits, lemma_logits = outputs

        word_ids = encoding.word_ids()
        first_word_offset = 1 if prepend else 0

        for i, (word, gold_lemma, gold_pos) in enumerate(
            zip(words, gold_lemmas, gold_upos, strict=True)
        ):
            total += 1
            word_id = first_word_offset + i
            upos_ok, lemma_tot, lemma_ok, pred_lemma, pred_upos = _process_multitask_word(
                word,
                gold_lemma,
                gold_pos,
                word_id,
                word_ids,
                upos_logits,
                lemma_logits,
                upos_id2label,
                id2label,
                candidate_ids,
                lexicon,
            )
            upos_total += 1
            upos_correct += upos_ok
            lemma_total += lemma_tot
            lemma_correct += lemma_ok

            if sent_idx < 3 and i < 5:
                samples.append(_make_sample(word, gold_lemma, pred_lemma, gold_pos, pred_upos))

    return {
        "lang": lang,
        "total_tokens": total,
        "upos_accuracy": round(upos_correct / max(upos_total, 1), 4),
        "upos_correct": upos_correct,
        "upos_total": upos_total,
        "lemma_accuracy": round(lemma_correct / max(lemma_total, 1), 4),
        "lemma_correct": lemma_correct,
        "lemma_total": lemma_total,
        "samples": samples,
    }


# --- eval_zh helpers ---


def _predict_zh_upos(char_label: list, offset: int, id2label: dict) -> str:
    """Predict UPOS from character-level label, or 'X' if unknown."""
    if char_label[offset] is not None:
        raw_label = id2label.get(char_label[offset], "O")
        return label_to_upos(raw_label)
    return "X"


def _process_zh_word(
    word: str,
    gold_lemma: str,
    gold_pos: str,
    offset: int,
    char_label: list,
    id2label: dict,
    exceptions: dict,
) -> tuple[int, int, int, str, str]:
    """Process one zh word.

    Return (upos_correct, lemma_total, lemma_correct, pred_lemma, pred_pos).
    """
    pred_pos = _predict_zh_upos(char_label, offset, id2label)
    upos_correct = 1 if pred_pos == gold_pos else 0

    pred_lemma = exceptions.get(word, word)
    if gold_pos in IDENTITY_UPOS:
        return upos_correct, 0, 0, pred_lemma, pred_pos

    lemma_correct = 1 if pred_lemma == gold_lemma else 0
    return upos_correct, 1, lemma_correct, pred_lemma, pred_pos


def eval_zh(sentences: list) -> dict:
    cfg = LANG_CONFIGS["zh"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    sess = ort.InferenceSession(cfg["onnx"], providers=["CPUExecutionProvider"])

    # Load zh exception lexicon
    exceptions = json.loads(Path("artifacts/lemma_zh/exceptions.json").read_text("utf-8"))
    bio_labels = json.loads(Path("data/processed/zh_bio/labels.json").read_text("utf-8"))
    id2label = {int(v): k for k, v in bio_labels["label2id"].items()}

    total = upos_total = upos_correct = 0
    lemma_total = lemma_correct = 0
    samples = []

    for sent_idx, sent in enumerate(sentences):
        words = sent["words"]
        gold_lemmas = sent["lemmas"]
        gold_upos = sent["upos"]

        chars, word_offsets = build_char_layout(words)
        n_chars = min(len(chars), MAX_LENGTH - 2)
        encoding = tokenizer(
            chars[:n_chars],
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        input_ids = np.array([encoding["input_ids"]], dtype=np.int64)
        attention_mask = np.array([encoding["attention_mask"]], dtype=np.int64)
        outputs = sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        preds = np.argmax(outputs[0], axis=-1)[0]

        char_label = decode_char_labels(encoding, preds, n_chars)

        for i, (word, gold_lemma, gold_pos) in enumerate(
            zip(words, gold_lemmas, gold_upos, strict=True)
        ):
            offset = word_offsets[i] if i < len(word_offsets) else 0
            if offset >= n_chars:
                continue
            total += 1

            upos_ok, lemma_tot, lemma_ok, pred_lemma, pred_pos = _process_zh_word(
                word, gold_lemma, gold_pos, offset, char_label, id2label, exceptions
            )
            upos_total += 1
            upos_correct += upos_ok
            lemma_total += lemma_tot
            lemma_correct += lemma_ok

            if sent_idx < 3 and i < 5:
                samples.append(_make_sample(word, gold_lemma, pred_lemma, gold_pos, pred_pos))

    return {
        "lang": "zh",
        "total_tokens": total,
        "upos_accuracy": round(upos_correct / max(upos_total, 1), 4),
        "upos_correct": upos_correct,
        "upos_total": upos_total,
        "lemma_accuracy": round(lemma_correct / max(lemma_total, 1), 4),
        "lemma_correct": lemma_correct,
        "lemma_total": lemma_total,
        "samples": samples,
    }


# --- main helpers ---


def _select_langs(lang_arg: str) -> list:
    if lang_arg == "all":
        return list(LANG_CONFIGS.keys())
    return [lang_arg]


def _evaluate_lang(lang: str, sentences: list) -> dict:
    if lang == "zh":
        return eval_zh(sentences)
    return eval_multitask(lang, sentences)


def _print_sample(s: dict) -> None:
    ml = "✓" if s["pred_lemma"] == s["gold_lemma"] else "✗"
    mu = "✓" if s["pred_upos"] == s["gold_upos"] else "✗"
    print(
        f"  {s['word']:12s} lemma: {s['gold_lemma']:12s}→{s['pred_lemma']:12s} {ml}  "
        f"upos: {s['gold_upos']:6s}→{s['pred_upos']:6s} {mu}"
    )


def _print_summary_table(results: list) -> None:
    print(f"\n{'=' * 70}")
    print(f"{'Lang':6s} {'Lemma':>10s} {'UPOS':>10s} {'Tokens':>8s}")
    print("-" * 40)
    for r in results:
        print(
            f"{r['lang']:6s} {r['lemma_accuracy']:10.4f} {r['upos_accuracy']:10.4f} "
            f"{r['total_tokens']:8d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="all")
    args = parser.parse_args()

    results = []
    for lang in _select_langs(args.lang):
        conllu_path = Path(f"data/handcraft/{lang}_test.conllu")
        if not conllu_path.exists():
            print(f"[{lang}] No handcraft file at {conllu_path}, skipping")
            continue

        sentences = read_conllu(str(conllu_path), lang=lang)
        print(f"[{lang}] {len(sentences)} sentences")

        r = _evaluate_lang(lang, sentences)
        results.append(r)
        print(
            f"  lemma: {r['lemma_correct']}/{r['lemma_total']} = {r['lemma_accuracy']:.4f}  "
            f"UPOS: {r['upos_correct']}/{r['upos_total']} = {r['upos_accuracy']:.4f}"
        )
        for s in r["samples"][:5]:
            _print_sample(s)

    _print_summary_table(results)


if __name__ == "__main__":
    main()
