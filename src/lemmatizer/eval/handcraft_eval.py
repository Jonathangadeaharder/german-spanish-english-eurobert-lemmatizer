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
MAX_LENGTH = 256


def label_to_upos(label: str) -> str:
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]
    return "X"


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

        # Prepend lang token for multilingual models
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

        outputs = sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        upos_logits, lemma_logits = outputs

        word_ids = encoding.word_ids()
        first_word_offset = 1 if prepend else 0

        for i, (word, gold_lemma, gold_pos) in enumerate(
            zip(words, gold_lemmas, gold_upos, strict=True)
        ):
            total += 1
            word_id = first_word_offset + i
            token_idx = None
            prev_wid = None
            for ti, wid in enumerate(word_ids):
                if wid is not None and wid == word_id and wid != prev_wid:
                    token_idx = ti
                    prev_wid = wid
                    break

            if token_idx is not None and token_idx < upos_logits.shape[1]:
                pred_upos_id = int(np.argmax(upos_logits[0][token_idx]))
                pred_upos = upos_id2label.get(pred_upos_id, "X")
            else:
                pred_upos = "X"

            upos_total += 1
            if pred_upos == gold_pos:
                upos_correct += 1

            if gold_pos not in IDENTITY_UPOS:
                lemma_total += 1
                # Try edit-tree prediction
                if token_idx is not None and token_idx < lemma_logits.shape[1]:
                    row = lemma_logits[0][token_idx]
                    valid = row[candidate_ids]
                    order = np.argsort(valid)[::-1][:12]
                    pred_lemma = word  # fallback identity
                    for offset in order:
                        label_id = int(candidate_ids[offset])
                        label = id2label.get(label_id, "UNKNOWN")
                        if label == "UNKNOWN":
                            continue
                        base_label = label.split("::", 1)[-1] if "::" in label else label
                        from lemmatizer.data.edit_trees import apply_edit_label

                        applied = apply_edit_label(word, base_label)
                        if applied is not None:
                            pred_lemma = applied
                            break
                    # Lexicon fallback
                    if pred_lemma == word and word in lexicon:
                        lex_entry = lexicon[word]
                        if isinstance(lex_entry, dict):
                            pred_lemma = lex_entry.get(gold_pos, next(iter(lex_entry.values())))
                        else:
                            pred_lemma = lex_entry
                else:
                    pred_lemma = word
                    if word in lexicon:
                        entry = lexicon[word]
                        if isinstance(entry, dict):
                            pred_lemma = entry.get(gold_pos, word)
                        else:
                            pred_lemma = entry

                if pred_lemma == gold_lemma:
                    lemma_correct += 1

            if sent_idx < 3 and i < 5:
                samples.append(
                    {
                        "word": word,
                        "gold_lemma": gold_lemma,
                        "pred_lemma": pred_lemma,
                        "gold_upos": gold_pos,
                        "pred_upos": pred_upos,
                    }
                )

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

        chars = []
        word_offsets = []
        for w in words:
            word_offsets.append(len(chars))
            chars.extend(list(w))

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

        word_ids = encoding.word_ids()
        char_label = [None] * n_chars
        prev_wid = None
        for ti, wid in enumerate(word_ids):
            if wid is None or wid == prev_wid:
                continue
            prev_wid = wid
            if wid < n_chars and ti < len(preds):
                char_label[wid] = int(preds[ti])

        for i, (word, gold_lemma, gold_pos) in enumerate(
            zip(words, gold_lemmas, gold_upos, strict=True)
        ):
            offset = word_offsets[i] if i < len(word_offsets) else 0
            if offset >= n_chars:
                continue
            total += 1

            if char_label[offset] is not None:
                raw_label = id2label.get(char_label[offset], "O")
                pred_pos = label_to_upos(raw_label)
            else:
                pred_pos = "X"

            upos_total += 1
            if pred_pos == gold_pos:
                upos_correct += 1

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="all")
    args = parser.parse_args()

    if args.lang == "all":
        langs = list(LANG_CONFIGS.keys())
    else:
        langs = [args.lang]

    results = []
    for lang in langs:
        conllu_path = Path(f"data/handcraft/{lang}_test.conllu")
        if not conllu_path.exists():
            print(f"[{lang}] No handcraft file at {conllu_path}, skipping")
            continue

        sentences = read_conllu(str(conllu_path), lang=lang)
        print(f"[{lang}] {len(sentences)} sentences")

        if lang == "zh":
            r = eval_zh(sentences)
        else:
            r = eval_multitask(lang, sentences)

        results.append(r)
        print(
            f"  lemma: {r['lemma_correct']}/{r['lemma_total']} = {r['lemma_accuracy']:.4f}  "
            f"UPOS: {r['upos_correct']}/{r['upos_total']} = {r['upos_accuracy']:.4f}"
        )
        for s in r["samples"][:5]:
            ml = "✓" if s["pred_lemma"] == s["gold_lemma"] else "✗"
            mu = "✓" if s["pred_upos"] == s["gold_upos"] else "✗"
            print(
                f"  {s['word']:12s} lemma: {s['gold_lemma']:12s}→{s['pred_lemma']:12s} {ml}  "
                f"upos: {s['gold_upos']:6s}→{s['pred_upos']:6s} {mu}"
            )

    print(f"\n{'=' * 70}")
    print(f"{'Lang':6s} {'Lemma':>10s} {'UPOS':>10s} {'Tokens':>8s}")
    print("-" * 40)
    for r in results:
        print(
            f"{r['lang']:6s} {r['lemma_accuracy']:10.4f} {r['upos_accuracy']:10.4f} "
            f"{r['total_tokens']:8d}"
        )


if __name__ == "__main__":
    main()
