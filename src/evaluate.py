import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoTokenizer

from edit_trees import apply_edit_label
from language_assets import language_assets
from multitask_model import EuroBertForUposLemma, EuroBertUposLemmaConfig

MAX_LENGTH = 256

LANGS = ["de", "es", "en"]
LANG_TOKENS = {
    "de": "[LANG_DE]",
    "es": "[LANG_ES]",
    "en": "[LANG_EN]",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def env_int(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_candidate_label_ids(id2label, lang=None):
    candidate_ids = {}

    if lang is not None:
        has_prefix = any(
            label.startswith(f"{lang}::")
            for label in id2label.values()
            if label != "UNKNOWN"
        )

        if has_prefix:
            ids = [
                int(label_id)
                for label_id, label in id2label.items()
                if label != "UNKNOWN" and label.startswith(f"{lang}::")
            ]
        else:
            ids = [
                int(label_id)
                for label_id, label in id2label.items()
                if label != "UNKNOWN"
            ]

        candidate_ids[lang] = np.array(sorted(ids), dtype=np.int64)
    else:
        for lang_key in LANGS:
            ids = [
                int(label_id)
                for label_id, label in id2label.items()
                if label != "UNKNOWN" and label.startswith(f"{lang_key}::")
            ]
            candidate_ids[lang_key] = np.array(sorted(ids), dtype=np.int64)

    return candidate_ids


def select_best_label_id(logits_row, candidate_ids):
    candidate_logits = logits_row[candidate_ids]
    best_offset = int(np.argmax(candidate_logits))
    return int(candidate_ids[best_offset])


def select_valid_label_id(logits_row, candidate_ids, id2label, lang, word, top_k=12):
    """Pick highest-logit candidate label whose edit applies to ``word``.

    Iterates candidate labels in descending logit order and returns the first
    (lang-prefix-stripped) label that ``apply_edit_label`` accepts. IDENTITY
    always applies, so it is the guaranteed terminal fallback. Model-internal
    selection only — no hand-written lemma rules.
    """
    candidate_ids = list(candidate_ids)
    candidate_logits = logits_row[candidate_ids]
    order = np.argsort(candidate_logits)[::-1]
    if top_k > 0:
        order = order[:top_k]

    for offset in order:
        label_id = candidate_ids[int(offset)]
        label = id2label.get(str(label_id), "UNKNOWN")
        if label == "UNKNOWN":
            continue
        base_label = strip_lang_prefix(label, lang)
        if apply_edit_label(word, base_label) is not None:
            return base_label

    return "IDENTITY"


def collect_word_predictions(
    upos_logits_row,
    lemma_logits_row,
    word_ids,
    candidate_ids,
    id2label,
    upos_id2label,
):
    raw_labels_by_word = {}
    constrained_ids_by_word = {}
    lemma_logits_by_word = {}
    upos_by_word = {}
    seen_word_ids = set()

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_word_ids:
            continue

        seen_word_ids.add(word_id)

        raw_label_id = int(np.argmax(lemma_logits_row[token_idx]))
        raw_labels_by_word[word_id] = id2label.get(str(raw_label_id), "UNKNOWN")

        constrained_ids_by_word[word_id] = select_best_label_id(
            lemma_logits_row[token_idx], candidate_ids
        )
        lemma_logits_by_word[word_id] = lemma_logits_row[token_idx]
        upos_label_id = int(np.argmax(upos_logits_row[token_idx]))
        upos_by_word[word_id] = upos_id2label.get(str(upos_label_id), "X")

    return raw_labels_by_word, constrained_ids_by_word, lemma_logits_by_word, upos_by_word


def resolve_prediction(word, upos, base_label, lexicon):
    if upos == "PROPN":
        return None, "propn", False

    if base_label is not None:
        applied = apply_edit_label(word, base_label)

        if applied is not None:
            return applied, "edit", False

        edit_failed = True
    else:
        edit_failed = False

    lexicon_entry = lexicon.get(word)

    if lexicon_entry is not None:
        if isinstance(lexicon_entry, dict):
            lexicon_lemma = lexicon_entry.get(upos, lexicon_entry.get(next(iter(lexicon_entry))))
        else:
            lexicon_lemma = lexicon_entry

        return lexicon_lemma, "lexicon", edit_failed

    return word, "identity", edit_failed


def ensure_upos(words, upos):
    if not upos:
        return ["_"] * len(words)

    if len(upos) < len(words):
        upos = list(upos) + ["_"] * (len(words) - len(upos))

    return upos[: len(words)]


def strip_lang_prefix(label, lang):
    prefix = f"{lang}::"
    if label.startswith(prefix):
        return label[len(prefix):]
    return label


def main():
    assets = language_assets()
    lang = assets.lang

    label2id = load_json(str(assets.label2id_path))
    upos_label2id = load_json(str(assets.upos_label2id_path))

    lexicon_path = assets.lexicon_path
    if lexicon_path.exists():
        lexicon = load_json(str(lexicon_path))
    else:
        lexicon = {}

    if not isinstance(lexicon, dict):
        lexicon = {}

    model_dir = os.getenv("MODEL_DIR", str(assets.merged_dir))
    use_lora = os.getenv("EVAL_USE_LORA", "").lower() in {"1", "true", "yes"}

    if not use_lora:
        model_config_path = Path(model_dir) / "config.json"
        if model_config_path.exists():
            model_config = load_json(str(model_config_path))
            label2id = model_config.get("lemma_label2id", label2id)
            upos_label2id = model_config.get("upos_label2id", upos_label2id)

    id2label = {str(v): k for k, v in label2id.items()}
    upos_id2label = {str(v): k for k, v in upos_label2id.items()}

    candidate_ids_by_lang = build_candidate_label_ids(id2label, lang=lang)
    candidate_ids = candidate_ids_by_lang[lang]

    multilingual_tokenizer_dir = os.getenv(
        "MULTILINGUAL_TOKENIZER_DIR", "artifacts/tokenizer"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        multilingual_tokenizer_dir, trust_remote_code=True
    )

    device = get_device()

    onnx_model_path = os.getenv("EVAL_ONNX_MODEL", "")
    ort_session = None
    if onnx_model_path:
        import onnxruntime as ort

        ort_session = ort.InferenceSession(
            onnx_model_path, providers=["CPUExecutionProvider"]
        )
        model = None
    elif use_lora:
        config = EuroBertUposLemmaConfig(
            base_model_name_or_path="EuroBERT/EuroBERT-210m",
            upos_label2id=upos_label2id,
            lemma_label2id=label2id,
        )
        base_model = EuroBertForUposLemma.from_pretrained(
            "EuroBERT/EuroBERT-210m",
            config=config,
            trust_remote_code=True,
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        model = EuroBertForUposLemma.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )

    if model is not None:
        model.to(device)
        model.eval()

    dataset = load_from_disk(str(assets.dataset_path))

    eval_limit = env_int("EVAL_LIMIT", 0)
    batch_size = max(1, env_int("EVAL_BATCH_SIZE", 8))
    valid_mask = os.getenv("EVAL_VALID_MASK", "1").lower() in {"1", "true", "yes"}

    rows = list(dataset["test"])

    if eval_limit > 0:
        rows = rows[: min(eval_limit, len(rows))]

    stats = {
        "total": 0,
        "upos_correct": 0,
        "lemma_total": 0,
        "lemma_correct": 0,
        "unknown": 0,
        "failed_apply": 0,
        "missing_prediction": 0,
        "oov_total": 0,
        "oov_correct": 0,
        "in_vocab_total": 0,
        "in_vocab_correct": 0,
        "edit_tree_total": 0,
        "edit_tree_correct": 0,
        "source_counts": Counter(),
        "pred_upos": Counter(),
        "upos": defaultdict(lambda: {"total": 0, "correct": 0}),
    }

    lang_token = LANG_TOKENS.get(lang)

    with torch.inference_mode():
        for batch_start in range(0, len(rows), batch_size):
            batch_rows = rows[batch_start : batch_start + batch_size]

            prepend_lang = (
                lang_token
                and lang_token in tokenizer.get_vocab()
                and os.getenv("EVAL_PREPEND_LANG", "").lower() in {"1", "true", "yes"}
            )

            if prepend_lang:
                batch_words = [[lang_token, *row["words"]] for row in batch_rows]
            else:
                batch_words = [row["words"] for row in batch_rows]

            encoded = tokenizer(
                batch_words,
                is_split_into_words=True,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            if ort_session is not None:
                ort_inputs = {
                    "input_ids": encoded["input_ids"].cpu().numpy(),
                    "attention_mask": encoded["attention_mask"].cpu().numpy(),
                }
                upos_logits, lemma_logits = ort_session.run(
                    ["upos_logits", "lemma_logits"], ort_inputs
                )
                upos_logits = np.asarray(upos_logits, dtype=np.float32)
                lemma_logits = np.asarray(lemma_logits, dtype=np.float32)
            else:
                model_inputs = {key: value.to(device) for key, value in encoded.items()}
                outputs = model(**model_inputs)
                upos_logits = outputs.logits[0].detach().cpu().numpy()
                lemma_logits = outputs.logits[1].detach().cpu().numpy()

            for batch_index, row in enumerate(batch_rows):
                words = row["words"]
                lemmas = row["lemmas"]
                gold_upos = ensure_upos(words, row.get("upos"))
                word_ids = encoded.word_ids(batch_index=batch_index)

                first_word_id = 1 if prepend_lang else 0

                (
                    raw_labels_by_word,
                    constrained_ids_by_word,
                    lemma_logits_by_word,
                    predicted_upos_by_word,
                ) = collect_word_predictions(
                    upos_logits[batch_index],
                    lemma_logits[batch_index],
                    word_ids,
                    candidate_ids,
                    id2label,
                    upos_id2label,
                )

                for word_offset, (word, gold_lemma) in enumerate(
                    zip(words, lemmas, strict=True)
                ):
                    word_id = first_word_id + word_offset
                    stats["total"] += 1

                    upos_tag = gold_upos[word_offset] if word_offset < len(gold_upos) else "_"
                    upos_bucket = stats["upos"][upos_tag]
                    upos_bucket["total"] += 1

                    predicted_upos = predicted_upos_by_word.get(word_id, "X")
                    stats["pred_upos"][predicted_upos] += 1

                    if predicted_upos == upos_tag:
                        stats["upos_correct"] += 1
                        upos_bucket["correct"] += 1

                    if upos_tag != "PROPN":
                        stats["lemma_total"] += 1
                        if word not in lexicon:
                            stats["oov_total"] += 1
                        else:
                            stats["in_vocab_total"] += 1

                        if predicted_upos == "PROPN":
                            predicted_lemma, source, failed_apply = resolve_prediction(
                                word,
                                predicted_upos,
                                None,
                                lexicon,
                            )
                        else:
                            stats["edit_tree_total"] += 1

                            raw_label = raw_labels_by_word.get(word_id)

                            if raw_label == "UNKNOWN":
                                stats["unknown"] += 1

                            constrained_id = constrained_ids_by_word.get(word_id)

                            if constrained_id is None:
                                stats["missing_prediction"] += 1
                                base_label = None
                            else:
                                constrained_label = id2label.get(str(constrained_id), "UNKNOWN")

                                if constrained_label == "UNKNOWN":
                                    stats["unknown"] += 1

                                base_label = strip_lang_prefix(constrained_label, lang)

                                if base_label == "UNKNOWN":
                                    base_label = None

                                if valid_mask and word_id in lemma_logits_by_word:
                                    base_label = select_valid_label_id(
                                        lemma_logits_by_word[word_id],
                                        candidate_ids,
                                        id2label,
                                        lang,
                                        word,
                                    )

                            predicted_lemma, source, failed_apply = resolve_prediction(
                                word,
                                predicted_upos,
                                base_label,
                                lexicon,
                            )

                        stats["source_counts"][source] += 1

                        if failed_apply:
                            stats["failed_apply"] += 1

                        if predicted_lemma == gold_lemma:
                            stats["lemma_correct"] += 1
                            stats["edit_tree_correct"] += 1

                            if word not in lexicon:
                                stats["oov_correct"] += 1
                            else:
                                stats["in_vocab_correct"] += 1

    report_path = assets.artifacts_dir / "eval_report.json"

    upos_report = {}

    for upos_tag, bucket in sorted(
        stats["upos"].items(),
        key=lambda item: (-item[1]["total"], item[0]),
    ):
        upos_total = bucket["total"] or 1
        upos_report[upos_tag] = {
            "total": bucket["total"],
            "accuracy": round(bucket["correct"] / upos_total, 4),
        }

    total = stats["total"] or 1
    lemma_total = stats["lemma_total"] or 1
    oov_total = stats["oov_total"] or 1
    in_vocab_total = stats["in_vocab_total"] or 1
    edit_tree_total = stats["edit_tree_total"] or 1

    summary = {
        "lang": lang,
        "total": stats["total"],
        "upos_accuracy": round(stats["upos_correct"] / total, 4),
        "lemma_accuracy": round(stats["lemma_correct"] / lemma_total, 4),
        "lemma_total": stats["lemma_total"],
        "unknown_rate": round(stats["unknown"] / lemma_total, 4),
        "failed_apply_rate": round(stats["failed_apply"] / lemma_total, 4),
        "missing_prediction_rate": round(stats["missing_prediction"] / lemma_total, 4),
        "oov_accuracy": round(stats["oov_correct"] / oov_total, 4),
        "in_vocab_accuracy": round(stats["in_vocab_correct"] / in_vocab_total, 4),
        "edit_tree_accuracy": (
            round(stats["edit_tree_correct"] / edit_tree_total, 4)
            if stats["edit_tree_total"] > 0
            else None
        ),
        "source_counts": dict(stats["source_counts"]),
        "upos_by_gold": upos_report,
    }

    report = {
        "config": {
            "eval_limit": eval_limit,
            "batch_size": batch_size,
            "device": str(device),
            "model_dir": model_dir,
        },
        "summary": summary,
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
