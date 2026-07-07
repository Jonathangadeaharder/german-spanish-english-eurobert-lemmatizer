"""Treebank evaluation for the EuroBERT lemmatizer.

Uses the shared EvalContext (label space, tokenizer, model backend, lexicon)
so there is a single source of truth across treebank and CEFR evaluation.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import numpy as np
from datasets import load_from_disk

from lemmatizer.data.edit_trees import apply_edit_label
from lemmatizer.eval.context import build_eval_context

MAX_LENGTH = 256


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def filter_test_rows_for_lang(rows, lang: str):
    """Filter a HuggingFace Dataset to rows matching ``lang``.

    Returns the original dataset unchanged when it has no ``lang`` column
    (single-language eval), so callers can pass mixed and single-language
    datasets uniformly.
    """
    if "lang" not in rows.column_names:
        return rows
    return rows.filter(lambda row: row["lang"] == lang)


def strip_lang_prefix(label: str, lang: str = "") -> str:
    """Strip a ``lang::`` prefix from a label."""
    if "::" in label:
        return label.split("::", 1)[1]
    return label


def ensure_upos(words: list[str], upos: list[str] | None) -> list[str]:
    if not upos:
        return ["_"] * len(words)
    if len(upos) < len(words):
        upos = list(upos) + ["_"] * (len(words) - len(upos))
    return upos[: len(words)]


def select_valid_label_id(
    logits_row: np.ndarray,
    candidate_ids,
    id2label: dict[str, str],
    lang: str,
    word: str,
    top_k: int = 12,
) -> str:
    """Pick highest-logit candidate label whose edit applies to ``word``.

    Iterates candidate labels in descending logit order and returns the first
    (lang-prefix-stripped) label that ``apply_edit_label`` accepts. IDENTITY
    always applies, so it is the guaranteed terminal fallback.
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
    upos_logits_row: np.ndarray,
    lemma_logits_row: np.ndarray,
    word_ids,
    candidate_ids,
    id2label: dict[str, str],
    upos_id2label: dict[str, str],
):
    """Per-word raw + constrained label selection and UPOS prediction."""
    raw_labels_by_word: dict[int, str] = {}
    constrained_ids_by_word: dict[int, int] = {}
    lemma_logits_by_word: dict[int, np.ndarray] = {}
    upos_by_word: dict[int, str] = {}
    seen_word_ids: set[int] = set()

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_word_ids:
            continue
        seen_word_ids.add(word_id)

        raw_label_id = int(np.argmax(lemma_logits_row[token_idx]))
        raw_labels_by_word[word_id] = id2label.get(str(raw_label_id), "UNKNOWN")
        constrained_ids_by_word[word_id] = _select_best_label_id(
            lemma_logits_row[token_idx], candidate_ids
        )
        lemma_logits_by_word[word_id] = lemma_logits_row[token_idx]
        upos_label_id = int(np.argmax(upos_logits_row[token_idx]))
        upos_by_word[word_id] = upos_id2label.get(str(upos_label_id), "X")

    return (
        raw_labels_by_word,
        constrained_ids_by_word,
        lemma_logits_by_word,
        upos_by_word,
    )


def _select_best_label_id(logits_row: np.ndarray, candidate_ids) -> int:
    candidate_logits = logits_row[candidate_ids]
    best_offset = int(np.argmax(candidate_logits))
    return int(candidate_ids[best_offset])


def resolve_prediction(
    word: str,
    upos: str,
    base_label: str | None,
    lexicon: dict,
    lang: str = "",
):
    """Resolve a predicted label to a final lemma.

    Order: PROPN skip → edit-tree → lexicon → identity.
    Returns (lemma, source, edit_failed).
    """
    if upos == "PROPN":
        return None, "propn", False

    edit_failed = False
    if base_label is not None:
        applied = apply_edit_label(word, base_label)
        if applied is not None:
            return applied, "edit", False
        edit_failed = True

    lexicon_entry = lexicon.get(word)
    if lexicon_entry is not None:
        if isinstance(lexicon_entry, dict):
            lexicon_lemma = lexicon_entry.get(upos, lexicon_entry.get(next(iter(lexicon_entry))))
        else:
            lexicon_lemma = lexicon_entry
        return lexicon_lemma, "lexicon", edit_failed

    return word, "identity", edit_failed


def main() -> None:
    ctx = build_eval_context()
    lang = ctx.lang

    dataset = load_from_disk(str(ctx.assets.dataset_path))
    rows = filter_test_rows_for_lang(dataset["test"], lang)
    rows = list(rows)

    eval_limit = env_int("EVAL_LIMIT", 0)
    if eval_limit > 0:
        rows = rows[: min(eval_limit, len(rows))]

    batch_size = max(1, env_int("EVAL_BATCH_SIZE", 8))
    valid_mask = os.getenv("EVAL_VALID_MASK", "1").lower() in {"1", "true", "yes"}
    first_word_id = ctx.first_word_offset()

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

    candidate_ids = ctx.candidate_ids
    id2label = ctx.id2label
    upos_id2label = ctx.upos_id2label
    lexicon = ctx.lexicon

    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        batch_words = [row["words"] for row in batch_rows]

        encoded = ctx.encode(batch_words)
        upos_logits, lemma_logits = ctx.backend.run(encoded)

        for batch_index, row in enumerate(batch_rows):
            words = row["words"]
            lemmas = row["lemmas"]
            gold_upos = ensure_upos(words, row.get("upos"))
            word_ids = encoded.word_ids(batch_index=batch_index)

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

            for word_offset, (word, gold_lemma) in enumerate(zip(words, lemmas, strict=True)):
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

                if upos_tag == "PROPN":
                    continue

                stats["lemma_total"] += 1
                if word not in lexicon:
                    stats["oov_total"] += 1
                else:
                    stats["in_vocab_total"] += 1

                if predicted_upos == "PROPN":
                    predicted_lemma, source, failed_apply = resolve_prediction(
                        word, predicted_upos, None, lexicon, lang=lang
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
                        word, predicted_upos, base_label, lexicon, lang=lang
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

    ctx.backend.close()

    upos_report: dict[str, dict] = {}
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
            "backend": ctx.backend.__class__.__name__,
        },
        "summary": summary,
    }

    report_path = ctx.assets.artifacts_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
