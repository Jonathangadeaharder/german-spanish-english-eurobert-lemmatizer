"""CEFR-level evaluation for the EuroBERT lemmatizer.

Refactored to use the shared EvalContext (same as treebank eval) and the
CefrDataSource strategy. This gives CEFR eval the same model backend support
(ONNX, merged, LoRA) and the same prediction logic as treebank eval.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

from lemmatizer.eval.cefr_data import cefr_data_source_from_env
from lemmatizer.eval.context import build_eval_context

MAX_LENGTH = 256


def env_int(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def find_term_index(words: list[str], term: str) -> int | None:
    term_lower = term.lower()
    for idx, word in enumerate(words):
        if word.lower() == term_lower:
            return idx
    return None


def main():
    ctx = build_eval_context()
    lang = ctx.lang

    # Load CEFR data
    source = cefr_data_source_from_env()
    rows = source.load(lang)
    eval_limit = env_int("EVAL_LIMIT", 0)
    if eval_limit > 0:
        rows = rows[: min(eval_limit, len(rows))]

    batch_size = max(1, env_int("EVAL_BATCH_SIZE", 8))

    print(f"CEFR evaluation for {lang}")
    print(f"  Data source: {source.__class__.__name__}")
    print(f"  Rows: {len(rows)}")
    print(f"  Lang token: {ctx.lang_token or 'none'}, prepend={ctx.prepend_lang}")
    print(f"  Backend: {ctx.backend.__class__.__name__}")

    if not rows:
        print("  No CEFR eval data found. Use EVAL_CEFR_SOURCE=treebank to generate.")
        return

    stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    sample_errors: dict[str, list] = defaultdict(list)

    first_word_offset = ctx.first_word_offset()

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        words_batch = []
        metadata = []

        for row in batch:
            words = row.sentence.split()
            idx = find_term_index(words, row.term)
            if idx is None:
                continue
            words_batch.append(words)
            metadata.append((row, idx))

        if not words_batch:
            continue

        encoded = ctx.encode(words_batch)
        upos_logits, lemma_logits = ctx.backend.run(encoded)

        for batch_index, (row, term_idx) in enumerate(metadata):
            words = row.sentence.split()
            word_id = first_word_offset + term_idx

            word_ids = encoded.word_ids(batch_index=batch_index)

            # Find the token index for this word
            token_idx = None
            for ti, wid in enumerate(word_ids):
                if wid == word_id:
                    token_idx = ti
                    break

            if token_idx is None:
                continue

            lemma_row = lemma_logits[batch_index][token_idx]

            # Predict lemma
            predicted_lemma, source, upos_tag, _ = ctx.predict_word(
                words[term_idx], "", lemma_row
            )

            if predicted_lemma is None:
                predicted_lemma = words[term_idx].lower()

            level = row.level
            correct = predicted_lemma.lower() == row.term.lower()

            stats[level]["total"] += 1
            if correct:
                stats[level]["correct"] += 1
            elif len(sample_errors[level]) < 8:
                sample_errors[level].append({
                    "term": row.term,
                    "level": level,
                    "sentence": row.sentence,
                    "predicted_lemma": predicted_lemma,
                    "source": source,
                })

    # Build report
    report = {"lang": lang, "levels": {}}
    total_correct = 0
    total_total = 0

    for level in ["A1", "A2", "B1", "B2", "C1"]:
        total = stats[level]["total"]
        correct = stats[level]["correct"]
        lo, hi = wilson_interval(correct, total)
        report["levels"][level] = {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "wilson_95_ci": [round(lo, 4), round(hi, 4)],
            "examples": sample_errors[level],
        }
        total_correct += correct
        total_total += total

    if total_total > 0:
        report["overall"] = {
            "correct": total_correct,
            "total": total_total,
            "accuracy": round(total_correct / total_total, 4),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    ctx.backend.close()


if __name__ == "__main__":
    main()
