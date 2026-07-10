"""CEFR vocabulary evaluation harness — the real acceptance gate.

Treebank test-set accuracy != CEFR-vocabulary accuracy. This harness reads
VocabLevels CSVs (one row per (lemma, POS) pair) and runs the *full*
production pipeline on each CEFR word in a treebank-sourced sentence:

    model forward -> constrained argmax -> lexicon fallback -> postprocess

NOT bare argmax. Word-level exact match. Lemma metric skips POS classes
where the lemma concept does not apply (PROPN/PUNCT/SYM/X/NUM); UPOS is
scored on all content tokens. Gates >90% on both metrics (nonzero exit on
failure).

Reproducible:
    uv run python -m lemmatizer.eval.cefr_eval --lang de
    uv run python -m lemmatizer.eval.cefr_eval --lang all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from lemmatizer.eval.cefr_data import LEVELS, build_sentence_index
from lemmatizer.eval.context import build_eval_context
from lemmatizer.languages import (
    LANGUAGE_NAMES,
    LANGUAGES,
    VOCAB_LEMMA_COLUMNS,
    vocab_levels_root,
)

# POS classes excluded from both lemma and UPOS metrics: lemma concept does
# not apply (PROPN/PUNCT/SYM/X/NUM), and these carry no semantic content
# worth gating UPOS on either.
NON_CONTENT_POS = {"PROPN", "PUNCT", "SYM", "X", "NUM"}

# Gate: every language must clear this on both metrics.
GATE_ACCURACY = 0.90

# Minimum fraction of vocab entries that must be evaluated (not skipped).
# Prevents a false-positive gate pass when most entries are dropped.
MIN_COVERAGE = 0.50

# Cap on per-level error examples retained in the report (keeps JSON bounded).
MAX_NEVER_CORRECT_EXAMPLES = 20

DEFAULT_OUT_DIR = Path("artifacts/cefr_eval")


@dataclass
class CefrVocabEntry:
    level: str
    term: str
    pos: str


def load_cefr_vocab_with_pos(lang: str) -> list[CefrVocabEntry]:
    """Load (lemma, POS) pairs from VocabLevels CSVs for one language.

    The VocabLevels schema's first column is the language lemma column
    (German_Lemma, French_Lemma, ...) and the POS column is named ``POS``.
    """
    vocab_dir = vocab_levels_root()
    lang_name = LANGUAGE_NAMES.get(lang, lang)
    entries: list[CefrVocabEntry] = []

    for level in LEVELS:
        csv_path = vocab_dir / lang_name / f"{level}.csv"
        if not csv_path.exists():
            continue
        # utf-8-sig strips a BOM if present (Windows exports).
        with csv_path.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if not fieldnames:
                continue
            # Use the registry's exact lemma column name; fall back to
            # first column only if the registry has no mapping.
            first_col = VOCAB_LEMMA_COLUMNS.get(lang) or fieldnames[0]
            header_pos = _resolve_pos_column(fieldnames)
            for row in reader:
                term = (row.get(first_col) or "").strip()
                pos = (row.get(header_pos) or "").strip() if header_pos else ""
                if term:
                    entries.append(CefrVocabEntry(level=level, term=term, pos=pos))

    return entries


def _resolve_pos_column(fieldnames: list[str] | None) -> str | None:
    if not fieldnames:
        return None
    for name in fieldnames:
        if name.strip().upper() == "POS":
            return name
    return None


def evaluate_language(lang: str, out_dir: Path, batch_size: int = 8) -> dict:
    """Run CEFR eval for one language. Returns the per-language report dict."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    ctx = build_eval_context(lang)
    try:
        first_word_offset = ctx.first_word_offset()

        vocab = load_cefr_vocab_with_pos(lang)
        sentence_index = build_sentence_index(lang)

        # Index vocab by (level, term) so each CEFR word is scored once.
        # Store pre-split words to avoid re-splitting every batch iteration.
        rows: list[tuple[CefrVocabEntry, list[str], int]] = []
        skipped_no_sentence = 0
        skipped_no_match = 0
        skipped_no_token = 0
        for entry in vocab:
            sentences = sentence_index.get(entry.term.lower(), [])
            if not sentences:
                skipped_no_sentence += 1
                continue
            # Try each available sentence until one matches the term,
            # so a tokenization mismatch on one sentence does not drop
            # the term from evaluation.
            matched = False
            for sentence in sentences:
                words = sentence.split()
                idx = _find_term_index(words, entry.term)
                if idx is not None:
                    rows.append((entry, words, idx))
                    matched = True
                    break
            if not matched:
                skipped_no_match += 1

        stats: dict[str, dict] = defaultdict(
            lambda: {
                "lemma_correct": 0,
                "lemma_total": 0,
                "lemma_errors": 0,
                "upos_correct": 0,
                "upos_total": 0,
                "upos_errors_count": 0,
                "never_correct": [],
                "upos_errors": [],
            }
        )

        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            words_batch = [words for _, words, _ in batch]
            encoded = ctx.encode(words_batch)
            upos_logits, lemma_logits = ctx.backend.run(encoded)

            for batch_index, (entry, words, term_idx) in enumerate(batch):
                word_ids = encoded.word_ids(batch_index=batch_index)
                token_idx = _first_token_for_word(word_ids, first_word_offset, term_idx)
                if token_idx is None:
                    skipped_no_token += 1
                    continue

                word = words[term_idx]
                # upos="" is safe: predict_word only uses it as a fallback when
                # upos_logits is None, which never happens here. We want UPOS
                # predicted from logits, not from the gold POS value.
                predicted_lemma, source, predicted_upos, _ = ctx.predict_word(
                    word,
                    "",
                    lemma_logits[batch_index][token_idx],
                    upos_logits[batch_index][token_idx],
                )
                gold_pos = entry.pos.upper() if entry.pos else ""

                # Lemma metric: skip POS where lemma concept does not apply.
                if gold_pos not in NON_CONTENT_POS:
                    stats[entry.level]["lemma_total"] += 1
                    if (
                        predicted_lemma is not None
                        and predicted_lemma.lower() == entry.term.lower()
                    ):
                        stats[entry.level]["lemma_correct"] += 1
                    else:
                        stats[entry.level]["lemma_errors"] += 1
                        if len(stats[entry.level]["never_correct"]) < MAX_NEVER_CORRECT_EXAMPLES:
                            stats[entry.level]["never_correct"].append(
                                {
                                    "term": entry.term,
                                    "level": entry.level,
                                    "pos": gold_pos,
                                    "predicted_lemma": predicted_lemma,
                                    "source": source,
                                }
                            )

                # UPOS metric: score on content tokens (skip non-content).
                if gold_pos and gold_pos not in NON_CONTENT_POS:
                    stats[entry.level]["upos_total"] += 1
                    if predicted_upos == gold_pos:
                        stats[entry.level]["upos_correct"] += 1
                    else:
                        stats[entry.level]["upos_errors_count"] += 1
                        if len(stats[entry.level]["upos_errors"]) < MAX_NEVER_CORRECT_EXAMPLES:
                            stats[entry.level]["upos_errors"].append(
                                {
                                    "term": entry.term,
                                    "level": entry.level,
                                    "gold_upos": gold_pos,
                                    "predicted_upos": predicted_upos,
                                }
                            )
    finally:
        ctx.backend.close()

    report = {"lang": lang, "levels": {}}
    for level in LEVELS:
        s = stats[level]
        lemma_acc = s["lemma_correct"] / s["lemma_total"] if s["lemma_total"] else 0.0
        upos_acc = s["upos_correct"] / s["upos_total"] if s["upos_total"] else 0.0
        report["levels"][level] = {
            "lemma_total": s["lemma_total"],
            "lemma_correct": s["lemma_correct"],
            "lemma_errors": s["lemma_errors"],
            "lemma_accuracy": round(lemma_acc, 4),
            "upos_total": s["upos_total"],
            "upos_correct": s["upos_correct"],
            "upos_errors": s["upos_errors_count"],
            "upos_accuracy": round(upos_acc, 4),
            "never_correct_examples": s["never_correct"],
            "upos_error_examples": s["upos_errors"],
        }

    total_lemma_c = sum(stats[lv]["lemma_correct"] for lv in LEVELS)
    total_lemma_t = sum(stats[lv]["lemma_total"] for lv in LEVELS)
    total_upos_c = sum(stats[lv]["upos_correct"] for lv in LEVELS)
    total_upos_t = sum(stats[lv]["upos_total"] for lv in LEVELS)
    vocab_total = len(vocab)
    evaluated = total_lemma_t
    coverage = evaluated / vocab_total if vocab_total else 0.0
    report["overall"] = {
        "lemma_accuracy": round(total_lemma_c / total_lemma_t, 4) if total_lemma_t else 0.0,
        "upos_accuracy": round(total_upos_c / total_upos_t, 4) if total_upos_t else 0.0,
        "lemma_total": total_lemma_t,
        "upos_total": total_upos_t,
        "vocab_total": vocab_total,
        "coverage": round(coverage, 4),
        "skipped_no_sentence": skipped_no_sentence,
        "skipped_no_match": skipped_no_match,
        "skipped_no_token": skipped_no_token,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{lang}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _strip_trailing_punct(word: str) -> str:
    """Strip trailing Unicode punctuation (any category P*).

    Handles Latin (``.``, ``,``) and CJK/Arabic marks (``。``, ``،``, ``؟``)
    so CEFR terms at sentence boundaries in any script are matched.
    """
    result = word
    while result and unicodedata.category(result[-1]).startswith("P"):
        result = result[:-1]
    return result


def _find_term_index(words: list[str], term: str) -> int | None:
    """Find the word index matching ``term``.

    Matches case-insensitively. Also strips trailing punctuation from each
    word so that a CEFR term at a sentence boundary (e.g. "Haus." in the
    raw UD text, or "字。" in Chinese) is still matched.
    """
    term_lower = term.lower()
    for idx, word in enumerate(words):
        if word.lower() == term_lower:
            return idx
    # Fallback: compare with trailing punctuation stripped, so words
    # adjacent to sentence-final punctuation are not silently skipped.
    for idx, word in enumerate(words):
        if _strip_trailing_punct(word.lower()) == term_lower:
            return idx
    return None


def _first_token_for_word(
    word_ids: list[int | None], first_word_offset: int, term_idx: int
) -> int | None:
    word_id = first_word_offset + term_idx
    for ti, wid in enumerate(word_ids):
        if wid == word_id:
            return ti
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CEFR vocabulary eval gate.")
    parser.add_argument(
        "--lang",
        required=True,
        choices=[s.lang for s in LANGUAGES] + ["all"],
        help="Language code (de/en/es/fr/sv/nl/ar/zh) or 'all'.",
    )
    def _positive_int(value: str) -> int:
        ivalue = int(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError(f"batch-size must be >= 1, got {value}")
        return ivalue

    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=int(os.getenv("EVAL_BATCH_SIZE", "8")),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.getenv("CEFR_EVAL_OUT", str(DEFAULT_OUT_DIR))),
    )
    args = parser.parse_args(argv)

    langs: list[str] = (
        [s.lang for s in LANGUAGES] if args.lang == "all" else [args.lang]
    )
    out_dir: Path = args.out_dir

    summary: dict[str, dict] = {}
    failed: list[str] = []
    for lang in langs:
        print(f"CEFR eval: {lang}", flush=True)
        try:
            report = evaluate_language(lang, out_dir, args.batch_size)
        except Exception as exc:  # noqa: BLE001 — CI gate must report all langs
            tb = traceback.format_exc()
            print(f"  {lang}: ERROR {type(exc).__name__}: {exc}\n{tb}", file=sys.stderr, flush=True)
            summary[lang] = {"error": str(exc), "type": type(exc).__name__}
            failed.append(lang)
            continue
        ov = report["overall"]
        summary[lang] = ov
        lemma_ok = ov["lemma_accuracy"] >= GATE_ACCURACY
        upos_ok = ov["upos_accuracy"] >= GATE_ACCURACY
        coverage_ok = ov["coverage"] >= MIN_COVERAGE
        status = "PASS" if lemma_ok and upos_ok and coverage_ok else "FAIL"
        print(
            f"  {lang}: lemma={ov['lemma_accuracy']:.4f} "
            f"upos={ov['upos_accuracy']:.4f} "
            f"coverage={ov['coverage']:.4f} [{status}]",
            flush=True,
        )
        if not (lemma_ok and upos_ok and coverage_ok):
            failed.append(lang)

    print(json.dumps({"summary": summary, "gate": GATE_ACCURACY}, indent=2))
    if failed:
        print(f"GATE FAILED: {failed} below {GATE_ACCURACY:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
