"""Duplicate sentence detector for CoNLL-U training data.

Finds exact and near-duplicate sentences within and across train/val files.
- Exact duplicate (case-insensitive, whitespace-normalized) → error
- Near-duplicate (Levenshtein distance < 3 tokens) → warning
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

TEXT_RE = re.compile(r"^#\s*text\s*=\s*(.*)$")


@dataclass
class DuplicateResult:
    duplicates: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.duplicates) == 0


def check_file(path: str | Path) -> DuplicateResult:
    text = Path(path).read_text(encoding="utf-8")
    return check_text(text)


def check_text(text: str) -> DuplicateResult:
    result = DuplicateResult()
    sentences = _extract_sentences(text)

    seen: dict[str, str] = {}
    for sent_id, sentence_text in sentences:
        normalized = _normalize(sentence_text)
        if normalized in seen:
            result.duplicates.append(
                f"Duplicate sentence in {sent_id}: '{sentence_text}' (same as '{seen[normalized]}')"
            )
        else:
            seen[normalized] = sent_id

    return result


def check_cross_file(
    train_text: str,
    val_text: str,
) -> DuplicateResult:
    result = DuplicateResult()

    train_sentences = _extract_sentences(train_text)
    val_sentences = _extract_sentences(val_text)

    train_normalized = {_normalize(text): sid for sid, text in train_sentences}

    for val_sid, val_text_str in val_sentences:
        val_norm = _normalize(val_text_str)
        if val_norm in train_normalized:
            result.duplicates.append(
                f"Duplicate across train/val: '{val_text_str}' "
                f"(val {val_sid} = train {train_normalized[val_norm]})"
            )
            continue
        for train_norm, train_sid in train_normalized.items():
            dist = _token_levenshtein(val_norm, train_norm)
            if 0 < dist < 3:
                result.warnings.append(
                    f"Near-duplicate: val {val_sid} '{val_text_str}' "
                    f"similar to train {train_sid} (distance={dist})"
                )

    return result


def _extract_sentences(text: str) -> list[tuple[str, str]]:
    sentences: list[tuple[str, str]] = []
    current_sent_id = ""
    current_text = ""

    for line in text.split("\n"):
        if line.startswith("# sent_id"):
            if "=" in line:
                current_sent_id = line.split("=", 1)[1].strip()
            else:
                current_sent_id = line.split(":", 1)[1].strip()
        elif line.startswith("# text"):
            if "=" in line:
                current_text = line.split("=", 1)[1].strip()
            else:
                current_text = line.split(":", 1)[1].strip()
        elif line.strip() == "":
            if current_sent_id and current_text:
                sentences.append((current_sent_id, current_text))
            current_sent_id = ""
            current_text = ""

    if current_sent_id and current_text:
        sentences.append((current_sent_id, current_text))

    return sentences


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _token_levenshtein(a: str, b: str) -> int:
    tokens_a = a.split()
    tokens_b = b.split()
    len_a, len_b = len(tokens_a), len(tokens_b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    prev = list(range(len_b + 1))
    for i in range(1, len_a + 1):
        curr = [i] + [0] * len_b
        for j in range(1, len_b + 1):
            cost = 0 if tokens_a[i - 1] == tokens_b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = curr

    return prev[len_b]
