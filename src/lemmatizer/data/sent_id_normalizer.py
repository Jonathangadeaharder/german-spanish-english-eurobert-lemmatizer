"""Sent_id normalizer for CoNLL-U training data.

Rewrites inconsistent sent_id formats to canonical:
    {lang}_{level}_{split}_{NNN}

Examples of input → output:
  a2-train-001 → de_a2_train_001
  de_train_b1_001 → de_b1_train_001
  handcraft-de-c2-train-1 → de_c2_train_001
  garbage_id → de_a1_train_001 (renumbered sequentially)
"""

from __future__ import annotations

import re
from pathlib import Path

SENT_ID_RE = re.compile(r"^#\s*sent_id\s*[:=]\s*(.+)$")

CANONICAL_RE = re.compile(
    r"^(?P<lang>[a-z]{2})_(?P<level>a[12]|b[12]|c[12])_(?P<split>train|val)_(?P<num>\d+)$"
)


def parse_sent_id(sent_id: str) -> tuple[str, str, str, int] | None:
    m = CANONICAL_RE.match(sent_id.strip())
    if not m:
        return None
    return (m["lang"], m["level"], m["split"], int(m["num"]))


def normalize_file(
    path: str | Path,
    lang: str,
    level: str,
    split: str,
) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return normalize_text(text, lang=lang, level=level, split=split)


def normalize_text(text: str, lang: str, level: str, split: str) -> str:
    lines = text.split("\n")
    counter = 0
    output: list[str] = []

    for line in lines:
        m = SENT_ID_RE.match(line)
        if m:
            counter += 1
            canonical = f"{lang}_{level}_{split}_{counter:03d}"
            output.append(f"# sent_id = {canonical}")
        else:
            output.append(line)

    return "\n".join(output)
