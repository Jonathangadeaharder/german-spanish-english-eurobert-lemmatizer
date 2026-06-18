from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from language_assets import (
    LANGS,
    LANGUAGE_NAMES,
    SPACY_MODELS,
    VOCAB_LEMMA_COLUMNS,
    normalize_lang,
    vocab_levels_root,
)

LEVELS = ("A1", "A2", "B1", "B2", "C1")


@dataclass(frozen=True)
class VocabEntry:
    lang: str
    level: str
    entry: str
    canonical: bool
    normalized: str
    audit_reason: str


def _basic_canonical(entry: str) -> tuple[bool, str]:
    if not entry:
        return False, "empty"
    if any(ch.isspace() for ch in entry):
        return False, "multi_word"
    if any(ch.isdigit() for ch in entry):
        return False, "contains_digit"
    return True, "basic"


def load_spacy(lang: str):
    model_name = SPACY_MODELS[normalize_lang(lang)]
    try:
        import spacy  # type: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "spaCy is required for this command. Install it and download "
            f"{model_name} with: python -m spacy download {model_name}"
        ) from exc

    try:
        return spacy.load(model_name)
    except OSError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            f"Missing spaCy model {model_name}. Download it with: "
            f"python -m spacy download {model_name}"
        ) from exc


def load_vocab_entries(root: Path | None = None) -> list[VocabEntry]:
    root = root or vocab_levels_root()
    entries: list[VocabEntry] = []

    for lang in LANGS:
        lang_dir = root / LANGUAGE_NAMES[lang]
        lemma_col = VOCAB_LEMMA_COLUMNS[lang]

        for level in LEVELS:
            path = lang_dir / f"{level}.csv"
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    entry = (row.get(lemma_col) or "").strip()
                    canonical, reason = _basic_canonical(entry)
                    entries.append(
                        VocabEntry(
                            lang=lang,
                            level=level,
                            entry=entry,
                            canonical=canonical,
                            normalized=entry.lower(),
                            audit_reason=reason,
                        )
                    )

    return entries


def build_inventory(root: Path | None = None) -> dict[str, object]:
    entries = load_vocab_entries(root)
    by_lang = {lang: {} for lang in LANGS}

    for entry in entries:
        by_lang[entry.lang][entry.normalized] = {
            "entry": entry.entry,
            "level": entry.level,
            "canonical": entry.canonical,
            "audit_reason": entry.audit_reason,
        }

    return {
        "schema": "canonical-vocab-inventory-v1",
        "languages": by_lang,
    }


def save_inventory(path: Path, root: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_inventory(root), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_inventory(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_terms(inventory: dict[str, object], lang: str) -> set[str]:
    lang = normalize_lang(lang)
    languages = inventory.get("languages", {})
    if not isinstance(languages, dict):
        return set()
    entries = languages.get(lang, {})
    if not isinstance(entries, dict):
        return set()
    return {
        key
        for key, value in entries.items()
        if isinstance(value, dict) and value.get("canonical") is True
    }


def lookup_level(inventory: dict[str, object], lang: str, term: str) -> str | None:
    languages = inventory.get("languages", {})
    if not isinstance(languages, dict):
        return None
    entries = languages.get(normalize_lang(lang), {})
    if not isinstance(entries, dict):
        return None
    record = entries.get(term.lower())
    if not isinstance(record, dict) or record.get("canonical") is not True:
        return None
    level = record.get("level")
    return level if isinstance(level, str) else None
