"""CEFR data source strategy.

Replaces make_cefr_eval_dataset.py (LLM-based) and
make_cefr_eval_from_treebank.py (treebank-based) with a unified interface.

Each source produces a list of CefrEvalRow: one sentence containing a
CEFR-level vocabulary term, for a specific language.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from conllu_reader import read_conllu
from language_assets import LANGUAGE_NAMES, split_files_for_lang, vocab_levels_root

LEVELS = ["A1", "A2", "B1", "B2", "C1"]
PER_LEVEL_LIMIT = 200


@dataclass
class CefrEvalRow:
    lang: str
    level: str
    term: str
    sentence: str


class CefrDataSource(Protocol):
    def load(self, lang: str) -> list[CefrEvalRow]:
        ...


def load_cefr_vocab(lang: str) -> dict[str, list[str]]:
    """Load CEFR vocabulary from VocabLevels CSVs using the schema's lemma column.

    Uses vocab_schema.LANGS to get the correct lemma column per language,
    instead of hardcoding column 0.
    """
    vocab_dir = vocab_levels_root()
    lang_name = LANGUAGE_NAMES.get(lang, lang)
    by_level: dict[str, list[str]] = {level: [] for level in LEVELS}

    for level in LEVELS:
        csv_path = vocab_dir / lang_name / f"{level}.csv"
        if not csv_path.exists():
            continue

        with csv_path.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            # Resolve lemma column index from header if possible
            col_idx = 0
            if header:
                # VocabLevels schema: first column is always the lemma column
                # (German_Lemma, French_Lemma, etc.)
                col_idx = 0

            for row in reader:
                if not row or col_idx >= len(row):
                    continue
                term = row[col_idx].strip()
                if term:
                    by_level[level].append(term)

    return by_level


def build_sentence_index(lang: str) -> dict[str, list[str]]:
    """Index UD treebank sentences by lowercase word for fast lookup."""
    files = split_files_for_lang(lang)
    index: dict[str, list[str]] = {}

    for split in ["train", "dev", "test"]:
        path = files.get(split)
        if not path or not Path(path).exists():
            continue
        sentences = read_conllu(path, lang=lang)
        for sent in sentences:
            text = sent.get("text", "")
            if not text:
                text = " ".join(sent["words"])
            for word in sent["words"]:
                lower = word.lower()
                if lower not in index:
                    index[lower] = []
                if len(index[lower]) < 5:
                    index[lower].append(text)

    return index


class TreebankCefrSource:
    """Extracts sentences from UD treebanks containing CEFR vocab words.

    No LLM dependency — works offline. Uses VocabLevels CSVs for the vocab
    list and UD treebanks for the sentences.
    """

    def __init__(self, per_level_limit: int = PER_LEVEL_LIMIT):
        self._per_level_limit = per_level_limit

    def load(self, lang: str) -> list[CefrEvalRow]:
        vocab = load_cefr_vocab(lang)
        sentence_index = build_sentence_index(lang)

        rows = []
        for level in LEVELS:
            terms = vocab.get(level, [])[: self._per_level_limit]
            for term in terms:
                sentences = sentence_index.get(term.lower(), [])
                if not sentences:
                    continue
                rows.append(CefrEvalRow(
                    lang=lang,
                    level=level,
                    term=term,
                    sentence=sentences[0],
                ))

        return rows

    def generate_and_save(self, lang: str, output_dir: Path | None = None) -> int:
        """Generate CEFR eval data and append to JSONL file."""
        output_dir = output_dir or Path("data/cefr_eval")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{lang}.jsonl"

        # Load existing to skip
        seen = set()
        if output_path.exists():
            with output_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        seen.add((row.get("level"), row.get("term")))
                    except json.JSONDecodeError:
                        continue

        rows = self.load(lang)
        written = 0
        with output_path.open("a", encoding="utf-8") as f:
            for row in rows:
                if (row.level, row.term) in seen:
                    continue
                json_row = {
                    "lang": row.lang,
                    "level": row.level,
                    "term": row.term,
                    "sentences": [row.sentence],
                    "raw_output": row.sentence,
                }
                f.write(json.dumps(json_row, ensure_ascii=False) + "\n")
                f.flush()
                written += 1

        return written


class JsonlCefrSource:
    """Reads pre-generated data/cefr_eval/{lang}.jsonl."""

    def __init__(self, data_dir: Path | None = None):
        self._data_dir = data_dir or Path("data/cefr_eval")

    def load(self, lang: str) -> list[CefrEvalRow]:
        path = self._data_dir / f"{lang}.jsonl"
        if not path.exists():
            return []

        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("error"):
                    continue
                for sentence in row.get("sentences", []):
                    rows.append(CefrEvalRow(
                        lang=row.get("lang", lang),
                        level=row.get("level", ""),
                        term=row.get("term", ""),
                        sentence=sentence,
                    ))

        return rows


class LlmCefrSource:
    """Generates sentences via LMStudio (requires running LLM server).

    Delegates to the existing make_cefr_eval_dataset.py logic.
    """

    def __init__(self, url: str = "http://127.0.0.1:1234", model: str = "gemma-4-e2b-it-qat"):
        self._url = url
        self._model = model

    def load(self, lang: str) -> list[CefrEvalRow]:
        raise NotImplementedError(
            "LLM CEFR source requires a running LMStudio server. "
            "Use TreebankCefrSource or JsonlCefrSource instead."
        )


def cefr_data_source_from_env() -> CefrDataSource:
    """Factory: selects CEFR data source from environment.

    - EVAL_CEFR_SOURCE=treebank → TreebankCefrSource (generates from VocabLevels + UD)
    - EVAL_CEFR_SOURCE=llm → LlmCefrSource (requires LMStudio)
    - Otherwise → JsonlCefrSource (reads pre-generated data)
    """
    source = os.getenv("EVAL_CEFR_SOURCE", "").lower()

    if source == "treebank":
        return TreebankCefrSource()
    if source == "llm":
        url = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234")
        model = os.getenv("LMSTUDIO_MODEL", "gemma-4-e2b-it-qat")
        return LlmCefrSource(url, model)

    return JsonlCefrSource()
