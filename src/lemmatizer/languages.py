"""Single source of truth for supported languages.

Adding a language = add one `LanguageSpec` entry to `LANGUAGES` below.
No other file in the codebase hardcodes a language list — trainers, data
pipeline, eval, and CLI all read from this registry.

Each spec binds a language code to:
- its training `Family` (which trainer module owns it)
- backbone model id, lang token, UD treebank source
- optional spaCy model + VocabLevels lemma column (None when N/A)

Derived helpers (`LANG_TOKENS`, `LANGUAGE_NAMES`, etc.) exist for backward
compat with code that iterates per-lang dicts; prefer `spec(lang)` / `LANGUAGES`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Family(str, Enum):
    """Which trainer module owns a language."""

    MULTITASK = "multitask"  # EuroBERT/BERT token-cls, upos + lemma (mlx_multitask)
    BYT5 = "byt5"  # ByT5 encoder + lemma head, byte-level (train_byt5)
    ZH_BIO = "zh_bio"  # bert-base-chinese BIO-POS (zh_bio)


@dataclass(frozen=True)
class LanguageSpec:
    """Everything the registry needs to know about one language.

    `ud_files` is derived from `ud_repo`-agnostic split paths; the repo/prefix
    pair is kept for `fetch_ud`.
    """

    lang: str
    name: str
    family: Family
    base_model: str
    lang_token: str
    ud_repo: str
    ud_prefix: str
    spacy_model: str | None = None
    vocab_lemma_column: str | None = None


def _gold_split(lang: str, split: str) -> str:
    """Standard UD gold path: data/gold/<lang>/<split>.conllu (dev→validation)."""
    fname = "dev" if split == "validation" else split
    return f"data/gold/{lang}/{fname}.conllu"


LANGUAGES: tuple[LanguageSpec, ...] = (
    LanguageSpec(
        lang="en",
        name="english",
        family=Family.MULTITASK,
        base_model="EuroBERT/EuroBERT-210m",
        lang_token="[LANG_EN]",
        ud_repo="UD_English-EWT",
        ud_prefix="en_ewt",
        spacy_model="en_core_web_lg",
        vocab_lemma_column="English_Lemma",
    ),
    LanguageSpec(
        lang="de",
        name="german",
        family=Family.MULTITASK,
        base_model="EuroBERT/EuroBERT-210m",
        lang_token="[LANG_DE]",
        ud_repo="UD_German-GSD",
        ud_prefix="de_gsd",
        spacy_model="de_core_news_lg",
        vocab_lemma_column="German_Lemma",
    ),
    LanguageSpec(
        lang="es",
        name="spanish",
        family=Family.MULTITASK,
        base_model="EuroBERT/EuroBERT-210m",
        lang_token="[LANG_ES]",
        ud_repo="UD_Spanish-AnCora",
        ud_prefix="es_ancora",
        spacy_model="es_core_news_lg",
        vocab_lemma_column="Spanish_Lemma",
    ),
    LanguageSpec(
        lang="fr",
        name="french",
        family=Family.MULTITASK,
        base_model="EuroBERT/EuroBERT-210m",
        lang_token="[LANG_FR]",
        ud_repo="UD_French-GSD",
        ud_prefix="fr_gsd",
        spacy_model="fr_core_news_lg",
        vocab_lemma_column="French_Lemma",
    ),
    LanguageSpec(
        lang="sv",
        name="swedish",
        family=Family.MULTITASK,
        base_model="vesteinn/ScandiBERT",
        lang_token="[LANG_SV]",
        ud_repo="UD_Swedish-Talbanken",
        ud_prefix="sv_talbanken",
        spacy_model=None,
        vocab_lemma_column="Swedish_Lemma",
    ),
    LanguageSpec(
        lang="ar",
        name="arabic",
        family=Family.BYT5,
        base_model="google/byt5-small",
        lang_token="[LANG_AR]",
        ud_repo="UD_Arabic-PADT",
        ud_prefix="ar_padt",
        spacy_model=None,
        vocab_lemma_column="Arabic_Lemma",
    ),
    LanguageSpec(
        lang="zh",
        name="chinese",
        family=Family.ZH_BIO,
        base_model="bert-base-chinese",
        lang_token="[LANG_ZH]",
        ud_repo="UD_Chinese-GSD",
        ud_prefix="zh_gsd",
        spacy_model=None,
        vocab_lemma_column="Chinese_Lemma",
    ),
)


# --- Registry lookups ---------------------------------------------------


def spec(lang: str) -> LanguageSpec:
    """Lookup a LanguageSpec by code. Raises ValueError if unknown.

    Normalizes aliases (e.g. "german" → "de") and case via `normalize_lang`
    so callers passing user input or env values resolve consistently.
    """
    resolved = normalize_lang(lang)
    for s in LANGUAGES:
        if s.lang == resolved:
            return s
    raise ValueError(
        f"Unsupported language '{resolved}'. Expected one of: {', '.join(lang_codes())}"
    )


def specs_for_family(family: Family) -> tuple[LanguageSpec, ...]:
    """All specs owned by a given trainer family."""
    return tuple(s for s in LANGUAGES if s.family == family)


def lang_codes() -> tuple[str, ...]:
    """All registered language codes."""
    return tuple(s.lang for s in LANGUAGES)


# --- Backward-compat derived views --------------------------------------
# Prefer `spec(lang)` / `LANGUAGES` in new code. These exist so existing
# callers (`LANG_TOKENS`, `LANGUAGE_NAMES`, etc.) keep working during the
# transition; they are pure projections over `LANGUAGES`.

DEFAULT_BASE_MODEL = "EuroBERT/EuroBERT-210m"

# Local multilingual tokenizer saved by dataset.main() (shared EuroBERT
# WordPiece + LANG_* special tokens). Used as the tokenizer fallback for
# languages without a dedicated BASE_MODELS entry, so per-language dataset
# builds resolve a local path instead of fetching EuroBERT remotely.
DEFAULT_LOCAL_TOKENIZER = "artifacts/tokenizer"

# Kept as a tuple for `from lemmatizer.languages import LANGS` callers.
LANGS = lang_codes()

LANG_TOKENS = {s.lang: s.lang_token for s in LANGUAGES}
LANGUAGE_NAMES = {s.lang: s.name for s in LANGUAGES}
BASE_MODELS = {s.lang: s.base_model for s in LANGUAGES}
SPACY_MODELS = {s.lang: s.spacy_model for s in LANGUAGES if s.spacy_model}
VOCAB_LEMMA_COLUMNS = {s.lang: s.vocab_lemma_column for s in LANGUAGES if s.vocab_lemma_column}

UD_FILES = {
    split: {s.lang: _gold_split(s.lang, split) for s in LANGUAGES}
    for split in ("train", "validation", "test")
}


# --- Path resolution ----------------------------------------------------


@dataclass(frozen=True)
class LanguageAssets:
    """Resolved per-language paths + model identifiers.

    Built from a LanguageSpec; environment variables override defaults.
    Kept frozen + paths-as-Path so callers can use `/` composition directly.
    """

    lang: str
    artifacts_dir: Path
    tokenizer_dir: Path
    dataset_path: Path
    output_dir: Path
    merged_dir: Path
    onnx_dir: Path
    web_model_dir: Path
    label2id_path: Path
    id2label_path: Path
    upos_label2id_path: Path
    upos_id2label_path: Path
    edit_trees_path: Path
    lexicon_path: Path
    exceptions_path: Path
    eval_report_path: Path
    base_model: str
    tokenizer_name: str


_ALIASES = {
    "english": "en",
    "german": "de",
    "deutsch": "de",
    "spanish": "es",
    "espanol": "es",
    "español": "es",
    "french": "fr",
    "francais": "fr",
    "français": "fr",
}


def normalize_lang(lang: str | None = None) -> str:
    """Resolve a lang code (case-insensitive, with name aliases).

    Honors `LEMMA_LANG` / `LANGUAGE` env vars when `lang` is None. Defaults
    to "de" so callers that historically defaulted to German still work.
    """
    resolved = lang or os.getenv("LEMMA_LANG") or os.getenv("LANGUAGE") or "de"
    resolved = resolved.strip().lower()
    resolved = _ALIASES.get(resolved, resolved)
    if resolved not in lang_codes():
        raise ValueError(
            f"Unsupported language '{resolved}'. Expected one of: {', '.join(lang_codes())}"
        )
    return resolved


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def vocab_levels_root() -> Path:
    configured = os.getenv("VOCAB_LEVELS_DIR")
    if configured:
        return Path(configured)
    return project_root().parent / "VocabLevels"


def split_files_for_lang(lang: str) -> dict[str, str]:
    """Gold CoNLL-U paths for a language's train/validation/test splits."""
    resolved = normalize_lang(lang)
    return {split: UD_FILES[split][resolved] for split in ("train", "validation", "test")}


def language_assets(lang: str | None = None) -> LanguageAssets:
    """Build a LanguageAssets (resolved paths) from the registry + env.

    Env overrides (per-language): ARTIFACTS_DIR, DATASET_PATH, OUTPUT_DIR,
    MERGED_DIR, ONNX_DIR, WEB_MODEL_DIR, TOKENIZER_DIR, TRAIN_WARM_START,
    TOKENIZER_NAME. Unset → registry-derived defaults.
    """
    resolved = normalize_lang(lang)
    s = spec(resolved)

    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", f"artifacts/lemma_{resolved}"))
    dataset_path = Path(
        os.getenv("DATASET_PATH", f"data/processed/eurobert_lemma_{resolved}_dataset")
    )
    output_dir = Path(os.getenv("OUTPUT_DIR", f"runs/eurobert-lemma-{resolved}-210m-lora"))
    merged_dir = Path(os.getenv("MERGED_DIR", f"models/eurobert-lemma-{resolved}-210m-merged"))
    onnx_dir = Path(os.getenv("ONNX_DIR", f"onnx/eurobert-lemma-{resolved}-210m"))
    web_model_dir = Path(os.getenv("WEB_MODEL_DIR", f"web/model/lemma_{resolved}"))
    tokenizer_dir = Path(os.getenv("TOKENIZER_DIR", str(artifacts_dir / "tokenizer")))

    base_model = os.getenv("TRAIN_WARM_START") or s.base_model
    # Tokenizer fallback chain: explicit env > per-language base model >
    # local multilingual tokenizer (saved at artifacts/tokenizer). The local
    # fallback avoids a remote HF fetch for languages without a dedicated
    # BASE_MODELS entry (e.g. en/de/es/fr fall back to the shared EuroBERT
    # tokenizer, which dataset.main() persists locally).
    local_tok = Path(DEFAULT_LOCAL_TOKENIZER)
    tokenizer_name = (
        os.getenv("TOKENIZER_NAME")
        or base_model
        or (str(local_tok) if local_tok.exists() else DEFAULT_BASE_MODEL)
    )

    return LanguageAssets(
        lang=resolved,
        artifacts_dir=artifacts_dir,
        tokenizer_dir=tokenizer_dir,
        dataset_path=dataset_path,
        output_dir=output_dir,
        merged_dir=merged_dir,
        onnx_dir=onnx_dir,
        web_model_dir=web_model_dir,
        label2id_path=artifacts_dir / "label2id.json",
        id2label_path=artifacts_dir / "id2label.json",
        upos_label2id_path=artifacts_dir / "upos_label2id.json",
        upos_id2label_path=artifacts_dir / "upos_id2label.json",
        edit_trees_path=artifacts_dir / "edit_trees.json",
        lexicon_path=artifacts_dir / "lexicon.json",
        exceptions_path=artifacts_dir / "exceptions.json",
        eval_report_path=artifacts_dir / "eval_report.json",
        base_model=base_model,
        tokenizer_name=tokenizer_name,
    )
