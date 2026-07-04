from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

LANGS = ("en", "de", "es", "fr", "ar", "sv", "zh")

LANGUAGE_NAMES = {
    "en": "english",
    "de": "german",
    "es": "spanish",
    "fr": "french",
    "ar": "arabic",
    "sv": "swedish",
    "zh": "chinese",
}

VOCAB_LEMMA_COLUMNS = {
    "en": "English_Lemma",
    "de": "German_Lemma",
    "es": "Spanish_Lemma",
    "fr": "French_Lemma",
    "ar": "Arabic_Lemma",
    "sv": "Swedish_Lemma",
    "zh": "Chinese_Lemma",
}

SPACY_MODELS = {
    "en": "en_core_web_lg",
    "de": "de_core_news_lg",
    "es": "es_core_news_lg",
    "fr": "fr_core_news_lg",
}

UD_FILES = {
    "train": {
        "de": "data/gold/de/train.conllu",
        "es": "data/gold/es/train.conllu",
        "en": "data/gold/en/train.conllu",
        "fr": "data/gold/fr/train.conllu",
        "ar": "data/gold/ar/train.conllu",
        "sv": "data/gold/sv/train.conllu",
        "zh": "data/gold/zh/train.conllu",
    },
    "validation": {
        "de": "data/gold/de/dev.conllu",
        "es": "data/gold/es/dev.conllu",
        "en": "data/gold/en/dev.conllu",
        "fr": "data/gold/fr/dev.conllu",
        "ar": "data/gold/ar/dev.conllu",
        "sv": "data/gold/sv/dev.conllu",
        "zh": "data/gold/zh/dev.conllu",
    },
    "test": {
        "de": "data/gold/de/test.conllu",
        "es": "data/gold/es/test.conllu",
        "en": "data/gold/en/test.conllu",
        "fr": "data/gold/fr/test.conllu",
        "ar": "data/gold/ar/test.conllu",
        "sv": "data/gold/sv/test.conllu",
        "zh": "data/gold/zh/test.conllu",
    },
}


@dataclass(frozen=True)
class LanguageAssets:
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


def normalize_lang(lang: str | None = None) -> str:
    resolved = lang or os.getenv("LEMMA_LANG") or os.getenv("LANGUAGE") or "de"
    resolved = resolved.strip().lower()

    aliases = {
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
    resolved = aliases.get(resolved, resolved)

    if resolved not in LANGS:
        raise ValueError(f"Unsupported language '{resolved}'. Expected one of: {', '.join(LANGS)}")

    return resolved


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def vocab_levels_root() -> Path:
    configured = os.getenv("VOCAB_LEVELS_DIR")
    if configured:
        return Path(configured)
    return project_root().parent / "VocabLevels"


def language_assets(lang: str | None = None) -> LanguageAssets:
    lang = normalize_lang(lang)
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", f"artifacts/lemma_{lang}"))
    dataset_path = Path(os.getenv("DATASET_PATH", f"data/processed/eurobert_lemma_{lang}_dataset"))
    output_dir = Path(os.getenv("OUTPUT_DIR", f"runs/eurobert-lemma-{lang}-210m-lora"))
    merged_dir = Path(os.getenv("MERGED_DIR", f"models/eurobert-lemma-{lang}-210m-merged"))
    onnx_dir = Path(os.getenv("ONNX_DIR", f"onnx/eurobert-lemma-{lang}-210m"))
    web_model_dir = Path(os.getenv("WEB_MODEL_DIR", f"web/model/lemma_{lang}"))
    tokenizer_dir = Path(os.getenv("TOKENIZER_DIR", str(artifacts_dir / "tokenizer")))

    return LanguageAssets(
        lang=lang,
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
    )


def split_files_for_lang(lang: str) -> dict[str, str]:
    lang = normalize_lang(lang)
    return {split: files[lang] for split, files in UD_FILES.items()}
