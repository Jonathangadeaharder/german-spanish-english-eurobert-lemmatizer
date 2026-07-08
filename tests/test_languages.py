"""Tests for the language registry: the single source of truth.

The core invariant: adding a language = one `LanguageSpec` entry in `LANGUAGES`.
No other module hardcodes a lang list. These tests pin that contract so a
future edit that re-introduces a parallel dict fails CI.
"""

from __future__ import annotations

import pytest

from lemmatizer.languages import (
    LANG_TOKENS,
    LANGUAGES,
    Family,
    lang_codes,
    language_assets,
    spec,
    specs_for_family,
    split_files_for_lang,
)

# --- Registry lookup ----------------------------------------------------


def test_spec_lookup_returns_registered_lang():
    s = spec("de")
    assert s.lang == "de"
    assert s.name == "german"
    assert s.family is Family.MULTITASK


@pytest.mark.parametrize("code", ["en", "de", "es", "fr", "ar", "sv", "zh"])
def test_every_registered_lang_lookupable(code):
    assert spec(code).lang == code


def test_spec_unknown_lang_raises():
    with pytest.raises(ValueError, match="Unsupported language"):
        spec("xx")


def test_spec_case_insensitive():
    assert spec("DE").lang == "de"
    assert spec("Ar").lang == "ar"


# --- Derived views stay in sync with registry ---------------------------


def test_lang_tokens_covers_all_registered_langs():
    assert set(LANG_TOKENS) == set(lang_codes())


def test_lang_codes_no_duplicates():
    codes = lang_codes()
    assert len(codes) == len(set(codes))


def test_specs_for_family_partitions_registry():
    families = set(Family)
    assert {s.family for s in LANGUAGES}.issubset(families)
    # Every spec appears in exactly one family's partition.
    partitioned = [s for f in families for s in specs_for_family(f)]
    assert sorted(s.lang for s in partitioned) == sorted(lang_codes())


def test_multitask_family_has_the_eurobert_langs():
    multitask = specs_for_family(Family.MULTITASK)
    assert {s.lang for s in multitask} >= {"en", "de", "es", "fr", "sv"}


def test_byt5_family_is_arabic_only():
    assert {s.lang for s in specs_for_family(Family.BYT5)} == {"ar"}


def test_zh_bio_family_is_chinese_only():
    assert {s.lang for s in specs_for_family(Family.ZH_BIO)} == {"zh"}


# --- Assets + paths -----------------------------------------------------


def test_language_assets_uses_registry_base_model(monkeypatch):
    for var in ("TRAIN_WARM_START", "TOKENIZER_NAME"):
        monkeypatch.delenv(var, raising=False)
    assert language_assets("sv").base_model == "vesteinn/ScandiBERT"
    assert language_assets("zh").base_model == "bert-base-chinese"
    assert language_assets("de").base_model == "EuroBERT/EuroBERT-210m"


def test_split_files_for_lang_uses_standard_paths():
    files = split_files_for_lang("es")
    assert files["train"] == "data/gold/es/train.conllu"
    assert files["validation"] == "data/gold/es/dev.conllu"
    assert files["test"] == "data/gold/es/test.conllu"


# --- The "piece of cake" invariant --------------------------------------
# Adding a language must be one entry. This test enforces that the derived
# dicts are projections of LANGUAGES — a future hard-coded dict that drifts
# from the registry will fail here.


def test_lang_tokens_all_match_spec_entries():
    for s in LANGUAGES:
        assert LANG_TOKENS[s.lang] == s.lang_token


def test_every_lang_has_unique_ud_repo():
    repos = [s.ud_repo for s in LANGUAGES]
    assert len(repos) == len(set(repos))
