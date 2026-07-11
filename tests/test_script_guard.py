"""Tests for the language script-plausibility guard.

AGENTS.md §5 (TDD): written before implementation.
The guard catches cross-language dataset corruption (e.g. Swedish rows
saved into a zh-named dataset directory because LEMMA_LANG was unset).
"""

from __future__ import annotations

import pytest

from lemmatizer.data.script_guard import assert_language_plausible


def test_zh_passes_on_real_chinese() -> None:
    words = ["然而", "这样", "的", "处理", "也", "衍生", "了", "一些", "问题"]
    assert_language_plausible("zh", words)


def test_zh_fails_on_swedish() -> None:
    words = ["Individuell", "beskattning", "av", "arbetsinkomster"]
    with pytest.raises(AssertionError, match="zh"):
        assert_language_plausible("zh", words)


def test_en_passes_on_english() -> None:
    words = ["The", "quick", "brown", "fox", "jumps"]
    assert_language_plausible("en", words)


def test_de_passes_on_german() -> None:
    words = ["Individuelle", "Besteuerung", "von", "Arbeitseinkommen"]
    assert_language_plausible("de", words)


def test_ar_passes_on_arabic() -> None:
    words = ["ميراث", "ب", "دولار", "الف"]
    assert_language_plausible("ar", words)


def test_ar_fails_on_english() -> None:
    words = ["The", "quick", "brown", "fox"]
    with pytest.raises(AssertionError, match="ar"):
        assert_language_plausible("ar", words)


def test_zh_fails_on_mixed_majority_swedish() -> None:
    """Corruption case: mostly Swedish with a few CJK chars."""
    words = ["Individuell", "beskattning", "av", "arbetsinkomster", "的"]
    with pytest.raises(AssertionError, match="zh"):
        assert_language_plausible("zh", words)


def test_empty_list_raises() -> None:
    with pytest.raises(AssertionError, match="empty"):
        assert_language_plausible("zh", [])
