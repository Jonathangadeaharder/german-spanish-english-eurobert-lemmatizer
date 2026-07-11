"""Script-plausibility guard for per-language dataset builds.

Catches cross-language dataset corruption: the bug where a make-dataset
run without LEMMA_LANG set silently wrote Swedish rows into a zh-named
directory. Called at the end of every per-language dataset build on a
sample of the collected words; raises (does not warn) on mismatch.
"""

from __future__ import annotations

import random

# Unicode block ranges for script detection.
CJK_RANGE = (0x4E00, 0x9FFF)
ARABIC_RANGE = (0x0600, 0x06FF)
LATIN_RANGES = [
    (0x0041, 0x005A),  # A-Z
    (0x0061, 0x007A),  # a-z
    (0x00C0, 0x024F),  # Latin Extended-A/B
]


def _in_range(ch: str, start: int, end: int) -> bool:
    cp = ord(ch)
    return start <= cp <= end


def _in_any_range(ch: str, ranges: list[tuple[int, int]]) -> bool:
    return any(_in_range(ch, s, e) for s, e in ranges)


def _is_cjk(ch: str) -> bool:
    return _in_range(ch, *CJK_RANGE)


def _is_arabic(ch: str) -> bool:
    return _in_range(ch, *ARABIC_RANGE)


def _is_latin(ch: str) -> bool:
    return _in_any_range(ch, LATIN_RANGES)


def _script_ratio(words: list[str], predicate) -> float:
    """Fraction of non-space/punct chars matching the predicate."""
    total = 0
    matching = 0
    for word in words:
        for ch in word:
            if ch.isspace() or ch in ".,;:!?\"'()[]{}—–-":
                continue
            if ch.isdigit():
                continue
            total += 1
            if predicate(ch):
                matching += 1
    return matching / max(total, 1)


def assert_language_plausible(lang: str, sample_words: list[str]) -> None:
    """Assert that sample_words' script is plausible for lang.

    Raises AssertionError if the script distribution doesn't match
    expectations for the declared language. Catches the corruption
    vector where wrong-language data is written to a lang-named dir.
    """
    if not sample_words:
        raise AssertionError(
            f"script_guard: empty word sample for lang={lang}; "
            "cannot verify dataset language plausibility"
        )

    # For small samples, use all words; for large, sample 100.
    if len(sample_words) > 100:
        rng = random.Random(42)
        sample_words = rng.sample(sample_words, 100)

    if lang == "zh":
        ratio = _script_ratio(sample_words, _is_cjk)
        if ratio < 0.5:
            raise AssertionError(
                f"script_guard: lang=zh but CJK char ratio is {ratio:.2f} "
                f"(expected ≥0.50). Sample: {sample_words[:5]}"
            )
    elif lang == "ar":
        ratio = _script_ratio(sample_words, _is_arabic)
        if ratio < 0.5:
            raise AssertionError(
                f"script_guard: lang=ar but Arabic char ratio is {ratio:.2f} "
                f"(expected ≥0.50). Sample: {sample_words[:5]}"
            )
    else:
        # en/de/es/fr/nl/sv — Latin script.
        ratio = _script_ratio(sample_words, _is_latin)
        if ratio < 0.5:
            raise AssertionError(
                f"script_guard: lang={lang} but Latin char ratio is "
                f"{ratio:.2f} (expected ≥0.50). Sample: {sample_words[:5]}"
            )
