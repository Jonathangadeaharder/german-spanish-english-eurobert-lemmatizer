"""Lemma consistency checker for CoNLL-U training data.

Validates language-specific lemma plausibility rules:
- PUNCT lemma equals form
- PROPN lemma capitalized
- German: NOUN capitalized, VERB infinitive (-en/-n)
- English: VERB base form (no -s/-ed/-ing)
- Spanish: VERB infinitive (-ar/-er/-ir)
- Chinese: lemma equals form
- Arabic: lemma in Arabic script
- No sense numbers in lemma (e.g., can-1)
- Same FORM+UPOS → same LEMMA within file
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
SENSE_NUMBER_RE = re.compile(r"-\d+$")


@dataclass
class LemmaCheckResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


def check_file(path: str | Path, lang: str) -> LemmaCheckResult:
    text = Path(path).read_text(encoding="utf-8")
    return check_text(text, lang=lang)


def check_text(text: str, lang: str) -> LemmaCheckResult:
    result = LemmaCheckResult()
    lines = text.split("\n")

    form_upos_lemma: dict[tuple[str, str], str] = {}
    current_line = 0

    for line in lines:
        current_line += 1
        if line.startswith("#") or line.strip() == "":
            continue

        cols = line.split("\t")
        if len(cols) != 10:
            continue

        _, form, lemma, upos = cols[0], cols[1], cols[2], cols[3]

        key = (form, upos)
        if key in form_upos_lemma:
            existing = form_upos_lemma[key]
            if existing != lemma:
                result.errors.append(
                    f"Line {current_line}: inconsistent lemma for "
                    f"FORM='{form}' UPOS={upos}: "
                    f"'{existing}' vs '{lemma}'"
                )
        else:
            form_upos_lemma[key] = lemma

        _check_sense_numbers(result, current_line, form, lemma, upos)
        _check_punct_identity(result, current_line, form, lemma, upos)
        _check_propn_capital(result, current_line, form, lemma, upos, lang)

        if lang == "de":
            _check_de_noun_capital(result, current_line, form, lemma, upos)
            _check_de_verb_infinitive(result, current_line, form, lemma, upos)
        elif lang == "en":
            _check_en_verb_base(result, current_line, form, lemma, upos)
        elif lang == "es":
            _check_es_verb_infinitive(result, current_line, form, lemma, upos)
        elif lang == "zh":
            _check_zh_lemma_equals_form(result, current_line, form, lemma, upos)
        elif lang == "ar":
            _check_ar_arabic_script(result, current_line, form, lemma, upos)

    return result


def _check_sense_numbers(
    result: LemmaCheckResult,
    line: int,
    form: str,
    lemma: str,
    upos: str,
) -> None:
    if SENSE_NUMBER_RE.search(lemma):
        result.errors.append(
            f"Line {line}: sense number in lemma '{lemma}' "
            f"(FORM='{form}', UPOS={upos}) — remove '-N' suffix"
        )


def _check_punct_identity(
    result: LemmaCheckResult,
    line: int,
    form: str,
    lemma: str,
    upos: str,
) -> None:
    if upos == "PUNCT" and form != lemma:
        result.errors.append(
            f"Line {line}: PUNCT lemma '{lemma}' != form '{form}' — "
            f"punctuation lemma must equal form"
        )


def _check_propn_capital(
    result: LemmaCheckResult,
    line: int,
    form: str,
    lemma: str,
    upos: str,
    lang: str,
) -> None:
    if upos != "PROPN":
        return
    if not lemma:
        return
    if lang == "zh" or lang == "ar":
        return
    if not lemma[0].isupper():
        result.errors.append(
            f"Line {line}: PROPN lemma '{lemma}' must start with uppercase (FORM='{form}')"
        )


def _check_de_noun_capital(
    result: LemmaCheckResult,
    line: int,
    form: str,
    lemma: str,
    upos: str,
) -> None:
    if upos != "NOUN" or not lemma:
        return
    if not lemma[0].isupper():
        result.errors.append(
            f"Line {line}: German NOUN lemma '{lemma}' must be capitalized (FORM='{form}')"
        )


def _check_de_verb_infinitive(
    result: LemmaCheckResult,
    line: int,
    form: str,
    lemma: str,
    upos: str,
) -> None:
    if upos != "VERB" or not lemma:
        return
    if not (lemma.endswith("en") or lemma.endswith("n")):
        result.errors.append(
            f"Line {line}: German VERB lemma '{lemma}' must be infinitive "
            f"(ending -en or -n, FORM='{form}')"
        )


_EN_VERB_SUFFIXES = ("s", "ed", "ing")


def _check_en_verb_base(
    result: LemmaCheckResult,
    line: int,
    form: str,
    lemma: str,
    upos: str,
) -> None:
    if upos != "VERB" or not lemma:
        return
    for suffix in _EN_VERB_SUFFIXES:
        if lemma.endswith(suffix) and len(lemma) > len(suffix):
            result.errors.append(
                f"Line {line}: English VERB lemma '{lemma}' has '-{suffix}' suffix — "
                f"must be base form (FORM='{form}')"
            )
            return


_ES_VERB_ENDINGS = ("ar", "er", "ir", "se")


def _check_es_verb_infinitive(
    result: LemmaCheckResult,
    line: int,
    form: str,
    lemma: str,
    upos: str,
) -> None:
    if upos != "VERB" or not lemma:
        return
    if not any(lemma.endswith(ending) for ending in _ES_VERB_ENDINGS):
        result.errors.append(
            f"Line {line}: Spanish VERB lemma '{lemma}' must be infinitive "
            f"(ending -ar/-er/-ir/-se, FORM='{form}')"
        )


def _check_zh_lemma_equals_form(
    result: LemmaCheckResult,
    line: int,
    form: str,
    lemma: str,
    upos: str,
) -> None:
    if upos == "PUNCT":
        return
    if form != lemma:
        result.errors.append(
            f"Line {line}: Chinese lemma '{lemma}' must equal form '{form}' (UPOS={upos})"
        )


def _check_ar_arabic_script(
    result: LemmaCheckResult,
    line: int,
    form: str,
    lemma: str,
    upos: str,
) -> None:
    if upos == "PUNCT":
        return
    if not lemma:
        return
    if not ARABIC_RANGE.search(lemma):
        result.errors.append(
            f"Line {line}: Arabic lemma '{lemma}' not in Arabic script (FORM='{form}', UPOS={upos})"
        )
