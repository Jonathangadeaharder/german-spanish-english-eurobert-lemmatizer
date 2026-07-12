"""Tests for lemma consistency checker.

TDD: written before implementation. Validates lemma plausibility
rules across languages (German, English, Spanish, Chinese, Arabic).
"""

from __future__ import annotations

from lemmatizer.data.lemma_checker import LemmaCheckResult, check_text

VALID_DE = """# sent_id = de_a1_train_001
# text = Das ist meine Mutter.
1\tDas\tder\tPRON\t_\t_\t_\t_\t_\t_
2\tist\tsein\tAUX\t_\t_\t_\t_\t_\t_
3\tmeine\tmein\tDET\t_\t_\t_\t_\t_\t_
4\tMutter\tMutter\tNOUN\t_\t_\t_\t_\t_\t_
5\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""


def test_valid_german_passes() -> None:
    result = check_text(VALID_DE, lang="de")
    assert result.passed, f"Expected pass, got errors: {result.errors}"


def test_punct_lemma_must_equal_form() -> None:
    text = VALID_DE.replace("5\t.\t.\tPUNCT", "5\t.\tDOT\tPUNCT")
    result = check_text(text, lang="de")
    assert not result.passed
    assert any("PUNCT" in e and "lemma" in e.lower() for e in result.errors)


def test_german_noun_lemma_must_be_capitalized() -> None:
    text = VALID_DE.replace("4\tMutter\tMutter\tNOUN", "4\tMutter\tmutter\tNOUN")
    result = check_text(text, lang="de")
    assert not result.passed
    assert any("NOUN" in e and "capital" in e.lower() for e in result.errors)


def test_german_verb_lemma_must_end_in_en_or_n() -> None:
    text = """# sent_id = de_a1_train_001
# text = Ich laufe.
1\tIch\tich\tPRON\t_\t_\t_\t_\t_\t_
2\tlaufe\tlauf\tVERB\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = check_text(text, lang="de")
    assert not result.passed
    assert any("VERB" in e and "infinitive" in e.lower() for e in result.errors)


def test_german_verb_lemma_en_passes() -> None:
    text = """# sent_id = de_a1_train_001
# text = Ich laufe.
1\tIch\tich\tPRON\t_\t_\t_\t_\t_\t_
2\tlaufe\tlaufen\tVERB\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = check_text(text, lang="de")
    assert result.passed, f"Expected pass, got errors: {result.errors}"


def test_same_form_upos_same_lemma() -> None:
    text = """# sent_id = de_a1_train_001
# text = Das ist gut.
1\tDas\tder\tDET\t_\t_\t_\t_\t_\t_
2\tist\tsein\tAUX\t_\t_\t_\t_\t_\t_
3\tgut\tgut\tADJ\t_\t_\t_\t_\t_\t_
4\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

# sent_id = de_a1_train_002
# text = Das ist schoen.
1\tDas\tdie\tDET\t_\t_\t_\t_\t_\t_
2\tist\tsein\tAUX\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = check_text(text, lang="de")
    assert not result.passed
    assert any("inconsistent" in e.lower() or "same form" in e.lower() for e in result.errors)


def test_propn_lemma_capitalized() -> None:
    text = """# sent_id = de_a1_train_001
# text = Berlin ist schoen.
1\tBerlin\tberlin\tPROPN\t_\t_\t_\t_\t_\t_
2\tist\tsein\tAUX\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = check_text(text, lang="de")
    assert not result.passed
    assert any("PROPN" in e and "uppercase" in e.lower() for e in result.errors)


def test_no_sense_numbers_in_lemma() -> None:
    text = """# sent_id = en_a1_train_001
# text = I can.
1\tI\tI\tPRON\t_\t_\t_\t_\t_\t_
2\tcan\tcan-1\tAUX\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = check_text(text, lang="en")
    assert not result.passed
    assert any("sense" in e.lower() or "number" in e.lower() for e in result.errors)


def test_english_verb_lemma_no_suffixes() -> None:
    text = """# sent_id = en_a1_train_001
# text = He jumps.
1\tHe\the\tPRON\t_\t_\t_\t_\t_\t_
2\tjumps\tjumps\tVERB\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = check_text(text, lang="en")
    assert not result.passed
    assert any("VERB" in e and ("-s" in e or "-ed" in e or "-ing" in e) for e in result.errors)


def test_english_verb_lemma_base_form_passes() -> None:
    text = """# sent_id = en_a1_train_001
# text = He jumps.
1\tHe\the\tPRON\t_\t_\t_\t_\t_\t_
2\tjumps\tjump\tVERB\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = check_text(text, lang="en")
    assert result.passed, f"Expected pass, got errors: {result.errors}"


def test_spanish_verb_lemma_correct_endings() -> None:
    text = """# sent_id = es_a1_train_001
# text = Yo hablo.
1\tYo\tyo\tPRON\t_\t_\t_\t_\t_\t_
2\thablo\thabl\tVERB\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = check_text(text, lang="es")
    assert not result.passed
    assert any("VERB" in e and ("ar" in e or "er" in e or "ir" in e) for e in result.errors)


def test_spanish_verb_lemma_ar_passes() -> None:
    text = """# sent_id = es_a1_train_001
# text = Yo hablo.
1\tYo\tyo\tPRON\t_\t_\t_\t_\t_\t_
2\thablo\thablar\tVERB\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = check_text(text, lang="es")
    assert result.passed, f"Expected pass, got errors: {result.errors}"


def test_chinese_lemma_equals_form() -> None:
    text = """# sent_id = zh_a1_train_001
# text = \u6211\u7231\u4f60\u3002
1\t\u6211\t\u6211\tPRON\t_\t_\t_\t_\t_\t_
2\t\u7231\t\u4e0d\tVERB\t_\t_\t_\t_\t_\t_
3\t\u4f60\t\u4f60\tPRON\t_\t_\t_\t_\t_\t_
4\t\u3002\t\u3002\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = check_text(text, lang="zh")
    assert not result.passed
    assert any("lemma" in e.lower() and "form" in e.lower() for e in result.errors)


def test_chinese_lemma_equals_form_passes() -> None:
    text = """# sent_id = zh_a1_train_001
# text = \u6211\u7231\u4f60\u3002
1\t\u6211\t\u6211\tPRON\t_\t_\t_\t_\t_\t_
2\t\u7231\t\u7231\tVERB\t_\t_\t_\t_\t_\t_
3\t\u4f60\t\u4f60\tPRON\t_\t_\t_\t_\t_\t_
4\t\u3002\t\u3002\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = check_text(text, lang="zh")
    assert result.passed, f"Expected pass, got errors: {result.errors}"


def test_arabic_lemma_in_arabic_script() -> None:
    text = """# sent_id = ar_a1_train_001
# text = test.
1\t\u0627\u0644\u0633\u0644\u0627\u0645\thello\tNOUN\t_\t_\t_\t_\t_\t_
2\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = check_text(text, lang="ar")
    assert not result.passed
    assert any("Arabic" in e or "script" in e.lower() for e in result.errors)


def test_arabic_lemma_in_arabic_script_passes() -> None:
    text = """# sent_id = ar_a1_train_001
# text = test.
1\t\u0627\u0644\u0633\u0644\u0627\u0645\t\u0633\u0644\u0627\u0645\tNOUN\t_\t_\t_\t_\t_\t_
2\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = check_text(text, lang="ar")
    assert result.passed, f"Expected pass, got errors: {result.errors}"


def test_lemma_check_result_dataclass() -> None:
    r = LemmaCheckResult()
    assert r.errors == []
    assert r.warnings == []
    assert r.passed
