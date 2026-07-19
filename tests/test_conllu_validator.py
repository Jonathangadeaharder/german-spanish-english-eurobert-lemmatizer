"""Tests for CoNLL-U format validator.

TDD: written before implementation. Validates structural correctness
of CoNLL-U files used for lemmatizer training data.
"""

from __future__ import annotations

from lemmatizer.data.conllu_validator import ValidationResult, validate_text

VALID_SENTENCE = """# sent_id = de_a1_train_001
# text = Das ist meine Mutter.
1\tDas\tder\tPRON\t_\t_\t_\t_\t_\t_
2\tist\tsein\tAUX\t_\t_\t_\t_\t_\t_
3\tmeine\tmein\tDET\t_\t_\t_\t_\t_\t_
4\tMutter\tMutter\tNOUN\t_\t_\t_\t_\t_\t_
5\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""


def test_valid_sentence_passes() -> None:
    result = validate_text(VALID_SENTENCE)
    assert result.passed, f"Expected pass, got errors: {result.errors}"
    assert result.sentence_count == 1
    assert result.token_count == 5


def test_two_valid_sentences_pass() -> None:
    text = VALID_SENTENCE + VALID_SENTENCE.replace("001", "002")
    result = validate_text(text)
    assert result.passed, f"Expected pass, got errors: {result.errors}"
    assert result.sentence_count == 2
    assert result.token_count == 10


def test_wrong_field_count_fails() -> None:
    text = """# sent_id = de_a1_train_001
# text = Das ist.
1\tDas\tder\tPRON\t_\t_\t_\t_
2\tist\tsein\tAUX\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = validate_text(text)
    assert not result.passed
    assert any(
        "field" in e.lower() or "column" in e.lower() or "tab" in e.lower() for e in result.errors
    )


def test_invalid_upos_fails() -> None:
    text = VALID_SENTENCE.replace("\tPRON\t", "\tNOUNZ\t")
    result = validate_text(text)
    assert not result.passed
    assert any("NOUNZ" in e or "UPOS" in e for e in result.errors)


def test_missing_sent_id_fails() -> None:
    text = """# text = Das ist.
1\tDas\tder\tPRON\t_\t_\t_\t_\t_\t_
2\tist\tsein\tAUX\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = validate_text(text)
    assert not result.passed
    assert any("sent_id" in e for e in result.errors)


def test_missing_text_fails() -> None:
    text = """# sent_id = de_a1_train_001
1\tDas\tder\tPRON\t_\t_\t_\t_\t_\t_
2\tist\tsein\tAUX\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = validate_text(text)
    assert not result.passed
    assert any("text" in e for e in result.errors)


def test_missing_trailing_blank_line_fails() -> None:
    text = """# sent_id = de_a1_train_001
# text = Das ist.
1\tDas\tder\tPRON\t_\t_\t_\t_\t_\t_
2\tist\tsein\tAUX\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_"""
    result = validate_text(text)
    assert not result.passed
    assert any("blank" in e.lower() or "trailing" in e.lower() for e in result.errors)


def test_duplicate_sent_id_fails() -> None:
    text = VALID_SENTENCE + VALID_SENTENCE
    result = validate_text(text)
    assert not result.passed
    assert any("duplicate" in e.lower() or "sent_id" in e for e in result.errors)


def test_text_mismatch_fails() -> None:
    text = """# sent_id = de_a1_train_001
# text = This is wrong text.
1\tDas\tder\tPRON\t_\t_\t_\t_\t_\t_
2\tist\tsein\tAUX\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = validate_text(text)
    assert not result.passed
    assert any("text" in e.lower() and "mismatch" in e.lower() for e in result.errors)


def test_empty_lemma_fails() -> None:
    text = """# sent_id = de_a1_train_001
# text = Das ist.
1\tDas\t\tPRON\t_\t_\t_\t_\t_\t_
2\tist\tsein\tAUX\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = validate_text(text)
    assert not result.passed
    assert any("lemma" in e.lower() and "empty" in e.lower() for e in result.errors)


def test_non_tab_separated_fails() -> None:
    text = """# sent_id = de_a1_train_001
# text = Das ist.
1 Das der PRON _ _ _ _ _ _
2 ist sein AUX _ _ _ _ _ _
3 . . PUNCT _ _ _ _ _ _

"""
    result = validate_text(text)
    assert not result.passed


def test_crlf_line_endings_fails() -> None:
    text = (
        "# sent_id = de_a1_train_001\r\n"
        "# text = Das ist.\r\n"
        "1\tDas\tder\tPRON\t_\t_\t_\t_\t_\t_\r\n"
        "2\tist\tsein\tAUX\t_\t_\t_\t_\t_\t_\r\n"
        "3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_\r\n"
        "\r\n"
    )
    result = validate_text(text)
    assert not result.passed
    assert any("CRLF" in e or "line ending" in e.lower() or "\\r" in e for e in result.errors)


def test_non_nfc_unicode_warns() -> None:
    # Combining characters: U + combining diaeresis instead of precomposed Ü
    text = (
        "# sent_id = de_a1_train_001\n"
        "# text = \u0055\u0308ber.\n"
        "1\t\u0055\u0308ber\t\u0055\u0308ber\tNOUN\t_\t_\t_\t_\t_\t_\n"
        "2\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_\n"
        "\n"
    )
    result = validate_text(text)
    assert len(result.warnings) > 0
    assert any("NFC" in w or "unicode" in w.lower() for w in result.warnings)


def test_text_matches_mwt_split_with_no_space_between_subwords() -> None:
    """Stanza MWT expansion (e.g. French "m'appelle" -> "m'" + "appelle") has
    no space between the sub-words; MISC is unused in this dataset, so the
    validator must not require a literal space there."""
    text = (
        "# sent_id = fr_a1_train_001\n"
        "# text = Je m'appelle Marie.\n"
        "1\tJe\tje\tPRON\t_\t_\t_\t_\t_\t_\n"
        "2\tm'\tje\tPRON\t_\t_\t_\t_\t_\t_\n"
        "3\tappelle\tappeler\tVERB\t_\t_\t_\t_\t_\t_\n"
        "4\tMarie\tMarie\tPROPN\t_\t_\t_\t_\t_\t_\n"
        "5\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_\n"
        "\n"
    )
    result = validate_text(text)
    assert result.passed, f"Expected pass, got errors: {result.errors}"


def test_text_matches_no_space_script_like_chinese() -> None:
    text = (
        "# sent_id = zh_a1_train_001\n"
        "# text = \u4e00\u5171\u591a\u5c11\u94b1\uff1f\n"
        "1\t\u4e00\u5171\t\u4e00\u5171\tADV\t_\t_\t_\t_\t_\t_\n"
        "2\t\u591a\u5c11\t\u591a\u5c11\tNUM\t_\t_\t_\t_\t_\t_\n"
        "3\t\u94b1\t\u94b1\tNOUN\t_\t_\t_\t_\t_\t_\n"
        "4\t\uff1f\t\uff1f\tPUNCT\t_\t_\t_\t_\t_\t_\n"
        "\n"
    )
    result = validate_text(text)
    assert result.passed, f"Expected pass, got errors: {result.errors}"


def test_validation_result_dataclass() -> None:
    vr = ValidationResult()
    assert vr.errors == []
    assert vr.warnings == []
    assert vr.sentence_count == 0
    assert vr.token_count == 0
    assert vr.passed
