"""Tests for duplicate sentence detector.

TDD: written before implementation. Finds exact and near-duplicate
sentences within and across train/val CoNLL-U files.
"""

from __future__ import annotations

from lemmatizer.data.duplicate_detector import (
    DuplicateResult,
    check_cross_file,
    check_text,
)

SENT_A = """# sent_id = de_a1_train_001
# text = Das ist meine Mutter.
1\tDas\tder\tPRON\t_\t_\t_\t_\t_\t_
2\tist\tsein\tAUX\t_\t_\t_\t_\t_\t_
3\tmeine\tmein\tDET\t_\t_\t_\t_\t_\t_
4\tMutter\tMutter\tNOUN\t_\t_\t_\t_\t_\t_
5\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""

SENT_B = """# sent_id = de_a1_train_002
# text = Ich trinke Wasser.
1\tIch\tich\tPRON\t_\t_\t_\t_\t_\t_
2\ttrinke\ttrinken\tVERB\t_\t_\t_\t_\t_\t_
3\tWasser\tWasser\tNOUN\t_\t_\t_\t_\t_\t_
4\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""


def test_no_duplicates_passes() -> None:
    result = check_text(SENT_A + SENT_B)
    assert result.passed, f"Expected pass, got: {result.errors}"
    assert len(result.duplicates) == 0


def test_exact_duplicate_within_file_fails() -> None:
    text = SENT_A + SENT_A
    result = check_text(text)
    assert not result.passed
    assert len(result.duplicates) >= 1
    assert any("Das ist meine Mutter" in d for d in result.duplicates)


def test_case_insensitive_duplicate_fails() -> None:
    sent_b_upper = SENT_A.replace("Das ist meine Mutter.", "DAS IST MEINE MUTTER.")
    text = SENT_A + sent_b_upper
    result = check_text(text)
    assert not result.passed
    assert len(result.duplicates) >= 1


def test_whitespace_normalized_duplicate_fails() -> None:
    sent_extra_space = SENT_A.replace("Das ist meine Mutter.", "Das  ist  meine  Mutter.")
    text = SENT_A + sent_extra_space
    result = check_text(text)
    assert not result.passed


def test_cross_file_exact_duplicate_fails() -> None:
    result = check_cross_file(train_text=SENT_A, val_text=SENT_A)
    assert not result.passed
    assert any("Das ist meine Mutter" in d for d in result.duplicates)


def test_near_duplicate_warns() -> None:
    sent_near = SENT_A.replace("Das ist meine Mutter.", "Das ist mein Vater.")
    sent_near = sent_near.replace("meine\tmein\tDET", "mein\tmein\tDET")
    sent_near = sent_near.replace("Mutter\tMutter\tNOUN", "Vater\tVater\tNOUN")
    result = check_cross_file(train_text=SENT_A, val_text=sent_near)
    assert len(result.warnings) > 0
    assert any("near-duplicate" in w.lower() or "similar" in w.lower() for w in result.warnings)


def test_different_sentences_no_warning() -> None:
    result = check_cross_file(train_text=SENT_A, val_text=SENT_B)
    assert result.passed
    assert len(result.warnings) == 0


def test_duplicate_result_dataclass() -> None:
    r = DuplicateResult()
    assert r.duplicates == []
    assert r.warnings == []
    assert r.passed
