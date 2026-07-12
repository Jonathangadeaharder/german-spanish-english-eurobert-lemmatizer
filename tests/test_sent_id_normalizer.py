"""Tests for sent_id normalizer.

TDD: written before implementation. Rewrites inconsistent sent_id
formats to canonical {lang}_{level}_{split}_{NNN} format.
"""

from __future__ import annotations

from lemmatizer.data.sent_id_normalizer import normalize_text, parse_sent_id

CANONICAL = """# sent_id = de_a1_train_001
# text = Das ist.
1\tDas\tder\tPRON\t_\t_\t_\t_\t_\t_
2\tist\tsein\tAUX\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""


def test_already_canonical_passes_through() -> None:
    result = normalize_text(CANONICAL, lang="de", level="a1", split="train")
    assert "# sent_id = de_a1_train_001" in result


def test_dashed_format_normalized() -> None:
    text = CANONICAL.replace("de_a1_train_001", "a2-train-001")
    result = normalize_text(text, lang="de", level="a2", split="train")
    assert "# sent_id = de_a2_train_001" in result


def test_wrong_order_normalized() -> None:
    text = CANONICAL.replace("de_a1_train_001", "de_train_b1_001")
    result = normalize_text(text, lang="de", level="b1", split="train")
    assert "# sent_id = de_b1_train_001" in result


def test_prefix_stripped_and_normalized() -> None:
    text = CANONICAL.replace("de_a1_train_001", "c1_train_001")
    result = normalize_text(text, lang="de", level="c1", split="train")
    assert "# sent_id = de_c1_train_001" in result


def test_long_prefix_normalized() -> None:
    text = CANONICAL.replace("de_a1_train_001", "handcraft-de-c2-train-1")
    result = normalize_text(text, lang="de", level="c2", split="train")
    assert "# sent_id = de_c2_train_001" in result


def test_unparseable_renumbered_sequentially() -> None:
    text = """# sent_id = garbage_id
# text = Das ist.
1\tDas\tder\tPRON\t_\t_\t_\t_\t_\t_
2\tist\tsein\tAUX\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

# sent_id = also_bad
# text = Ich laufe.
1\tIch\tich\tPRON\t_\t_\t_\t_\t_\t_
2\tlaufe\tlaufen\tVERB\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = normalize_text(text, lang="de", level="a1", split="train")
    assert "# sent_id = de_a1_train_001" in result
    assert "# sent_id = de_a1_train_002" in result


def test_renumbered_sequentially() -> None:
    text = """# sent_id = de_a1_train_005
# text = Das ist.
1\tDas\tder\tPRON\t_\t_\t_\t_\t_\t_
2\tist\tsein\tAUX\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

# sent_id = de_a1_train_003
# text = Ich laufe.
1\tIch\tich\tPRON\t_\t_\t_\t_\t_\t_
2\tlaufe\tlaufen\tVERB\t_\t_\t_\t_\t_\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_

"""
    result = normalize_text(text, lang="de", level="a1", split="train")
    assert "# sent_id = de_a1_train_001" in result
    assert "# sent_id = de_a1_train_002" in result


def test_parse_sent_id_canonical() -> None:
    parsed = parse_sent_id("de_a1_train_001")
    assert parsed is not None
    assert parsed == ("de", "a1", "train", 1)


def test_parse_sent_id_garbage_returns_none() -> None:
    assert parse_sent_id("garbage_id") is None
