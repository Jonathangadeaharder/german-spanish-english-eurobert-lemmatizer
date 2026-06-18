import csv

from vocab_inventory import build_inventory, canonical_terms, lookup_level


def write_vocab_file(path, lemma_col, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[lemma_col, "T1", "T2"])
        writer.writeheader()
        writer.writerows(rows)


VALUES = {
    "A1": "alpha",
    "A2": "beta",
    "B1": "gamma",
    "B2": "delta",
    "C1": "epsilon",
}


def test_build_inventory_marks_basic_single_word_entries(tmp_path):
    for dirname, lemma_col in [
        ("english", "English_Lemma"),
        ("german", "German_Lemma"),
        ("spanish", "Spanish_Lemma"),
    ]:
        for level in ["A1", "A2", "B1", "B2", "C1"]:
            write_vocab_file(
                tmp_path / dirname / f"{level}.csv",
                lemma_col,
                [{lemma_col: f"{dirname}_{VALUES[level]}", "T1": "x", "T2": "y"}],
            )

    inventory = build_inventory(tmp_path)

    assert "english_alpha" in canonical_terms(inventory, "en")
    assert lookup_level(inventory, "de", "german_delta") == "B2"


def test_build_inventory_rejects_multi_word_entries(tmp_path):
    for dirname, lemma_col in [
        ("english", "English_Lemma"),
        ("german", "German_Lemma"),
        ("spanish", "Spanish_Lemma"),
    ]:
        for level in ["A1", "A2", "B1", "B2", "C1"]:
            value = (
                "two words"
                if dirname == "english" and level == "A1"
                else f"{dirname}_{VALUES[level]}"
            )
            write_vocab_file(
                tmp_path / dirname / f"{level}.csv",
                lemma_col,
                [{lemma_col: value, "T1": "x", "T2": "y"}],
            )

    inventory = build_inventory(tmp_path)

    assert "two words" not in canonical_terms(inventory, "en")
    assert lookup_level(inventory, "en", "two words") is None
