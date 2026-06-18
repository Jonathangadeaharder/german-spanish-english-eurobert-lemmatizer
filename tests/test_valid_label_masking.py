import numpy as np

from evaluate import select_valid_label_id


def _row(values):
    return np.array(values, dtype=np.float32)


def test_picks_highest_logit_valid_label():
    # "Hunde" -> "Hund": prefix 4 ("Hund"), delete "e" -> P4|S0|De|I
    id2label = {"0": "IDENTITY", "1": "P4|S0|De|I"}
    logits = _row([0.1, 5.0])
    candidate_ids = [0, 1]
    label = select_valid_label_id(logits, candidate_ids, id2label, lang="de", word="Hunde")
    assert label == "P4|S0|De|I"
    from edit_trees import apply_edit_label

    assert apply_edit_label("Hunde", label) == "Hund"


def test_skips_invalid_label_falls_to_next():
    # Highest-logit label is structurally invalid for the word; selector must skip it.
    # "P3|S0|Dxyz|I" requires word[3:]=="xyz"; for "Hund" that fails -> skip.
    id2label = {"0": "IDENTITY", "1": "P3|S0|Dxyz|I"}
    logits = _row([0.1, 9.0])
    candidate_ids = [0, 1]
    label = select_valid_label_id(logits, candidate_ids, id2label, lang="de", word="Hund")
    assert label == "IDENTITY"


def test_identity_terminal_fallback_when_nothing_applies():
    id2label = {"0": "P9|S9|Dnope|I"}
    logits = _row([5.0])
    candidate_ids = [0]
    label = select_valid_label_id(logits, candidate_ids, id2label, lang="de", word="ab")
    assert label == "IDENTITY"


def test_strips_lang_prefix():
    id2label = {"0": "de::IDENTITY", "1": "de::P4|S0|De|I"}
    logits = _row([0.1, 9.0])
    candidate_ids = [0, 1]
    label = select_valid_label_id(logits, candidate_ids, id2label, lang="de", word="Hunde")
    assert label == "P4|S0|De|I"
