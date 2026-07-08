"""Tests for the ByT5 Arabic lemma dataset builder.

Per AGENTS.md §5 (TDD): written BEFORE the implementation. These tests pin
the contract that `make_byt5_dataset` must satisfy:
- lemma vocabulary built from train, excluding PROPN, with special tokens.
- byte-level encoding of words with byte→word span mapping for pooling.
- PROPN words masked to PAD_LABEL (-100).
- per-sentence output: input_ids, word_byte_spans, labels, words, lemmas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lemmatizer.data.byt5_dataset import (
    BYTE_ID_OFFSET,
    PAD_LABEL,
    build_lemma_vocab,
    encode_sentence,
)


@pytest.fixture
def mini_conllu(tmp_path: Path) -> Path:
    """A 2-sentence Arabic-ish CoNLL-U file for testing."""
    content = """# sent_id = test-1
# text = كتب الولد
1\tكَتَبَ\tكَتَبَ\tVERB\t_\t_\t_\t_\t_\t_
2\tالْوَلَدُ\tوَلَد\tNOUN\t_\t_\t_\t_\t_\t_

# sent_id = test-2
# text = ذهب أحمد
1\tذَهَبَ\tذَهَبَ\tVERB\t_\t_\t_\t_\t_\t_
2\tأَحْمَد\t_\tPROPN\t_\t_\t_\t_\t_\t_

"""
    p = tmp_path / "train.conllu"
    p.write_text(content, encoding="utf-8")
    return p


def test_build_lemma_vocab_excludes_propn(mini_conllu: Path):
    vocab, id2lemma = build_lemma_vocab([str(mini_conllu)])
    # 3 lemmas: كَتَبَ, وَلَد, ذَهَبَ (the PROPN's lemma is "_" → excluded;
    # also PROPN tokens excluded). Plus special tokens <PAD>, <UNK>, <IDENTITY>.
    assert "<PAD>" in vocab
    assert "<UNK>" in vocab
    assert "<IDENTITY>" in vocab
    assert "كَتَبَ" in vocab
    assert "وَلَد" in vocab
    assert "ذَهَبَ" in vocab
    # No "_" placeholder lemma (PROPN with lemma="_" must be skipped).
    assert "_" not in vocab
    # id2lemma is the inverse.
    assert id2lemma[vocab["كَتَبَ"]] == "كَتَبَ"


def test_build_lemma_vocab_returns_inverse(mini_conllu: Path):
    vocab, id2lemma = build_lemma_vocab([str(mini_conllu)])
    for lemma, idx in vocab.items():
        assert id2lemma[idx] == lemma


def test_encode_sentence_produces_byte_spans_and_labels(mini_conllu: Path):
    vocab, _ = build_lemma_vocab([str(mini_conllu)])
    words = ["كَتَبَ", "الْوَلَدُ"]
    lemmas = ["كَتَبَ", "وَلَد"]
    upos_tags = ["VERB", "NOUN"]

    encoded = encode_sentence(words, lemmas, upos_tags, vocab)

    # input_ids: 1D byte array, length = sum of byte lengths + separators.
    assert encoded["input_ids"].ndim == 1
    assert encoded["input_ids"].shape[0] > 0

    # word_byte_spans: (N_words, 2), one row per word.
    assert encoded["word_byte_spans"].shape[1] == 2
    assert encoded["word_byte_spans"].shape[0] == 2  # 2 words

    # labels: one per word, first is lemma id, second is lemma id.
    assert encoded["labels"].shape[0] == 2
    assert encoded["labels"][0] == vocab["كَتَبَ"]
    assert encoded["labels"][1] == vocab["وَلَد"]

    # Each span must be non-empty and within bounds.
    for start, end in encoded["word_byte_spans"]:
        assert 0 <= start < end <= encoded["input_ids"].shape[0]


def test_encode_sentence_masks_propn_to_pad_label(mini_conllu: Path):
    vocab, _ = build_lemma_vocab([str(mini_conllu)])
    words = ["ذَهَبَ", "أَحْمَد"]
    lemmas = ["ذَهَبَ", "_"]  # PROPN lemma is "_"
    upos_tags = ["VERB", "PROPN"]

    encoded = encode_sentence(words, lemmas, upos_tags, vocab)

    # VERB keeps its lemma label.
    assert encoded["labels"][0] == vocab["ذَهَبَ"]
    # PROPN is masked to PAD_LABEL.
    assert encoded["labels"][1] == PAD_LABEL


def test_encode_sentence_unknown_lemma_gets_unk(mini_conllu: Path):
    vocab, _ = build_lemma_vocab([str(mini_conllu)])
    words = ["كَتَبَ", "كلمةغيرموجودة"]
    lemmas = ["كَتَبَ", "lemma_not_in_vocab"]
    upos_tags = ["VERB", "NOUN"]

    encoded = encode_sentence(words, lemmas, upos_tags, vocab)

    assert encoded["labels"][0] == vocab["كَتَبَ"]
    assert encoded["labels"][1] == vocab["<UNK>"]


def test_encode_sentence_byte_ids_match_byt5_vocabulary(mini_conllu: Path):
    """ByT5 SentencePiece maps byte value b -> id b + 3 (ids 0/1/2 are
    PAD/EOS/UNK). Regression for the wrong `b if b > 1 else b + 256` mapping.
    """
    vocab, _ = build_lemma_vocab([str(mini_conllu)])
    # ASCII word "AB" so byte values are deterministic: 0x41, 0x42.
    words = ["AB"]
    lemmas = ["AB"]
    upos_tags = ["NOUN"]

    encoded = encode_sentence(words, lemmas, upos_tags, vocab)
    ids = list(np.array(encoded["input_ids"]))

    # Layout: EOS, 'A', 'B', space, EOS.
    assert ids[0] == 1  # BYT5_EOS
    assert ids[1] == ord("A") + BYTE_ID_OFFSET
    assert ids[2] == ord("B") + BYTE_ID_OFFSET
    assert ids[3] == ord(" ") + BYTE_ID_OFFSET
    assert ids[-1] == 1  # trailing EOS
    # Every non-special id must be in the byte range [3, 258].
    for i in ids[1:-1]:
        assert 3 <= i <= 258


def test_encode_sentence_literal_zero_byte_maps_to_offset(mini_conllu: Path):
    """A word containing a literal 0x00 byte must not collide with PAD(0)
    or EOS(1); it must map to 0 + BYTE_ID_OFFSET = 3."""
    vocab, _ = build_lemma_vocab([str(mini_conllu)])
    word = "a\x00b"
    encoded = encode_sentence([word], [word], ["NOUN"], vocab)
    ids = list(np.array(encoded["input_ids"]))
    # The 0x00 byte sits between 'a' (0x61) and 'b' (0x62).
    assert 0 not in ids[1:-1]
    assert 1 not in ids[1:-1]
    assert BYTE_ID_OFFSET in ids[1:-1]  # 0x00 -> 3
    assert ord("a") + BYTE_ID_OFFSET in ids[1:-1]
    assert ord("b") + BYTE_ID_OFFSET in ids[1:-1]


# --- UPOS encoding (added per OCR finding: the upos2id paths had no coverage) ---


def test_encode_sentence_upos_known_tags_mapped_to_ids(mini_conllu: Path):
    """Known UPOS tags map to their upos2id ids (PROPN included)."""
    vocab, _ = build_lemma_vocab([str(mini_conllu)])
    upos2id = {"VERB": 5, "NOUN": 7, "PROPN": 9}
    words = ["كَتَبَ", "الْوَلَدُ", "أَحْمَد"]
    lemmas = ["كَتَبَ", "وَلَد", "_"]
    upos_tags = ["VERB", "NOUN", "PROPN"]

    encoded = encode_sentence(words, lemmas, upos_tags, vocab, upos2id=upos2id)

    upos_labels = list(np.array(encoded["upos_labels"]))
    assert upos_labels == [5, 7, 9]


def test_encode_sentence_unknown_upos_masked_to_pad_label(mini_conllu: Path):
    """Unknown UPOS tags are masked to PAD_LABEL in upos_labels."""
    vocab, _ = build_lemma_vocab([str(mini_conllu)])
    upos2id = {"VERB": 5, "NOUN": 7}  # no PROPN
    words = ["كَتَبَ", "أَحْمَد"]
    lemmas = ["كَتَبَ", "أَحْمَد"]
    upos_tags = ["VERB", "PROPN"]

    encoded = encode_sentence(
        words, lemmas, upos_tags, vocab, upos2id=upos2id, unknown_upos_seen=set()
    )

    upos_labels = list(np.array(encoded["upos_labels"]))
    assert upos_labels == [5, PAD_LABEL]


def test_encode_sentence_upos_labels_present_when_upos2id_given(mini_conllu: Path):
    """`upos_labels` key is present in output when upos2id is provided."""
    vocab, _ = build_lemma_vocab([str(mini_conllu)])
    encoded = encode_sentence(["كَتَبَ"], ["كَتَبَ"], ["VERB"], vocab, upos2id={"VERB": 0})
    assert "upos_labels" in encoded


def test_encode_sentence_upos_labels_absent_when_upos2id_none(mini_conllu: Path):
    """`upos_labels` key is absent when upos2id is None (lemma-only build)."""
    vocab, _ = build_lemma_vocab([str(mini_conllu)])
    encoded = encode_sentence(["كَتَبَ"], ["كَتَبَ"], ["VERB"], vocab, upos2id=None)
    assert "upos_labels" not in encoded


def test_encode_sentence_propn_included_in_upos_labels(mini_conllu: Path):
    """PROPN words are masked in `labels` but KEPT in `upos_labels` (UPOS
    head must learn all tags, unlike the lemma head which skips PROPN)."""
    vocab, _ = build_lemma_vocab([str(mini_conllu)])
    upos2id = {"VERB": 5, "PROPN": 9}
    words = ["كَتَبَ", "أَحْمَد"]
    lemmas = ["كَتَبَ", "_"]  # PROPN lemma "_" → masked in labels
    upos_tags = ["VERB", "PROPN"]

    encoded = encode_sentence(words, lemmas, upos_tags, vocab, upos2id=upos2id)

    labels = list(np.array(encoded["labels"]))
    upos_labels = list(np.array(encoded["upos_labels"]))
    # Lemma label for PROPN is masked...
    assert labels[1] == PAD_LABEL
    # ...but the UPOS label is retained (not masked).
    assert upos_labels[1] == 9
