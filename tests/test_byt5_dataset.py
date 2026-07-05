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

import pytest

from lemmatizer.data.byt5_dataset import (
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
