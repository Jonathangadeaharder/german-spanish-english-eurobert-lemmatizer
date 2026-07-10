from lemmatizer.eval.cefr_eval import (
    GATE_ACCURACY,
    SKIP_POS_FOR_LEMMA,
    CefrVocabEntry,
    _find_term_index,
    _first_token_for_word,
    _resolve_pos_column,
)


def test_skip_pos_set_excludes_proper_nouns_and_punctuation():
    assert "PROPN" in SKIP_POS_FOR_LEMMA
    assert "PUNCT" in SKIP_POS_FOR_LEMMA
    assert "SYM" in SKIP_POS_FOR_LEMMA
    assert "X" in SKIP_POS_FOR_LEMMA
    assert "NUM" in SKIP_POS_FOR_LEMMA
    assert "NOUN" not in SKIP_POS_FOR_LEMMA
    assert "VERB" not in SKIP_POS_FOR_LEMMA


def test_gate_accuracy_is_90_percent():
    assert GATE_ACCURACY == 0.90


def test_resolve_pos_column_finds_pos_header_case_insensitive():
    assert _resolve_pos_column(["German_Lemma", "POS"]) == "POS"
    assert _resolve_pos_column(["Lemma", "pos"]) == "pos"
    assert _resolve_pos_column(["Lemma", "Pos"]) == "Pos"


def test_resolve_pos_column_returns_none_when_absent():
    assert _resolve_pos_column(["Lemma", "English"]) is None
    assert _resolve_pos_column(None) is None


def test_find_term_index_exact_match():
    assert _find_term_index(["Das", "Haus", "ist"], "Haus") == 1


def test_find_term_index_case_insensitive():
    assert _find_term_index(["das", "HAUS", "ist"], "Haus") == 1


def test_find_term_index_not_found():
    assert _find_term_index(["Der", "Hund"], "Haus") is None


def test_find_term_index_empty_words():
    assert _find_term_index([], "Haus") is None


def test_first_token_for_word_no_lang_token_offset_zero():
    word_ids = [0, 1, 1, 2, None]
    assert _first_token_for_word(word_ids, first_word_offset=0, term_idx=1) == 1


def test_first_token_for_word_with_lang_token_offset_one():
    # Lang token is word 0; first real word is word 1. term_idx=1 → word_id=2.
    word_ids = [0, 1, 2, 2, 3]
    assert _first_token_for_word(word_ids, first_word_offset=1, term_idx=1) == 2


def test_first_token_for_word_missing_returns_none():
    assert _first_token_for_word([None, 0], first_word_offset=1, term_idx=5) is None


def test_cefr_vocab_entry_carries_level_term_pos():
    entry = CefrVocabEntry(level="A1", term="Haus", pos="NOUN")
    assert entry.level == "A1"
    assert entry.term == "Haus"
    assert entry.pos == "NOUN"
