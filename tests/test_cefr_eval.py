from lemmatizer.eval.cefr_eval import (
    GATE_ACCURACY,
    SKIP_POS_FOR_LEMMA,
    CefrVocabEntry,
    _find_term_index,
    _first_token_for_word,
    _resolve_pos_column,
    main,
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


def test_resolve_pos_column_handles_whitespace():
    # VocabLevels exports may carry headers with leading/trailing spaces.
    assert _resolve_pos_column(["Lemma", " POS "]) == " POS "
    assert _resolve_pos_column(["Lemma", "pos "]) == "pos "


def test_find_term_index_exact_match():
    assert _find_term_index(["Das", "Haus", "ist"], "Haus") == 1


def test_find_term_index_case_insensitive():
    assert _find_term_index(["das", "HAUS", "ist"], "Haus") == 1


def test_find_term_index_not_found():
    assert _find_term_index(["Der", "Hund"], "Haus") is None


def test_find_term_index_empty_words():
    assert _find_term_index([], "Haus") is None


def test_find_term_index_strips_trailing_punctuation():
    # Sentence-final words from raw UD text carry punctuation attached.
    assert _find_term_index(["Ich", "gehe", "nach", "Hause."], "Hause") == 3
    assert _find_term_index(["Wort,"], "Wort") == 0
    assert _find_term_index(["Ende!"], "Ende") == 0


def test_find_term_index_strips_cjk_and_arabic_punctuation():
    # Non-Latin sentence-final punctuation must also be stripped.
    assert _find_term_index(["汉字。"], "汉字") == 0
    assert _find_term_index(["كلمة،"], "كلمة") == 0
    assert _find_term_index(["سؤال؟"], "سؤال") == 0


def test_first_token_for_word_no_lang_token_offset_zero():
    word_ids = [0, 1, 1, 2, None]
    assert _first_token_for_word(word_ids, first_word_offset=0, term_idx=1) == 1


def test_first_token_for_word_with_lang_token_offset_one():
    # Lang token is word 0; first real word is word 1. term_idx=1 → word_id=2.
    word_ids = [0, 1, 2, 2, 3]
    assert _first_token_for_word(word_ids, first_word_offset=1, term_idx=1) == 2


def test_first_token_for_word_missing_returns_none():
    assert _first_token_for_word([None, 0], first_word_offset=1, term_idx=5) is None


def test_first_token_for_word_none_not_equal_to_zero():
    # Special tokens have word_id None; a target word_id of 0 must not match
    # None (None == 0 is False). Guards against a falsy-check refactor.
    assert _first_token_for_word([None, 0], first_word_offset=0, term_idx=0) == 1


def test_cefr_vocab_entry_carries_level_term_pos():
    entry = CefrVocabEntry(level="A1", term="Haus", pos="NOUN")
    assert entry.level == "A1"
    assert entry.term == "Haus"
    assert entry.pos == "NOUN"


def test_main_gate_passes_when_above_threshold(monkeypatch, tmp_path):
    """Gate returns 0 when every language clears GATE_ACCURACY."""
    passing_report = {
        "levels": {},
        "overall": {
            "lemma_accuracy": GATE_ACCURACY + 0.01,
            "upos_accuracy": GATE_ACCURACY + 0.01,
            "lemma_total": 10,
            "upos_total": 10,
        },
    }
    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.evaluate_language",
        lambda lang, out_dir, bs: passing_report,
    )
    rc = main(["--lang", "de", "--out-dir", str(tmp_path)])
    assert rc == 0


def test_main_gate_passes_at_exact_boundary(monkeypatch, tmp_path):
    """Gate returns 0 when accuracy is exactly GATE_ACCURACY (>= comparison)."""
    boundary_report = {
        "levels": {},
        "overall": {
            "lemma_accuracy": GATE_ACCURACY,
            "upos_accuracy": GATE_ACCURACY,
            "lemma_total": 10,
            "upos_total": 10,
        },
    }
    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.evaluate_language",
        lambda lang, out_dir, bs: boundary_report,
    )
    rc = main(["--lang", "de", "--out-dir", str(tmp_path)])
    assert rc == 0


def test_main_lang_all_passes_when_all_clear(monkeypatch, tmp_path):
    """--lang all returns 0 when every language clears the gate."""
    passing_report = {
        "levels": {},
        "overall": {
            "lemma_accuracy": GATE_ACCURACY + 0.01,
            "upos_accuracy": GATE_ACCURACY + 0.01,
            "lemma_total": 10,
            "upos_total": 10,
        },
    }
    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.evaluate_language",
        lambda lang, out_dir, bs: passing_report,
    )

    class _FakeSpec:
        __slots__ = ("lang",)

        def __init__(self, lang):
            self.lang = lang

    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.LANGUAGES",
        [_FakeSpec("de"), _FakeSpec("en")],
    )
    rc = main(["--lang", "all", "--out-dir", str(tmp_path)])
    assert rc == 0


def test_main_lang_all_fails_when_one_below(monkeypatch, tmp_path):
    """--lang all returns 1 when any language fails the gate."""
    passing_report = {
        "levels": {},
        "overall": {
            "lemma_accuracy": GATE_ACCURACY + 0.01,
            "upos_accuracy": GATE_ACCURACY + 0.01,
            "lemma_total": 10,
            "upos_total": 10,
        },
    }
    failing_report = {
        "levels": {},
        "overall": {
            "lemma_accuracy": GATE_ACCURACY - 0.05,
            "upos_accuracy": GATE_ACCURACY + 0.01,
            "lemma_total": 10,
            "upos_total": 10,
        },
    }

    def fake_eval(lang, out_dir, bs):
        return failing_report if lang == "de" else passing_report

    monkeypatch.setattr("lemmatizer.eval.cefr_eval.evaluate_language", fake_eval)

    class _FakeSpec:
        __slots__ = ("lang",)

        def __init__(self, lang):
            self.lang = lang

    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.LANGUAGES",
        [_FakeSpec("de"), _FakeSpec("en")],
    )
    rc = main(["--lang", "all", "--out-dir", str(tmp_path)])
    assert rc == 1


def test_main_gate_fails_when_lemma_below_threshold(monkeypatch, tmp_path):
    """Gate returns 1 when lemma falls below GATE_ACCURACY."""
    failing_report = {
        "levels": {},
        "overall": {
            "lemma_accuracy": GATE_ACCURACY - 0.05,
            "upos_accuracy": GATE_ACCURACY + 0.01,
            "lemma_total": 10,
            "upos_total": 10,
        },
    }
    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.evaluate_language",
        lambda lang, out_dir, bs: failing_report,
    )
    rc = main(["--lang", "de", "--out-dir", str(tmp_path)])
    assert rc == 1


def test_main_gate_fails_when_upos_below_threshold(monkeypatch, tmp_path):
    """Gate returns 1 when UPOS (not lemma) falls below GATE_ACCURACY."""
    report = {
        "levels": {},
        "overall": {
            "lemma_accuracy": GATE_ACCURACY + 0.01,
            "upos_accuracy": GATE_ACCURACY - 0.05,
            "lemma_total": 10,
            "upos_total": 10,
        },
    }
    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.evaluate_language",
        lambda lang, out_dir, bs: report,
    )
    rc = main(["--lang", "de", "--out-dir", str(tmp_path)])
    assert rc == 1


def test_main_handles_exception_as_failure(monkeypatch, tmp_path):
    """A language raising during eval is caught and counts as a gate failure."""
    def raising_eval(lang, out_dir, bs):
        raise RuntimeError("model load failed")

    monkeypatch.setattr("lemmatizer.eval.cefr_eval.evaluate_language", raising_eval)
    rc = main(["--lang", "de", "--out-dir", str(tmp_path)])
    assert rc == 1
