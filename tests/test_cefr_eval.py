from pathlib import Path

from lemmatizer.eval.cefr_eval import (
    GATE_ACCURACY,
    NON_CONTENT_POS,
    CefrVocabEntry,
    _find_term_index,
    _first_token_for_word,
    _resolve_pos_column,
    main,
)
from lemmatizer.languages import LanguageAssets


def test_skip_pos_set_excludes_proper_nouns_and_punctuation():
    assert "PROPN" in NON_CONTENT_POS
    assert "PUNCT" in NON_CONTENT_POS
    assert "SYM" in NON_CONTENT_POS
    assert "X" in NON_CONTENT_POS
    assert "NUM" in NON_CONTENT_POS
    assert "NOUN" not in NON_CONTENT_POS
    assert "VERB" not in NON_CONTENT_POS


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


def _make_report(
    lemma_accuracy=GATE_ACCURACY + 0.01,
    upos_accuracy=GATE_ACCURACY + 0.01,
    lemma_total=10,
    upos_total=10,
    vocab_total=10,
    coverage=1.0,
):
    """Factory for eval report dicts used by gate tests."""
    return {
        "levels": {},
        "overall": {
            "lemma_accuracy": lemma_accuracy,
            "upos_accuracy": upos_accuracy,
            "lemma_total": lemma_total,
            "upos_total": upos_total,
            "vocab_total": vocab_total,
            "coverage": coverage,
        },
    }


class _FakeSpec:
    __slots__ = ("lang",)

    def __init__(self, lang):
        self.lang = lang


def _fake_assets_for_dir(lang, artifacts_dir):
    """Return LanguageAssets with label2id_path under a test-controlled dir."""
    base = Path(artifacts_dir) / f"lemma_{lang}"
    base.mkdir(parents=True, exist_ok=True)
    (base / "label2id.json").write_text('{"a": 0}', encoding="utf-8")
    return LanguageAssets(
        lang=lang,
        artifacts_dir=base,
        tokenizer_dir=base / "tokenizer",
        dataset_path=Path(f"data/processed/eurobert_lemma_{lang}_dataset"),
        output_dir=Path(f"runs/eurobert-lemma-{lang}-210m-lora"),
        merged_dir=Path(f"models/eurobert-lemma-{lang}-210m-merged"),
        onnx_dir=Path(f"onnx/eurobert-lemma-{lang}-210m"),
        web_model_dir=Path(f"web/model/lemma_{lang}"),
        label2id_path=base / "label2id.json",
        id2label_path=base / "id2label.json",
        upos_label2id_path=base / "upos_label2id.json",
        upos_id2label_path=base / "upos_id2label.json",
        edit_trees_path=base / "edit_trees.json",
        lexicon_path=base / "lexicon.json",
        exceptions_path=base / "exceptions.json",
        eval_report_path=base / "eval_report.json",
        base_model=None,
        tokenizer_name=None,
    )


def _patch_langs(monkeypatch, langs):
    """Monkeypatch LANGUAGES so --lang choices are test-isolated."""
    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.LANGUAGES",
        [_FakeSpec(code) for code in langs],
    )


def _patch_langs_with_assets(monkeypatch, langs, artifacts_dir):
    """Patch LANGUAGES and language_assets so label2id.json resolves under artifacts_dir."""
    _patch_langs(monkeypatch, langs)
    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.language_assets",
        lambda lang: _fake_assets_for_dir(lang, artifacts_dir),
    )


def test_main_gate_passes_when_above_threshold(monkeypatch, tmp_path):
    """Gate returns 0 when every language clears GATE_ACCURACY."""
    _patch_langs_with_assets(monkeypatch, ["de"], tmp_path)
    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.evaluate_language",
        lambda lang, out_dir, bs: _make_report(),
    )
    rc = main(["--lang", "de", "--out-dir", str(tmp_path)])
    assert rc == 0


def test_main_gate_passes_at_exact_boundary(monkeypatch, tmp_path):
    """Gate returns 0 when accuracy is exactly GATE_ACCURACY (>= comparison)."""
    _patch_langs_with_assets(monkeypatch, ["de"], tmp_path)
    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.evaluate_language",
        lambda lang, out_dir, bs: _make_report(
            lemma_accuracy=GATE_ACCURACY, upos_accuracy=GATE_ACCURACY
        ),
    )
    rc = main(["--lang", "de", "--out-dir", str(tmp_path)])
    assert rc == 0


def test_main_lang_all_passes_when_all_clear(monkeypatch, tmp_path):
    """--lang all returns 0 when every language clears the gate."""
    _patch_langs_with_assets(monkeypatch, ["de", "en"], tmp_path)
    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.evaluate_language",
        lambda lang, out_dir, bs: _make_report(),
    )
    rc = main(["--lang", "all", "--out-dir", str(tmp_path)])
    assert rc == 0


def test_main_lang_all_fails_when_one_below(monkeypatch, tmp_path):
    """--lang all returns 1 when any language fails the gate."""
    _patch_langs_with_assets(monkeypatch, ["de", "en"], tmp_path)

    def fake_eval(lang, out_dir, bs):
        if lang == "de":
            return _make_report(lemma_accuracy=GATE_ACCURACY - 0.05)
        return _make_report()

    monkeypatch.setattr("lemmatizer.eval.cefr_eval.evaluate_language", fake_eval)
    rc = main(["--lang", "all", "--out-dir", str(tmp_path)])
    assert rc == 1


def test_main_gate_fails_when_lemma_below_threshold(monkeypatch, tmp_path):
    """Gate returns 1 when lemma falls below GATE_ACCURACY."""
    _patch_langs_with_assets(monkeypatch, ["de"], tmp_path)
    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.evaluate_language",
        lambda lang, out_dir, bs: _make_report(lemma_accuracy=GATE_ACCURACY - 0.05),
    )
    rc = main(["--lang", "de", "--out-dir", str(tmp_path)])
    assert rc == 1


def test_main_gate_fails_when_upos_below_threshold(monkeypatch, tmp_path):
    """Gate returns 1 when UPOS (not lemma) falls below GATE_ACCURACY."""
    _patch_langs_with_assets(monkeypatch, ["de"], tmp_path)
    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.evaluate_language",
        lambda lang, out_dir, bs: _make_report(upos_accuracy=GATE_ACCURACY - 0.05),
    )
    rc = main(["--lang", "de", "--out-dir", str(tmp_path)])
    assert rc == 1


def test_main_handles_exception_as_failure(monkeypatch, tmp_path):
    """A language raising during eval is caught and counts as a gate failure."""
    _patch_langs_with_assets(monkeypatch, ["de"], tmp_path)

    def raising_eval(lang, out_dir, bs):
        raise RuntimeError("model load failed")

    monkeypatch.setattr("lemmatizer.eval.cefr_eval.evaluate_language", raising_eval)
    rc = main(["--lang", "de", "--out-dir", str(tmp_path)])
    assert rc == 1


def test_main_gate_fails_when_coverage_too_low(monkeypatch, tmp_path):
    """Gate returns 1 when coverage falls below MIN_COVERAGE."""
    from lemmatizer.eval.cefr_eval import MIN_COVERAGE

    _patch_langs_with_assets(monkeypatch, ["de"], tmp_path)
    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.evaluate_language",
        lambda lang, out_dir, bs: _make_report(
            lemma_total=5,
            upos_total=5,
            vocab_total=100,
            coverage=MIN_COVERAGE - 0.01,
        ),
    )
    rc = main(["--lang", "de", "--out-dir", str(tmp_path)])
    assert rc == 1


def test_main_skips_lang_when_artifacts_missing(monkeypatch, tmp_path):
    """A language whose label2id.json is missing is skipped, not crashed."""
    _patch_langs(monkeypatch, ["de"])

    def fake_missing_assets(lang):
        return LanguageAssets(
            lang=lang,
            artifacts_dir=Path("/nonexistent/artifacts/lemma_de"),
            tokenizer_dir=Path("/nonexistent/artifacts/lemma_de/tokenizer"),
            dataset_path=Path("data/processed/dataset"),
            output_dir=Path("runs/out"),
            merged_dir=Path("models/merged"),
            onnx_dir=Path("onnx/model"),
            web_model_dir=Path("web/model"),
            label2id_path=Path("/nonexistent/artifacts/lemma_de/label2id.json"),
            id2label_path=Path("/nonexistent/id2label.json"),
            upos_label2id_path=Path("/nonexistent/upos_label2id.json"),
            upos_id2label_path=Path("/nonexistent/upos_id2label.json"),
            edit_trees_path=Path("/nonexistent/edit_trees.json"),
            lexicon_path=Path("/nonexistent/lexicon.json"),
            exceptions_path=Path("/nonexistent/exceptions.json"),
            eval_report_path=Path("/nonexistent/eval_report.json"),
            base_model=None,
            tokenizer_name=None,
        )

    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.language_assets", fake_missing_assets
    )
    rc = main(["--lang", "de", "--out-dir", str(tmp_path)])
    assert rc == 0


def test_main_all_skips_returns_zero_when_no_artifacts(monkeypatch, tmp_path):
    """--lang all returns 0 (not 1) when all languages lack artifacts."""
    _patch_langs(monkeypatch, ["de", "en"])

    def fake_missing_assets(lang):
        return LanguageAssets(
            lang=lang,
            artifacts_dir=Path(f"/nonexistent/artifacts/lemma_{lang}"),
            tokenizer_dir=Path(f"/nonexistent/artifacts/lemma_{lang}/tokenizer"),
            dataset_path=Path(f"data/processed/dataset_{lang}"),
            output_dir=Path(f"runs/out_{lang}"),
            merged_dir=Path(f"models/merged_{lang}"),
            onnx_dir=Path(f"onnx/model_{lang}"),
            web_model_dir=Path(f"web/model_{lang}"),
            label2id_path=Path(f"/nonexistent/artifacts/lemma_{lang}/label2id.json"),
            id2label_path=Path("/nonexistent/id2label.json"),
            upos_label2id_path=Path("/nonexistent/upos_label2id.json"),
            upos_id2label_path=Path("/nonexistent/upos_id2label.json"),
            edit_trees_path=Path("/nonexistent/edit_trees.json"),
            lexicon_path=Path("/nonexistent/lexicon.json"),
            exceptions_path=Path("/nonexistent/exceptions.json"),
            eval_report_path=Path("/nonexistent/eval_report.json"),
            base_model=None,
            tokenizer_name=None,
        )

    monkeypatch.setattr(
        "lemmatizer.eval.cefr_eval.language_assets", fake_missing_assets
    )
    rc = main(["--lang", "all", "--out-dir", str(tmp_path)])
    assert rc == 0
