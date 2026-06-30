import json

from evaluate_cefr import (
    find_term_index,
    load_cefr_rows,
    resolve_base_model_source,
    wilson_interval,
)


def test_wilson_interval_zero_total():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_all_correct():
    lo, hi = wilson_interval(100, 100)
    assert lo > 0.95
    assert hi >= 0.9999


def test_wilson_interval_none_correct():
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0
    assert hi < 0.1


def test_wilson_interval_symmetry():
    lo1, hi1 = wilson_interval(30, 100)
    lo2, hi2 = wilson_interval(70, 100)
    assert abs(hi1 - (1 - lo2)) < 0.001
    assert abs(lo1 - (1 - hi2)) < 0.001


def test_wilson_interval_custom_z():
    lo_90, hi_90 = wilson_interval(50, 100, z=1.645)
    lo_95, hi_95 = wilson_interval(50, 100, z=1.96)
    assert lo_90 > lo_95
    assert hi_90 < hi_95


def test_find_term_index_exact():
    assert find_term_index(["The", "cat", "sat"], "cat") == 1


def test_find_term_index_case_insensitive():
    assert find_term_index(["The", "Cat", "sat"], "cat") == 1


def test_find_term_index_not_found():
    assert find_term_index(["The", "dog", "sat"], "cat") is None


def test_find_term_index_empty():
    assert find_term_index([], "cat") is None


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )


def test_load_cefr_rows_parses_valid(tmp_path):
    rows = [
        {
            "lang": "de",
            "level": "A1",
            "term": "Haus",
            "sentences": ["Das Haus ist groß."],
            "raw_output": "",
        },
        {
            "lang": "de",
            "level": "A2",
            "term": "lernen",
            "sentences": ["Ich lerne Deutsch."],
            "raw_output": "",
        },
    ]
    _write_jsonl(tmp_path / "test.jsonl", rows)
    result = load_cefr_rows(tmp_path / "test.jsonl")
    assert len(result) == 2
    assert result[0]["term"] == "Haus"
    assert result[1]["sentence"] == "Ich lerne Deutsch."


def test_load_cefr_rows_skips_errors(tmp_path):
    rows = [
        {
            "lang": "de",
            "level": "A1",
            "term": "Haus",
            "sentences": ["Das Haus."],
            "raw_output": "",
        },
        {
            "lang": "de",
            "level": "A1",
            "term": "bad",
            "sentences": [],
            "raw_output": "",
            "error": "timeout",
        },
    ]
    _write_jsonl(tmp_path / "test.jsonl", rows)
    result = load_cefr_rows(tmp_path / "test.jsonl")
    assert len(result) == 1


def test_load_cefr_rows_skips_bad_json(tmp_path):
    (tmp_path / "test.jsonl").write_text("not json\n", encoding="utf-8")
    result = load_cefr_rows(tmp_path / "test.jsonl")
    assert result == []


def test_load_cefr_rows_nonexistent(tmp_path):
    result = load_cefr_rows(tmp_path / "nope.jsonl")
    assert result == []


def test_resolve_base_model_source_with_adapter(tmp_path):
    config = {"base_model_name_or_path": "models/my-base"}
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    assert resolve_base_model_source(str(tmp_path)) == "models/my-base"


def test_resolve_base_model_source_no_adapter(tmp_path):
    assert resolve_base_model_source(str(tmp_path)) == "EuroBERT/EuroBERT-210m"


def test_resolve_base_model_source_empty_base(tmp_path):
    config = {"base_model_name_or_path": ""}
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    assert resolve_base_model_source(str(tmp_path)) == "EuroBERT/EuroBERT-210m"
