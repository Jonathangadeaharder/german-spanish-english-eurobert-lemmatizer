from lemmatizer.eval.evaluate_cefr import (
    find_term_index,
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
