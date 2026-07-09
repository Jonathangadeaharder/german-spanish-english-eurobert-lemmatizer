"""Tests for subtitle_pipeline CoNLL-U annotation fixes.

Pins the contracts fixed in the OCR review:
- spaCy: relative head index per sentence, one block per sentence.
- stanza: depparse processor included; failure threshold raises.
- vtt_parser: CJK path splits on ASCII punctuation too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from lemmatizer.data.subtitle_pipeline import (
    FAILURE_THRESHOLD,
    _raise_on_excess_failures,
    annotate_with_spacy,
    annotate_with_stanza,
)


@dataclass
class _FakeTok:
    text: str
    lemma_: str
    pos_: str
    dep_: str
    head: Any
    i: int


@dataclass
class _FakeSpan:
    text: str
    start: int
    tokens: list[_FakeTok]

    def __iter__(self):
        return iter(self.tokens)

    @property
    def end(self) -> int:
        return self.start + len(self.tokens)


@dataclass
class _FakeDoc:
    sents: list[_FakeSpan]


class _FakeNlp:
    def __init__(self, docs: list[_FakeDoc]):
        self._docs = docs
        self.pipe_calls = 0

    def pipe(self, sentences: list[str]):
        self.pipe_calls += 1
        return self._docs


def _make_two_sentence_doc() -> _FakeDoc:
    # Two sentences in one doc to verify the per-sentence head index
    # is relative (resets per sentence), not absolute (doc-level).
    #
    # Sentence 0: tokens A(root), B(head=A), C(head=B)
    # Sentence 1: tokens D(root), E(head=D)
    # i values are doc-level (absolute) to mimic spaCy's token.head.i.
    a = _FakeTok("A", "a", "NOUN", "ROOT", None, 0)
    b = _FakeTok("B", "b", "NOUN", "nsubj", a, 1)
    c = _FakeTok("C", "c", "NOUN", "obj", b, 2)
    a.head = a
    d = _FakeTok("D", "d", "NOUN", "ROOT", None, 3)
    e = _FakeTok("E", "e", "NOUN", "nsubj", d, 4)
    d.head = d
    sent0 = _FakeSpan("A B C", 0, [a, b, c])
    sent1 = _FakeSpan("D E", 3, [d, e])
    return _FakeDoc([sent0, sent1])


def _parse_conllu_blocks(text: str) -> list[list[list[str]]]:
    blocks = [b for b in text.split("\n\n") if b.strip()]
    parsed = []
    for block in blocks:
        rows = []
        for line in block.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.split("\t"))
        parsed.append(rows)
    return parsed


class TestSpacyHeadIndexPerSentence:
    def test_relative_head_index_resets_per_sentence(self):
        doc = _make_two_sentence_doc()
        nlp = _FakeNlp([doc])
        out = annotate_with_spacy(["A B C D E"], "de", nlp=nlp)
        # One result element per spaCy sentence.
        blocks = [_parse_conllu_blocks(b)[0] for b in out]
        assert len(blocks) == 2, "should emit one block per spaCy sentence"

        # Sentence 0: A is root (head 0), B→A (head 1), C→B (head 2)
        heads0 = [r[6] for r in blocks[0]]
        assert heads0 == ["0", "1", "2"]

        # Sentence 1: D is root (head 0), E→D (head 1) — NOT absolute.
        heads1 = [r[6] for r in blocks[1]]
        assert heads1 == ["0", "1"]

    def test_one_block_per_sentence(self):
        doc = _make_two_sentence_doc()
        nlp = _FakeNlp([doc])
        out = annotate_with_spacy(["A B C D E"], "de", nlp=nlp)
        # One result element per spaCy sentence (not merged into one).
        assert len(out) == 2
        b0 = _parse_conllu_blocks(out[0])[0]
        b1 = _parse_conllu_blocks(out[1])[0]
        assert b0[0][1] == "A"
        assert b1[0][1] == "D"

    def test_token_ids_per_sentence(self):
        doc = _make_two_sentence_doc()
        nlp = _FakeNlp([doc])
        out = annotate_with_spacy(["A B C D E"], "de", nlp=nlp)
        blocks = [_parse_conllu_blocks(b)[0] for b in out]
        # Token id column is 1-based per sentence.
        ids0 = [r[0] for r in blocks[0]]
        ids1 = [r[0] for r in blocks[1]]
        assert ids0 == ["1", "2", "3"]
        assert ids1 == ["1", "2"]

    def test_absolute_head_index_bug_would_fail(self):
        # Pin the fix: with the OLD buggy code (token.head.i + 1), the
        # second sentence's E token would get head=5 (absolute), not 1.
        # The corrected relative formula gives token.head.i - start + 1.
        doc = _make_two_sentence_doc()
        nlp = _FakeNlp([doc])
        out = annotate_with_spacy(["A B C D E"], "de", nlp=nlp)
        blocks = [_parse_conllu_blocks(b)[0] for b in out]
        e_head = blocks[1][1][6]  # E is row index 1
        assert e_head == "1", f"E head must be relative (1), not absolute (5); got {e_head}"


class TestSpacyFailures:
    def test_raises_when_over_half_fail(self):
        # 3 sentences, all raise inside doc.sents → >50% → RuntimeError.
        class _BoomDoc:
            @property
            def sents(self):
                raise RuntimeError("boom")

        class BoomNlp:
            def pipe(self, sentences):
                return [_BoomDoc() for _ in sentences]

        with pytest.raises(RuntimeError, match="spacy"):
            annotate_with_spacy(["x", "y", "z"], "de", nlp=BoomNlp())

    def test_no_raise_when_below_threshold(self):
        # 3 sentences, 1 fails inside doc.sents (<50%) → OK, returns
        # results for the 2 successful docs (2 sentences each = 4 blocks).
        good = _make_two_sentence_doc()

        class _BoomDoc:
            @property
            def sents(self):
                raise RuntimeError("boom")

        class MixedNlp:
            def __init__(self):
                self.calls = 0

            def pipe(self, sentences):
                results = []
                for _ in sentences:
                    if self.calls == 1:
                        self.calls += 1
                        results.append(_BoomDoc())
                        continue
                    self.calls += 1
                    results.append(good)
                return results

        out = annotate_with_spacy(["s1", "s2", "s3"], "de", nlp=MixedNlp())
        # 2 successful docs × 2 sentences each = 4 blocks.
        assert len(out) == 4


class TestStanzaDepparseProcessor:
    def test_processors_string_includes_depparse(self):
        # The default-None branch builds a Pipeline. Patch stanza.Pipeline
        # to capture the kwargs.
        captured: dict[str, Any] = {}

        class FakePipeline:
            def __init__(self, **kwargs: Any):
                captured.update(kwargs)

            def __call__(self, text: str):
                return self

            @property
            def sentences(self):
                return []

        import sys

        fake_mod = type(sys)("stanza")
        fake_mod.Pipeline = FakePipeline
        old = sys.modules.get("stanza")
        sys.modules["stanza"] = fake_mod
        try:
            annotate_with_stanza(["x"], "sv", nlp=None)
        finally:
            if old is not None:
                sys.modules["stanza"] = old
            else:
                del sys.modules["stanza"]

        processors = captured.get("processors", "")
        assert "depparse" in processors.split(","), (
            f"depparse must be in processors, got {processors!r}"
        )


class TestFailureThresholdHelper:
    def test_no_raise_at_or_below_threshold(self):
        # failures/total == threshold should NOT raise (> strict).
        _raise_on_excess_failures(1, 2, "x")  # 0.5, not > 0.5

    def test_raises_above_threshold(self):
        with pytest.raises(RuntimeError, match="3/4"):
            _raise_on_excess_failures(3, 4, "x")

    def test_zero_total_no_raise(self):
        _raise_on_excess_failures(0, 0, "x")

    def test_threshold_constant(self):
        assert FAILURE_THRESHOLD == 0.5


class TestVttParserCjkSplit:
    def test_cjk_path_splits_on_ascii_punctuation(self):
        from lemmatizer.data.vtt_parser import blocks_to_sentences

        # Mixed CJK + ASCII punctuation: previously the CJK path only
        # split on 。！？, leaving "你world." glued to the next sentence.
        text = "你好。你world. Another sentence here."
        out = blocks_to_sentences([text])
        joined = " | ".join(out)
        assert "world. Another" not in joined, f"ASCII period should split CJK text; got {joined!r}"

    def test_cjk_path_still_splits_on_cjk_punctuation(self):
        from lemmatizer.data.vtt_parser import blocks_to_sentences

        # Each segment must be >= 5 chars to pass the min-length filter.
        text = "你好世界今天。天气真不错啊。我们明天再见吧。"
        out = blocks_to_sentences([text])
        assert len(out) == 3

    def test_ascii_path_unchanged(self):
        from lemmatizer.data.vtt_parser import blocks_to_sentences

        text = "Hello world. Another one here! Third?"
        out = blocks_to_sentences([text])
        assert len(out) == 3
