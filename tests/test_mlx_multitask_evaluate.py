"""Regression tests for lemmatizer.train.mlx_multitask.evaluate.

Pins the truncation behavior: a row whose `upos_labels` has fewer non-masked
entries than `words` (e.g. an unknown UPOS tag mapped to -100) yields fewer
token positions than words. Rather than raising, evaluate() truncates the
word lists to match the available positions (legitimate MAX_LENGTH case).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lemmatizer.train import mlx_multitask as mt


def _assets(tmp_path: Path) -> SimpleNamespace:
    label2id = {"de::IDENTITY": 0, "UNKNOWN": 1}
    upos_label2id = {"NOUN": 0, "VERB": 1}
    (tmp_path / "label2id.json").write_text(json.dumps(label2id))
    (tmp_path / "upos_label2id.json").write_text(json.dumps(upos_label2id))
    (tmp_path / "lexicon.json").write_text("{}")
    return SimpleNamespace(
        label2id_path=tmp_path / "label2id.json",
        upos_label2id_path=tmp_path / "upos_label2id.json",
        lexicon_path=tmp_path / "lexicon.json",
    )


class _FakeModel:
    def __call__(self, input_ids, attention_mask):
        B, T = np.array(input_ids).shape
        upos = np.zeros((B, T, 2), dtype=np.float32)
        lemma = np.zeros((B, T, 2), dtype=np.float32)
        import mlx.core as mx

        return mx.array(upos), mx.array(lemma)

    def eval(self):
        return None


def _row(words, upos_labels):
    n = len(upos_labels)
    return {
        "input_ids": list(range(n)),
        "attention_mask": [1] * n,
        "labels": [0] * n,
        "upos_labels": list(upos_labels),
        "words": list(words),
        "lemmas": list(words),
        "upos": list(words),
    }


def test_evaluate_truncates_on_alignment_mismatch(tmp_path: Path):
    """Unknown UPOS (-100) drops a position -> words truncated, no raise."""
    # 3 words, but upos_labels has only 2 non-masked entries (one is -100),
    # so word_positions returns 2 positions for 3 words.
    row = _row(["w1", "w2", "w3"], [0, -100, 1])

    result = mt.evaluate(
        _FakeModel(),
        [row],
        lang="de",
        assets=_assets(tmp_path),
        batch_size=1,
        split="test",
    )

    # Word lists truncated to match the 2 available positions.
    assert len(row["words"]) == 2
    assert row["words"] == ["w1", "w2"]
    assert len(row["lemmas"]) == 2
    assert len(row["upos"]) == 2
    assert "lemma_accuracy" in result


def test_evaluate_accepts_aligned_row(tmp_path: Path):
    """Aligned rows (positions == words) do not raise."""
    row = _row(["w1", "w2"], [0, 1])
    result = mt.evaluate(
        _FakeModel(),
        [row],
        lang="de",
        assets=_assets(tmp_path),
        batch_size=1,
        split="test",
    )
    assert "lemma_accuracy" in result
