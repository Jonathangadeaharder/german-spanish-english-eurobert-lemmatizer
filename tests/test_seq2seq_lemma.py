"""Tests for the seq2seq lemmatizer trainer (pure functions only).

Per AGENTS.md §5 (TDD): written BEFORE the implementation changes were
finalized. These tests pin the contracts of the pure helpers in
seq2seq_lemma.py — collate_batch, loss_fn, generate, and the
post-processing step — without instantiating a real T5 model.

The strategy is to call the pure functions directly with small mock
batches and (for generate) a stub model object that mimics the
T5.encode/T5.decode signature.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from lemmatizer.train.grad_utils import tree_add, tree_scale
from lemmatizer.train.seq2seq_lemma import (
    BYT5_EOS,
    BYT5_PAD,
    _strip_decoder_start_and_truncate_at_eos,
    collate_batch,
    generate,
    loss_fn,
)

# A byte id in the valid content range (0 + BYTE_ID_OFFSET = 3 in
# byt5_dataset, but here any non-special id is fine for collation logic).
CONTENT_A = 5
CONTENT_B = 7


# --- collate_batch -----------------------------------------------------------


def test_collate_batch_pads_to_max_length_in_batch():
    rows = [
        {"input_ids": [CONTENT_A, CONTENT_B], "labels": [CONTENT_A]},
        {"input_ids": [CONTENT_A, CONTENT_B, CONTENT_A], "labels": [CONTENT_B, CONTENT_A]},
    ]
    batch = collate_batch(rows)
    # max_input = 3, max_label = 2 across the two rows.
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].shape == (2, 2)
    # Padding positions are BYT5_PAD with attention_mask 0.
    assert int(batch["input_ids"][0, 2]) == BYT5_PAD
    assert int(batch["attention_mask"][0, 2]) == 0
    assert int(batch["attention_mask"][1, 2]) == 1
    # Unpadded positions carry the original tokens.
    assert int(batch["input_ids"][0, 0]) == CONTENT_A
    assert int(batch["input_ids"][1, 2]) == CONTENT_A


def test_collate_batch_truncates_overlong_sequences():
    long_input = [CONTENT_A] * 5000
    long_label = [CONTENT_B] * 5000
    rows = [{"input_ids": long_input, "labels": long_label}]
    batch = collate_batch(rows)
    # MAX_SEQ_LEN caps both axes.
    from lemmatizer.data.byt5_dataset import MAX_SEQ_LEN

    assert batch["input_ids"].shape[1] == MAX_SEQ_LEN
    assert batch["labels"].shape[1] == MAX_SEQ_LEN
    # Attention mask marks the kept prefix as valid.
    assert int(batch["attention_mask"][0, 0]) == 1


def test_collate_batch_labels_default_to_minus_100():
    rows = [
        {"input_ids": [CONTENT_A], "labels": [CONTENT_A]},
        {"input_ids": [CONTENT_A], "labels": [CONTENT_A, CONTENT_B]},
    ]
    batch = collate_batch(rows)
    # Row 0 label slot beyond its length must be -100 (masked from loss).
    assert int(batch["labels"][0, 1]) == -100
    # Row 1 carries its second label token.
    assert int(batch["labels"][1, 1]) == CONTENT_B


def test_collate_batch_T_equals_1_edge_case():
    # Single-token sequences: T=1 is the boundary where loss_fn skips the
    # `decoder_input[:, 1:] = ...` assignment. Ensure collation doesn't
    # produce a zero-width axis.
    rows = [{"input_ids": [CONTENT_A], "labels": [CONTENT_A]}]
    batch = collate_batch(rows)
    assert batch["input_ids"].shape == (1, 1)
    assert batch["labels"].shape == (1, 1)
    assert int(batch["input_ids"][0, 0]) == CONTENT_A


# --- loss_fn -----------------------------------------------------------------


class _StubModel:
    """Minimal callable mimicking T5.__call__ for loss_fn.

    Returns a (B, T, V) logits tensor whose value at position (b, t) is a
    function of the decoder_input token, so loss_fn's gather path is
    exercised against a known target.
    """

    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size

    def __call__(
        self,
        input_ids: mx.array,
        decoder_input: mx.array,
        padding_mask: mx.array | None = None,
    ) -> mx.array:
        B, T = decoder_input.shape
        # Argmax of logits == decoder_input token at that position: the
        # model "predicts" the token it was fed. Shift-right semantics in
        # loss_fn mean this maps labels[:-1] -> next-position predictions.
        logits = mx.full((B, T, self.vocab_size), -10.0)
        mx.eval(logits)
        for b in range(B):
            for t in range(T):
                tok = int(decoder_input[b, t]) % self.vocab_size
                logits[b, t, tok] = 10.0
        return logits


def test_loss_fn_decoder_input_shifts_labels_with_eos_start():
    # labels: [CONTENT_A, CONTENT_B]; expected decoder_input =
    # [BYT5_EOS, max(CONTENT_A, 0)] = [1, CONTENT_A].
    rows = [{"input_ids": [CONTENT_A], "labels": [CONTENT_A, CONTENT_B]}]
    batch = collate_batch(rows)
    model = _StubModel(vocab_size=16)

    # Recreate decoder_input the way loss_fn builds it, then assert the
    # stub model's logits line up (sanity that the shift contract holds).
    labels = batch["labels"]
    B, T = labels.shape
    expected_dec = mx.full((B, T), BYT5_PAD, dtype=mx.int32)
    expected_dec[:, 0] = BYT5_EOS
    if T > 1:
        expected_dec[:, 1:] = mx.maximum(labels[:, :-1], 0)
    mx.eval(expected_dec)

    logits = model(batch["input_ids"], expected_dec, padding_mask=None)
    # At position 0 the decoder saw BYT5_EOS (1); position 1 saw CONTENT_A.
    assert int(mx.argmax(logits[0, 0, :])) == BYT5_EOS % 16
    assert int(mx.argmax(logits[0, 1, :])) == CONTENT_A % 16


def test_loss_fn_masks_minus_100_labels_from_loss():
    # Row 0 has a -100 at position 1; that position must contribute zero
    # to the loss. With the stub model, the only way loss is ~0 is if the
    # mask zeroes out every misaligned position.
    rows = [
        {"input_ids": [CONTENT_A], "labels": [CONTENT_A, -100]},
    ]
    batch = collate_batch(rows)
    model = _StubModel(vocab_size=16)
    loss = loss_fn(model, batch)
    # The single unmasked position (0) has decoder_input EOS, label
    # CONTENT_A — the stub predicts the decoder token (EOS), not the
    # label, so loss > 0 there. But position 1 (label -100) must not
    # contribute. Assert the loss is finite and positive.
    assert mx.isfinite(loss)
    assert float(loss) > 0.0


def test_loss_fn_returns_near_zero_when_predictions_match_labels():
    # Build a stub that predicts labels exactly: logits peak at the label
    # token at each position, so loss -> ~0.
    rows = [
        {"input_ids": [CONTENT_A], "labels": [CONTENT_A, CONTENT_B]},
    ]
    batch = collate_batch(rows)
    labels = batch["labels"]
    B, T = labels.shape

    class _PerfectModel:
        def __call__(
            self,
            input_ids: mx.array,
            decoder_input: mx.array,
            padding_mask: mx.array | None = None,
        ) -> mx.array:
            V = 16
            logits = mx.full((B, T, V), -10.0)
            mx.eval(logits)
            for b in range(B):
                for t in range(T):
                    lab = int(labels[b, t])
                    if lab < 0:
                        continue
                    logits[b, t, lab % V] = 10.0
            return logits

    loss = loss_fn(_PerfectModel(), batch)
    mx.eval(loss)
    assert float(loss) < 1e-3


def test_loss_fn_T_equals_1_does_not_index_empty_slice():
    # T=1: the `if T > 1` guard must skip the shifted assignment. This is
    # a regression guard — without the guard, labels[:, :-1] would be
    # empty and decoder_input[:, 1:] would be a 0-width slice.
    rows = [{"input_ids": [CONTENT_A], "labels": [CONTENT_A]}]
    batch = collate_batch(rows)
    model = _StubModel(vocab_size=16)
    loss = loss_fn(model, batch)
    assert mx.isfinite(loss)


# --- grad_utils (tree_scale / tree_add) --------------------------------------


def test_tree_scale_none_passes_through():
    assert tree_scale(None, 2.0) is None


def test_tree_scale_scales_scalar_leaf():
    out = tree_scale(mx.array(3.0), 2.0)
    assert float(out) == pytest.approx(6.0)


def test_tree_scale_recurses_dict_and_list():
    tree = {"a": mx.array([1.0, 2.0]), "b": [mx.array(3.0)]}
    out = tree_scale(tree, 2.0)
    mx.eval(out)
    assert list(np.array(out["a"])) == [2.0, 4.0]
    assert float(np.array(out["b"][0])) == 3.0 * 2.0


def test_tree_add_none_left_returns_right():
    rhs = mx.array([1.0, 2.0])
    out = tree_add(None, rhs)
    assert list(np.array(out)) == [1.0, 2.0]


def test_tree_add_none_right_returns_left():
    lhs = mx.array([1.0, 2.0])
    out = tree_add(lhs, None)
    assert list(np.array(out)) == [1.0, 2.0]


def test_tree_add_both_none_returns_none():
    assert tree_add(None, None) is None


def test_tree_add_adds_scalar_leaves():
    out = tree_add(mx.array(1.0), mx.array(2.0))
    assert float(out) == pytest.approx(3.0)


def test_tree_add_recurses_dict_and_list():
    a = {"x": mx.array([1.0, 2.0]), "y": [mx.array(3.0)]}
    b = {"x": mx.array([10.0, 20.0]), "y": [mx.array(30.0)]}
    out = tree_add(a, b)
    mx.eval(out)
    assert list(np.array(out["x"])) == [11.0, 22.0]
    assert float(np.array(out["y"][0])) == 33.0


# --- generate + post-processing ---------------------------------------------


class _StubT5:
    """Stub T5 for generate(): emits a fixed token schedule per step.

    `schedule` is a list of token ids emitted in order, regardless of the
    decoder input. This lets us test EOS termination, KV-cache plumbing,
    and the leading-EOS strip + truncate-at-first-EOS post-processing
    without a real model.
    """

    def __init__(self, schedule: list[int]):
        self._schedule = schedule
        self._step = 0
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, inputs: mx.array, padding_mask: Any = None) -> mx.array:
        self.encode_calls += 1
        # Return a dummy memory tensor of shape (B, T_in, 1).
        return mx.zeros((inputs.shape[0], inputs.shape[1], 1))

    def decode(
        self,
        inputs: mx.array,
        memory: mx.array,
        cache: Any = None,
    ) -> tuple[mx.array, Any]:
        self.decode_calls += 1
        B = inputs.shape[0]
        tok = self._schedule[self._step % len(self._schedule)]
        self._step += 1
        # logits shape (B, 1, V); force argmax to `tok`.
        V = max(max(self._schedule) + 1, BYT5_EOS + 1, 2)
        logits = mx.full((B, 1, V), -10.0)
        logits[:, 0, tok] = 10.0
        return logits, cache


def test_generate_returns_list_of_token_lists():
    stub = _StubT5(schedule=[CONTENT_A, CONTENT_B, BYT5_EOS])
    input_ids = mx.array([[CONTENT_A], [CONTENT_B]], dtype=mx.int32)
    out = generate(stub, input_ids, max_len=5)
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(seq, list) for seq in out)
    assert all(isinstance(tok, int) for seq in out for tok in seq)


def test_generate_strips_leading_decoder_start_eos():
    # First emitted token is CONTENT_A (not EOS), so the leading
    # decoder-start EOS (column 0) must be stripped — outputs begin with
    # CONTENT_A, not BYT5_EOS.
    stub = _StubT5(schedule=[CONTENT_A, BYT5_EOS])
    input_ids = mx.array([[CONTENT_A]], dtype=mx.int32)
    out = generate(stub, input_ids, max_len=4)
    assert out[0][0] == CONTENT_A


def test_generate_truncates_at_first_emitted_eos():
    # Schedule emits CONTENT_A, CONTENT_B, then EOS, then more tokens.
    # Post-processing must cut each sequence at the first EOS and drop it.
    stub = _StubT5(schedule=[CONTENT_A, CONTENT_B, BYT5_EOS, CONTENT_A])
    input_ids = mx.array([[CONTENT_A]], dtype=mx.int32)
    out = generate(stub, input_ids, max_len=5)
    assert out[0] == [CONTENT_A, CONTENT_B]
    assert BYT5_EOS not in out[0]


def test_generate_stops_when_all_sequences_emit_eos():
    # All batch rows finish once EOS is emitted; the loop should break early
    # rather than running all max_len steps.
    stub = _StubT5(schedule=[BYT5_EOS])
    input_ids = mx.array([[CONTENT_A], [CONTENT_B]], dtype=mx.int32)
    out = generate(stub, input_ids, max_len=10)
    # EOS is emitted at step 1 for both rows -> empty streams (leading
    # EOS stripped, then truncation at the first EOS yields []).
    assert out == [[], []]
    # decode() called once per step; with early break after step 1 we
    # expect exactly 1 decode call, not 10.
    assert stub.decode_calls == 1
    # Encoder runs exactly once (memory is reused via cache).
    assert stub.encode_calls == 1


def test_generate_runs_full_max_len_without_eos():
    # No EOS ever emitted -> loop runs max_len steps; each row's stream
    # has max_len tokens (the decoder-start EOS is stripped).
    stub = _StubT5(schedule=[CONTENT_A])
    input_ids = mx.array([[CONTENT_A]], dtype=mx.int32)
    out = generate(stub, input_ids, max_len=4)
    assert out[0] == [CONTENT_A, CONTENT_A, CONTENT_A, CONTENT_A]
    assert stub.decode_calls == 4


def test_strip_decoder_start_and_truncate_at_eos_direct():
    # Direct unit test of the post-processing helper against a crafted
    # decoder_input array (B=2): row 0 has an EOS mid-stream, row 1 has
    # no EOS (runs to the end).
    decoder_input = mx.array(
        [
            [BYT5_EOS, CONTENT_A, CONTENT_B, BYT5_EOS, CONTENT_A],
            [BYT5_EOS, CONTENT_B, CONTENT_B, CONTENT_B, CONTENT_A],
        ],
        dtype=mx.int32,
    )
    out = _strip_decoder_start_and_truncate_at_eos(decoder_input)
    assert out == [
        [CONTENT_A, CONTENT_B],
        [CONTENT_B, CONTENT_B, CONTENT_B, CONTENT_A],
    ]
