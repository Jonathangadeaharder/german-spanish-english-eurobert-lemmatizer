"""ByT5 encoder + lemma classification head for Arabic lemmatization (MLX).

Architecture: `google/byt5-small` encoder (12 layers, d_model=1472, byte-level)
→ per-word mean-pool over byte representations → linear head over the Arabic
lemma vocabulary (~13.5K classes from UD Arabic-PADT, PROPN excluded).

This is the constrained classification synthesis:
- Byte-level morphological understanding (ByT5 pre-trained on raw UTF-8,
  natively aware of Arabic root-and-pattern morphology).
- Zero hallucination (classification, not generation).
- Fast single-pass inference.
- Token-level with sentence context: full sentence in, classify each word's
  lemma in place (mirrors the EuroBERT token-classification pipeline shape).

Per-word pooling: ByT5 tokenizes to bytes; each input word spans multiple
byte positions. We mean-pool each word's byte representations into a single
word vector using a byte→word index map, then classify. This is the
"pool the byte-level representations for each word" directive.

PROPN tokens are masked to label -100 (parity with the EuroBERT path).
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from lemmatizer.train.byt5_encoder import T5

BYT5_MODEL_ID = "google/byt5-small"
# Cached snapshot dirs (config + weights may resolve to different revisions).
# _resolve_byt5_path falls back to snapshot_download when the cache is absent.
BYT5_SMALL_SNAPSHOT = (
    "~/.cache/huggingface/hub/models--google--byt5-small/"
    "snapshots/68377bdc18a2ffec8a0533fef03b1c513a4dd49d"
)
BYT5_SMALL_WEIGHTS = (
    "~/.cache/huggingface/hub/models--google--byt5-small/"
    "snapshots/6f07f879d308b7b762708b50c83d41b27e329992/model.safetensors"
)
PAD_LABEL = -100


def _resolve_byt5_path(local_path: str, filename: str = "") -> str:
    """Return an existing byt5-small cache path, or fetch via snapshot_download.

    Falls back to huggingface_hub.snapshot_download when the hardcoded cache
    snapshot is absent (e.g. fresh machine, different HF_HOME).
    """
    expanded = os.path.expanduser(local_path)
    target = os.path.join(expanded, filename) if filename else expanded
    if os.path.exists(target):
        return target
    from huggingface_hub import snapshot_download

    return str(snapshot_download(repo_id=BYT5_MODEL_ID, allow_patterns=[filename or "*"]))


def load_byt5_config() -> SimpleNamespace:
    """Load byt5-small config from the HF cache into a SimpleNamespace."""
    cfg_path = _resolve_byt5_path(BYT5_SMALL_SNAPSHOT, "config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    return SimpleNamespace(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        d_kv=cfg["d_kv"],
        num_layers=cfg["num_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        num_heads=cfg["num_heads"],
        layer_norm_epsilon=cfg["layer_norm_epsilon"],
        feed_forward_proj=cfg["feed_forward_proj"],
        relative_attention_num_buckets=cfg["relative_attention_num_buckets"],
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        decoder_start_id=cfg.get("decoder_start_token_id", 0),
    )


class ByT5EncoderLemmaClassifier(nn.Module):
    """ByT5 encoder + per-word pooling + linear lemma classification head.

    The encoder is the pretrained byt5-small backbone (fine-tunable). The
    head is a single `nn.Linear(d_model, num_lemmas)` over the mean-pooled
    per-word byte representations.
    """

    def __init__(self, num_lemmas: int, dropout: float = 0.1):
        super().__init__()
        config = load_byt5_config()
        self.t5 = T5(config)
        self.hidden_size = config.d_model
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.hidden_size, num_lemmas)
        self._weights_loaded = False

    def load_pretrained(self) -> None:
        """Load byt5-small encoder weights into the T5 backbone."""
        weights_path = _resolve_byt5_path(BYT5_SMALL_WEIGHTS, "model.safetensors")
        weights = mx.load(weights_path)
        sanitized = T5.sanitize(weights)
        missing = self.t5.load_weights(list(sanitized.items()), strict=False)
        # Decoder weights (4 of them) are expected-missing — encoder-only path.
        if missing and len(missing) > 4:
            raise RuntimeError(
                f"Unexpected missing weights after load: {len(missing)} tensors. "
                f"Expected ≤4 (decoder-only). First: {list(missing)[:3]}"
            )
        self._weights_loaded = True

    def __call__(
        self,
        input_ids: mx.array,
        word_byte_spans: mx.array,
    ) -> mx.array:
        """Forward pass.

        Args:
            input_ids: (B, T) byte-level token ids.
            word_byte_spans: (B, N_words, 2) start/end byte indices per word.
                Words with no bytes (span [0,0)) are masked to PAD_LABEL upstream.

        Returns:
            logits: (B, N_words, num_lemmas) per-word lemma classification logits.
        """
        enc_out = self.t5.encode(input_ids)  # (B, T, d_model)
        enc_out = self.dropout(enc_out)
        logits = self._pool_and_classify(enc_out, word_byte_spans)
        return logits

    def _pool_and_classify(
        self,
        enc_out: mx.array,
        word_byte_spans: mx.array,
    ) -> mx.array:
        """Mean-pool each word's byte representations, then classify.

        Args:
            enc_out: (B, T, d_model)
            word_byte_spans: (B, N_words, 2)

        Returns:
            (B, N_words, num_lemmas)
        """
        B, T, D = enc_out.shape

        # Build a per-word, per-byte mask: mask[b, w, t] = 1 if byte t is in
        # word w's span [start, end), else 0. Vectorized via broadcasting.
        byte_idx = mx.arange(T)  # (T,)
        starts = word_byte_spans[:, :, 0:1]  # (B, N_words, 1)
        ends = word_byte_spans[:, :, 1:2]  # (B, N_words, 1)
        in_span = (byte_idx >= starts) & (byte_idx < ends)  # (B, N_words, T)
        in_span = in_span.astype(mx.float32)

        # Mean pool: (B, N_words, T) @ (B, T, D) → (B, N_words, D), then divide
        # by per-word byte count to get the mean.
        pooled = mx.einsum("bnt,btd->bnd", in_span, enc_out)
        word_lens = in_span.sum(axis=2, keepdims=True)  # (B, N_words, 1)
        # Avoid div-by-zero for empty spans (e.g. padded word slots).
        safe_lens = mx.maximum(word_lens, mx.array(1.0))
        pooled = pooled / safe_lens

        # Zero out pooled vectors for empty spans (zero byte count) so they
        # don't produce spurious logits. The loss mask (PAD_LABEL) handles
        # these upstream, but this keeps the forward numerically clean.
        empty = (word_lens == 0).astype(pooled.dtype)
        pooled = pooled * (1.0 - empty)

        return self.classifier(pooled)


def masked_cross_entropy(logits: mx.array, labels: mx.array) -> mx.array:
    """Mean cross-entropy over non-PAD labels.

    Args:
        logits: (B, N_words, num_lemmas)
        labels: (B, N_words) int32, with PAD_LABEL (-100) for masked positions.

    Returns:
        scalar loss.
    """
    # Flatten for cross_entropy: (B*N_words, num_lemmas) and (B*N_words,)
    B, N, C = logits.shape
    flat_logits = logits.reshape(B * N, C)
    safe_labels = mx.maximum(labels.reshape(B * N), 0)
    per_token = nn.losses.cross_entropy(flat_logits, safe_labels, reduction="none")
    mask = (labels.reshape(B * N) != PAD_LABEL).astype(per_token.dtype)
    loss = (per_token * mask).sum() / mx.maximum(mask.sum(), mx.array(1.0))
    return loss
