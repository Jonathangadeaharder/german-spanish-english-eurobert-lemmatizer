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
    # If local_path already points to an existing file, return it directly.
    if os.path.isfile(expanded):
        return expanded
    # If local_path is a directory and filename is given, join them.
    if filename and os.path.isdir(expanded):
        joined = os.path.join(expanded, filename)
        if os.path.isfile(joined):
            return joined
    # Fall back to snapshot_download.
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
    """ByT5 encoder + per-word pooling + lemma and UPOS classification heads.

    The encoder is the pretrained byt5-small backbone (fine-tunable). The
    lemma head is `nn.Linear(d_model, num_lemmas)` over the mean-pooled
    per-word byte representations. An optional UPOS head
    `nn.Linear(d_model, num_upos)` provides joint UPOS tagging.
    """

    def __init__(self, num_lemmas: int, dropout: float = 0.1, num_upos: int = 0):
        super().__init__()
        config = load_byt5_config()
        self.t5 = T5(config)
        self.hidden_size = config.d_model
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.hidden_size, num_lemmas)
        self.upos_classifier = nn.Linear(self.hidden_size, num_upos) if num_upos > 0 else None
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
    ) -> tuple[mx.array, mx.array | None]:
        """Forward pass.

        Args:
            input_ids: (B, T) byte-level token ids.
            word_byte_spans: (B, N_words, 2) start/end byte indices per word.
                Words with no bytes (span [0,0)) are masked to PAD_LABEL upstream.

        Returns:
            Always a (lemma_logits, upos_logits) tuple. upos_logits is None
            when the UPOS head is absent, so callers need no isinstance check.
        """
        enc_out = self.t5.encode(input_ids)  # (B, T, d_model)
        enc_out = self.dropout(enc_out)
        pooled = self._pool(enc_out, word_byte_spans)
        lemma_logits = self.classifier(pooled)
        if self.upos_classifier is not None:
            upos_logits = self.upos_classifier(pooled)
            return lemma_logits, upos_logits
        return lemma_logits, None

    def _pool(
        self,
        enc_out: mx.array,
        word_byte_spans: mx.array,
    ) -> mx.array:
        """Mean-pool each word's byte representations.

        Args:
            enc_out: (B, T, d_model)
            word_byte_spans: (B, N_words, 2)

        Returns:
            (B, N_words, d_model) pooled word vectors.
        """
        B, T, D = enc_out.shape

        byte_idx = mx.arange(T)  # (T,)
        starts = word_byte_spans[:, :, 0:1]  # (B, N_words, 1)
        ends = word_byte_spans[:, :, 1:2]  # (B, N_words, 1)
        in_span = (byte_idx >= starts) & (byte_idx < ends)  # (B, N_words, T)
        in_span = in_span.astype(mx.float32)

        pooled = mx.einsum("bnt,btd->bnd", in_span, enc_out)
        word_lens = in_span.sum(axis=2, keepdims=True)  # (B, N_words, 1)
        safe_lens = mx.maximum(word_lens, mx.array(1.0))
        pooled = pooled / safe_lens

        empty = (word_lens == 0).astype(pooled.dtype)
        pooled = pooled * (1.0 - empty)

        return pooled


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
