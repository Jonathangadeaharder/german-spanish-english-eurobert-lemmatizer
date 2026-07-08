"""Single source of truth for label-ID mapping.

Replaces the 4 scattered sparse→contiguous remap blocks in evaluate.py and
evaluate_cefr.py. The model's classifier head outputs contiguous 0..N-1
logits, but label files may use sparse multilingual IDs (e.g. French has
1384 labels with max ID 5363). This class resolves the mapping once and
exposes contiguous IDs for logits indexing.
"""

from __future__ import annotations

import numpy as np


class LabelSpace:
    """Resolves the sparse multilingual → contiguous label-ID mapping."""

    def __init__(self, label2id: dict[str, str]):
        self._raw_label2id = label2id
        # Build id2label (id → label) from label2id (label → id).
        # label2id keys are label strings, values are string IDs.
        self._raw_id2label = {str(v): k for k, v in label2id.items()}

        # Sort by sparse ID value to get deterministic contiguous ordering.
        self._all_ids_sorted = sorted(int(v) for v in label2id.values())
        self._max_id = max(self._all_ids_sorted) if self._all_ids_sorted else 0
        self._num_labels = len(self._all_ids_sorted)
        # Sparse → contiguous: needed when max_id exceeds the number of labels.
        # Uses > (not >=) so a contiguous 0..N-1 set with UNKNOWN at position 0
        # doesn't trigger a spurious remap (N-1 == N-1 is fine).
        self._needs_remap = self._max_id > self._num_labels

        if self._needs_remap:
            self._sparse_to_contiguous = {
                sparse: contiguous for contiguous, sparse in enumerate(self._all_ids_sorted)
            }
        else:
            self._sparse_to_contiguous = {i: i for i in self._all_ids_sorted}

        # Build contiguous id2label by looking up each sparse ID in raw_id2label.
        self._contiguous_id2label = {
            str(contiguous): self._raw_id2label[str(sparse)]
            for contiguous, sparse in enumerate(self._all_ids_sorted)
        }

    @property
    def needs_remap(self) -> bool:
        return self._needs_remap

    @property
    def id2label(self) -> dict[str, str]:
        """Contiguous id2label for argmax lookups into model logits."""
        return self._contiguous_id2label

    @property
    def remapped_label2id(self) -> dict[str, str]:
        """label2id with contiguous values, safe for EuroBertUposLemmaConfig."""
        if not self._needs_remap:
            return self._raw_label2id
        return {
            label: str(self._sparse_to_contiguous[int(sparse_id)])
            for label, sparse_id in self._raw_label2id.items()
        }

    def candidate_ids(self, lang: str) -> np.ndarray:
        """Contiguous candidate IDs for a language's `lang::` prefixed labels.

        Falls back to all non-UNKNOWN labels if no `lang::` prefix is found.
        """
        labels = self._contiguous_id2label
        has_prefix = any(v.startswith(f"{lang}::") for v in labels.values() if v != "UNKNOWN")

        if has_prefix:
            ids = [
                int(k) for k, v in labels.items() if v != "UNKNOWN" and v.startswith(f"{lang}::")
            ]
        else:
            ids = [int(k) for k, v in labels.items() if v != "UNKNOWN"]

        return np.array(sorted(ids), dtype=np.int64)

    def config_label2id(self) -> dict[str, str]:
        """label2id safe for EuroBertUposLemmaConfig (contiguous, ≤ classifier size)."""
        return self.remapped_label2id
