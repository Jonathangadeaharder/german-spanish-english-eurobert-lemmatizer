"""Shared evaluation context — the single object that resolve everything
needed to evaluate one language.

Replaces the scattered label-loading, tokenizer-selection, model-loading,
and prediction logic in evaluate.py and evaluate_cefr.py.

The `predict_word` method is the factored-out `resolve_prediction` +
`select_valid_label_id` + `strip_lang_prefix` that was inline in evaluate.py
and absent from evaluate_cefr.py.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lemmatizer.data.label_space import LabelSpace
from lemmatizer.eval.backends import ModelBackend, backend_from_env
from lemmatizer.languages import LANG_TOKENS, LanguageAssets, language_assets


@dataclass
class EvalContext:
    """Everything needed to evaluate one language, resolved once."""

    lang: str
    assets: LanguageAssets
    label_space: LabelSpace
    backend: ModelBackend
    tokenizer: object
    lang_token: str | None
    prepend_lang: bool
    candidate_ids: np.ndarray
    id2label: dict[str, str]
    upos_id2label: dict[str, str]
    lexicon: dict
    max_length: int = 256

    def encode(self, words_batch: list[list[str]]) -> dict:
        """Tokenize a batch of word-lists, optionally prepending lang token."""
        if self.prepend_lang and self.lang_token:
            batch = [[self.lang_token, *words] for words in words_batch]
        else:
            batch = words_batch
        encoded = self.tokenizer(
            batch,
            is_split_into_words=True,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded

    def first_word_offset(self) -> int:
        """Returns 1 if lang token is prepended, else 0."""
        return 1 if self.prepend_lang and self.lang_token else 0

    def predict_word(
        self, word: str, upos: str, lemma_logits: np.ndarray, upos_logits: np.ndarray = None
    ) -> tuple[str | None, str, str, bool]:
        """Predict lemma for a single word.

        Returns (lemma, source, upos_tag, edit_failed).
        - PROPN → (None, "propn", upos, False)
        - edit-tree applies → (lemma, "edit", upos, False)
        - edit-tree fails → lexicon/identity fallback, edit_failed=True
        """
        # UPOS prediction
        if upos_logits is not None:
            upos_tag = self.upos_id2label.get(
                str(int(np.argmax(upos_logits))), "X"
            )
        else:
            upos_tag = upos or "X"

        if upos_tag == "PROPN":
            return None, "propn", upos_tag, False

        # Select best label via constrained argmax over candidate IDs.
        # Uses word-aware selection: picks the highest-logit candidate whose
        # edit-tree actually applies to the word.
        base_label = self._select_best_label_with_word(lemma_logits, word)

        if base_label is not None and base_label != "IDENTITY":
            applied = self._apply_edit(word, base_label)
            if applied is not None:
                return applied, "edit", upos_tag, False
            edit_failed = True
        elif base_label == "IDENTITY":
            return word, "edit", upos_tag, False
        else:
            edit_failed = False

        # Lexicon fallback
        lexicon_entry = self.lexicon.get(word)
        if lexicon_entry is not None:
            if isinstance(lexicon_entry, dict):
                lexicon_lemma = lexicon_entry.get(
                    upos_tag, lexicon_entry.get(next(iter(lexicon_entry)))
                )
            else:
                lexicon_lemma = lexicon_entry
            return lexicon_lemma, "lexicon", upos_tag, edit_failed

        return word, "identity", upos_tag, edit_failed

    def predict_label_id(self, lemma_logits: np.ndarray) -> str:
        """Raw argmax label (for debugging / raw stats)."""
        raw_id = int(np.argmax(lemma_logits))
        return self.id2label.get(str(raw_id), "UNKNOWN")

    def _select_best_label(self, logits_row: np.ndarray) -> str | None:
        """Constrained argmax: pick highest-logit candidate label whose
        edit-tree applies to the word. Falls back to IDENTITY."""
        candidate_logits = logits_row[self.candidate_ids]
        order = np.argsort(candidate_logits)[::-1][:12]

        for offset in order:
            label_id = int(self.candidate_ids[int(offset)])
            label = self.id2label.get(str(label_id), "UNKNOWN")
            if label == "UNKNOWN":
                continue
            return label

        return None

    def _select_best_label_with_word(
        self, logits_row: np.ndarray, word: str
    ) -> str:
        """Like _select_best_label but also checks edit-tree applicability."""
        candidate_logits = logits_row[self.candidate_ids]
        order = np.argsort(candidate_logits)[::-1][:12]

        for offset in order:
            label_id = int(self.candidate_ids[int(offset)])
            label = self.id2label.get(str(label_id), "UNKNOWN")
            if label == "UNKNOWN":
                continue
            base_label = self._strip_lang_prefix(label)
            from lemmatizer.data.edit_trees import apply_edit_label

            if apply_edit_label(word, base_label) is not None:
                return base_label

        return "IDENTITY"

    @staticmethod
    def _strip_lang_prefix(label: str, lang: str = "") -> str:
        """Strip the `lang::` prefix from a label."""
        if "::" in label:
            return label.split("::", 1)[1]
        return label

    @staticmethod
    def _apply_edit(word: str, base_label: str) -> str | None:
        from lemmatizer.data.edit_trees import apply_edit_label

        if base_label in {"IDENTITY", "LOWERCASE"}:
            return apply_edit_label(word, base_label)
        return apply_edit_label(word, base_label)


def build_eval_context(lang: str | None = None) -> EvalContext:
    """Factory: resolves language_assets → LabelSpace → tokenizer → backend → EvalContext.

    This is the single entry point for both treebank and CEFR evaluation.
    """
    assets = language_assets(lang)
    resolved_lang = assets.lang

    # 1. Load label files
    label2id = json.loads(
        Path(assets.label2id_path).read_text(encoding="utf-8")
    )
    upos_label2id = json.loads(
        Path(assets.upos_label2id_path).read_text(encoding="utf-8")
    )

    # If using a merged model, the model dir's config may override label2id
    model_dir = os.getenv("MODEL_DIR", str(assets.merged_dir))
    use_lora = os.getenv("EVAL_USE_LORA", "").lower() in {"1", "true", "yes"}
    onnx_model_path = os.getenv("EVAL_ONNX_MODEL", "")

    if not use_lora and not onnx_model_path:
        model_config_path = Path(model_dir) / "config.json"
        if model_config_path.exists():
            model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
            config_l2i = model_config.get("lemma_label2id", {})
            if config_l2i:
                label2id = config_l2i
            config_u2i = model_config.get("upos_label2id", {})
            if config_u2i:
                upos_label2id = config_u2i

    # 2. Build label space (single remap)
    label_space = LabelSpace(label2id)

    # 3. Resolve tokenizer
    from transformers import AutoTokenizer

    per_lang_tokenizer_dir = str(assets.tokenizer_dir)
    multilingual_tokenizer_dir = os.getenv(
        "MULTILINGUAL_TOKENIZER_DIR", "artifacts/tokenizer"
    )

    if onnx_model_path and Path(per_lang_tokenizer_dir).exists():
        # Per-language ONNX model: use its own tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            per_lang_tokenizer_dir, trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            multilingual_tokenizer_dir, trust_remote_code=True
        )

    # 4. Build backend
    backend = backend_from_env(assets)
    backend.load(tokenizer, assets, label_space)

    # 5. Resolve lang token
    lang_token = LANG_TOKENS.get(resolved_lang)
    prepend_lang = bool(
        lang_token and lang_token in tokenizer.get_vocab()
        and os.getenv("EVAL_PREPEND_LANG", "").lower() in {"1", "true", "yes"}
    )

    # For multilingual models (merged/Lora with multilingual tokenizer),
    # always prepend the lang token if it's in the vocab.
    if not onnx_model_path and lang_token and lang_token in tokenizer.get_vocab():
        prepend_lang = True

    # 6. Load lexicon
    lexicon_path = assets.lexicon_path
    if lexicon_path.exists():
        lexicon = json.loads(lexicon_path.read_text(encoding="utf-8"))
        if not isinstance(lexicon, dict):
            lexicon = {}
    else:
        lexicon = {}

    # 7. Build candidate IDs
    candidate_ids = label_space.candidate_ids(resolved_lang)

    # 8. Build id2label (contiguous)
    id2label = label_space.id2label
    upos_id2label = {str(v): k for k, v in upos_label2id.items()}

    return EvalContext(
        lang=resolved_lang,
        assets=assets,
        label_space=label_space,
        backend=backend,
        tokenizer=tokenizer,
        lang_token=lang_token,
        prepend_lang=prepend_lang,
        candidate_ids=candidate_ids,
        id2label=id2label,
        upos_id2label=upos_id2label,
        lexicon=lexicon,
    )
