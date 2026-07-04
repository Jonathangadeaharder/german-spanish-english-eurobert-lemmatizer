"""Strategy interface for model backends.

Replaces the 3-way inline branches (ONNX/LoRA/merged) in evaluate.py and
the 2-way branch (LoRA/merged, no ONNX) in evaluate_cefr.py.

Each backend implements `run(encoded) -> (upos_logits, lemma_logits)` so the
evaluation loop is backend-agnostic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class ModelBackend(Protocol):
    """Interface for model backends used by the evaluation pipeline."""

    def load(
        self,
        tokenizer,
        assets,
        label_space,
    ) -> None:
        """Load the model, tokenizer-aware assets, and label space."""
        ...

    def run(self, encoded: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Run inference. Returns (upos_logits, lemma_logits) as numpy arrays."""
        ...

    def close(self) -> None:
        """Release resources (model, ONNX session, etc.)."""
        ...


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OnnxBackend:
    """Loads an ONNX int8 model via onnxruntime.

    Uses the per-language tokenizer (not the multilingual one) because
    per-language ONNX models (sv, ar) have their own embedding tables
    that don't include [LANG_*] tokens from the multilingual tokenizer.
    """

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._session = None

    def load(self, tokenizer, assets, label_space) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            self._model_path, providers=["CPUExecutionProvider"]
        )

    def run(self, encoded: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        ort_inputs = {
            "input_ids": encoded["input_ids"].cpu().numpy(),
            "attention_mask": encoded["attention_mask"].cpu().numpy(),
        }
        upos_logits, lemma_logits = self._session.run(
            ["upos_logits", "lemma_logits"], ort_inputs
        )
        return (
            np.asarray(upos_logits, dtype=np.float32),
            np.asarray(lemma_logits, dtype=np.float32),
        )

    def close(self) -> None:
        self._session = None


class MergedBackend:
    """Loads a merged EuroBertForUposLemma model from a directory."""

    def __init__(self, model_dir: str, device: str = ""):
        self._model_dir = model_dir
        self._device = device or get_device()
        self._model = None

    def load(self, tokenizer, assets, label_space) -> None:
        from multitask_model import EuroBertForUposLemma

        model_config_path = Path(self._model_dir) / "config.json"

        if model_config_path.exists():
            # Load directly from model dir, using its own config.json.
            # But override lemma_label2id with the contiguous version if the
            # model's config has sparse IDs that don't match the classifier.
            config_data = json.loads(model_config_path.read_text(encoding="utf-8"))
            config_l2i = config_data.get("lemma_label2id", {})
            config_max = max(int(v) for v in config_l2i.values()) if config_l2i else 0
            classifier_size = self._get_classifier_size(self._model_dir)

            if classifier_size is not None and config_max > classifier_size:
                # Config has sparse IDs — use contiguous label2id instead
                from multitask_model import EuroBertUposLemmaConfig

                upos_label2id = json.loads(
                    Path(assets.upos_label2id_path).read_text(encoding="utf-8")
                )
                config = EuroBertUposLemmaConfig(
                    base_model_name_or_path=config_data.get(
                        "base_model_name_or_path", "EuroBERT/EuroBERT-210m"
                    ),
                    upos_label2id=upos_label2id,
                    lemma_label2id=label_space.config_label2id(),
                )
                self._model = EuroBertForUposLemma.from_pretrained(
                    self._model_dir,
                    config=config,
                    ignore_mismatched_sizes=True,
                    trust_remote_code=True,
                )
            else:
                self._model = EuroBertForUposLemma.from_pretrained(
                    self._model_dir,
                    ignore_mismatched_sizes=True,
                    trust_remote_code=True,
                )
        else:
            self._model = EuroBertForUposLemma.from_pretrained(
                self._model_dir, trust_remote_code=True
            )

        self._model.resize_token_embeddings(len(tokenizer))

        self._model.to(self._device)
        self._model.eval()

    def _get_classifier_size(self, model_dir: str) -> int | None:
        """Read the lemma_classifier weight shape from the safetensors checkpoint."""
        safetensors_path = Path(model_dir) / "model.safetensors"
        if not safetensors_path.exists():
            return None
        try:
            from safetensors import safe_open

            with safe_open(str(safetensors_path), framework="pt") as f:
                for key in f.keys():
                    if key == "lemma_classifier.weight":
                        return f.get_tensor(key).shape[0]
        except Exception:
            pass
        return None

    def run(self, encoded: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        import torch

        model_inputs = {
            key: value.to(self._device) for key, value in encoded.items()
        }
        with torch.inference_mode():
            outputs = self._model(**model_inputs)
        upos_logits = outputs.logits[0].detach().cpu().numpy()
        lemma_logits = outputs.logits[1].detach().cpu().numpy()
        return upos_logits, lemma_logits

    def close(self) -> None:
        self._model = None


class LoraBackend:
    """Loads a base EuroBERT model + LoRA adapter."""

    def __init__(self, adapter_dir: str, base_model: str = "", device: str = ""):
        self._adapter_dir = adapter_dir
        self._base_model = base_model or "EuroBERT/EuroBERT-210m"
        self._device = device or get_device()
        self._model = None

    def load(self, tokenizer, assets, label_space) -> None:
        from peft import PeftModel

        from multitask_model import EuroBertForUposLemma, EuroBertUposLemmaConfig

        upos_label2id = json.loads(
            Path(assets.upos_label2id_path).read_text(encoding="utf-8")
        )
        config = EuroBertUposLemmaConfig(
            base_model_name_or_path=self._base_model,
            upos_label2id=upos_label2id,
            lemma_label2id=label_space.config_label2id(),
        )
        base_model = EuroBertForUposLemma.from_pretrained(
            self._base_model,
            config=config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
        )
        base_model.resize_token_embeddings(len(tokenizer))
        self._model = PeftModel.from_pretrained(base_model, self._adapter_dir)

        self._model.to(self._device)
        self._model.eval()

    def run(self, encoded: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        import torch

        model_inputs = {
            key: value.to(self._device) for key, value in encoded.items()
        }
        with torch.inference_mode():
            outputs = self._model(**model_inputs)
        upos_logits = outputs.logits[0].detach().cpu().numpy()
        lemma_logits = outputs.logits[1].detach().cpu().numpy()
        return upos_logits, lemma_logits

    def close(self) -> None:
        self._model = None


def backend_from_env(assets) -> ModelBackend:
    """Factory: selects backend from environment variables.

    - EVAL_ONNX_MODEL set → OnnxBackend
    - EVAL_USE_LORA=true → LoraBackend
    - Otherwise → MergedBackend (default to assets.merged_dir)
    """
    onnx_model_path = os.getenv("EVAL_ONNX_MODEL", "")
    if onnx_model_path:
        return OnnxBackend(onnx_model_path)

    use_lora = os.getenv("EVAL_USE_LORA", "").lower() in {"1", "true", "yes"}
    if use_lora:
        adapter_dir = os.getenv("MODEL_DIR", str(assets.output_dir))
        base_model = os.getenv("BASE_MODEL", "")
        return LoraBackend(adapter_dir, base_model)

    model_dir = os.getenv("MODEL_DIR", str(assets.merged_dir))
    return MergedBackend(model_dir)
