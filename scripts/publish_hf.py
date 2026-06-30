"""Publish per-language int8 ONNX lemmatizers + assets to HuggingFace."""

import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo

USER = "Jonandrop"
LANGS = {
    "de": {"name": "German", "treebank": "UD German-GSD", "lemma": 91.09, "upos": 90.24},
    "en": {"name": "English", "treebank": "UD English-EWT", "lemma": 92.43, "upos": 86.34},
    "es": {"name": "Spanish", "treebank": "UD Spanish-AnCora", "lemma": 94.99, "upos": 94.16},
    "fr": {"name": "French", "treebank": "UD French-GSD", "lemma": 94.07, "upos": 97.46},
}
FP_LEMMA = {"de": 95.27, "en": 96.72, "es": 98.65, "fr": 94.07}

ROOT = Path(__file__).resolve().parent.parent
STAGE = Path("/tmp/hf_publish")


def card(lang: str, meta: dict) -> str:
    return f"""---
license: apache-2.0
language: {lang}
library_name: onnx
tags:
  - lemmatization
  - token-classification
  - eurobert
  - onnx
  - int8
base_model: EuroBERT/EuroBERT-210m
---

# EuroBERT-210m Lemmatizer ({meta["name"]}, int8 ONNX)

Per-language lemmatizer + UPOS tagger fine-tuned from
[EuroBERT/EuroBERT-210m](https://huggingface.co/EuroBERT/EuroBERT-210m) on {meta["treebank"]}.
Predicts an **edit-tree label** per token; applying it to the surface word yields the lemma.
Dynamic-int8 quantized ONNX (~204 MB).

## Accuracy (UD test, dynamic int8, valid-label masking ON)

| Metric | int8 | fp (reference) |
|--------|------|----------------|
| Lemma accuracy | {meta["lemma"]}% | {FP_LEMMA[lang]}% |
| UPOS accuracy | {meta["upos"]}% | — |

int8 trades accuracy for size; the fp16/torch model is several points higher (UPOS head is
most affected by quantization).

## CRITICAL: valid-label masking

Plain argmax over edit-tree labels often picks a label that is structurally invalid for the
word (its delete-segment doesn't match) — applying it fails and accuracy collapses. At
inference you MUST iterate candidate labels in descending logit order and pick the first
whose edit **applies** to the word (`IDENTITY` is the guaranteed fallback). This is
model-internal selection, not a hand-written rule.

## Files
- `model.int8.onnx` — inputs `input_ids`, `attention_mask`; outputs `upos_logits`, `lemma_logits`
- `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`
- `id2label.json` / `label2id.json` — edit-tree lemma labels
- `upos_id2label.json` / `upos_label2id.json` — UPOS labels
- `edit_trees.json` — edit-tree label definitions
- `lexicon.json` — fallback {{word: lemma}} lexicon

## Usage (sketch)
```python
import onnxruntime as ort
sess = ort.InferenceSession("model.int8.onnx")
upos_logits, lemma_logits = sess.run(["upos_logits", "lemma_logits"], {{
    "input_ids": ids, "attention_mask": mask}})
# decode: for each word, pick highest-logit lemma label whose edit applies (see masking note),
# apply it to the word; fall back to lexicon, then identity.
```
"""


def main():
    api = HfApi()
    STAGE.mkdir(parents=True, exist_ok=True)

    for lang, meta in LANGS.items():
        dst = STAGE / lang
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

        merged = ROOT / f"models/eurobert-lemma-{lang}-210m-merged"
        arts = ROOT / f"artifacts/lemma_{lang}"
        onnx_src = ROOT / f"onnx/eurobert-lemma-{lang}-210m/model.int8.onnx"

        shutil.copy(onnx_src, dst / "model.int8.onnx")
        for f in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]:
            shutil.copy(merged / f, dst / f)
        for f in [
            "id2label.json",
            "label2id.json",
            "upos_id2label.json",
            "upos_label2id.json",
            "edit_trees.json",
            "lexicon.json",
        ]:
            shutil.copy(arts / f, dst / f)

        (dst / "README.md").write_text(card(lang, meta))

        repo_id = f"{USER}/eurobert-lemma-{lang}-210m"
        print(f"=== {repo_id} ===")
        create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=str(dst), repo_type="model")
        print(f"published https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
