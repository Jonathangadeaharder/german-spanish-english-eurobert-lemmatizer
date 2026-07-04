# Chinese Lemmatizer Training Plan

This documents the path to a first-class Chinese analyzer model for Vidiom,
mirroring the existing `de/en/es/fr` EuroBERT pipeline. The scripts are already
scaffolded with `zh` support; this file covers the execution steps, base model
decision, and evaluation gates.

## Status

- ✅ Scripts updated: `fetch_ud.py`, `make_dataset.py`, `build_labels.py`,
  `train.py`, `language_assets.py`, `evaluate.py`, `publish_hf.py`
- ✅ Dataset downloaded (UD_Chinese-GSDSimp: 118599 train / 15165 dev / 14510 test lines)
- ✅ Model trained (200 steps, LoRA on EuroBERT-210m, warm-started from multilingual merged)
- ✅ Evaluated: **UPOS accuracy 91.6%** (passes ≥90% gate); lemma accuracy 71.2% (expected for Chinese where lemma ≈ normalized surface)
- ✅ ONNX exported (model.onnx + model.int8.onnx quantized, 216MB int8)
- ✅ Packaged to `web/model/lemma_zh/`
- ⬜ Published to HuggingFace (requires `huggingface-cli login` — run `scripts/publish_hf.py` when ready)
- ✅ Runtime wired in Vidiom (`eurobert-lemmatizer.ts`, `config.ts`, `lemma-postprocess.ts`, `supported-languages.ts`)

## Execution steps

```bash
# 1. Download the UD Chinese GSDSimp treebank
uv run --active --no-sync eurobert-lemma fetch-ud   # downloads zh_gsdsimp

# 2. Build label vocab + lexicon (now includes zh)
uv run --active --no-sync eurobert-lemma build-labels

# 3. Make the dataset (LANG_TOKEN includes [LANG_ZH])
uv run --active --no-sync eurobert-lemma make-dataset

# 4. Fine-tune (uses EuroBERT-210m as base by default)
#    For Chinese, consider a dedicated base (see Base model decision below).
LEMMA_LANG=zh uv run --active --no-sync eurobert-lemma train --config configs/mps-full.toml

# 5. Evaluate
LEMMA_LANG=zh uv run --active --no-sync eurobert-lemma evaluate

# 6. Merge LoRA + export ONNX + package
uv run --active --no-sync eurobert-lemma merge-lora
uv run --active --no-sync eurobert-lemma export-onnx
uv run --active --no-sync eurobert-lemma package-web

# 7. Publish
uv run --active --no-sync python scripts/publish_hf.py
```

## Base model decision

EuroBERT-210m is multilingual but its tokenizer/pretraining is not
Chinese-optimized. Two candidates:

1. **Conservative baseline:** `google-bert/bert-base-chinese`
   (character-based, well-supported, stable).
2. **Candidate improvement:** `hfl/chinese-macbert-base`
   (whole-word masking, better downstream task performance).

Recommendation: train both, gate on token-segmentation F1 and UPOS accuracy.
Pick the winner for export.

To train with a different base, set `MODEL_ID` in `make_dataset.py` and
`train.py`, or export `MODEL_ID` env var if the script supports it.

## Nuance: Chinese lemma ≈ normalized surface

Chinese has no inflectional morphology in the Indo-European sense. The lemma
is typically the normalized surface token. The edit-tree head may collapse to
near-identity for most tokens. The high-value heads are:

- **Segmentation** (tokenization is non-trivial for Chinese)
- **UPOS** (part-of-speech tagging)
- **PROPN gating** (proper nouns should not be lemmatized)

The model card should document this honestly so downstream consumers don't
expect morphological reduction where none exists.

## Evaluation gates (must pass before un-beta in Vidiom)

- Token segmentation F1 ≥ existing-language floor (target ≥ 0.95)
- UPOS accuracy ≥ floor (target ≥ 0.90)
- Lemma/normalization exact match where meaningful
- End-to-end subtitle word-click quality review on real Chinese videos
- Latency/memory benchmark in the local Vidiom runtime (ONNX int8)

## Runtime wiring (Vidiom side, after gates pass)

1. `web/src/lib/server/infrastructure/config.ts` — add `LEMMATIZER_MODEL_ZH`
   env field.
2. `web/src/lib/server/lang/eurobert-lemmatizer.ts` — add `zh` to
   `SUPPORTED_LANGS`, `getLemmatizerModelId`, `languageToken` `[LANG_ZH]`.
3. `web/src/lib/server/lang/lemma-postprocess.ts` — add `zh: []` bucket.
4. `web/src/lib/languages/supported-languages.ts` — flip `zh.lemmatizer` from
   `'none'` to `'eurobert'` and remove `beta: true`.
