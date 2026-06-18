# Resume Guide — EuroBERT Per-Language Lemmatizer

State at wrap-up (2026-06-09). Read this before continuing.

## Current accuracy (UD test, applied lemma, valid-label masking ON)

| Lang | Treebank | Lemma | UPOS | failed_apply | SOTA ref | Realistic max |
|------|----------|-------|------|--------------|----------|---------------|
| DE   | GSD      | 95.27% | 96.04% | 0% | ~96% (Stanza, GSD is hard) | 96.5–97.5% |
| EN   | EWT      | 96.72% | 96.94% | 0% | 97.21% (Stanza) | 97.5–98.5% |
| ES   | AnCora   | 98.65% | 98.87% | 0% | 99.95% (UDPipe) | 99.0–99.5% |

## The key fix: valid-label masking

Root cause of the old DE 89.5%: the model's argmax edit-tree label is often
structurally invalid *for the word* (`apply_edit_label` returns None →
`failed_apply` was 19.35%). The same phenomenon = the 87% "spurious_edit" errors.

Fix = `select_valid_label_id` (`src/evaluate.py`): walk candidate labels in
descending logit order, return the first whose `apply_edit_label(word, label)`
succeeds; `IDENTITY` is the guaranteed terminal fallback. Model-internal
selection — NOT a hand-written lemma rule (preserves the no-postproc architecture).
Toggle: `EVAL_VALID_MASK` (default `1`). Tests: `tests/test_valid_label_masking.py`.

**This logic must exist in every runtime** or accuracy regresses to ~89.5%:
- Python: `src/evaluate.py::select_valid_label_id`
- Web JS: `web/postprocess.js::selectValidLabel` (mirror), called in `web/demo.js`.

## Re-merge a model from its adapter (no retraining)

Checkpoints were deleted; final LoRA adapters kept under `runs/eurobert-lemma-{lang}-210m-lora/`.
```
LEMMA_LANG=de uv run python -m merge_lora   # uses warm-start base + adapter -> models/eurobert-lemma-de-210m-merged
```
Warm-start base (required): `models/eurobert-multilingual-lemma-210m-merged`.

## Evaluate
```
LEMMA_LANG=de uv run python -m evaluate                       # torch merged model
EVAL_ONNX_MODEL=onnx/eurobert-lemma-de-210m/model.int8.onnx \
  LEMMA_LANG=de uv run python -m evaluate                     # ONNX (int8) parity check
```
Report: `artifacts/lemma_{lang}/eval_report.json`.

## ONNX export (per-language, int8)
```
for L in de en es; do
  MERGED_DIR=models/eurobert-lemma-$L-210m-merged \
  ONNX_DIR=onnx/eurobert-lemma-$L-210m \
  EXPORT_DTYPE=fp32 QUANTIZE=int8 uv run python -m export_onnx
done
```
Outputs `model.onnx` (fp32 base, ~813 MB) + `model.int8.onnx` (~204 MB; embeddings
dominate, so int8 doesn't shrink as much as a typical 4x). Deployed artifact is
`model.int8.onnx`; the fp32 base was deleted (regenerable from the merged model).
`EXPORT_DTYPE=fp16` + `QUANTIZE=none` reproduces the fp16 export (~407 MB).

### int8 accuracy cost (measured, masking ON) — chosen deploy
Dynamic int8 degrades these models noticeably, especially the UPOS head:
| Lang | lemma fp(torch) | lemma int8 | UPOS fp | UPOS int8 |
|------|------|------|------|------|
| DE | 95.27% | 91.09% | 96.04% | 90.24% |
| EN | 96.72% | 92.43% | 96.94% | 86.34% |
| ES | 98.65% | 94.99% | 98.87% | 94.16% |

If accuracy matters more than size, re-export `EXPORT_DTYPE=fp16 QUANTIZE=none` —
fp16 matches torch. Better int8 (per-channel / static / exclude UPOS-head ops)
is unexplored and could recover most of the gap.

### Web wiring TODO for int8 deploy
`web/demo.js` loads `model/model.onnx`. For int8, point it at `model.int8.onnx`
and re-run `package-web` so `selectValidLabel` (already ported to `web/postprocess.js`)
runs against the quantized model.

## To push accuracy further (next round, optional)
- **DE → ~96–97%**: class-weighted / focal lemma loss favoring IDENTITY/LOWERCASE,
  then retrain DE (`LEMMA_LANG=de uv run python -m train`, ~6h MPS) + re-merge.
  Residual errors are spurious edits on common words (CEFR A1 still weakest).
- **EN → ~98%**: edit-tree/lexicon coverage for irregular plurals + participle
  over-stemming.
- ES is near ceiling; lowest ROI.

## Don't
- Don't use the multilingual dataset for per-language training (it's the small gold set).
- Don't add hand-written lemma post-processing rules (architecture decision: corrections learned via training; masking is the only allowed inference-time step).
