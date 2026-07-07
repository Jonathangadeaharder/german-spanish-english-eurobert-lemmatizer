# EuroBERT Lemmatizer

Per-language UPOS + lemma taggers trained on Apple MLX, evaluated with a
multi-backend Python stack (ONNX / merged / LoRA), and **served in the
browser via Transformers.js + onnxruntime-web — no server**.

Seven languages share one repo, three model families:

| family | languages | backbone |
|---|---|---|
| multilingual multitask | de, en, es, fr, sv | EuroBERT-210m / ScandiBERT |
| ByT5 encoder + lemma head | ar | google/byt5-small |
| BERT BIO-POS | zh | bert-base-chinese (via openmed) |

## Held-out validation (de / es / en)

| language | UPOS acc | lemma acc | tokens |
|---|---|---|---|
| Spanish  | 96.7% | 96.2% | 212 |
| German   | 94.4% | 91.8% | 196 |
| English  | 93.1% | 94.1% | 204 |

Lemmas are predicted via language-prefixed **edit-trees** with a gold-lexicon
backoff, gated off for `PROPN`.

## Repository layout

```text
src/lemmatizer/
  cli.py              Typer entry point (registry-driven dispatch)
  languages.py        LANGUAGES registry — single source of truth
  data/               CoNLL-U reader, UD fetch, labels, dataset builders
  train/              TrainOptions + train_language() dispatcher + MLX trainers
  eval/               EvalContext + backends + treebank/CEFR eval
  export/             ByT5 ONNX bridge + web packaging
  inference/           Postprocess rules
data/                 Gold UD + processed HF datasets
artifacts/            Built lexicons / edit_trees / id2label (serving assets)
web/                  Browser runtime (demo.js + ONNX model bundle)
```

## Adding a language

One entry in `src/lemmatizer/languages.py`:

```python
LanguageSpec(
    lang="it", name="italian", family=Family.MULTITASK,
    base_model="EuroBERT/EuroBERT-210m", lang_token="[LANG_IT]",
    ud_repo="UD_Italian-ISDT", ud_prefix="it_isdt",
    spacy_model="it_core_news_lg", vocab_lemma_column="Italian_Lemma",
),
```

Then: `eurobert-lemma fetch-ud --lang it` → `build-labels` → `make-dataset`
→ `train --lang it --checkpoint EuroBERT/EuroBERT-210m` → `evaluate`.

No other file edits. Trainers, data pipeline, eval, and CLI all read from
the `LANGUAGES` registry. A new *family* (rare) needs a `Family` member + a
`run()` in a new `train/` module + one branch in `train_language()`.

## Setup

```bash
uv sync
```

## CLI

```bash
uv run eurobert-lemma --help
uv run eurobert-lemma fetch-ud
uv run eurobert-lemma prepare
uv run eurobert-lemma train --lang de
uv run eurobert-lemma evaluate --model-dir models/eurobert-lemma-de-210m-merged
uv run eurobert-lemma export-onnx --lang ar
uv run eurobert-lemma package-web
```

Each MLX trainer also accepts its own argparse flags — see
`uv run python -m lemmatizer.train.mlx_multitask --help` etc.

## Runtime

The browser runtime is `web/demo.js`:

```js
await lemmatize("Die besseren Ergebnisse wurden veröffentlicht.", "de");
await lemmatize("Los mejores resultados fueron publicados.", "es");
await lemmatize("The better results were published.", "en");
```

Returns `{ word, upos, lemma, lang }`; `lemma` is `null` when UPOS is `PROPN`.

## Known gaps

- MLX→ONNX export only exists for Arabic (`export.byt5_onnx`). The
  de/en/es/fr/sv/zh ONNX bundles currently ship from prior PyTorch exports;
  regenerating them from MLX checkpoints needs a new bridge.
- `openmed` (ZH trainer only) is an undeclared optional dependency — install
  separately if training Chinese.
