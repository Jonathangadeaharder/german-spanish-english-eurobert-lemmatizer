# German, Spanish, English EuroBERT UPOS/Lemma tagger

One multilingual EuroBERT-210m token classifier for German, Spanish, and English that predicts UPOS for every token and predicts lemmas only when the token is not `PROPN`.

## What this repository does

- Builds language-prefixed edit-tree labels
- Builds a per-language lexicon backoff from train/dev data
- Converts gold CoNLL-U data into a token-classification dataset
- Fine-tunes `EuroBERT/EuroBERT-210m`
- Evaluates UPOS accuracy and lemma accuracy with lemma gated off for `PROPN`
- Merges LoRA adapters into the base model
- Exports the merged model to ONNX for Transformers.js
- Provides a browser/runtime postprocessor in JavaScript

## Repository layout

```text
data/
  gold/
    de/
    es/
    en/
  processed/

artifacts/
src/
web/
```

## Data requirements

Place gold-only UD CoNLL-U files here:

```text
data/gold/de/train.conllu
data/gold/de/dev.conllu
data/gold/de/test.conllu

data/gold/es/train.conllu
data/gold/es/dev.conllu
data/gold/es/test.conllu

data/gold/en/train.conllu
data/gold/en/dev.conllu
data/gold/en/test.conllu
```

Recommended treebanks:

- German: UD German-HDT or UD German-GSD
- Spanish: UD Spanish-AnCora or UD Spanish-GSD
- English: UD English-EWT

You can download the recommended gold files with:

```bash
uv run --active --no-sync eurobert-lemma fetch-ud
```

## Setup

```bash
bash scripts/bootstrap_arm64_venv.sh
source .venv/bin/activate
```

## Apple Silicon Setup

On the M5 Pro, use a native arm64 environment for training and benchmarking:

```bash
/opt/homebrew/bin/uv python install 3.13
bash scripts/bootstrap_arm64_venv.sh
source .venv/bin/activate
uv run --active --no-sync eurobert-lemma benchmark-mps --profile smoke
```

If you need to invoke uv manually, point it at the native Homebrew Python:

```bash
/opt/homebrew/bin/uv sync --python 3.13
```

## Pipeline

```bash
uv run --active --no-sync eurobert-lemma prepare
uv run --active --no-sync eurobert-lemma train --profile mps-full
uv run --active --no-sync eurobert-lemma evaluate --profile mps-full
uv run --active --no-sync eurobert-lemma merge --profile mps-full
uv run --active --no-sync eurobert-lemma export-onnx --profile mps-full
uv run --active --no-sync eurobert-lemma package-web --profile mps-full
```

To continue an interrupted run in the same output directory instead of starting fresh, set `TRAIN_RESUME_FROM=auto` or pass `--resume-from auto` to `uv run --active --no-sync eurobert-lemma train`.

## CLI

The primary entrypoint is `uv run --active --no-sync eurobert-lemma`.

```bash
uv run --active --no-sync eurobert-lemma --help
uv run --active --no-sync eurobert-lemma train --profile mps-full
uv run --active --no-sync eurobert-lemma benchmark-mps --profile smoke
```

## Runtime

The runtime entry point is `web/demo.js`.

```js
await lemmatize("Die besseren Ergebnisse wurden veröffentlicht.", "de");
await lemmatize("Los mejores resultados fueron publicados.", "es");
await lemmatize("The better results were published.", "en");
```

The runtime returns `{ word, upos, lemma, lang }` and sets `lemma` to `null` when the predicted UPOS is `PROPN`.
