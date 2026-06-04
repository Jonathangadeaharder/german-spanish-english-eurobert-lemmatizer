# Multilingual EuroBERT Lemmatizer Plan

## Goal

Build one EuroBERT-610m multilingual token classifier for German, Spanish, and English. The model predicts one language-specific edit-tree label per original word. Runtime applies the predicted edit label in JavaScript and returns `{ word, lemma, lang }`.

Do not train three separate models for v1.

## Fixed decisions

- Base model: `EuroBERT/EuroBERT-610m`
- Languages: German, Spanish, English
- Task: token classification
- Input: sentence + language code
- Output: one edit-tree label per token
- Runtime output: `[{ "word": "...", "lemma": "...", "lang": "de" }]`
- Training framework: Hugging Face Transformers
- Fine-tuning method: LoRA first
- Deployment target: ONNX + Transformers.js
- Gold data format: CoNLL-U

Language codes:

- `de` = German
- `es` = Spanish
- `en` = English

The model must not infer language in v1. The caller passes it.

Example API:

```js
await lemmatize("Die besseren Ergebnisse wurden veröffentlicht.", "de")
await lemmatize("Los mejores resultados fueron publicados.", "es")
await lemmatize("The better results were published.", "en")
```

## Main change: language-prefixed labels

Use language-prefixed labels instead of shared labels:

```text
de::IDENTITY
de::LOWERCASE
de::P4|S0|Ds|I

es::IDENTITY
es::LOWERCASE
es::P4|S0|Ds|I

en::IDENTITY
en::LOWERCASE
en::P4|S0|Ds|I
```

This avoids German, Spanish, and English edit patterns colliding.

At runtime, only accept labels that start with the requested language prefix.

If the model predicts an English label for a German word, ignore it and fall back.

## Repository layout

```text
german-spanish-english-eurobert-lemmatizer/
  README.md
  requirements.txt

  data/
    gold/
      de/
        train.conllu
        dev.conllu
        test.conllu
      es/
        train.conllu
        dev.conllu
        test.conllu
      en/
        train.conllu
        dev.conllu
        test.conllu
    processed/

  artifacts/
    label2id.json
    id2label.json
    edit_trees.json
    exceptions.json

  src/
    conllu_reader.py
    edit_trees.py
    build_labels.py
    make_dataset.py
    train.py
    evaluate.py
    merge_lora.py
    export_onnx.py

  web/
    postprocess.js
    demo.js
```

## Data placement

Put the CoNLL-U files here:

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

Use Universal Dependencies treebanks.

Recommended first choices:

- `de`: UD German-HDT or UD German-GSD
- `es`: UD Spanish-AnCora or UD Spanish-GSD
- `en`: UD English-EWT

Keep the test sets gold-only. Never add silver or teacher-generated data to test.

## Step 1: Update CoNLL-U reader

Add language metadata in `src/conllu_reader.py`.

## Step 2: Keep edit tree logic language-neutral

Keep `src/edit_trees.py` language-neutral. The edit operation itself does not need to know the language.

Language prefixing happens outside:

```python
full_label = f"{lang}::{base_label}"
```

## Step 3: Build multilingual label inventory

Build labels from the gold train/dev splits only.

Rules:

- Count labels per language-prefixed label
- Keep `UNKNOWN`
- Keep `IDENTITY` and `LOWERCASE` for every language even if rare
- Persist `label2id.json`, `id2label.json`, and `edit_trees.json`

Run:

```bash
PYTHONPATH=src python src/build_labels.py
```

## Step 4: Add language token to input

Prepend a fake language token as the first word in each sentence:

```text
[LANG_DE] Die besseren Ergebnisse wurden veröffentlicht .
[LANG_ES] Los mejores resultados fueron publicados .
[LANG_EN] The better results were published .
```

The language token gets label `-100` and is ignored by the loss.

Add these special tokens to the tokenizer:

```text
[LANG_DE]
[LANG_ES]
[LANG_EN]
```

## Step 5: Convert multilingual CoNLL-U to dataset

Create a single concatenated dataset across `de`, `es`, and `en`.

Save it to:

```text
data/processed/eurobert_multilingual_lemma_dataset
```

Save the tokenizer with the added language tokens to:

```text
artifacts/tokenizer
```

## Step 6: Train multilingual EuroBERT

Use:

- `MODEL_ID = "EuroBERT/EuroBERT-610m"`
- `TOKENIZER_DIR = "artifacts/tokenizer"`
- `DATASET_PATH = "data/processed/eurobert_multilingual_lemma_dataset"`
- `OUTPUT_DIR = "runs/eurobert-multilingual-lemma-610m-lora"`

Load tokenizer from `TOKENIZER_DIR`.

After creating the model, resize embeddings because the language tokens were added:

```python
model.resize_token_embeddings(len(tokenizer))
```

## Step 7: Evaluate per language

Report separate scores for `de`, `es`, and `en`.

When applying a predicted label:

```python
full_label = id2label.get(pred_id, "UNKNOWN")

expected_prefix = f"{lang}::"

if not full_label.startswith(expected_prefix):
    pred_lemma = word
else:
    base_label = full_label[len(expected_prefix):]
    pred_lemma = apply_edit_label(word, base_label)
```

If `pred_lemma is None`, fall back to `word`.

Acceptance criteria:

```text
de lemma_accuracy >= 0.95
es lemma_accuracy >= 0.95
en lemma_accuracy >= 0.95

failed_apply <= 0.01 per language
wrong-language-label rate <= 0.02 per language
```

## Step 8: JavaScript postprocessor must be language-aware

Replace `web/postprocess.js` so it strips a language prefix, applies an edit label, tokenizes simple text, and maps language codes to `[LANG_*]` tokens.

## Step 9: JavaScript runtime API

Update `web/demo.js` so the caller passes language explicitly.

Required API:

```js
await lemmatize(text, lang)
```

Valid `lang` values:

- `de`
- `es`
- `en`

The runtime should:

- tokenize the input
- prepend the correct language token
- validate that predicted labels match the requested language prefix
- apply edit-tree labels in JavaScript
- fall back to the original word on invalid or mismatched labels

## Step 10: Multilingual command sequence

Run:

```bash
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

PYTHONPATH=src python src/build_labels.py
PYTHONPATH=src python src/make_dataset.py
PYTHONPATH=src python src/train.py
PYTHONPATH=src python src/evaluate.py
PYTHONPATH=src python src/merge_lora.py
PYTHONPATH=src python src/export_onnx.py
```

Copy web artifacts:

```bash
mkdir -p web/model

cp -r onnx/eurobert-multilingual-lemma-610m/* web/model/
cp artifacts/id2label.json web/model/
cp artifacts/edit_trees.json web/model/

cp models/eurobert-multilingual-lemma-610m-merged/config.json web/model/
cp models/eurobert-multilingual-lemma-610m-merged/tokenizer.json web/model/
cp models/eurobert-multilingual-lemma-610m-merged/tokenizer_config.json web/model/
cp models/eurobert-multilingual-lemma-610m-merged/special_tokens_map.json web/model/
```

## Dumb-model one-sentence instruction

Build one EuroBERT-610m multilingual token classifier for German, Spanish, and English by prepending `[LANG_DE]`, `[LANG_ES]`, or `[LANG_EN]`, training it to predict language-prefixed edit-tree labels like `de::P4|S0|Ds|I`, evaluating by applying those labels back into lemmas, exporting to ONNX, and using JavaScript only for tokenization, language-prefix validation, and edit-tree postprocessing.
