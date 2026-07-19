# Handcraft lemmatizer data (manual only)

**No generation scripts.** All train/val CoNLL-U for lemmatizer training is curated by hand.

## What lives here

| Path | Role | Status |
|------|------|--------|
| `{lang}_test.conllu` | Small hand-curated eval set (100 sentences/lang) | **Keep** — clean reference |
| `{lang}/train/` | Per-language training sentences (CoNLL-U) | **Empty** — add manually |
| `{lang}/val/` | Per-language validation sentences (CoNLL-U) | **Empty** — add manually |

Languages: `de`, `en`, `es`, `fr`, `nl`, `sv`, `ar`, `zh`.

## Gold UD treebanks (primary training source)

Official UD train/dev/test splits live under `data/gold/{lang}/`. Fetch with:

```bash
uv run eurobert-lemma fetch-ud --lang de   # or omit --lang for all
```

Do **not** replace gold `train.conllu` / `dev.conllu` with handcraft-only content.

## Manual QA bar (every sentence, every token)

Before any file is accepted:

1. **Real language** — no word salad, no LLM garbage, grammatically coherent at the stated CEFR level.
2. **FORM / LEMMA / UPOS** — citation lemmas; UPOS matches usage in context.
3. **No `UPOS=X`** on content words (punctuation `PUNCT` only where appropriate).
4. **Consistency** — same `(FORM, UPOS)` → same `LEMMA` within the file.
5. **Language rules** — run validators after editing:

```bash
uv run python -m lemmatizer.data.conllu_validator   # via tests
uv run pytest tests/test_lemma_checker.py tests/test_conllu_validator.py -q
```

Per-language lemma rules: `src/lemmatizer/data/lemma_checker.py`.

## Removed (2026-07-15)

Automated Stanza generation (`_lib/`, `generate_*.py`, sentence Python modules, `merge_gold.py`, LLM validation scripts) was deleted after audits showed systemic annotation corruption. See `docs/data_audit/handcraft_synthesis.md`.
