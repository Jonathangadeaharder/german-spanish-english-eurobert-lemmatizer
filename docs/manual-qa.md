# Lemmatizer training data — manual QA only

See also: [data/handcraft/README.md](../../data/handcraft/README.md)

## Sources

| Source | Path | Use |
|--------|------|-----|
| UD treebank | `data/gold/{lang}/train.conllu`, `dev.conllu`, `test.conllu` | **Primary** — `uv run eurobert-lemma fetch-ud` |
| Handcraft train/val | `data/handcraft/{lang}/train/`, `val/` | **Manual** — empty until curated |
| Handcraft eval | `data/handcraft/{lang}_test.conllu` | ONNX spot-check benchmark |

## Validators (read-only)

```bash
uv run pytest tests/test_conllu_validator.py tests/test_lemma_checker.py -q
```

## Removed (2026-07-15)

- `data/handcraft/_lib/` — Stanza batch generation, `merge_gold.py`
- `data/handcraft/**/generate_*.py` — per-wave generators (de/en/es)
- `scripts/llm_validate_gold.py`, `loop_until_clean.sh`, `run_validation.sh`
- `src/lemmatizer/data/zh_augment*.py`
- All auto-generated handcraft train/val `.conllu` (~49k sentences) — audit showed systemic corruption

Do not reintroduce bulk NLP rewrite pipelines. Every token is hand-verified or from UD gold.
