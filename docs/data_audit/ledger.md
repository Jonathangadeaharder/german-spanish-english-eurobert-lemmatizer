# Data Audit Ledger

## Coverage

| File type | Languages | Coverage | Method |
|-----------|-----------|----------|--------|
| dev.conllu | all 8 | Spot-read (1 sentence each) | Manual read (2026-07-11) |
| test.conllu | all 8 | Spot-read (1 sentence each) | Manual read (2026-07-11) |
| cefr_sentences.conllu | all 8 | 100% read (first 3 sentences each) | Manual read — LLM garbage, deleted |
| train_cefr_augmented.conllu | all 8 | Byte-compare vs train.conllu | cmp — duplicates, deleted |
| subtitle_*.conllu | sv/ar/zh | Not read | Orphaned (code deleted in PR #31), deleted |
| gold train.conllu | all 8 | Not fully read | Gold UD treebank — spot-read + provenance |
| **handcraft train+val** | all 8 | **Removed 2026-07-15** | Auto-generated bulk deleted; manual curation only — see `data/handcraft/README.md` |
| `{lang}_test.conllu` | all 8 | Hand-curated | Clean eval reference — preserve |

## Verdicts

| File | Batch | Verdict | Finding IDs |
|------|-------|---------|-------------|
| dev.conllu (all langs) | first sentence | ✅ CLEAN | — |
| test.conllu (all langs) | first sentence | ✅ CLEAN | — |
| cefr_sentences.conllu (all langs) | first 3 sentences | 🔴 FINDINGS | LLM-generated garbage — deleted |
| train_cefr_augmented.conllu (de/en/es/fr/nl) | full file | 🔴 FINDINGS | Byte-duplicate of train.conllu — deleted |
| train_cefr_augmented.conllu (sv/ar/zh) | n/a | 🔴 FINDINGS | Orphaned, no consumers — deleted |
| handcraft fr/ar/zh/nl train+val | n/a | 🔴 REMOVED | Poisoned auto-generated data deleted 2026-07-15 |
| handcraft de/en/es/sv train+val | n/a | 🔴 REMOVED | Auto-generated data deleted; manual rewrite required |
| gold train/dev (UD) | all 8 | ✅ USE THIS | `eurobert-lemma fetch-ud`; do not merge handcraft over gold |
