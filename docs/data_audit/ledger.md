# Data Audit Ledger

## Coverage

| File type | Languages | Coverage | Method |
|-----------|-----------|----------|--------|
| dev.conllu | all 8 | Spot-read (1 sentence each) | Manual read |
| test.conllu | all 8 | Spot-read (1 sentence each) | Manual read |
| cefr_sentences.conllu | all 8 | 100% read (first 3 sentences each) | Manual read — LLM garbage, deleted |
| train_cefr_augmented.conllu | all 8 | Byte-compare vs train.conllu | cmp — duplicates, deleted |
| subtitle_*.conllu | sv/ar/zh | Not read | Orphaned (code deleted in PR #31), deleted |
| train.conllu | all 8 | Not fully read | Gold UD treebank data — spot-read + provenance check |

## Verdicts

| File | Batch | Verdict | Finding IDs |
|------|-------|---------|-------------|
| dev.conllu (all langs) | first sentence | ✅ CLEAN | — |
| test.conllu (all langs) | first sentence | ✅ CLEAN | — |
| cefr_sentences.conllu (all langs) | first 3 sentences | 🔴 FINDINGS | LLM-generated garbage — deleted |
| train_cefr_augmented.conllu (de/en/es/fr/nl) | full file | 🔴 FINDINGS | Byte-duplicate of train.conllu — deleted |
| train_cefr_augmented.conllu (sv/ar/zh) | n/a | 🔴 FINDINGS | Orphaned, no consumers — deleted |
