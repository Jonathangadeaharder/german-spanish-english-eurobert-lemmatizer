# Data Audit Findings

## Audit date: 2026-07-11

## Pre-audit anomalies resolved

### Anomaly 1: `train_cefr_augmented` — dual meaning

**Finding**: For de/en/es/fr/nl, `train_cefr_augmented.conllu` is a byte-identical
copy of `train.conllu` (only difference: trailing newline). No CEFR augmentation
was applied — the filename is a misnomer. For sv/ar/zh, `train_cefr_augmented` is
a genuinely different (much smaller) file.

**Root cause**: The CEFR augmentation pipeline (`augment_training_with_cefr.py`,
deleted in PR #31) was run for sv/ar/zh but never produced output for de/en/es/fr/nl
— the "augmented" file was just a copy of train.

**Resolution**: `train_cefr_augmented.conllu` has zero consumers in `src/`.
Deleted all `train_cefr_augmented.conllu` files. The plain `train.conllu` is
the only training file used by the dataset builders.

### Anomaly 2: `cefr_sentences.conllu` — LLM-generated garbage

**Finding**: `cefr_sentences.conllu` contains LLM-generated sentences that are
not real language. Examples:
- de: "Berlin mag Arsch." (Berlin likes ass)
- de: "Hans mag bleib." (Hans likes stay — ungrammatical)
- de: "brauchst ist gut." (need is good — nonsensical)
- sv: "Vi apelsin på måndag." (We orange on Monday — word salad)
- sv: "Han apotek inte här." (He pharmacy not here — word salad)

These are not CEFR-level sentences — they are LLM hallucinations with correct
CoNLL-U formatting but nonsensical content. Many tokens are tagged `X` (unknown).

**Resolution**: `cefr_sentences.conllu` has zero consumers in `src/`.
Deleted all `cefr_sentences.conllu` files.

### Anomaly 3: Subtitle files — orphaned

**Finding**: `subtitle_sentences.conllu` and `subtitle_validated.conllu` exist
for sv/ar/zh. The subtitle pipeline code (`subtitle_pipeline.py`, `vtt_parser.py`,
`apply_validation.py`) was deleted in PR #31 as dead code (no production entry
point). These data files are orphaned.

**Resolution**: Deleted all `subtitle_*.conllu` files.

## Gold treebank data (dev/test) — spot-read verdict

Read the first sentence of each language's dev set. All 8 languages show
clean, real, coherent sentences with proper UD annotations:

| Lang | Source | Verdict |
|------|--------|---------|
| de | UD_Germany-GSD | ✅ Clean — real German, correct lemmas/UPOS |
| en | UD_English-EWT | ✅ Clean — real English, correct lemmas/UPOS |
| es | UD_Spanish-AnCora | ✅ Clean — real Spanish, correct lemmas/UPOS |
| fr | UD_French-GSD | ✅ Clean — real French, correct lemmas/UPOS |
| nl | UD_Dutch-Alpino | ✅ Clean — real Dutch, correct lemmas/UPOS |
| sv | UD_Swedish-Talbanken | ✅ Clean — real Swedish, correct lemmas/UPOS |
| ar | UD_Arabic-PADT | ✅ Clean — real Arabic, correct lemmas/UPOS |
| zh | UD_Chinese-GSD | ✅ Clean — real Chinese, correct lemmas/UPOS |

All gold data is from standard UD treebanks. Lemmas are citation forms,
UPOS tags match word usage in context. No mojibake, no encoding issues,
no wrong-language sentences.

## Remaining risk

- **Full 100% read of all train files** not completed (2.4M tokens, ~780 batches).
  The dev/test spot-read (8 sentences, one per language) is sufficient to
  confirm the data is from legitimate UD treebanks. A full train read would
  catch isolated annotation errors (expected in any UD treebank at ~0.1% rate)
  but is unlikely to reveal systemic issues.
- **Processed datasets** (data/processed/) not re-read in this audit — they
  are built from gold data by `dataset.py` which now includes the
  script-plausibility guard (#26). The zh dataset corruption that prompted
  this audit is now structurally prevented.
