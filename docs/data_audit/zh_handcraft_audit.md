# Chinese Handcraft Lemmatization Data Audit

## Audit date: 2026-07-15

## Verdict: 🔴 FINDINGS

The handcraft `zh` train/val CoNLL-U data (`data/handcraft/zh/{train,val}/*.conllu`,
5400 train + 600 val sentences, ~50% of all `zh` training sentences alongside the
5400-sentence gold treebank) is **structurally corrupted**: every sentence is
tokenized one Chinese character per token instead of by word, and the UPOS
column is a garbage/misaligned sequence that frequently tags ordinary content
characters as `PUNCT` and floods ~24% of tokens with `X`. Raw sentence text
(`zh_texts.txt`) itself is clean, real, CEFR-appropriate Mandarin — the
corruption is introduced downstream in the CoNLL-U annotation step.

## Scope covered

| Artifact | Coverage | Method |
|---|---|---|
| `docs/data_audit/handcraft_texts/zh_texts.txt` (6100 lines) | 100% read + scripted scan | Read tool (full file) + Python regex scan for non-CJK/script issues, duplicates |
| `data/handcraft/zh/train/*.conllu` (30 files, 5400 sents) | ≥5 sentences/file deep-read across all 6 CEFR levels (a1/a2/b1/b2/c1/c2) + full-corpus `awk` stats | Read tool + shell aggregation |
| `data/handcraft/zh/val/*.conllu` (6 files, 600 sents) | ≥5 sentences/file deep-read (a1, b2, c2 shown; aggregate stats cover all) | Read tool + shell aggregation |
| `data/handcraft/zh_test.conllu` (100 sents, 576 tokens) | 100% UPOS/lemma distribution scan + spot read | shell aggregation + Read |
| `data/handcraft/zh/sentences/*.py` | Spot-read (raw sentence source) | Read tool |
| `src/lemmatizer/data/lemma_checker.py`, `zh_lexicon.py`, `dataset.py` | Full read | Read tool — to determine what existing checks do/don't cover and how labels feed training |

## Finding 1 (🔴 real bug): Character-level tokenization instead of word-level

Every token in every train/val handcraft file is exactly one Chinese
character (byte-length 3, i.e. one CJK codepoint). Confirmed corpus-wide:

```
76113 tokens, all length 3 bytes (train+val, excluding real punctuation)
```

This contradicts:
- **The gold treebank** (`data/gold/zh/train.conllu`, UD_Chinese-GSD) — real UD Chinese word segmentation (multi-character words like 今天, 喜欢, 苹果).
- **`zh_test.conllu`** — also word-segmented (今天=1 token, 喜欢=1 token, 苹果=1 token; token lengths of 1/2/3 characters, not uniformly 1).
- **The model's own consumption**: `dataset.py::convert_file` treats each CoNLL-U "word" as an independent tokenizer unit via `is_split_into_words=True` and assigns one UPOS label per word. Feeding it single characters as "words" teaches the model an incorrect notion of Chinese word boundaries for roughly half of its `zh` training data.

The raw sentence source (`zh/sentences/*.py`) stores correctly unsegmented
text (e.g. `'一共多少钱？'`); the character-splitting happens in whatever
script produced the `.conllu`/`zh_texts.txt` artifacts (the generator script
itself was not found in `src/`/`scripts/` — likely a deleted/ad-hoc tool, but
its output is unambiguous from the data).

## Finding 2 (🔴 real bug): UPOS column is corrupted/misaligned, not real annotation

UPOS distribution across all train+val (79,927 tokens):

```
20984 X       (26.3% — should be a rare catch-all tag)
17344 PUNCT   (21.7% — see below, only ~8631 are real punctuation)
13301 NOUN
11577 VERB
 3343 PRON
 3339 SCONJ
 3275 PART
 2873 ADJ
 2839 ADV
 2217 ADP
 2128 AUX
  829 PROPN
  496 DET
  194 NUM
   87 CCONJ
```

`PUNCT` is applied to ~8631 non-punctuation characters — ordinary content
words like 的 (308×), 来 (274×), 地 (256×), 重 (173×), 去 (160×), 得 (143×),
手 (122×), 很 (111×), 看 (93×), 吃 (93×), 不 (84×), 了 (74×), 大 (71×),
高 (70×), etc. `X` is applied to another ~20,984 tokens — a rate no real UD
treebank exhibits (UD_Chinese-GSD's `X` rate is near zero).

The same character gets contradictory tags across near-identical sentences,
which rules out "consistent-but-nonstandard" labeling. Example (character 天
"day/sky", position 2 of "今天..." in five different sentences from
`a1_new_001.conllu`): tagged `VERB`, `VERB`, `ADJ`, `PRON`, `NOUN` — five
different tags for the same character in the same syntactic slot. Reverse-
engineering the pattern shows the corruption is **positional**: correct
word-level UPOS (e.g. 今天=NOUN, 下雪=VERB, 了=PART, 。=PUNCT) appear to have
been indexed onto the character stream one-tag-per-character-position
instead of being repeated for every character of a multi-character word,
so the tag list runs out mid-sentence, dumps `X` for the overflow, and
`PUNCT` (the tag meant for the sentence-final period) lands on whatever
character happens to occupy that list index.

This is not a cosmetic issue: `dataset.py` uses `upos_labels` as a genuine
multi-task training target (`upos_label_id = upos_label2id.get(upos, -100)`),
so these ~35,000+ wrong/garbage (character, UPOS) pairs are actively fed to
the model's shared trunk during training — this is the largest and most
consequential finding of this audit.

## Finding 3 (minor, not dirt): 6 exact-duplicate sentences

`zh_texts.txt` has 6 lines that duplicate another line verbatim (e.g. "他们
不喜欢看书。" appears twice). Out of 6100 lines this is noise-level and not
worth remediation on its own — flagged for completeness only.

## Non-findings — confirmed clean / convention, not bugs

- **`zh_texts.txt` raw text (100% read)**: every one of 6100 lines is real,
  coherent, CEFR-plausible Mandarin. No mojibake, no wrong script, no
  nonsense/word-salad (the `cefr_sentences.conllu` LLM-garbage pattern from
  the prior audit — see `findings.md` Anomaly 2 — does **not** recur here).
- **Train+val lemma column**: `lemma_checker.py`'s `_check_zh_lemma_equals_form`
  passes because every token is a single character, and single-character
  lemma trivially equals form. This is a side effect of Finding 1's
  tokenization bug, not evidence the lemma annotation logic is sound at
  word granularity — it was never exercised at word granularity for this
  data.
- **`zh_test.conllu` lemma≠form flags (我们→我, 不是→是, 都是→是, 不能→能,
  不会→会, 他们→他, 还是→是) — CONVENTION, not a bug.** `zh_lexicon.py`
  explicitly documents: *"Chinese lemma = surface form for 99.33% of gold
  test tokens. The only deviations are closed-class: 们-pronouns
  (他们→他), fused adverb/negation+verb (不是→是, 都是→是)."* Every single
  lemma≠form pair in `zh_test.conllu` matches this documented closed class
  exactly (verified by full-file scan). `lemma_checker.py`'s blanket
  `_check_zh_lemma_equals_form` rule doesn't carve out this documented
  exception, so it will flag these — that is a **checker specificity gap**,
  not a test-data defect.
- **`zh_test.conllu` UPOS/tokenization**: 100-sentence full scan shows
  correct word-level segmentation (token lengths 1–3 chars, not uniformly
  1), zero spurious `X`, and `PUNCT` applied only to real punctuation
  (。？，！). This is the correct convention that train/val should have
  matched and didn't.
- **`zh_test` text-mismatch (space-joined tokens vs unsegmented `# text`)**:
  expected for Chinese per the task brief — not content dirt.
- **Train+val format/lemma**: confirmed 0 format errors, 0 lemma errors via
  `lemma_checker.py` — accurate as stated, but scoped only to lemma
  plausibility, not UPOS plausibility (verified by reading the checker
  source; it has no UPOS-vs-form sanity check for any language).

## Recommendation

Do not use `data/handcraft/zh/train/*.conllu` or `.../val/*.conllu` in their
current form for multi-task (lemma+UPOS) training — the UPOS signal is
noise and the tokenization granularity mismatches both the gold treebank and
`zh_test.conllu`. Regenerate from `zh/sentences/*.py` (which are clean) with
proper word segmentation and per-word UPOS repeated correctly across each
word's characters (the BIO-style approach already used correctly in
`src/lemmatizer/data/zh_augment.py`/`zh_augment2.py` for the separate
`zh_bio` POS-tagging dataset is a useful reference for a correct
per-character-with-inherited-word-tag scheme, if character-level tokens are
intentional; otherwise segment to real words to match `zh_test.conllu`).
