# Arabic Handcraft Data Audit

## Audit date: 2026-07-15

## Scope

- `docs/data_audit/handcraft_texts/ar_texts.txt` (6,100 lines) + `ar_meta.tsv` — full read (automated scan of every line + manual spot-reads across the whole file).
- `data/handcraft/ar/train/*.conllu` (5,400 sentences, 30 files) and `data/handcraft/ar/val/*.conllu` (600 sentences, 6 files) — full automated pass over every token (68,743 train + 7,550 val tokens) plus manual deep-read of ≥3 sentences per CEFR level (A1–C2) from val and additional train samples.
- `data/handcraft/ar_test.conllu` (100 sentences, 557 tokens) — full automated pass, used as the clean-baseline comparator.
- `data/handcraft/ar/sentences/*.py` — spot-checked to confirm these are raw-text sources (no lemma/UPOS authored here; annotations are produced by a downstream pipeline).
- `src/lemmatizer/data/lemma_checker.py` — read to understand exactly what the "0 lemma errors" claim actually validates.

## Verdict: 🔴 FINDINGS (not clean)

The "0 format, 0 lemma errors" claim in the task brief is **technically true but misleading** — the per-file lemma checker only checks Arabic-script-ness, PUNCT identity, and *within-file* FORM+UPOS→LEMMA consistency. It cannot see the two real defects below, both of which are systematic and material.

## Summary table

| Check | Result |
|---|---|
| Latin-script / mojibake corruption (texts, FORM, LEMMA) | ✅ CLEAN — 0 hits anywhere |
| Non-Arabic lemma for non-PUNCT tokens | ✅ CLEAN — 0 hits |
| Stray non-Arabic punctuation | 🟡 1 hit (fullwidth `？`, propagated end-to-end) |
| Exact duplicate sentences | 🟡 7 duplicate pairs, 1 crosses train↔val |
| Templated-sentence grammar defect | 🟡 60 sentences with a dangling gerund ("وإرساء") |
| و-prefix root stripped as false conjunction (lemma bug) | 🔴 125 tokens / 122 sentences, train+val only, 0 in test |
| Unlemmatized "tail collapse" (lemma=form, UPOS=X) | 🔴 27,723 train + 3,384 val tokens (40–45%) in 55%+ of sentences; 0 in test |
| UPOS tag inconsistency for common function words | 🔴 392 distinct forms with ≥2 UPOS tags; X = 43.7% of train / 48.1% of val vs. 0% of test |

## Findings

### 1. Script integrity — CLEAN

Scanned all 6,100 lines of `ar_texts.txt` for Latin letters, digits, and any character outside the Arabic Unicode blocks + standard punctuation. Also scanned every FORM and LEMMA column across all 68,743 train + 7,550 val + 557 test tokens for Latin characters or non-Arabic-script lemmas on non-PUNCT tokens. **Zero hits.** No mojibake, no Latin corruption, no wrong-script lemmas anywhere in the handcraft set.

### 2. One stray non-Arabic punctuation character

Line 1056 of `ar_texts.txt` (`ar_a2_train_156`, file `ar/train/a2_new_001.conllu`) uses the **fullwidth question mark `？` (U+FF1F, CJK punctuation)** instead of the Arabic question mark `؟` (U+061F):

> كيف أذهب إلى وسط المدينة من هنا？

This propagated consistently through the pipeline: it's in the Python source (`sentences/a2_train_new_001.py:157`), the `# text` comment, and token 8 (FORM=LEMMA=`？`, UPOS=PUNCT). Cosmetic/minor, but wrong for MSA text and inconsistent with the rest of the corpus, which correctly uses `؟` for all other ~1,600 questions.

The same sentence also illustrates the lemma bug in §5: token 4 `وسط` ("middle/downtown", a NOUN whose root starts with و) is lemmatized as `سط` — the و is wrongly stripped.

### 3. Duplicate sentences (7 pairs, including cross-split leakage)

7 exact-duplicate `text` strings exist across different source files:

| Duplicate text | Locations |
|---|---|
| هل يشرب أبي الشاي؟ | train a1_new_001 #84, train a2_new_004 #671 |
| هل تكتب أمي الواجب؟ | train a1_new_001 #137, train a2_new_004 #768 |
| هل تكتب أختي الواجب؟ | train a1_new_003 #459, train a2_new_005 #816 |
| هل يشرب أبي الحليب؟ | train a1_new_003 #538, train a2_new_003 #548 |
| هل أشرب أنا الحليب؟ | train a1_new_005 #837, train a2_new_003 #531 |
| هل يشرب صديقي الشاي؟ | train a1_new_005 #867, train a2_new_005 #806 |
| **هل تشرب أختي الحليب؟** | **train a2_new_001 #079** ↔ **val a1 #051** |

The first six are just A1/A2-level cross-contamination within train (harmless — same split). The last one is a genuine **train↔val leak**: the identical sentence appears verbatim in both the A2 train file and the A1 val file, which inflates val "accuracy" for that item since the model will have seen it verbatim in training. Minor in isolation (1 sentence / 6,000), but worth de-duplicating.

### 4. Templated C1/C2 sentences: one dangling grammatical construction

The B2–C2 texts are heavily combinatorial (a fixed subject/topic clause + one of ~10–30 interchangeable closing clauses), which is a reasonable design for vocabulary coverage at scale and is *not itself* a defect — all other templates checked are grammatically complete. However, one closing-clause template is broken:

> …يساعد على ترسيخ دعائم الحكم الرشيد **وإرساء** لتحقيق التنمية المستدامة…

`وإرساء` ("and establishing") is a gerund that requires a noun object (e.g., "وإرساء العدالة" — "and establishing justice"); as written it dangles with no object before the next templated clause starts. This exact broken clause occurs **60 times** in `ar_texts.txt` (3 base sentences × ~20 closing-clause variants), always as a mid-sentence fragment, e.g.:

> إن التطور التكنولوجي المتسارع في العصر الحديث يساعد على ترسيخ دعائم الحكم الرشيد وإرساء لتحقيق التنمية المستدامة والازدهار للمجتمع بأكمله.

This is a genuine MSA coherence defect in the sentence-generation template, not an annotation error.

### 5. 🔴 Real lemma bug: root-initial و mistaken for the conjunction prefix

125 tokens (114 train + 11 val, **0 in test**) across 122 distinct sentences have the lemma wrongly computed by stripping a leading و as if it were the coordinating conjunction "and", even though the و is part of the word's actual root — producing lemmas that are not real Arabic words:

| FORM | Wrong LEMMA produced | Correct LEMMA (seen elsewhere in the same corpus) |
|---|---|---|
| وجبة (meal) | جبة | وجبة — 56/56 occurrences wrong, **100%** |
| وقت (time) | قت | وقت — 32/40 occurrences wrong (80%) |
| وقتاً (time, acc.) | قتاً | — 32/32 wrong (100%; correct form `وقتاً` never seen) |
| وسائل (means) | سائل | وسائل — 3/23 wrong |
| وسط (middle) | سط | — 1/1 wrong (only occurrence) |
| ورش (workshops) | رش | — 1/1 wrong (only occurrence) |

This is provably a bug, not a valid alternate convention: the *same* surface form (`وقت` NOUN, `وسائل` NOUN) is correctly lemmatized with the leading و preserved in the majority of its occurrences (`وقتهم`→`وقتهم`, `وقتك`→`وقتك`, `وسيلة`→`وسيلة`, `وسائل`→`وسائل` 20×) but incorrectly stripped in others. Crucially, the correct and incorrect variants of a given (FORM, UPOS) pair **never co-occur within the same file** — which is exactly why `lemma_checker.py`'s within-file consistency check reports 0 errors: the inconsistency is only visible corpus-wide, across files, which the checker never compares.

### 6. 🔴 Critical: massive unlemmatized "tail collapse" (lemma=form, UPOS=X)

This is the dominant real defect and far larger than the "0 lemma errors" claim suggests. In a large majority of B1–C2 sentences (and some A1/A2), only the first few tokens are genuinely lemmatized/POS-tagged — from some point onward, **every remaining non-punctuation token has LEMMA identical to FORM and UPOS=X**, i.e. it was never actually analyzed. Example (`ar_c2_val_082`):

```
1  مما     مِمَّا   CCONJ
2  لا      لا      PART
3  شك      شَكّ    VERB
4  فيه     فيه     X   ← degenerate tail starts here
5  أن      أن      X
6  الحفاظ  الحفاظ  X
7  على     على     X
8  الهوية  الهوية  X
...(all 19 remaining tokens: X, lemma=form)...
```

Measured across the whole corpus (trailing run of ≥3 non-PUNCT tokens with LEMMA==FORM and UPOS==X):

| Split | Sentences w/ degenerate tail | Tokens affected |
|---|---|---|
| train | 3,002 / 5,400 (**55.6%**) | 27,723 / 68,743 (**40.3%**) |
| val | 328 / 600 (**54.7%**) | 3,384 / 7,550 (**44.8%**) |
| test | 0 / 100 (**0.0%**) | 0 / 557 (**0.0%**) |

`ar_test.conllu` has zero instances of this pattern — it is completely clean. This is a stark train/val-vs-test quality gap: whatever annotation pipeline produced the handcraft train/val CoNLL-U files (an LLM- or morphological-analyzer-based tagger, based on the sentence Python sources containing only raw text) systematically fails partway through longer/more complex sentences and falls back to copying the surface form with a placeholder tag, rather than erroring out or being caught by validation.

**This is not a benign convention.** `X` is treated in the codebase (`byt5_lemma_model.py`, `oracle_ceilings.py`, `handcraft_eval.py`, etc.) as `IDENTITY_UPOS` — "no morphology to undo, lemma==form" — and is excluded from *evaluation* accuracy metrics. But the training loop (`train_byt5.py::collate_batch`) builds `labels` and `upos_labels` from every token with no such exclusion, so during actual gradient updates the model is directly taught that ~40% of common Arabic words — including real content words like `الحفاظ` ("preservation"), `تنظيم` ("organization"), `الأولويات` ("priorities") — are UPOS=X and require no lemmatization (keep attached `ال-`/`و-`/`ب-` clitics as-is). This actively corrupts both the lemma and POS heads of the multitask model on the affected tokens, it just isn't visible in eval stats because eval explicitly skips scoring on `X`.

### 7. UPOS tagging inconsistency for common function words

Downstream of (or alongside) #6, high-frequency function words are inconsistently tagged across the corpus — 392 distinct FORMs have ≥2 different UPOS tags. Worst offenders by frequency:

| FORM | Total | UPOS distribution |
|---|---|---|
| في (in) | 3,099 | ADP: 2,152 / X: 947 |
| من (from) | 2,579 | ADP: 2,029 / X: 550 |
| إلى (to) | 1,330 | ADP: 881 / X: 449 |
| أن (that) | 1,286 | SCONJ: 1,063 / X: 223 |
| هل (question particle) | 407 | VERB: 215 / PART: 192 |
| على (on) | 768 | X: 458 / ADP: 310 |

`هل` being tagged `VERB` 215 times is a genuine mistagging (it's an invariant question particle, correctly `PART` the other 192 times) — independent of the tail-collapse pattern since `هل` is always the first token of its sentence. The overall corpus-wide UPOS=X rate is 43.7% (train) / 48.1% (val) vs. 0% (test), consistent with §6.

### 8. و-attachment tokenization convention differs between test and train/val (previously known)

Confirmed the known issue: in `ar_test.conllu`, 14/100 sentences tokenize an attached conjunction و- as a standalone token (e.g. text `وأشرب` → tokens `و` + `أشرب`), which does not match simple space-based reconstruction of the `# text` line. Train/val never split و- this way — it stays attached to its host word. This is a tokenization-convention inconsistency between splits, not text corruption (whitespace-insensitive reconstruction of `ar_test.conllu` is byte-exact for all 100 sentences). No new information here beyond what was already flagged; included for completeness.

## What is genuinely clean

- Arabic script integrity: perfect, in text, FORM, and LEMMA, across all 6,100 sentences and all 76,850 CoNLL-U tokens (train+val+test).
- MSA coherence: the overwhelming majority of sentences (both hand-varied A1/A2/B1 sentences and templated B2–C2 sentences) are grammatical, sensible MSA appropriate to their CEFR level.
- `ar_test.conllu`: clean by every automated metric checked (no lemma bug, no tail-collapse, no duplicate-with-train/val beyond the one val leak noted).

## Recommendation

Do not rely on the per-file `lemma_checker.py` result as a cleanliness signal for this dataset — it structurally cannot detect either real defect found here (cross-file lemma inconsistency, or lemma=form/UPOS=X tail collapse). Before using this train/val data as-is:
1. Fix or filter the 125 و-stripped lemmas (small, mechanical).
2. Investigate and re-run whatever annotation step produces the tail-collapse pattern (§6) — this affects ~40% of all train/val tokens and is the dominant quality issue in this dataset.
3. De-duplicate the one train↔val leaked sentence.
4. Optionally fix the 60-instance dangling "وإرساء" template and the single fullwidth `？`.
