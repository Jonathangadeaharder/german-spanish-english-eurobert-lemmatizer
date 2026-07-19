# French Handcraft Lemmatization Data Audit

## Audit date: 2026-07-15

## Verdict: 🔴 FINDINGS

Raw sentence text (`fr_texts.txt`) is almost entirely clean, real, CEFR-appropriate
French — only 6 minor content errors in 6100 lines. The real dirt is downstream:
the train/val CoNLL-U annotation pipeline **systematically fails on any sentence
containing a French elision/contraction** (`l'idée`, `d'un`, `m'appelle`, `qu'il`,
etc.). The elided token is kept as one un-split blob tagged `X` with
`lemma = surface form`, and the corruption **cascades to every token after it in
the same sentence**. This hits **55.8% of all train/val sentences** and **25.1%
of all train/val tokens** — none of which is caught by the "0 lemma errors" claim,
because `lemma_checker.py` has no French-specific plausibility rule and the
corruption is internally self-consistent. This is the same failure pattern as the
`zh` handcraft audit (`zh_handcraft_audit.md`), just triggered by apostrophes
instead of character-level segmentation.

## Scope covered

| Artifact | Coverage | Method |
|---|---|---|
| `docs/data_audit/handcraft_texts/fr_texts.txt` (6100 lines) | 100% read (chunked) + scripted regex scan (elisions, stray English, accents, dupes, double-spaces) | Read tool + `rg` |
| `docs/data_audit/handcraft_texts/fr_meta.tsv` (6100 lines) | Spot-checked against flagged text lines to resolve `sent_id`/file | Read tool + `sed` |
| `data/handcraft/fr/train/*.conllu` (30 files, 5400 sents, ~72k tokens) | ≥5 sentences/file deep-read across all 6 CEFR levels (a1–c2) + full-corpus `rg`/Python aggregation | Read tool + shell/Python |
| `data/handcraft/fr/val/*.conllu` (6 files, 600 sents) | ≥5 sentences/file deep-read (a1, a2, c2 shown; aggregate stats cover all) | Read tool + shell/Python |
| `data/handcraft/fr_test.conllu` (100 sents, 631 tokens) | 100% structural scan (token-concat-vs-text reconstruction, hyphen/lemma cross-check, `!` search) + spot read | Python script + Read tool |
| `src/lemmatizer/data/lemma_checker.py` | Full read — to determine what the "0 lemma errors" claim actually covers for French | Read tool |

## Finding 1 (🔴 real bug): Elision cascade corrupts 25% of train/val tokens

Whenever a sentence contains an un-split elided/contracted form (`l'idée`,
`d'un`, `m'appelle`, `n'a`, `c'est`, `s'il`, hyphenated inversions like
`vas-tu`, idioms like `Au revoir`), that token — **and every token after it in
the same sentence** — gets `UPOS = X` and `LEMMA = FORM` (no reduction at all).
Tokens *before* the trigger stay correctly tagged. Confirmed corpus-wide:

```
84,646 total tokens (train+val)
21,243 tokens tagged X            (25.1%)
6,000  total sentences
3,347  sentences with ≥1 X token  (55.8%)
```

X-tag density rises with CEFR level (more elisions/complex syntax): a1 ≈9%,
b1 ≈21%, c2 ≈37% of tokens per file. Example (`fr/train/a1_new_001.conllu`,
sentence `fr_a1_train_006`, text `Au revoir, à bientôt!` — no elision at all,
just an idiom the tagger didn't recognize):

```
1  Au       Au       X
2  revoir   revoir   X
3  ,        ,        PUNCT
4  à        à        X
5  bientôt  bientôt  X
6  !        !        PUNCT
```

Same corpus, sentence `fr_a1_train_002` (`Je m'appelle Marie.`): `Je` is
correctly `PRON`/`je`, but `m'appelle` and even the proper noun `Marie` (which
gets `PROPN` correctly elsewhere, e.g. `fr_a1_train_003`'s `Paul`) both collapse
to `X`/verbatim-form once the elision hits.

This directly contradicts the "0 format, 0 lemma errors" premise: it's not a
format error (columns are well-formed) and not a lemma error *by the existing
checker's definition* (X-tagged rows have `lemma == form`, which trivially
"passes" every check), but it is completely wrong/uninformative supervision —
the model is taught that ~1 in 4 French tokens simply don't get lemmatized at
all, at every CEFR level, in both train and val.

## Finding 2 (🔴 real bug): systematic mis-lemmatization of "de" → "un" inside corrupted spans

Inside these same corrupted sentences, standalone preposition `de` is
frequently re-tagged `DET` with `LEMMA = un` — i.e. confused with the
indefinite article — instead of `ADP`/`de`:

```
1,603×  de → de   (ADP, correct)
1,160×  de → un   (DET, wrong — should be ADP/de)
```

Every single occurrence of `de`+`DET` maps to lemma `un` (100% internally
consistent — confirmed by exhaustive scan), which is exactly why
`lemma_checker.py`'s only relevant rule ("same FORM+UPOS → same LEMMA within
file") never flags it: the bug is self-consistent nonsense, not a
contradiction. Example (`fr/train/c2_new_001.conllu`, `fr_c2_train_023`,
"...l'idée d'un critère **de** démarcation unique..."):

```
8  de  un  DET   _  _  _  _  _  _
```

`de` here is unambiguously the preposition "of" in "critère de démarcation"
(criterion of demarcation) — lemma `un` is flatly wrong. This is the same
elision-cascade failure as Finding 1, just landing on a specific closed-class
word often enough to be independently quantifiable (1,160 occurrences across
train+val).

## Finding 3 (root cause, not itself a bug): `lemma_checker.py` has no French rule

Reading `check_text()` in `lemma_checker.py`: the `elif lang == ...` chain
handles `de`, `en`, `es`, `zh`, `ar` — **`fr` is not handled at all.** French
files only get the four generic checks (sense-number suffix, `PUNCT`
lemma==form, `PROPN` capitalization, FORM+UPOS internal consistency). None of
these can detect "25% of tokens are `X` with `lemma==form`" or "`de` lemmatizes
to `un`" — both pass trivially. **The "0 format, 0 lemma errors" claim is
accurate for what the checker actually tests, and does not mean the data is
clean.** Same root cause and same masking effect as documented for `zh` in
`zh_handcraft_audit.md` Finding 2 / Non-findings.

## Finding 4 (🟡 minor, real bug): `fr_texts.txt` content errors — 6 total

Full 6100-line read + scripted scan found 6 genuine French-language errors
(not annotation/format issues, just bad source sentences):

| Line | Text | Issue |
|---|---|---|
| 2132 | `je planerais des espèces locales` | wrong verb — "planer" (to glide) instead of "planter" (to plant) |
| 2828 | `Il est rare que une œuvre...` | missing elision — should be `qu'une` |
| 3921 | `...lorsque un État est incapable...` | missing elision — should be `lorsqu'un` |
| 4799 | `...lorsque une histoire migre...` | missing elision — should be `lorsqu'une` |
| 5746 | `L'education à l'environnement...` | missing accent — should be `L'éducation` |
| 5855 | `...la littérature romantique, expressing la division...` | stray English gerund — should be e.g. `exprimant` |

All 6 map to real `sent_id`s in train (`fr_b1_train_332`, `fr_b2_train_128`,
`fr_c1_train_321`, `fr_c2_train_299`, plus two more) via `fr_meta.tsv` — these
sentences also happen to be inside the Finding-1 X-cascade (their apostrophes/
missing-elisions are exactly the kind of token that triggers it), so the bad
source text and the bad annotation are not independent problems, but each
would need to be fixed anyway even if the other were.

## Non-findings — confirmed clean / not real bugs

- **`fr_texts.txt` bulk quality**: 6094 of 6100 lines (99.9%) are clean,
  coherent, CEFR-plausible French prose. No mojibake, no wrong script, no
  double-spaces, no word-salad.
- **14 exact-duplicate lines** in `fr_texts.txt` (e.g. `Le chat dort sur le
  canapé.`, `J'aime ma famille.`) — all short, generic A1-level sentences;
  natural convergence at that vocabulary size, not a generation bug. Noise-
  level, not worth remediation.
- **`Les critiques du law and economics questionnent...` (line 4687)** —
  looks like stray English at first glance, but "law and economics" is a
  standard untranslated academic/legal term used as-is in French scholarship
  (l'école "law and economics"). Not an error.
- **`fr_test.conllu` hyphen/space "mismatch"** — investigated exhaustively
  (reconstructed every sentence by concatenating token FORMs and diffing
  against `# text`, plus a FORM/LEMMA hyphen cross-check): **zero mismatches
  found across all 100 sentences / 631 tokens.** `fr_test.conllu` correctly
  splits inversions/imperatives into 3 tokens (`As` / `-` / `tu`), consistently,
  every time. This "known" issue is **not reproducible** in the current file —
  either already fixed or not applicable to French; flagging as resolved/non-
  issue rather than carrying it forward.
- **`fr_test.conllu` tokenization convention vs train/val**: `fr_test.conllu`
  splits hyphenated inversions into real tokens and is the *correct* reference
  convention that train/val's `X`-tag fallback should have matched and didn't
  (same relationship as `zh_test.conllu` vs `zh` train/val in the prior audit).

## Finding 5 (🟡 minor, real bug, isolated): `fr_test.conllu` column-5 contamination

`sent_id = handcraft-fr-10`, text `N'oublie pas tes clés !`:

```
6  !  !  PUNCT  !  _  _  _  _  _
```

Column 5 (XPOS) is `!` instead of `_` (every other `!`/`?` token in the file
has `_` there — confirmed by exhaustive `grep`). Isolated single-cell
copy/paste artifact — 1 of 631 tokens, 1 of 100 sentences. Real bug, trivial
scope; does not affect FORM/LEMMA/UPOS (columns 2–4 are correct).

## Recommendation

Do not use `data/handcraft/fr/train/*.conllu` or `.../val/*.conllu` in their
current form for lemma/UPOS training — Finding 1 means a random ~1-in-4 tokens
carries zero real lemmatization signal, and Finding 2 means a chunk of those
actively teach a wrong `de→un` mapping. Root cause is the annotation
generator's elision/contraction handling (mirrors the `zh` character-
segmentation bug): regenerate by first splitting elided/hyphenated forms into
their real UD tokens (matching `fr_test.conllu`'s convention — e.g.
`m'appelle` → `m'` + `appelle`, `l'idée` → `l'` + `idée`) before running
POS/lemma tagging, so the `X`-tag fallback stops triggering and stops
cascading. Separately, add an `fr` branch to `lemma_checker.py` (e.g. flag
`UPOS == "X"` above a low threshold, and/or verb-lemma-ends-in-`-er/-ir/-re`
sanity check) so this class of bug is caught by CI instead of silently
passing. Fix the 6 `fr_texts.txt` content errors (Finding 4) and the single
`fr_test.conllu` column-5 artifact (Finding 5) while touching these files.
