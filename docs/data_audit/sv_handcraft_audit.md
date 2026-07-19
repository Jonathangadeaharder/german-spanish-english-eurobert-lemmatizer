# Swedish Handcraft Lemmatization Data Audit

## Audit date: 2026-07-15

## Verdict: 🔴 FINDINGS (not clean)

The "0 format, 0 lemma errors" claim is **incomplete**: the per-file checker
cannot see cross-file lemma/UPOS inconsistencies, and two such inconsistencies
turn out to be genuine, quantifiable lemma bugs (§1). On top of that, one
train file has a systematic adjective-agreement grammar defect (§2), and four
recurring B2 sentence templates contain a systematic bare-infinitive-for-
finite-verb grammar defect across both train and val (§3). The B1/C1/C2 texts
are also flagged for extreme combinatorial repetition (§4/§5), which is not
an error but undermines "handcraft" diversity. `sv_test.conllu` is clean on
every check below.

## Scope covered

| Artifact | Coverage | Method |
|---|---|---|
| `docs/data_audit/handcraft_texts/sv_texts.txt` (6,100 lines) + `sv_meta.tsv` | 100% read (full file, all CEFR levels, train+val+test) | Read tool, full-file reads in ~500-line windows |
| `data/handcraft/sv/train/*.conllu` (30 files, 5,400 sentences) | 100% automated token/UPOS/lemma scan + manual deep-read of ≥5 sentences per CEFR level | `rg`/`awk` aggregation + Read tool |
| `data/handcraft/sv/val/*.conllu` (6 files, 600 sentences) | 100% automated scan + manual spot-read | `rg`/`awk` aggregation |
| `data/handcraft/sv_test.conllu` (100 sentences) | 100% automated scan, used as clean-baseline comparator | `rg`/`awk` aggregation |

## Summary table

| Check | Result |
|---|---|
| Non-Swedish characters / mojibake / duplicate consecutive words / exact-duplicate lines | ✅ CLEAN — 0 hits anywhere in `sv_texts.txt` |
| Format (structure, tab-separated columns, `_` placeholders) | ✅ CLEAN, confirmed |
| UPOS=`X` tail-collapse (the pattern that corrupted `ar`/`zh` handcraft data) | ✅ CLEAN — 0 `X` tags anywhere in `sv` train/val/test |
| Cross-file lemma/UPOS consistency for common word forms | 🔴 **2 real bugs**: `åt` (28/31 wrong) and `igår`/`imorgon` (94/114 wrong POS, 13 wrong lemma) |
| A1 adjective–subject number agreement | 🔴 80 sentences, 1 train file only (`a1_new_003.conllu`) |
| B2 bare-infinitive-for-finite-verb (4 templates) | 🔴 109 sentences: 99 train + 10 val, **0 test** |
| B1 CEFR-level calibration | 🟡 stylistically indistinguishable from A2 (no hard error) |
| C1/C2 combinatorial template repetition | 🟡 ≥66% of C1 sentences share one 5-slot template; C2 similarly templated |
| Self-referential pronoun oddity (`hjälpa mig/dig`) | 🔵 2 sentences, grammatically valid but odd |
| `sv_test.conllu` | ✅ CLEAN on every check above (pre-existing `.`/`!` text mismatch noted in task brief, not re-verified here) |

## Findings

### 1. 🔴 Real lemma/UPOS bug: `åt` and `igår`/`imorgon` systematically mistagged (train + val, 0 in test)

The per-file `lemma_checker.py`-style check only validates *within-file*
FORM+UPOS→LEMMA consistency, so it cannot see that the *same* form is
lemmatized differently depending on which file it lands in. Corpus-wide
aggregation across all 30 train + 6 val files finds two concrete defects:

**`åt` (past tense of `äta`, "ate") — 28/31 occurrences wrong, 0 in test:**

| Split | Correct (`äta`, VERB) | Wrong (`åt`, ADP) |
|---|---|---|
| train | 3 | 25 |
| val | 0 | 3 |
| test | 0 (not present) | 0 |

Example (`a2_new_002.conllu`, `sv_a2_train_XXX`, text = "Min kompis åt en
stor pizza igår."):

```
3	åt	åt	ADP	_	_	_	_	_	_
```

`åt` is a real Swedish preposition ("towards/for") in other contexts, but in
every one of these 28 sentences it is unambiguously the irregular past tense
of "äta" (to eat) — e.g. "Erik åt lunch på restaurang igår" = "Erik ate lunch
at a restaurant yesterday." The annotator defaulted to the preposition
reading and never disambiguated by context. This is a straightforward,
provable **wrong lemma** (`åt` should be `äta`) and **wrong UPOS** (should be
VERB), not a valid alternate analysis — confirmed by the 3 occurrences
elsewhere in the same corpus that get it right.

**`igår` ("yesterday") and `imorgon` ("tomorrow") — invariant temporal
adverbs, but only 5/90 and 15/24 occurrences (respectively) are tagged ADV:**

| Form | ADV (correct) | ADP | NOUN | DET | PROPN (wrong lemma `Igår`/`Imorgon`) | VERB (wrong lemma `igå`) |
|---|---|---|---|---|---|---|
| `igår` | 5 | 20 | 48 | 4 | 6 | 7 |
| `imorgon` | 15 | 0 | 1 | 3 | 1 | 0 |

- 7 sentences give `igår` the invented, non-existent lemma **`igå`** with
  UPOS=VERB (e.g. `train/a2_new_002.conllu`: "Min kompis åt en stor pizza
  igår." → token 7 `igår igå VERB`) — the tagger appears to have stripped a
  trailing "-r" as if `igår` were a present-tense verb form, producing a lemma
  that is not a real Swedish word.
- 6 sentences capitalize it as **`Igår`** with UPOS=PROPN (treating "yesterday"
  as a proper noun).
- The remaining 72 wrong instances (ADP/DET/NOUN for `igår`; ADJ/DET/NOUN/PROPN
  for `imorgon`) keep the correct lemma string but assign an impossible POS
  for an invariant adverb.
- By contrast, every other temporal adverb checked (`idag` 47×, `nu` 103×,
  `ikväll` 15×, `snart` 23×, `ofta` 169×) is **100% consistently ADV** — this
  is not a general problem with temporal adverbs, it is specific to `igår`/
  `imorgon`.
- `sv_test.conllu` gets both words right 100% of the time (1/1 `igår`=ADV,
  3/3 `imorgon`=ADV) — the bug is confined to train/val, exactly like the
  `åt` bug above.

**Net effect on the "0 lemma errors" claim**: at least 42 tokens (28 `åt` +
13 `igår`/`Igår`/`igå` + 1 `Imorgon`) across ~40 distinct sentences in
train+val carry a lemma that is objectively wrong, not just a labeling
convention the within-file checker happens not to flag. A further ~80 tokens
carry a wrong UPOS with a technically-correct lemma string.

### 2. 🔴 Real grammar bug: adjective–subject number agreement, A1 train only (80 sentences)

`data/handcraft/sv/train/a1_new_003.conllu` (lines 510–597 of `sv_texts.txt`)
contains a block of "[pronoun] är [adjective]" template sentences where every
plural-subject sentence (`Vi`/`Ni`/`De`, "we/you-pl/they") keeps the
**singular** predicative adjective form instead of the required plural `-a`
ending:

```
Vi är trött.   → should be "Vi är trötta."
Ni är hungrig. → should be "Ni är hungriga."
De är glad.    → should be "De är glada."
```

This is basic, unambiguous A1-level Swedish grammar (predicative adjectives
agree in number with the subject) and the errors are 100% systematic across
the block: every `Vi`/`Ni`/`De` + adjective sentence is wrong except the 4
that use the indeclinable adjective `redo` ("ready"), which happens to be
identical in singular and plural. 80 of the 84 sentences in this block are
affected. This does not corrupt the *lemma* annotation (the lemma of `trött`
is trivially `trött` whether the surface form is grammatical or not — see
CoNLL-U spot check below), but it means ~80 objectively ungrammatical
Swedish sentences are present as "clean" A1 training data:

```
# sent_id = sv_a1_train_510
# text = Vi är trött.
3	trött	trött	ADJ	_	_	_	_	_	_
```

Confined to this one file; `val/a1.conllu` and `sv_test.conllu` do not
reproduce this pattern (checked: their `Vi/Ni/De är X` sentences are all
locative, e.g. "Vi är i skolan", which correctly need no agreement).

### 3. 🔴 Real grammar bug: bare infinitive used where a finite verb is required, B2 train + val (109 sentences, 0 in test)

Four recurring B2 sentence-opener templates put the embedded clause's verb in
the **bare infinitive** where standard Swedish requires either a present-tense
finite verb (after the subordinator `att`), a past participle (after `har`),
or the particle `att` before the infinitive (after `väljer`/`kommer`):

| Template | Correct form needed | Count (train) | Count (val) |
|---|---|---|---|
| "Utan tillräckliga resurser blir det svårt att `[subj]` `[bare-inf]`…" | present tense, e.g. `satsar` not `satsa` | 23 | 0 |
| "I de flesta europeiska länder väljer `[subj]` `[bare-inf]`…" | needs `att` before infinitive | 18 | 1 |
| "Under det senaste året / På grund av den snabba utvecklingen har `[subj]` `[bare-inf]`…" | past participle, e.g. `satsat` not `satsa` | 37 | 3 |
| "Med rätt stöd och vägledning kommer `[subj]` `[bare-inf]`…" | needs `att` before infinitive | 20 | 4 |
| **Total** | | **99** | **10** (109 combined) |

Example (`b2_new_001.conllu`, `sv_b2_train_011`):

```
# text = Utan tillräckliga resurser blir det svårt att företag satsa mer på hållbarhet och minska avfallet.
7	att	att	PART	_	_	_	_	_	_
8	företag	företag	NOUN	_	_	_	_	_	_
9	satsa	satsa	VERB	_	_	_	_	_	_
```

Correct Swedish here is "…svårt att företag **satsar** mer…" (present tense
— `att` after `svårt` is the subordinating "that", not an infinitive marker,
so it needs a finite verb). Because Swedish verb lemmas equal the infinitive
form, this defect happens not to corrupt the LEMMA column (`satsa`→`satsa` is
technically a valid lemma pair) — which is exactly why the existing checker
reports 0 errors here — but the model is still being trained on ~109 sentences
of objectively incorrect Swedish syntax. Confined entirely to B2 (train files
`b2_new_001–005.conllu` + `val/b2.conllu`); `sv_test.conllu` has zero
instances of any of the four templates.

### 4. 🟡 CEFR-level calibration: B1 reads as A2, not more advanced

`data/handcraft/sv/train/b1_new_*.conllu` and `val/b1.conllu` (lines
1801–2700 of `sv_texts.txt`) consist almost entirely of simple habitual
sentences ("Min pappa brukar…", "X tycker om att…", "Vi planerar att resa
till…") that are grammatically and structurally indistinguishable from the
preceding A2 block (lines 901–1800) — no exact-duplicate sentences (verified,
0 overlap), but the same register, same sentence length, same lack of
subordinate clauses/hypotheticals/opinions that would normally distinguish
B1 from A2. The jump in complexity only appears abruptly at line 2701, where
the B2 files begin. Not a hard error, but worth flagging if CEFR-appropriate
progression matters for training.

### 5. 🟡 Extreme combinatorial repetition at C1/C2 (diversity concern, not an error)

- **C1** (`c1_new_*.conllu`, `val/c1.conllu`): ≥591 of the 900 train sentences
  (66%+) are one fixed 5-slot template — "Det är `[1 of ~15 adjectives]` att
  `[1 of ~15 verbs]` `[1 of ~10 nouns]`en för att `[1 of ~15 verbs]` `[1 of
  ~15 nouns]` i `[1 of ~8 locative nouns]`en." — filled combinatorially, e.g.
  "Det är väsentligt att strukturera ekonomin för att uppnå de uppsatta målen
  i kommunen." Every instance checked is grammatically valid Swedish, but the
  lexical/syntactic diversity is extremely low for content billed as
  hand-crafted C1 material.
- **C2** (`c2_new_*.conllu`, `val/c2.conllu`): similarly dominated by a
  handful of long bureaucratic-register templates ("Det framstår som
  uppenbart att `[actor]` `[adverb]` `[verb phrase]`…", "Eftersom `[actor]`
  `[verb phrase]`, `[consequence]` för att `[purpose]`.") cycling through
  ~15 interchangeable actor nouns (`arbetsgruppen`, `kommissionen`,
  `ledningen`, …) and ~10–15 interchangeable verb phrases/objects/purposes.
  Grammatically sound (spot-checked, including lemma correctness on complex
  forms like `problematiserat`→`problematisera`, `förslagets`→`förslag`) but,
  like C1, mechanically generated rather than naturally varied.

This does not affect lemma or format correctness (confirmed by spot checks)
but means the effective lemma/lexical diversity of the C1/C2 splits is far
lower than the raw sentence count (900 train + 100 val each) suggests.

### 6. 🔵 Minor: 2 self-referential pronoun sentences (A2)

`train/a2_new_005.conllu` line "Du kan hjälpa dig med läxan idag." and
`train/a2_new_004.conllu`-area line "Jag kan hjälpa mig med väskan efter
skolan." both use a subject pronoun and object pronoun of the same person
("you help yourself", "I help myself"), which is grammatically valid but
semantically odd for a sentence pattern clearly intended to vary the object
independently of the subject (compare neighboring sentences like "Johan kan
hjälpa dig med läxan imorgon."). Almost certainly an artifact of combinatorial
generation not excluding subject==object pairs; negligible at 2/6,100 lines.

## What is genuinely clean

- **Text integrity**: all 6,100 lines of `sv_texts.txt` read start to finish.
  No non-Swedish/mojibake characters, no duplicate consecutive words, no
  exact-duplicate lines, every line properly terminated with `.`/`!`/`?`.
  Nothing resembling the prior `cefr_sentences.conllu` LLM-garbage pattern
  ("Vi apelsin på måndag") appears anywhere.
- **No tail-collapse**: unlike the `ar`/`zh` handcraft audits, `sv` train+val
  has **zero** `UPOS=X` tokens anywhere (verified full corpus-wide tag
  distribution) — every token got a real POS tag attempt.
- **A1–C2 sentence content** (outside the specific defects above) is
  coherent, natural, CEFR-plausible Swedish — verified by deep-reading ≥30
  sentences per level across both train and val.
- **`sv_test.conllu`**: clean on every automated check run in this audit
  (no `åt`/`igår`/`imorgon` mistagging, no B1-b2-style bare-infinitive
  defect, no A1 agreement defect). The pre-existing `.`/`!` text mismatch
  noted in the task brief was not independently re-verified in this pass.

## Recommendation

1. Fix the `åt`→`äta`/VERB and `igår`/`imorgon`→ADV lemma/UPOS bugs (§1) —
   small, mechanical, ~42 tokens with a genuinely wrong lemma across ~40
   sentences. This is the most important fix since it's a real lemma error
   the existing per-file checker structurally cannot detect.
2. Fix or drop the 80 ungrammatical A1 sentences in `a1_new_003.conllu`
   (§2) — trivial single-file, single-block fix (add `-a` to 15 adjective
   lemma stems for the `Vi`/`Ni`/`De` rows).
3. Fix or drop the 109 ungrammatical B2 sentences across the 4 templates
   (§3) — regenerate with correct present-tense/past-participle/`att`+infinitive
   forms.
4. Consider regenerating B1 with genuinely more complex sentences than A2
   (§4), and diversifying the C1/C2 templates (§5) if lemma/lexical coverage
   at those levels matters for training quality — neither is a hard blocker.
