# English Handcraft Lemmatization Data Audit

## Audit date: 2026-07-15

## Verdict: 🟡 FINDINGS (text is clean; annotations have real, systematic lemma/UPOS bugs at low but non-trivial rate)

The raw sentence text (`en_texts.txt`, 6100 lines = 5400 train + 600 val + 100
test) is **100% clean, coherent, real English** — no word salad, no wrong
language, no LLM garbage. Train+val format/lemma-checker errors are correctly
reported as 0/0, but `lemma_checker.py` only checks a few narrow rules (PUNCT
lemma=form, PROPN capitalization, EN verb suffix) and has no dictionary/word-
plausibility check, so it misses real bugs. Deep-reading and full-corpus
`awk`/`rg` scans surface **~128 real lemma/UPOS annotation bugs** across
~87,195 tokens (0.15% token error rate), concentrated in two reproducible
patterns: (1) a deterministic lemma-truncation bug for a fixed set of verb
stems, and (2) mis-tagging of hyphenated compounds and possessive-adjacent
nouns. `en_test.conllu`'s 3 `lemma_checker` flags are confirmed false
positives; its `$50` tokenization is a real but minor `SpaceAfter` omission.

## Scope covered

| Artifact | Coverage | Method |
|---|---|---|
| `docs/data_audit/handcraft_texts/en_texts.txt` (6100 lines) | 100% read | Read tool, full file, chunked |
| `data/handcraft/en/train/*.conllu` (30 files, 5400 sents) | ≥5 sentences/file deep-read (start/mid/end) across all 6 CEFR levels (a1/a2/b1/b2/c1/c2) + full-corpus `awk`/`rg` aggregation | Read tool + shell aggregation |
| `data/handcraft/en/val/*.conllu` (6 files, 600 sents) | ≥5 sentences/file deep-read + aggregate stats | Read tool + shell aggregation |
| `data/handcraft/en_test.conllu` (100 sents, ~880 tokens) | Full `lemma_checker` re-run + spot-read of flagged lines + `X`/hyphen scan | shell + Read tool |
| `src/lemmatizer/data/lemma_checker.py` | Full read | To determine what existing checks do/don't cover |
| `docs/data_audit/handcraft_texts/en_meta.tsv` | Aggregated (per-source-file sentence counts) | `awk`/`sort`/`uniq` — confirmed 5400+600+100=6100 matches `en_texts.txt` exactly |

## Finding 1 (🔴 real bug, systematic): Deterministic lemma truncation for a fixed set of verb stems

A specific set of verb lemmas is **truncated to a short garbage prefix every
single time they occur**, corpus-wide — not random noise, but a reproducible
per-lemma defect (whatever generated these lemmas has a bad/truncated
dictionary entry for exactly these words):

| Correct lemma | Bug lemma | Occurrences | Forms affected |
|---|---|---|---|
| embed | `emb` | 17 | embedded, embedding |
| exceed | `exce` | 22 | exceeds, exceeded, exceeding |
| assess | `asse` | 6 | assessing, assessed |
| need | `ne` | 12 | need, needs, needed, needing |
| bring | `br` | 8 | bring, brings, brought |
| press | `pre` | 4 | press, presses, pressing |
| bypass | `bypa` | 3 | bypassed |
| precede | `prec` | 4 | preceding |
| scale | `scal` | 2 | scaling |
| bias | `bia` | 3 | (skill-biased) |
| impede | `imp` | 1 | impeding |
| infringe | `infr` | 1 | infringed |
| ban | `bann` | 2 | Banning (doubled consonant not undone) |
| map | `mapp` | 3 | mapping (doubled consonant not undone) |
| war | `warr` | 1 | warring (doubled consonant not undone) |

**Total: 89 occurrences.** All UPOS tags for these tokens are correct (VERB);
only the lemma is wrong. `lemma_checker.py`'s `_check_en_verb_base` only flags
lemmas that end in `-s`/`-ed`/`-ing`, so none of these (`emb`, `ne`, `br`,
`pre`, etc.) trip it — a checker blind spot, not evidence of correctness.

Example (`data/handcraft/en/train/c2_new_005.conllu:60`):

```60:60:data/handcraft/en/train/c2_new_005.conllu
9	embedded	emb	VERB	_	_	_	_	_	_
```

## Finding 2 (🔴 real bug): Hyphenated compounds mis-tagged with the head word's tag/lemma, dropping the modifier — wrong UPOS

The corpus correctly lemmatizes the *majority* of hyphenated compounds as the
full form (e.g. `cross-border`→`cross-border`/ADJ, `quasi-experimental`→
`quasi-experimental`/ADJ, `decision-making`→`decision-making`/NOUN — dozens of
correct examples). But 24 compounds instead take only the **head component's**
lemma and POS, which is objectively wrong (ADP/PRON/ADV/VERB where the token
functions as a NOUN or ADJ):

| Form (single token) | Bug lemma/UPOS | Should be |
|---|---|---|
| trade-off (×2) | off/ADP | trade-off/NOUN |
| follow-on | on/ADP | follow-on/NOUN or ADJ |
| plug-in | in/ADP | plug-in/NOUN |
| opt-out | out/ADP | opt-out/NOUN |
| knowing-how | how/ADV | knowing-how/NOUN |
| knowing-that | that/PRON | knowing-that/NOUN |
| skill-biased | bia/VERB | skill-biased/ADJ (also Finding 1 truncation) |
| ill-suited | suit/VERB | ill-suited/ADJ |
| short-lived | live/VERB | short-lived/ADJ |
| reason-giving | give/VERB | reason-giving/NOUN |
| skull-bound | bind/VERB | skull-bound/ADJ |
| quality-adjusted (×2) | adjust/VERB | quality-adjusted/ADJ |
| placebo-controlled | control/VERB | placebo-controlled/ADJ |
| age-weighted | weight/VERB | age-weighted/ADJ |
| open-textured | texture/VERB | open-textured/ADJ |
| puzzle-solving | solve/VERB | puzzle-solving/NOUN |
| taken-for-granted | grant/VERB | taken-for-granted/ADJ |
| liberty-focused | focu/VERB | liberty-focused/ADJ (also Finding 1 truncation) |
| usage-based / market-based / dignity-based / conscience-based / treaty-based | base/VERB | \<word\>/ADJ |

**Total: 24 occurrences**, all in C1/C2 train files (`c2_new_001..005`,
`c2_new_003`, `c2_new_004`, `c1_new_002`). Example
(`data/handcraft/en/train/c2_new_001.conllu:2145`):

```2144:2146:data/handcraft/en/train/c2_new_001.conllu
19	the	the	DET	_	_	_	_	_	_
20	trade-off	off	ADP	_	_	_	_	_	_
21	obsolete	obsolete	ADJ	_	_	_	_	_	_
```

Minor/non-bug sibling pattern (noted, not counted as a bug): 8 compounds
(`exposure-dependent`, `context-dependent`, `policy-ready`, `value-laden`,
`double-blind`, `rural-urban`, `agro-industrial`, `counter-majoritarian`,
`finance-centric`) get a correct ADJ tag but a head-only lemma
(`dependent`/`ready`/`laden`/... instead of the full hyphenated form). UPOS is
right here, so this is an internal **lemma-convention inconsistency**
(same corpus lemmatizes structurally identical compounds both ways), not a
hard error.

## Finding 3 (🔴 real bug): `AUX`/`be` mis-assigned to ordinary nouns, mostly after a possessive

6 tokens get lemma `be`/UPOS `AUX` despite not being any form of "be":

| Sentence (excerpt) | Mis-tagged token | Location |
|---|---|---|
| "...my sister's **birthday**." | birthday → be/AUX | `train/a2_new_001.conllu:380` |
| "...our grandma's **birthday** at home..." | birthday → be/AUX | `val/a2.conllu:364` |
| "...my wife's **parents**." | parents → be/AUX | `val/a2.conllu:506` |
| "...on New Year's **Eve**." | Eve → be/AUX | `train/a2_new_002.conllu:559` |
| "...my family's **help**." | help → be/AUX | `train/a2_new_004.conllu:716` |
| "...the neighbours' **dog** this weekend." | dog → be/AUX | `train/a2_new_004.conllu:2145` |

4/6 immediately follow a possessive noun (`sister's`, `grandma's`, `wife's`,
`neighbours'`) — a reproducible trigger, not random. Should be NOUN with the
word's own lemma (birthday, parents, Eve, help, dog).

## Finding 4 (🔴 real bug): Neighbor-token lemma/UPOS duplication (annotation desync)

Distinct from Finding 3, isolated cases where a token's lemma+UPOS is a
verbatim copy of an unrelated neighboring token's, rather than its own:

```1558:1565:data/handcraft/en/train/a2_new_001.conllu
1	They	they	PRON	_	_	_	_	_	_
2	cannot	can	AUX	_	_	_	_	_	_
3	come	not	PART	_	_	_	_	_	_
4	to	to	ADP	_	_	_	_	_	_
5	the	the	DET	_	_	_	_	_	_
6	party	the	DET	_	_	_	_	_	_
7	tonight	tonight	ADV	_	_	_	_	_	_
8	.	.	PUNCT	_	_	_	_	_	_
```

`party` (token 6) gets `the`/DET, duplicating token 5. `come` (token 3) gets
`not`/PART — a leftover of splitting `cannot`'s negation, never reassigned to
its own lemma `come`/VERB.

Three more instances of the same pattern:
- `weekend` → `this`/DET (duplicate of preceding `this`) — `train/a2_new_004.conllu:2147`
- `home` → `at`/ADP (duplicate of preceding `at`) — `val/a2.conllu:366`
- `film` → `the`/DET (duplicate of preceding `the`), in the same sentence
  `enjoy` → `not`/PART (same cascading pattern as `come` above) —
  `val/a2.conllu:1054-1056`

**Total: 6 tokens across 4 sentences.**

## Finding 5 (🔴 real bug): Wrong dictionary headword (homograph confusion)

```3192:3192:data/handcraft/en/train/c1_new_003.conllu
8	collages	college	NOUN	_	_	_	_	_	_
```
"collages" (plural of *collage*, an artwork) is lemmatized as `college` (a
different word entirely). 1 occurrence.

## Finding 6 (🔴 real bug): `UPOS=X` applied to ordinary nouns/verbs/adjectives

```
train/a1_new_001.conllu:6   friend  friend  X   (should be NOUN — "Hello, my friend.")
train/a1_new_001.conllu:62  see     see     X   (should be VERB — "Goodbye, see you.")
train/a1_new_001.conllu:103 sir     sir     X   (should be NOUN — "Good evening, sir.")
train/a1_new_001.conllu:129 class   class   X   (should be NOUN — "Good afternoon, class.")
train/a1_new_004.conllu:934 Wi-Fi   Wi-Fi   X   (should be NOUN/PROPN — "The Wi-Fi is slow.")
train/a1_new_004.conllu:936 slow    slow    X   (should be ADJ — "The Wi-Fi is slow.")
```

**Total: 6 occurrences**, all in `a1` train files. Not counted as bugs: `avant`
/`garde` (`c1_new_003.conllu`, tokenized halves of "avant-garde") and
`bonjour` (`en_test.conllu`, "She said *bonjour*...") — `X` for
untranslated/foreign lexical items is defensible per UD convention.

## `en_test.conllu` spot-check results

- **`lemma_checker.py` 3 flags on `pass`/`bring`/`need` — confirmed FALSE
  POSITIVES.** All three are correctly annotated base-form VERB lemmas
  (`pass`=`pass`, `bring`=`bring`, `need`=`need`); the checker's
  `_check_en_verb_base` naively flags any lemma whose *last characters*
  spell `-s`/`-ed`/`-ing`, without checking whether that's a real
  inflectional suffix. `pass` ends in literal `-s`, `bring` ends in literal
  `-ing`, `need` ends in literal `-ed` — coincidental substrings, not
  inflection. This is a checker specificity gap, not a data defect.
- **`$50` text/token mismatch — confirmed, minor, real.** `data/handcraft/en_test.conllu:1066`
  has `# text = The shirt costs $50 at the store.` but tokenizes `$` and `50`
  as two separate tokens with `MISC=_` (no `SpaceAfter=No`). Reconstructing
  text from tokens naively would yield `$ 50` not `$50`.
- No instances of the truncation bug (Finding 1), compound mis-tagging
  (Finding 2), or `X`-mistagging (Finding 6) were found in `en_test.conllu`
  itself — those are train/val-only defects.
- Deep-read of 10+ additional test sentences (simple A1-style and complex
  C1-style) found otherwise correct, real, coherent annotations.

## Non-findings — confirmed clean / convention, not bugs

- **`en_texts.txt` (100% read, 6100 lines)**: every line is real, coherent
  English appropriate to its stated CEFR band. A1/A2 are simple everyday
  sentences; C1/C2 are dense, formal academic-register prose (economics, law,
  linguistics, sociology, philosophy) — sophisticated but grammatically valid
  and coherent, not nonsense or LLM word salad. No wrong-language content
  (one apparent "mixed language" hit, *"She said bonjour and waved goodbye"*,
  is ordinary code-switching/quotation, not an error). No exact/near
  duplicate spam detected during the full read.
- **Train+val format/lemma_checker**: confirmed 0 format errors, 0
  `lemma_checker.py` errors — accurate as stated, but the checker has no
  dictionary/word-plausibility rule for any language (verified by reading
  its source), so it cannot and does not catch Findings 1–6 above.
- **Sentence-count coverage**: `en_meta.tsv` confirms `en_texts.txt`'s 6100
  lines = 5400 train + 600 val + 100 test sentences exactly, with 1:1
  correspondence to the `.conllu` files' `# sent_id` counts.
- **Overwhelming majority of hyphenated compounds** (~90+, e.g.
  `cross-border`, `semi-structured`, `quasi-experimental`, `decision-making`,
  `laissez-faire`, `p-hacking`) are correctly lemmatized as the full compound
  form with correct UPOS.

## Summary

- **Sentences scanned**: 6100/6100 raw text (100%), 6100/6100 sentences have
  gold `.conllu` annotations (5400 train + 600 val + 100 test); ≥5
  sentences/file deep-read across all 6 CEFR levels for both train and val,
  plus full-corpus scripted scans (87,195 tokens) for the bug patterns above.
- **Dirty/word-salad findings in raw text**: 0. Text is CLEAN.
- **Real lemma/UPOS annotation bugs**: ~128 tokens (89 truncated-lemma +
  24 mis-tagged compounds + 6 AUX/be-after-possessive + 6 neighbor-duplication
  + 1 wrong-headword + 6 X-mistagged ≈ 128 of 87,195 tokens, ≈0.15%), all in
  train/val; `en_test.conllu` is clean of these patterns.
- **False positives confirmed**: `lemma_checker.py`'s 3 flags on
  `pass`/`bring`/`need` in `en_test.conllu` (naive suffix-substring matching).
- **Overall verdict**: 🟡 **FINDINGS** — text is CLEAN, annotations are
  >99.8% clean by token count but have a low-rate, non-random, reproducible
  set of lemma/UPOS defects (concentrated in specific verb-lemma truncations
  and hyphenated-compound handling) that a dictionary-free rule checker like
  `lemma_checker.py` structurally cannot detect.
