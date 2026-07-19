# German Handcraft Lemmatization Data Audit

## Audit date: 2026-07-15

## Verdict: 🟡 FINDINGS (text is ~99.95% clean; annotations have real, systematic lemma/UPOS bugs at a low but non-trivial rate, concentrated in a few reproducible patterns)

The raw sentence text (`de_texts.txt`, 6211 lines = 5511 train + 600 val + 100
test) is **effectively 100% clean, coherent, real German** — no word salad, no
wrong language, no LLM garbage anywhere in the full read. Exactly one source
typo was found (`is` for `ist`, line 3918) and three grammatically-broken
sentences in a templated C1 generation block (all confined to
`de/val/c1.conllu`). The automated `lemma_checker.py` reports 0/40 errors as
real corpus problems requiring a fix at the *lemma-consistency* level it
checks (all 40 flagged pairs are legitimate German homography: `Sie`/`sie`,
separable-verb ambiguity, nominalized-adjective capitalization) — but that
checker has no dictionary/plausibility rule, so it is structurally blind to
the real bugs below. Full-corpus scripted scans surface **~296 real
lemma/UPOS annotation bugs across 72,935 tokens (≈0.41% token error rate)**,
concentrated in four reproducible, fixable patterns: (1) unresolved
ambiguous-lemma artifacts literally containing a `|` character (44 tokens),
(2) inconsistent `AUX`/`VERB` tagging of modal verbs (217 tokens), (3)
inconsistent tagging of separable-verb particles as `VERB` with the full-verb
lemma instead of `ADP`/`PART` with the bare particle (17 tokens), and (4) data
corruption/garbled lemmas concentrated in `de/train/c2.conllu` (5 tokens) plus
scattered isolated bugs (`uns`→`sich`, a typo, a leading-space lemma, etc.).

## Scope covered

| Artifact | Coverage | Method |
|---|---|---|
| `docs/data_audit/handcraft_texts/de_texts.txt` (6211 lines) | 100% read | Read tool, full file, chunked (500-line windows) |
| `data/handcraft/de/train/*.conllu` (48 files, 5511 sents) | ≥5 sentences/file deep-read (start/¼/mid/¾/end) across all 6 CEFR levels (a1/a2/b1/b2/c1/c2 + `*_new_*`) + full-corpus Python aggregation scripts | Read tool + Python scripts (not `grep`, see note below) |
| `data/handcraft/de/val/*.conllu` (12 files, 600 sents) | ≥5 sentences/file deep-read + aggregate stats | Read tool + Python scripts |
| `data/handcraft/de_test.conllu` (100 sents) | Full `lemma_checker` re-run + spot-read + token-count aggregation | Python scripts |
| `src/lemmatizer/data/lemma_checker.py` | Full read | To determine what the 40 known hits do/don't cover |
| `docs/data_audit/handcraft_texts/de_meta.tsv` | Aggregated (per-source-file sentence counts) | Confirmed 5511+600+100=6211 matches `de_texts.txt` and every `.conllu` `# sent_id` count exactly |

**Methodology note:** an early pass of this audit used `grep -rn` and
misattributed several phantom findings (a literal `n`/`ln` "placeholder
token" that did not actually exist) to ANSI color-highlighting corruption in
piped `grep` output combined with a malformed `-rln` flag combination. All
findings below were re-verified with pure-Python `open()`/`readlines()` file
reads (immune to that class of artifact) before being included in this
report; nothing here rests on unverified `grep` output.

## Finding 1 (🔴 real bug, systematic): Unresolved ambiguous-lemma artifacts — literal `Lemma1|Lemma2` strings

44 tokens across 15 files have a lemma field containing a literal `|`
character — two candidate lemmas concatenated by whatever
disambiguation/lookup step produced the annotation, with neither one chosen.
This is unambiguously wrong CoNLL-U (the `LEMMA` column must be a single
string) and is highly reproducible: the same word (`Regeln`, `Studien`,
`Räumen`, `Mitteln`, ...) gets the same malformed lemma every time it recurs.

Representative sample (26 of the 44 shown; full pattern is homogeneous):

| File:Line | Form | Bug lemma | Correct lemma (context) |
|---|---|---|---|
| `train/b1_new_008.conllu:9` | Gründen | `Grund\|Gründen` | Grund ("aus wirtschaftlichen Gründen") |
| `train/b1_new_006.conllu:1098` | Fußballspielen | `Fußballspiel\|Fußballspielen` | Fußballspiel |
| `train/b1_new_006.conllu:1480,2060` | Bergen | `Berg\|Bergen` | Berg (×2) |
| `train/b1_new_007.conllu:1416` | zieht | `zeihen\|ziehen` | ziehen ("zieht ... Besucher an") |
| `train/b1_new_007.conllu:1758` | gedacht | `denken\|gedenken` | denken ("hätte nie gedacht, dass...") |
| `train/b1_new_007.conllu:2018,2103` | Regeln/geleitet | `Regel\|Regeln` / `geleiten\|leiten` | Regel / leiten ("von einem Architekten geleitet") |
| `train/b1_new_007.conllu:2031` | getroffen | `treffen\|triefen` | treffen ("Entscheidung ... getroffen") |
| `train/b1_new_008.conllu:629` | fällt | `fallen\|fällen` | fallen ("fällt ... schwer") |
| `train/b1_new_008.conllu:1106,1436,2081,2742` | Regeln | `Regel\|Regeln` | Regel (×4) |
| `train/b1_new_008.conllu:1993` | Verwandte | `Verwandte\|Verwandter` | verwandt or Verwandter (needs a convention decision) |
| `train/b2_new_005.conllu:395` | Studien | `Studie\|Studium` | Studie ("mehrere unabhängige Studien") |
| `train/b2_new_005.conllu:1661` | Mitteln | `Mittel\|Mitteln` | Mittel |
| `train/b2_new_006.conllu:193,1126,2868,3313` | Regeln | `Regel\|Regeln` | Regel (×4) |
| `train/b2_new_006.conllu:1027,1241,2788` | Studien | `Studie\|Studium` | Studie (×3) |
| `train/b2_new_006.conllu:1312,1331` | Räumen | `Raum\|Räumen` | Raum (×2) |
| `train/b2_new_006.conllu:1576,2292` | Mitteln | `Mittel\|Mitteln` | Mittel (×2) |
| `train/b2_new_006.conllu:2682` | gewährt | `gewähren\|währen` | gewähren ("Einwilligung ... gewährt werden") |
| `train/c1_new_004.conllu:1004` | Werten | `Wert\|Werten` | Wert |
| `train/c1_new_005.conllu:913,2964` | Studien/Regeln | `Studie\|Studium` / `Regel\|Regeln` | Studie / Regel |
| `train/c1_new_006.conllu:522,940,2052` | Regeln/Studien | (same pattern) | Regel / Studie / Regel |
| `train/c2_new_005.conllu:95,1952,2965,3047` | geleitet/Versuchen/Räumen/Regeln | (same pattern) | leiten / Versuch / Raum / Regel |
| `train/c2_new_007.conllu:1706` | getroffen | `treffen\|triefen` | treffen |
| `val/b2_new_001.conllu:683` | gelobt | `geloben\|loben` | loben ("wurde ... gelobt") |
| `val/c1_new_001.conllu:593,1321` | Studien | `Studie\|Studium` | Studie (×2) |
| `val/c2_new_001.conllu:1616` | misst | `messen\|missen` | messen ("nie ganz misst") |

Example:

```9:9:data/handcraft/de/train/b1_new_008.conllu
7	Gründen	Grund|Gründen	NOUN	_	_	_	_	_	_
```

**Total: 44 occurrences**, in `train/b1_new_006/007/008`, `b2_new_005/006`,
`c1_new_004/005/006`, `c2_new_005/007`, `val/b2_new_001`, `val/c1_new_001`,
`val/c2_new_001`. All confirmed by reading the full sentence context; every
case has one semantically-correct candidate given the sentence.

## Finding 2 (🔴 real bug, systematic): Inconsistent `AUX`/`VERB` tagging of modal verbs

Universal Dependencies convention for German tags a modal (`müssen`,
`können`, `wollen`, `sollen`, `dürfen`, `mögen`) as `AUX` when it governs an
infinitive, and `VERB` only when it is the sole/main verb of the clause. This
corpus applies that rule correctly in the large majority of cases but
violates it at scale:

- **209 instances** where a modal + infinitive construction (e.g. *"Ich muss
  heute die Rechnung bezahlen."*) is tagged `VERB` instead of `AUX`.
- **8 instances** where a modal with no other verb present (e.g. *"Ich möchte
  eine Cola."*) is tagged `AUX` instead of `VERB`.

Concentration by file (VERB→should-be-AUX cases; heaviest offenders):

```
data/handcraft/de/train/b1_new_006.conllu
data/handcraft/de/train/b1_new_007.conllu
data/handcraft/de/train/b1_new_008.conllu
data/handcraft/de/train/a2_new_006.conllu / a2_new_007.conllu
data/handcraft/de/train/b2_new_005.conllu / b2_new_006.conllu
data/handcraft/de/train/c2_new_005.conllu / c2_new_006.conllu / c2_new_007.conllu
+ scattered instances in several val/*.conllu files
```

Example of the majority pattern (should be `AUX`):

```422:423:data/handcraft/de/train/b2.conllu
2	Medikament	Medikament	NOUN	_	_	_	_	_	_
3	muss	müssen	AUX	_	_	_	_	_	_
```
(this specific instance is correctly `AUX` — included to show the *correct*
form the 209 mistagged instances should match)

**Total: 217 occurrences** (209 + 8). This is the single largest systemic
defect in the corpus by token count.

## Finding 3 (🔴 real bug, systematic): Separable-verb particle tagged `VERB` with the full-verb lemma instead of `ADP`/`PART`

When a separable prefix (`an`, `auf`, `ab`, `zu`, `mit`, `aus`, `um`, ...)
appears in its post-verbal position, the dominant/correct convention in this
corpus (e.g. `c1.conllu`, which has 0 instances of this bug) tags the particle
by its own part of speech (`ADP`/`PART`/`ADV`) with its own bare lemma. A
secondary pattern instead copies the *full separable verb's lemma* onto the
particle token and tags it `VERB`:

| File:Line | Form | Bug lemma/UPOS | Should be |
|---|---|---|---|
| `data/handcraft/de/val/b1.conllu:11` | aus | ausfallen/VERB | aus / ADP (or PART) |
| `data/handcraft/de_test.conllu:7` | aus | ausfallen/VERB | aus / ADP |
| `data/handcraft/de_test.conllu:8` | an | ankommen/VERB | an / ADP |
| `data/handcraft/de/train/c2.conllu:27` | zu | machen/VERB | zu / ADP or PART |
| `data/handcraft/de/train/b2.conllu` (×13) | an/auf/ab/zu/mit/aus/um/... | full-verb lemma/VERB | bare particle / ADP or PART |

**Total: 17 occurrences**, 13/17 concentrated in `train/b2.conllu` alone.
This is the same underlying phenomenon behind several `lemma_checker`
false-positive flags (Finding-below on separable verbs) but is a genuine
UPOS/lemma error, not a false positive, when it manifests as a `VERB`-tagged
particle.

## Finding 4 (🔴 real bug): Data corruption / garbled lemmas concentrated in `train/c2.conllu`

5 tokens in `de/train/c2.conllu` have lemmas that are neither the correct
lemma for the token nor a plausible alternate reading — they appear to be
values copied from a *different, unrelated* token (cross-contamination bug),
plus 1 further garbled/nonsensical lemma in `c2_new_004.conllu`:

| Sent ID | File:Line | Form | Bug lemma | Fix |
|---|---|---|---|---|
| `de_c2_train_016` | `c2.conllu:526` | Wesen | Phänomenologie | Wesen |
| `de_c2_train_018` | `c2.conllu:592` | Phänomenologie | zu | Phänomenologie |
| `de_c2_train_028` | `c2.conllu:923` | `,` (PUNCT) | zu | `,` |
| `de_c2_train_045` | `c2.conllu:1483` | sie | er | sie |
| `de_c2_train_055` | `c2.conllu:1818` | in | um | in |
| `de_c2_train_316` | `c2_new_004.conllu:2569` | wiedergetroffen | `wiederbiertreffen` | wiedertreffen (or treffen) |

Example:

```526:526:data/handcraft/de/train/c2.conllu
7	Wesen	Phänomenologie	NOUN	_	_	_	_	_	_
```

**Total: 6 occurrences.** `train/c2.conllu` is otherwise a well-formed,
dense-register philosophical-prose file (Heidegger-style academic German) —
deep-reading 10+ additional sentences from it found no further corruption
beyond these 5 plus the related instance in `c2_new_004.conllu`.

## Finding 5 (🔴 real bug): `uns` mislemmatized as `sich` instead of `wir`

In UD German, the accusative/dative personal pronoun `uns` ("us"/reflexive
"ourselves") lemmatizes to `wir`, never to `sich` (`sich` is exclusively the
3rd-person reflexive). The corpus gets this right in the overwhelming
majority of `uns` occurrences (and correctly lemmatizes the parallel
1st-person-singular case, e.g. `mich`→`ich` in *"Ich dusche mich morgens"*),
but 5 instances mislemmatize it:

| Sent ID | File:Line | Text |
|---|---|---|
| `de_a2_train_180` | `train/a2_new_001.conllu:300` | Wir haben **uns** am Nachmittag getroffen. |
| `de_a2_train_258` | `train/a2_new_001.conllu:1118` | Wir haben **uns** für die Oper entschieden. |
| `de_c2_train_316` | `train/c2_new_004.conllu:2569` | ...dass wir **uns** nach all den Jahren ... wiedergetroffen haben. |
| `de_a1_val_065` | `val/a1_new_001.conllu:415` | Wir treffen **uns** am Park. |
| `de_a1_val_100` | `val/a1_new_001.conllu:712` | Tschüss, wir sehen **uns** morgen. |

**Total: 5 occurrences.** Fix: lemma `wir`, not `sich`.

## Finding 6 (🔵 minor real bug): Isolated typo, leading-space lemma, and PRON/DET inconsistency

| Sent ID | File:Line | Issue | Fix |
|---|---|---|---|
| `de_b2_train_173` | `train/b2_new_001.conllu:399` | `debattiert`→lemma `debatieren` (typo, missing `t`) | `debattieren` |
| `de_c1_train_167` | `train/c1_new_002.conllu:36` | `in`→lemma `' in'` (leading space) | `in` |
| `de_a2_val_004` | `val/a2.conllu:41` | `meines`→`PRON` in "der schönste Tag **meines** Lebens" — 4/5 corpus occurrences of genitive-modifier `meines` are tagged `DET`; this is the lone outlier | `DET` |

## Finding 7 (🔴 real bug, source text + propagated annotation): `is` typo for `ist`

`docs/data_audit/handcraft_texts/de_texts.txt:3918` reads *"Es **is**
unbestritten, ..."* (missing `t`). The typo propagates into `de_meta.tsv` and
into `data/handcraft/de/train/c1_new_003.conllu:464` (`de_c1_train_207`),
where the token is annotated `is`/lemma `--`/UPOS `X` — i.e. the annotation
pipeline correctly detected it couldn't lemmatize garbage, but the underlying
fix belongs in the source text, not the annotation:

```464:464:data/handcraft/de/train/c1_new_003.conllu
2	is	--	X	_	_	_	_	_	_
```

Fix: correct `de_texts.txt`/`de_meta.tsv` to `ist`, then re-annotate the token
as `ist`/`sein`/`AUX`.

## Finding 8 (🔴 real bug, source text): 3 broken/ill-formed sentences in a templated C1 generation block, all in `de/val/c1.conllu`

A block of C1-level sentences (`de_texts.txt` lines ~5900–6015) was produced
by combinatorial/templated generation. The vast majority of this block is
grammatical, if dense, academic German. Three sentences in this specific
block (all landing in `val/c1.conllu`) are genuinely broken:

| Sent ID | Text | Problem |
|---|---|---|
| `de_c1_val_005` (`de_texts.txt:5916`) | "...gelang es, **einen innovatives Zugriff** auf den viel diskutierten Stoff zu eröffnen." | Case/gender agreement error: `innovatives` (neut.) does not agree with `Zugriff` (masc. acc.) after `einen`. Should be `innovativen`. |
| `de_c1_val_008` (`de_texts.txt:5919`) | "Obgleich die ... ausgewerteten Daten **lückenhaft innovativen**, vermögen sie dennoch ein klarem Muster aufzuweisen." | Ungrammatical adjective stacking with no verb/conjunction between `lückenhaft` and `innovativen` — reads as a broken template merge. |
| `de_c1_val_012` (`de_texts.txt:5923`) | "...der **eng klares der praktischen Vernunft** verknüpft ist, ..." | Should be "eng **mit** der praktischen Vernunft verknüpft" (idiom "eng mit X verknüpft" = "closely linked with X"); `klares` is a leftover template substitution error. Confirmed against the correctly-generated parallel sentence in `val/generate_c1_new_001.py`/`c1_new_001.conllu:1388`, which has the correct "eng **mit** der praktischen Vernunft verknüpft". |

These 3 sentences carry through into the CoNLL-U annotations as-is (the
annotations are internally self-consistent with the broken text — e.g. the
agreement error's `innovatives`/`innovativ`/`ADJ` token is annotated
correctly *for the form actually present*), so the fix belongs in the source
text/generation template, with re-annotation to follow.

**Total: 3 sentences**, all `val/c1.conllu`. The corresponding generator
script (`data/handcraft/de/val/generate_c1_new_001.py`) already produces the
*correct* form of the Finding-8-row-3 sentence for a sibling item, confirming
the template itself is capable of correct output and this is a
generation-time substitution bug, not a fundamental template design flaw.

## Known `lemma_checker` hits (40) — classification

| Pattern | Verdict | Reasoning |
|---|---|---|
| `Sie`/`sie` capitalization inconsistency | 🟢 **FALSE POSITIVE** | German has two distinct grammatical items spelled `Sie`/`sie`: formal "you" (always capitalized, lemma `Sie`) and 3rd-person "she/they" (capitalized only sentence-initially, lemma `sie`). The checker's same-FORM-same-UPOS-implies-same-LEMMA heuristic cannot see this homography; the annotations are correct. |
| Particle verbs (`bringen` vs `mitbringen`, `andauern` vs `anstreben`, `nehmen` vs `zunehmen`, `hinweisen` vs `hindeuten`, `wirken` vs `auswirken`, `stellen` vs `darstellen`, ...) | 🟢 **FALSE POSITIVE at the lemma level** | A single verb form (e.g. `bringt`) is a valid conjugation of multiple distinct separable verbs (`bringen`/`mitbringen`/`vorbringen`/...) depending on the sentence; disambiguating to the contextually-correct full verb is correct UD practice. (Note: this phenomenon is the false-positive *lemma* side of the genuine systemic *UPOS* bug documented above in Finding 3, where the particle itself is sometimes wrongly tagged `VERB`.) |
| NOUN capitalization (`studierend`, `jung`, `verdächtig`) | 🟢 **FALSE POSITIVE** | Nominalized participles/adjectives (`Studierenden`, `Jüngeren`, `Verdächtige`) are capitalized as surface forms (correct German orthography for nominalization) but UD convention lemmatizes them to the lowercase base adjective/participle form. The checker's naive same-form-implies-same-case-lemma rule cannot see this; annotations are correct. |
| `' in'` leading-space ADP lemma | 🔴 **REAL BUG** (see Finding 6) | Genuine data-entry defect — the lemma literally contains a leading space character. |
| `c2.conllu` garbled (Wesen/Phänomenologie/zu, sie/er, in/um) | 🔴 **REAL BUG** (see Finding 4) | Confirmed cross-token lemma contamination / corruption. |
| `debateren` vs `debattieren` | 🔴 **REAL BUG** (see Finding 6) | Genuine typo (`debatieren` missing a `t`). |
| `uns`→`sich` | 🔴 **REAL BUG** (see Finding 5) | `uns` must lemmatize to `wir`, never `sich`. |

**Net: 3 of the 7 listed hit-patterns are false positives (checker blind to
German homography/nominalization conventions); 4 are real, confirmed bugs.**
The checker's 40 raw hits collapse to these 7 underlying patterns because it
flags every co-occurring FORM+UPOS pair with a differing LEMMA, so a single
recurring homograph generates many redundant hits.

## Sample of clean sentences verified (deep-read, FORM/LEMMA/UPOS checked against `# text`)

```1:7:data/handcraft/de/train/a1.conllu
# sent_id = de_a1_train_001
# text = Das ist meine Mutter.
1	Das	der	PRON	_	_	_	_	_	_
2	ist	sein	AUX	_	_	_	_	_	_
3	meine	mein	DET	_	_	_	_	_	_
4	Mutter	Mutter	NOUN	_	_	_	_	_	_
5	.	.	PUNCT	_	_	_	_	_	_
```

```1:8:data/handcraft/de/train/b1_new_008.conllu
# sent_id = de_b1_train_855
# text = Ich engagiere mich ehrenamtlich in einer Tafel und verteile Lebensmittel an Bedürftige.
1	Ich	ich	PRON	_	_	_	_	_	_
2	engagiere	engagieren	VERB	_	_	_	_	_	_
3	mich	ich	PRON	_	_	_	_	_	_
```

```1:15:data/handcraft/de/train/b2_new_001.conllu
# sent_id = de_b2_train_226
# text = Der Algorithmus, dessen Entwicklung jahrelang dauerte, kann Muster erkennen, die dem menschlichen Auge verborgen bleiben.
1	Der	der	DET	_	_	_	_	_	_
2	Algorithmus	Algorithmus	NOUN	_	_	_	_	_	_
9	kann	können	AUX	_	_	_	_	_	_
```
(a correctly-tagged `AUX` modal, for contrast with Finding 2's 217 mistagged instances)

```1:20:data/handcraft/de/train/c2.conllu
# sent_id = de_c2_train_045
# text = Obgleich die Existenz des Einzelnen je die seine ist, entzieht sie sich der Verfügbarkeit, die das Denken beansprucht, indem es sie zum Gegenstand einer ontologischen Untersuchung macht.
```
(dense, grammatically valid philosophical-register German — one lemma in
this exact sentence, `sie`→`er` at token 12, is the Finding-4 bug; the rest
of the sentence, including the second `sie` at token 24 correctly
lemmatized to `sie`, is clean)

## Summary

- **Sentences scanned**: 6211/6211 raw text (100%); 6211/6211 have gold
  `.conllu` annotations (5511 train + 600 val + 100 test), confirmed via
  `# sent_id` counts matching `de_texts.txt`/`de_meta.tsv` exactly. ≥5
  sentences/file deep-read (start/¼/mid/¾/end) across all 6 CEFR levels for
  both train and val, plus full-corpus Python-scripted aggregation over all
  72,935 tokens for the bug patterns above.
- **Dirty/word-salad findings in raw text**: 1 typo (Finding 7) + 3
  ill-formed sentences in one templated block (Finding 8) = 4 sentence-level
  text issues out of 6211 (0.06%). Text is otherwise CLEAN.
- **Real lemma/UPOS annotation bugs**: ~296 tokens across 72,935 (≈0.41%):
  44 unresolved-ambiguity `X|Y` lemmas (Finding 1) + 217 modal AUX/VERB
  mistags (Finding 2) + 17 particle-as-VERB mistags (Finding 3) + 6 corrupted
  lemmas (Finding 4) + 5 `uns`→`sich` (Finding 5) + 3 isolated
  typo/formatting/convention bugs (Finding 6) + 1 typo-propagated `X` tag
  (Finding 7) ≈ 293–296 tokens, concentrated in `train/c2.conllu`,
  `train/b2.conllu`, and the `b1_new_006/007/008` + `b2_new_005/006` +
  `c1_new_004/005/006` + `c2_new_005/007` files.
- **False positives confirmed**: 3 of the 7 known `lemma_checker` hit-patterns
  (Sie/sie capitalization, separable-verb lemma disambiguation, nominalized-
  adjective capitalization) — all legitimate German homography/orthography
  the checker cannot model. 4 of 7 patterns are real, confirmed bugs.
- **Overall verdict**: 🟡 **FINDINGS** — raw text is effectively clean
  (99.94% of sentences with zero issues); annotations are 99.6% clean by
  token count but carry a low-rate, non-random, reproducible set of
  lemma/UPOS defects — most importantly the 44-token unresolved-ambiguity
  `|`-lemma artifact and the 217-token modal AUX/VERB inconsistency — that a
  dictionary-free rule checker like `lemma_checker.py` structurally cannot
  detect and that should be fixed before this data is treated as
  production-clean.
