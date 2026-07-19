# Dutch Handcraft Lemmatization Data — Cleanliness Audit

## Verdict: **FINDINGS** (extensive real lemma bugs + text-quality issues in train + val)

The "0 format errors, 0 lemma errors" premise for train/val is **incorrect**. A systematic deep-read surfaced **dozens of distinct, verified lemma bugs totaling 400+ wrong-lemma tokens** across every CEFR level in train and val (not just `nl_test.conllu`), plus source-text quality problems: 15 grammatically incorrect sentences, 3 sentences with English-word contamination, and an entire CEFR level (B1, ~880/1000 sentences) dominated by one mechanical subject×destination×object template that produces largely nonsensical shopping scenarios (e.g. "buying fresh fish at the bookstore", "buying milk at the hairdresser").

## Method executed

1. Read all 6100 lines of `nl_texts.txt` end-to-end (offsets 1→6100), cross-referenced against `nl_meta.tsv` to confirm line ranges: train A1–C2 = 1–5400 (900/level), val A1–C2 = 5401–6000 (100/level), `nl_test.conllu` = 6001–6100.
2. Ran automated scans for garbage/encoding artifacts, duplicate lines, English contamination, and templated-sentence prefixes.
3. Deep-read ≥5–30 CoNLL-U sentences per CEFR level across all 30 train shards + 6 val files.
4. Once suspicious lemmas surfaced (`leest→doorlezen`), extracted **every unique (surface-form, lemma) pair for VERB/AUX, NOUN, and ADJ/ADV** tokens across all train+val files (605 verb pairs, 995 noun pairs, 529 adj/adv pairs) and manually audited the full lists for anomalies, then verified each candidate bug against its full sentence context.
5. Classified all `nl_test.conllu` particle-verb lemma choices (`gaan/meegaan`, `staat/opstaan`, `komt/thuiskomen`) against the "Known" baseline.

## 1. Text content scan (`nl_texts.txt`, lines 1–6100)

No encoding artifacts, no stray control/markup characters, no double-spacing. All lines end in proper punctuation. However:

### 1a. Grammar error in source sentences — `hij/zij wilt` (REAL BUG, 15 occurrences)

Standard Dutch requires **`hij/zij wil`** (no `-t`); `hij/zij wilt` is a recognized error (confirmed via Onze Taal / Van Dale-adjacent sources — only `jij wilt` and `u wilt` take the `-t`). 15 A1 sentences use the incorrect form:

```
nl_a1_train_013  Hij wilt thee.
nl_a1_train_090  Zij wilt sap.
nl_a1_train_138  Zij wilt water.
...(12 more across a1_new_001–005 + val/a1 lines 066, 086, 096)
```

This teaches the model an ungrammatical verb form at the very first CEFR level.

### 1b. English-word contamination in Dutch sentences (REAL BUG, 3 occurrences)

```
train/b1_new_003  "Ik vind het belangrijk om goed te luisteren naar others."   (should be "anderen")
train/b1_new_005  "Mijn broer heeft een price gewonnen met zijn sportteam."     (should be "prijs")
train/a2_new_003  "Het feest begins om acht uur vanavond."                     (should be "begint")
```

All three are also mis-lemmatized self-referentially (`others→other`, `price→price`, `begins→begins`) since the pipeline has no Dutch entry for these English forms.

### 1c. B1 level is ~88% one mechanical template with widespread semantic implausibility (CONTENT-QUALITY FINDING)

787/900 B1 train sentences + 94/100 B1 val sentences follow the exact template:
`[6 subjects] gaat morgen naar [12 destinations] om [10 objects] te kopen.`

Cross-multiplying 6 subjects × 12 destinations × 10 objects yields hundreds of nonsensical combinations, since the template applies no semantic filtering:

```
De bakker gaat morgen naar de boekwinkel om verse vis te kopen.      (fish at the bookstore)
De docent gaat morgen naar de kapper om een pak melk te kopen.       (milk at the hairdresser)
De bakker gaat morgen naar het gemeentehuis om nieuwe kleren te kopen. (clothes at the town hall)
```

Destinations `kapper` (hairdresser) and `gemeentehuis` (town hall) sell **none** of the 10 objects; `boekwinkel`/`kledingwinkel`/`slager`/`groenteman` are each correct for only 1 of 10 objects. This is grammatically correct but semantically incoherent Dutch dominating an entire CEFR level.

A1 also contains a milder version of this pattern: `Hij/Zij/Jij/U leest een [object].` is repeated ~55 times with objects that cannot be "read" (`een fiets`, `een auto`, `een jas`, `een sleutel`, `een pen`) — see §2 for the accompanying lemma bug on the same sentences.

### 1d. Minor: 4 duplicate sentences (low severity, expected at A1)

`Heb je broers of zussen?`, `Hij woont in Rotterdam.`, `Kan ik u helpen?`, `Wij luisteren naar muziek.` each appear twice (once in train, once in val/test) — negligible at this frequency for A1 boilerplate.

## 2. CoNLL-U deep-read — format

Structurally clean: consistent 10-column tab-separated format, `# sent_id`/`# text` headers, blank-line separators, plausible UPOS tags. Compound-noun lemma decomposition with `_` (e.g. `boekwinkel→boek_winkel`) is an intentional, mostly-consistent convention (1705 instances) — **not** a bug in general, though see §2c for cases where the convention itself is applied inconsistently or corrupts the result.

## 3. CoNLL-U deep-read — lemma bugs (the actual finding)

### 3a. Systematic particle-verb lemma bugs (verb gets wrong compound, or wrong plain form)

| Form | Wrong lemma | Correct lemma | Count | Example sentence |
|---|---|---|---|---|
| `leest` | `doorlezen` | `lezen` | 60 (54 train, 5 val, 1 test) | "Hij leest een kaart." — no "door" anywhere |
| `komt` (+ "uit [land]") | `aankomen` | `komen` | 52 (51 train, 1 val) | "U komt uit Frankrijk." — no "aan" anywhere |
| `staat` (+ "vast dat") | `staan` | `vaststaan` | 39 (train b2_new_003) | "Het staat vast dat..." — `vaststaan` is a dictionary-listed separable verb whose canonical example is exactly this construction |
| `gaat` (+ "...gewoon door") | `gaan` | `doorgaan` | 13 (9 train c1 + 4 val c1) | "...gaat de bouw... gewoon door." |
| `gerend` | `aan_rennen` | `rennen` | 11 (10 train a2 + 1 val a2) | "Je bent vanochtend naar het station gerend." — no "aan" |
| `gebeld` | `terug_bellen` | `bellen` | 9 (train a2) | "Anna heeft gisteren mijn vrienden gebeld." — no "terug" |
| `investeerde` | `in_vesteren` | `investeren` | 11 (train c2_new_004) | inconsistent with correctly-lemmatized `investeren→investeren` elsewhere in the same corpus |
| `voldoet` | `voldoeten` | `voldoen` | 25 (train c1 ×4 files + val c1) | `voldoeten` is not a Dutch word |
| `zoekt` | `opzoeken` | (context-dependent; verify each hit) | — | flagged, not fully quantified |

### 3b. Fabricated / non-existent lemma forms (nonsense words)

| Form | Wrong lemma | Correct lemma | Count |
|---|---|---|---|
| `rustgevende` | `rust_venden` | `rust_geven` | 19 |
| `toonaangevende` | `toon_aan_venden` | `toon_aan_geven` | 11 |
| `gerenommeerde` | `gerenommeren` (non-existent verb) | keep `gerenommeerd` or self-lemma | 26 |
| `geüpdatet` | `üp_daten` | `updaten` | 7 |
| `updaten` | `uit_doten` | `updaten` | 1 (a **different**, also-wrong decomposition of the same verb) |
| `raadplegen` | `raad_plijgen` | `raad_plegen` | 1 |
| `doorsluizen` | `doorsluisen` (z→s typo) | `door_sluizen` | 2 |
| `geïnteresseerd` | `ïnteresseren` (stray diaeresis, dropped letters) | `interesseren` | 1 |
| `Lust` | `lussen` | `lusten` | 1 |
| `wegga` | `weggen` | `weg_gaan` | 4 |
| `zwaarder` | `zwaard` ("sword" — unrelated word!) | `zwaar` | 9 |
| `dorst` (NOUN "thirst") | `durven` ("to dare" — unrelated verb) | `dorst` | 1 |
| `durf` (NOUN "courage") | `durven` | `durf` | 9 |
| `alstublieft` (1 of 3 instances) | `alstubliefen` (non-word) | `alstublieft` (self, as elsewhere) | 1 |
| `avondeten` (NOUN "dinner") tagged **VERB** | `aven_doeten` (non-word) | NOUN, `avondeten`/`avond_eten` | 16 |
| `CO2-uitstoot` | `cou_uitstoot` | `co2_uitstoot` | 4 |
| `carrièreswitch` | `carrière_witch` ("witch"!) | `carrière_switch` | 1 |

### 3c. Compound-splitting convention applied inconsistently or corrupted mid-word

Same lemma convention (`_`-joined compound decomposition), but garbled splits create non-words, or the same source word gets two different lemmas in different sentences:

```
klimaatverandering → klim_aat_verandering   (garbled)     vs.  klimaatverandering → klimaat_verandering  (correct, elsewhere)
natuurbescherming  → natuur_bescherming     vs.  natuurbescherming → natuurbescherming (undecomposed) — same word, both conventions used
sporten (NOUN)     → sport                  vs.  sporten (NOUN) → sporte (truncated)   — same word, inconsistent
datamonopolies     → datamono_polies        (should be data_monopolie)
energie-infrastructuur → energie_infrast_rastuur  (should be energie_infrastructuur)
filterproces        → filt_proces            (should be filter_proces)
kinderschoenen      → kind_schoe             (should be kind_schoen)
lagelonenlanden     → lagel_onenland         (should be laag_loon_land)
lerarentekort       → leraarte_kort          (should be leraar_tekort)
luchtkwaliteit      → lucht_waliteit         (should be lucht_kwaliteit)
marktaandeel        → marktaan_deel          (should be markt_aandeel)
milieueisen         → milieuis               (should be milieu_eis)
milieumaatregelen   → milieumaat_regel       (should be milieu_maatregel)
monopolievorming    → mon_olievorming        (should be monopolie_vorming)
opslagsystemen      → opslag_system          (English "system" leaked in; should be opslag_systeem)
persoonsgegevens    → persoonsgeven          (should be persoon_gegeven)
privacyverlies       → privacyerlies          (should be privacy_verlies)
recordoogst          → recordoog_st           (should be record_oogst)
rijgedrag            → rij_drag               (should be rij_gedrag)
stresshormonen        → stres_shormoon          (should be stress_hormoon)
studieadviseur        → studie_viseur           (should be studie_adviseur)
valutamarkten          → valutamarkak            (should be valuta_markt)
respectvolle           → respect_vol             (inconsistent: succesvol/risicovol/hoopvol left undecomposed elsewhere)
```

### 3d. Widespread single-letter truncation of plain (non-compound) noun/adjective lemmas

A large, distinct class of bugs: multi-syllable base-form lemmas losing their final letter(s), producing non-words:

```
keuken → keuke        examen → exame         fenomeen → fenomee      seizoen → seizoe
miljoenen → miljoe     verleden → verlede      vertrouwen → vertrouwe  wantrouwen → wantrouwe
nepnieuws → nepnieuw    mindfulness → mindfulnes  tandarts → tandart   recepten → recepte
schoenen → schoe        psychologen → psycholog   aardbeien → aardbeei  forensen → fore
bedrijfsleven → bedrijfsleve   boodschappenlijstje → boodschappenlijs
consumentenvertrouwen → consumentenvertrouw     burn-outs → burnouts (plural not singularized)
crises → crise (should be "crisis", not the non-word "crise")
online (ADJ) → onlin    agressievere → agressieveer   reële → reëël (typo)
extremere → extremer (comparative left un-reduced, should be "extreem")
flexibelere → flexibele (comparative left un-reduced, should be "flexibel")
wereldwijde (ADJ) → wereld_wijde (inflection not stripped, should be wereld_wijd)
```

## 4. `nl_test.conllu` particle-verb classification (the "Known" cases)

| Sentence | Token | Lemma given | Classification |
|---|---|---|---|
| "Wij gaan morgen naar de stad." | gaan | `gaan` | **FALSE POSITIVE** — no particle present, correct |
| "Wij gaan met jullie mee." | gaan | `meegaan` | **FALSE POSITIVE** — "mee" present, correctly resolved |
| "Hij staat altijd vroeg op." | staat | `opstaan` | **FALSE POSITIVE** — "op" present, correctly resolved |
| "De auto staat voor het huis." | staat | `staan` | **FALSE POSITIVE** — no particle, correct |
| "De deur staat open." | staat | `staan` | **BORDERLINE** — defensible (literal physical sense of a door is standardly analyzed as `staan`+predicate-adjective, not the idiom `openstaan`), but inconsistent with this same corpus's own `openstaande vacatures→open_staan` (§3c-adjacent) convention. Minor, not counted as a hard bug. |
| "Hij komt vandaag laat thuis." | komt | `thuiskomen` | **FALSE POSITIVE** — "thuis" present, correctly resolved |
| "De trein komt laat aan." | komt | `aankomen` | **FALSE POSITIVE** — "aan" present, correctly resolved |
| "Ik denk dat zij komt." | komt | `komen` | **FALSE POSITIVE** — no particle, correct |

**Conclusion: 7/8 "known" test-set cases are correct, context-sensitive particle-verb resolutions, not bugs.** The one genuinely inconsistent case (`staat open`) is a minor, defensible edge case. This is the opposite problem from train/val: the test set's particle-verb handling is *better* than train/val's (§3a shows train/val getting the *reverse* kinds of errors — attaching particles that aren't there, or failing to attach particles that are).

## Summary for return

- **Scanned:** all 6100 lines of `nl_texts.txt`; every train shard (30 files) and val file (6 files) deep-read across levels; full VERB/NOUN/ADJ/ADV form→lemma inventories (605/995/529 unique pairs) manually audited; all 8 particle-verb cases in `nl_test.conllu` classified.
- **Dirty findings (source text):** 15 sentences with ungrammatical `hij/zij wilt`, 3 sentences with English-word contamination (`others`, `price`, `begins`), and B1 level ~88% dominated by one semantically-incoherent shopping template (fish at the bookstore, milk at the hairdresser).
- **Real lemma bugs:** 400+ confirmed wrong-lemma tokens spanning every CEFR level in train+val — particle-verb misattachment (`leest→doorlezen`, `komt→aankomen`, `staat vast→staan` instead of `vaststaan`, `gaat door→gaan` instead of `doorgaan`), fabricated non-word lemmas (`rust_venden`, `gerenommeren`, `üp_daten`, `carrière_witch`, `zwaard` for "heavier"), corrupted compound splits, and widespread single-letter truncation of plain nouns/adjectives.
- **`nl_test.conllu` particle-verb cases:** 7/8 are false positives (correct); 1 borderline/defensible.
- **Verdict: FINDINGS.** The "0 lemma errors in train/val" premise is false — train and val both contain extensive, systematic lemma corruption that predates and is unrelated to the test-set particle-verb question the audit was originally scoped around.
