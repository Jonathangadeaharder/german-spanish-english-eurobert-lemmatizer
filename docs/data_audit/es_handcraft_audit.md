# Spanish Handcraft Lemmatization Data — Cleanliness Audit

## Verdict: **FINDINGS** (real lemma bugs present in train + val)

The prose content is clean (no word salad, nonsense, or wrong-language contamination). However, the CoNLL-U deep-read uncovered a **systematic lemma bug** that the "0 lemma errors" assumption did not account for: **277 confirmed wrong lemmas** across **16 files** (13 train shards + 3 val files), i.e. this is *not* limited to `es_test.conllu`'s known tokenizer-spacing artifact.

## Method executed

1. Read all ~6100 lines of `es_texts.txt` end-to-end in sequential chunks (offsets 1→6100).
2. Cross-referenced `es_meta.tsv` to confirm line ranges 1–6000 map to train/val (`es/train/*.conllu`, `es/val/*.conllu`) and 6001–6100 map to `es_test.conllu`.
3. Deep-read ≥5 sentences per CEFR level (A1–C2) in both train and val CoNLL-U files.
4. Once a suspicious lemma surfaced (`tribunales → tribunale`), ran a systematic regex/Python sweep across **all 36 train+val `.conllu` files** to quantify the bug's true scope, and inspected the shared generator script to find the root cause.

## 1. Text content scan (`es_texts.txt`, lines 1–6000 = train/val)

**Clean.** All 6000 lines are well-formed, grammatical Spanish. Content is deliberately graded by CEFR level:

- **A1/A2** (~lines 1–1600, 6001–6100 test tail): short, simple everyday sentences ("Hola, amigo mío.", "Llueve mucho en otoño aquí en mi ciudad.").
- **B1/B2** (~1600–3600): compound sentences on concrete topics (housing, law, health, economics).
- **C1/C2** (~3600–6000): dense academic register spanning philosophy of mind, jurisprudence, economics, literary theory, history, climate science, international relations — sophisticated but coherent and topically consistent throughout.

No word salad, no nonsense strings, no wrong-language contamination, no encoding artifacts found in any of the 6000 train/val lines. (Lines 6001–6100 belong to `es_test.conllu`, out of scope per task — matches the pre-known ¿-spacing tokenizer artifact, not dirty content.)

## 2. CoNLL-U deep-read — format

Structurally clean across all levels sampled (A1–C2, train shards `*_new_00[1-5].conllu` and all 6 val files): consistent 10-column tab-separated format, `# sent_id` / `# text` headers present, blank-line sentence separators intact, UPOS tags plausible (DET/AUX/PRON choices for `mío`, `se→él`, etc. follow AnCora-style UD conventions).

## 3. CoNLL-U deep-read — lemma bug (the actual finding)

### Root cause

`train/generate_b2_new_00{2,3,4,5}.py` (identical files, byte-for-byte) post-process Stanza's lemma output with this heuristic (`normalize_token`, ~line 207):

```python
if upos == "NOUN" and lemma:
    lemma = lemma.lower()
    if lemma.endswith("es") and fl.endswith("es") and len(lemma) > 3:
        lemma = lemma[:-2] if lemma.endswith("iones") else lemma[:-1]
```

This assumes any noun ending in `-es` needs only its final `-s` stripped to reach the lemma (correct for e.g. `coches→coche`). But it fires whenever **Stanza's own lemma output happens to still end in `-es`** (i.e. whenever Stanza's model failed and fell back to returning the surface form) — which is common for:

- Plurals of consonant-final nouns, where Spanish morphology requires stripping the full `-es`, not just `-s` (`tribunal-es→tribunal`, not `tribunale`).
- Nouns/adjectives ending in `-dad` (`ciudad-es→ciudad`, not `ciudade`).
- Singular nouns that merely *happen* to end in `-es` (`mes→me`!) or invariant day-names (`martes→marte`, `jueves→jueve`, `viernes→vierne`, `lunes→lune`).

Crucially, this same pipeline gets the *n*-stem irregulars (`imágenes→imagen`, `jóvenes→joven`, `órdenes→orden`, `regímenes→régimen`) **right** in every file — because Stanza's lemmatizer succeeds on those, so the buggy fallback branch never triggers. The inconsistency (some `-es` nouns correct, others wrong, even within one sentence) confirms this is a **Stanza-lemma-fallback + naive suffix-strip** interaction bug, not a hand-written annotation error — and it evidently was not caught by the repo's own `check_text()` lemma checker.

### Confirmed occurrence counts

| File | Wrong lemmas |
|---|---|
| `train/b2_new_004.conllu` | 71 |
| `train/b2_new_002.conllu` | 63 |
| `train/b2_new_005.conllu` | 52 |
| `train/b2_new_003.conllu` | 53 |
| `train/a2_new_001.conllu` | 6 |
| `train/a2_new_003.conllu` | 5 |
| `train/a2_new_004.conllu` | 5 |
| `train/b1_new_001.conllu` | 5 |
| `train/a2_new_002.conllu` | 4 |
| `train/b1_new_002.conllu` | 3 |
| `train/b1_new_003.conllu` | 2 |
| `train/c2_new_003.conllu` | 2 |
| `train/c2_new_005.conllu` | 1 |
| `val/b1.conllu` | 3 |
| `val/a2.conllu` | 1 |
| `val/b2.conllu` | 1 |
| **Total** | **277** (272 train / 5 val) |

~86% of the errors are concentrated in the 4 B2 train shards that share the buggy generator script (`b2_new_002.conllu` through `b2_new_005.conllu`); `b2_new_001.conllu` uses a different generator and is unaffected. The remaining ~14% are scattered one-offs in other levels/files where Stanza independently failed on a particular word.

### Representative examples (verified by direct file read, not grep — grep's ANSI highlight corrupted captured output during initial scanning and was discarded as a false lead)

| Sentence (excerpt) | Token | Wrong lemma | Correct lemma |
|---|---|---|---|
| `b2_new_003.conllu:37` — "...siguen siendo largos en **tribunales** de primera instancia." | tribunales | `tribunale` | `tribunal` |
| `b2_new_004.conllu:1104` — "...recibió quejas de humedad..." (hospital context) | hospitales | `hospitale` | `hospital` |
| `b2_new_005.conllu:351` — "...crecimiento desigual entre **sectores**..." | sectores | `sectore` | `sector` |
| `b2_new_002.conllu:206` | materiales | `materiale` | `material` |
| `val/b2.conllu:847` — "Fortalecer clínicas rurales... cada **mes**." | mes | `me` | `mes` |
| `val/b1.conllu` — "...el próximo **viernes**." | viernes | `vierne` | `viernes` (invariant) |
| `val/a2.conllu` / `val/b1.conllu` — "los **martes**..." | martes | `marte` | `martes` (invariant; `marte`≈"Mars" is a different word) |
| `train/b2_new_005.conllu:1670` — sentence-initial "**Certificaciones** ..." | Certificaciones | `certificacione` | `certificación` |

Two additional minor/debatable edge cases (foreign technical terms, not part of the systematic bug, low priority): `sorites` (Greek loan, "sorites paradox") → `sorite`, and `omnes` (Latin legal phrase *erga omnes*) → `omne`. Both should arguably be invariant.

## Summary for return

- **Scanned:** 6000 `es_texts.txt` train/val lines (full file) + ≥5 CoNLL-U sentences per CEFR level (A1–C2) in train and val, escalated to a full 36-file/~103k-token sweep once the bug was found.
- **Dirty findings (word salad/nonsense/wrong language):** 0. Prose content is clean throughout.
- **Real bugs:** **277 wrong lemmas** (272 train / 5 val), a systematic Stanza-fallback + naive-suffix-strip bug concentrated in `b2_new_002–005.conllu`, plus scattered day-name (`martes/jueves/viernes/lunes`) and `mes` mislemmatizations bleeding into `val/a2.conllu`, `val/b1.conllu`, `val/b2.conllu`.
- **Verdict: FINDINGS** — the "0 lemma errors in train/val" premise is incorrect; val is not clean, and train has hundreds of wrong lemmas needing correction (regenerate affected sentences or patch `tribunale`-style outputs programmatically).
