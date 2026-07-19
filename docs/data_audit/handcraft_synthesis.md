# Handcraft Training Data Audit — Synthesis

**Date:** 2026-07-15  
**Scope:** all handcraft train+val (+test spot-check) across 8 languages  
**Method:** automated `lemma_checker` / `conllu_validator` + 8 parallel language subagents (full text scan + CoNLL-U deep-read)  
**Parent docs:** per-lang reports `*_handcraft_audit.md`

## Coverage

| Lang | Sentences (train+val+test texts) | Train+val tokens | Text scan | CoNLL deep-read |
|------|----------------------------------|------------------|-----------|-----------------|
| de | 6211 | 72118 | 100% | ≥5/file all levels |
| en | 6100 | 86387 | 100% | ≥5/file all levels |
| es | 6100 | 85021 | 100% | ≥5/file + full sweep |
| fr | 6100 | 84646 | 100% | ≥5/file + full stats |
| nl | 6100 | 83644 | 100% | ≥5/file + lemma inventory |
| sv | 6100 | 72858 | 100% | ≥5/file + full token scan |
| ar | 6100 | 76293 | 100% | ≥5/file + full token scan |
| zh | 6100 | 84826 | 100% | ≥5/file + full stats |
| **Total** | **~48911** | **~645k** | **100% texts** | **all files sampled** |

## Verdict by language

| Lang | Text quality | Annotation quality | Overall | Severity driver |
|------|--------------|--------------------|---------|-----------------|
| de | ~99.9% clean | ~99.6% | 🟡 FINDINGS | Separable verbs, modal AUX/VERB, `|`-ambiguity lemmas, c2 corruption |
| en | CLEAN | ~99.8% | 🟡 FINDINGS | Truncated lemmas (`embed→emb`, `need→ne`, `bring→br`) |
| es | CLEAN | dirty pockets | 🟡 FINDINGS | Bad `-es` strip (`tribunales→tribunale`, `mes→me`, `martes→marte`) |
| fr | ~99.9% clean | **DIRTY** | 🔴 FINDINGS | **25.1% tokens `UPOS=X`** — elision/contraction not tokenized |
| nl | dirty pockets | **DIRTY** | 🔴 FINDINGS | Wrong particle lemmas, truncated nouns, B1 shopping-template nonsense |
| sv | grammar issues | dirty pockets | 🟡 FINDINGS | `åt` past-of-äta→ADP; adjective agreement; B2 bare infinitives |
| ar | mostly clean | **DIRTY** | 🔴 FINDINGS | **44.1% tokens `UPOS=X`** (tail collapse); و-stripping (`وقت→قت`) |
| zh | CLEAN | **DIRTY** | 🔴 FINDINGS | **100% char-level tokens**; **24.7% `UPOS=X`**; PUNCT on content chars |

## Empirically verified critical metrics (train+val)

```
lang  tokens   UPOS=X%   sents_with_X%   notes
de    72118    0.0%      0.0%
en    86387    0.0%      0.1%
es    85021    0.0%      0.0%
fr    84646    25.1%     55.8%           elisions as single X tokens
nl    83644    0.0%      0.0%
sv    72858    0.0%      0.0%
ar    76293    44.1%     82.8%           analysis tail collapse
zh    84826    24.7%     91.1%           + 100% single-char forms
```

## What `lemma_checker` missed

Checker reported train+val clean for en/es/fr/nl/sv/ar/zh. That is **false confidence**:

1. **No `fr` language rules** — only generic checks; X-tagged contractions pass.
2. **Per-file FORM+UPOS consistency** — misses cross-file wrong lemmas and within-file consistent-but-wrong labels (`de→un`).
3. **No dictionary / lemma-plausibility** — truncated lemmas (`emb`, `tribunale`, `keuke`) invisible.
4. **zh `lemma==form` rule** too strict on test closed-class exceptions; too weak on train char-level + UPOS corruption.

## Bottom line (updated 2026-07-15)

**Automated handcraft generation removed.** All poisoned train/val `.conllu` files deleted.

- **Train/val handcraft:** empty — curate manually per `data/handcraft/README.md`.
- **Gold UD `data/gold/*/train.conllu`:** primary training source (`eurobert-lemma fetch-ud`).
- **`{lang}_test.conllu`:** clean hand-curated eval sets — keep.

## Per-lang reports

- [de_handcraft_audit.md](de_handcraft_audit.md)
- [en_handcraft_audit.md](en_handcraft_audit.md)
- [es_handcraft_audit.md](es_handcraft_audit.md)
- [fr_handcraft_audit.md](fr_handcraft_audit.md)
- [nl_handcraft_audit.md](nl_handcraft_audit.md)
- [sv_handcraft_audit.md](sv_handcraft_audit.md)
- [ar_handcraft_audit.md](ar_handcraft_audit.md)
- [zh_handcraft_audit.md](zh_handcraft_audit.md)
