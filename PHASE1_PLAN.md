# Phase 1 Improvement Plan

## Current State

**Silver Data:**
- DE: 500 entries (~2500 sentences)
- EN: 200 entries (~1000 sentences)  
- ES: 200 entries (~1000 sentences)

**Held-Out Validation Results:**
- DE: 92.3% lemma, 93.9% UPOS, 15 errors
- EN: 92.6% lemma, 95.1% UPOS, 15 errors
- ES: 96.2% lemma, 95.3% UPOS, 8 errors

## Error Analysis

### German (15 errors)
**Identity errors (12/15):** Model predicts IDENTITY when lemma differs
- Inflected adjectives: `aufmerksamer→aufmerksam`, `nächstes→nächster`
- Plural nouns: `Großeltern→Großelter`, `Zutaten→zutat`
- Contractions: `ins→in`, `beim→bei`
- Verb participles: `veröffentlichten→veröffentlichen`, `halfen→helfen`
- Pronouns: `uns→sich`

**Edit-failures (3/15):** Edit-tree label can't apply
- `Wissenschaftlerinnen→wissenschaftlerin` (feminine plural -innen)
- `renovierte→renoviert` (adj inflection)
- `Unwetters→unwetter` (genitive -s)

### English (15 errors)
**Plural nouns:** `children→child`, `punishments→punishment`, `administrations→administration`
**Verb over-stemming:** `crumbling→crumbl`, `predicted→predicte`
**DET lemma:** `an→a` (4 occurrences)
**Participle issues:** `shattered→shatter`, `complicated→complicate`

### Spanish (8 errors)
**Conditional auxiliaries:** `Habríamos→haber`, `Habrías→haber`, `hubieras→haber`
**Plural nouns:** `revolucionarios→revolucionario`, `hallazgos→hallazgo`
**Reflexive verbs:** `esforzarnos→esforzarse`

## Phase 1 Implementation

### 1. Targeted Silver Data Generation (1500 entries per language)

**Create `src/generate_targeted_silver.py`:**
- Language-specific prompts targeting problem patterns
- DE: Focus on inflected adjectives, plural nouns, contractions
- EN: Focus on irregular plurals, participles, DET contexts
- ES: Focus on conditionals, reflexives, plurals

**Prompt templates:**

**German:**
```
Schreibe 5 deutsche Sätze mit flektierten Adjektiven (Endungen -e, -en, -er, -es).
Beispiele: "der schnelle Zug", "eine schöne Blume", "mit großem Erfolg"
Gib nur die Sätze zurück, eine pro Zeile.
```

```
Schreibe 5 deutsche Sätze mit Pluralnomen.
Beispiele: "die Kinder", "viele Bücher", "alle Studenten"
Gib nur die Sätze zurück, eine pro Zeile.
```

```
Schreibe 5 deutsche Sätze mit Kontraktionen (ins, beim, zum, zur, vom).
Beispiele: "Ich gehe ins Kino", "Er ist beim Arzt", "Wir fahren zum Strand"
Gib nur die Sätze zurück, eine pro Zeile.
```

**English:**
```
Write 5 English sentences with irregular plural nouns.
Examples: "children play", "men work", "women lead", "people gather"
Return only sentences, one per line.
```

```
Write 5 English sentences with past participles as adjectives.
Examples: "the broken window", "a shattered dream", "complicated rules"
Return only sentences, one per line.
```

**Spanish:**
```
Escribe 5 oraciones en español con verbos en condicional compuesto.
Ejemplos: "habríamos ido", "habrías sabido", "habrían llegado"
Devuelve solo las oraciones, una por línea.
```

```
Escribe 5 oraciones en español con verbos reflexivos conjugados.
Ejemplos: "nos esforzamos", "se levantaron", "me desperté"
Devuelve solo las oraciones, una por línea.
```

**Target:** 300 batches × 5 sentences = 1500 sentences per language

### 2. Post-Processing Rules

**Create `src/postprocess_rules.py`:**

```python
"""Deterministic post-processing rules for common lemma patterns."""

# German contractions
DE_CONTRACTIONS = {
    'ins': 'in',
    'beim': 'bei',
    'zum': 'zu',
    'zur': 'zu',
    'vom': 'von',
    'im': 'in',
    'am': 'an',
    'ans': 'an',
    'aufs': 'auf',
    'durchs': 'durch',
    'fürs': 'für',
    'hinterm': 'hinter',
    'hinters': 'hinter',
    'nebenm': 'neben',
    'übers': 'über',
    'unterm': 'unter',
    'unters': 'unter',
    'vorm': 'vor',
    'vors': 'vor',
}

# English irregular plurals
EN_IRREGULAR_PLURALS = {
    'children': 'child',
    'men': 'man',
    'women': 'woman',
    'people': 'person',
    'mice': 'mouse',
    'geese': 'goose',
    'teeth': 'tooth',
    'feet': 'foot',
    'oxen': 'ox',
    'lice': 'louse',
    'dice': 'die',
    'cacti': 'cactus',
    'fungi': 'fungus',
    'nuclei': 'nucleus',
    'syllabi': 'syllabus',
    'alumni': 'alumnus',
    'criteria': 'criterion',
    'phenomena': 'phenomenon',
    'data': 'datum',
    'media': 'medium',
    'analyses': 'analysis',
    'bases': 'basis',
    'crises': 'crisis',
    'diagnoses': 'diagnosis',
    'hypotheses': 'hypothesis',
    'oases': 'oasis',
    'parentheses': 'parenthesis',
    'synopses': 'synopsis',
    'theses': 'thesis',
}

# Spanish reflexive verb patterns
def es_reflexive_lemma(word: str) -> str | None:
    """Convert conjugated reflexive to infinitive + se."""
    if word.endswith('nos'):
        # nosotros form: esforzarnos -> esforzarse
        stem = word[:-3]
        return stem + 'se'
    if word.endswith('os'):
        # vosotros form: esforzaros -> esforzarse
        stem = word[:-2]
        return stem + 'se'
    if word.endswith('se'):
        # ellos/ellas form: esforzarse -> esforzarse
        return word
    return None


def apply_postprocess_rules(word: str, lang: str, pred_lemma: str, pred_upos: str) -> str:
    """Apply deterministic rules to improve lemma predictions."""
    word_lower = word.lower()
    
    if lang == 'de':
        # Check contractions
        if word_lower in DE_CONTRACTIONS:
            return DE_CONTRACTIONS[word_lower]
    
    elif lang == 'en':
        # Check irregular plurals
        if word_lower in EN_IRREGULAR_PLURALS:
            return EN_IRREGULAR_PLURALS[word_lower]
    
    elif lang == 'es':
        # Check reflexive verbs
        if pred_upos == 'VERB':
            reflexive = es_reflexive_lemma(word_lower)
            if reflexive:
                return reflexive
    
    return pred_lemma
```

### 3. Integration

**Modify `src/evaluate.py`:**

```python
from postprocess_rules import apply_postprocess_rules

def resolve_prediction(word, upos, base_label, lexicon, lang='de'):
    if upos == "PROPN":
        return None, "propn", False
    
    if base_label is not None:
        applied = apply_edit_label(word, base_label)
        if applied is not None:
            # Apply post-processing rules
            final = apply_postprocess_rules(word, lang, applied, upos)
            return final, "edit", False
        edit_failed = True
    else:
        edit_failed = False
    
    lexicon_lemma = lexicon.get(word)
    if lexicon_lemma is not None:
        # Apply post-processing rules
        final = apply_postprocess_rules(word, lang, lexicon_lemma, upos)
        return final, "lexicon", edit_failed
    
    # Apply post-processing rules even for identity
    final = apply_postprocess_rules(word, lang, word, upos)
    return final, "identity", edit_failed
```

### 4. Retraining

**Steps:**
1. Generate targeted silver data (1500 entries per language)
2. Rebuild datasets with expanded silver data
3. Retrain models with new data
4. Evaluate on held-out set
5. Compare improvements

**Expected improvements:**
- DE: 92.3% → 95%+ (better inflection handling)
- EN: 92.6% → 95%+ (better plurals and participles)
- ES: 96.2% → 97%+ (better conditionals and reflexives)

## Implementation Checklist

- [ ] Create `src/generate_targeted_silver.py`
- [ ] Generate 1500 DE silver entries (inflections, plurals, contractions)
- [ ] Generate 1500 EN silver entries (irregular plurals, participles)
- [ ] Generate 1500 ES silver entries (conditionals, reflexives)
- [ ] Create `src/postprocess_rules.py`
- [ ] Add DE contraction rules
- [ ] Add EN irregular plural dictionary
- [ ] Add ES reflexive verb handler
- [ ] Integrate post-processing into `resolve_prediction`
- [ ] Update `validate_held_out.py` to use post-processing
- [ ] Rebuild datasets with expanded silver data
- [ ] Retrain DE model
- [ ] Retrain EN model
- [ ] Retrain ES model
- [ ] Evaluate on held-out set
- [ ] Export ONNX models
- [ ] Package browser bundles
- [ ] Update web demo

## Testing

**Unit tests for post-processing:**
```python
def test_de_contractions():
    assert apply_postprocess_rules('ins', 'de', 'ins', 'ADP') == 'in'
    assert apply_postprocess_rules('beim', 'de', 'beim', 'ADP') == 'bei'

def test_en_irregular_plurals():
    assert apply_postprocess_rules('children', 'en', 'children', 'NOUN') == 'child'
    assert apply_postprocess_rules('men', 'en', 'men', 'NOUN') == 'man'

def test_es_reflexives():
    assert apply_postprocess_rules('esforzarnos', 'es', 'esforzarnos', 'VERB') == 'esforzarse'
```

## Success Criteria

- DE held-out lemma accuracy: ≥95%
- EN held-out lemma accuracy: ≥95%
- ES held-out lemma accuracy: ≥97%
- Edit-failure rate: <5% for all languages
- All unit tests pass
- Ruff lint clean
