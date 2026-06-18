from __future__ import annotations

DE_CONTRACTIONS: dict[str, str] = {
    "ins": "in",
    "beim": "bei",
    "zum": "zu",
    "zur": "zu",
    "vom": "von",
    "im": "in",
    "am": "an",
    "ans": "an",
    "aufs": "auf",
    "durchs": "durch",
    "fürs": "für",
    "hinterm": "hinter",
    "hinters": "hinter",
    "übers": "über",
    "unterm": "unter",
    "unters": "unter",
    "vorm": "vor",
    "vors": "vor",
    "nebenm": "neben",
    "nebens": "neben",
    "zwischenm": "zwischen",
    "zwischens": "zwischen",
    "überm": "über",
    "erm": "er",
    "s": "es",
}

EN_IRREGULAR_PLURALS: dict[str, str] = {
    "children": "child",
    "men": "man",
    "women": "woman",
    "people": "person",
    "mice": "mouse",
    "geese": "goose",
    "teeth": "tooth",
    "feet": "foot",
    "oxen": "ox",
    "lice": "louse",
    "dice": "die",
    "cacti": "cactus",
    "fungi": "fungus",
    "nuclei": "nucleus",
    "syllabi": "syllabus",
    "alumni": "alumnus",
    "criteria": "criterion",
    "phenomena": "phenomenon",
    "data": "datum",
    "media": "medium",
    "analyses": "analysis",
    "bases": "basis",
    "crises": "crisis",
    "diagnoses": "diagnosis",
    "hypotheses": "hypothesis",
    "oases": "oasis",
    "parentheses": "parenthesis",
    "synopses": "synopsis",
    "theses": "thesis",
    "stimuli": "stimulus",
    "syllabuses": "syllabus",
    "cactuses": "cactus",
    "appendices": "appendix",
    "indices": "index",
    "matrices": "matrix",
    "vertices": "vertex",
    "corpora": "corpus",
    "genera": "genus",
    "stigmata": "stigma",
    "dogmata": "dogma",
    "lemmata": "lemma",
    "strata": "stratum",
    "errata": "erratum",
    "agenda": "agendum",
    "bacteria": "bacterium",
    "curricula": "curriculum",
    "memoranda": "memorandum",
    "referenda": "referendum",
    "spectra": "spectrum",
    "aquaria": "aquarium",
    "quora": "quorum",
    "symposia": "symposium",
    "encomia": "encomium",
    "enigmas": "enigma",
    "formulas": "formula",
    "stigmas": "stigma",
    "atlases": "atlas",
    "beaux": "beau",
    "bureaux": "bureau",
    "tableaux": "tableau",
    "chateaux": "chateau",
    "plateaux": "plateau",
    "portmanteaux": "portmanteau",
    "trousseaux": "trousseau",
}

EN_DET_LEMMAS: dict[str, str] = {
    "an": "a",
}

ES_REFLEXIVE_SUFFIXES: list[tuple[str, str]] = [
    ("nos", "se"),
]


def es_reflexive_lemma(word: str) -> str | None:
    lower = word.lower()
    for suffix, replacement in ES_REFLEXIVE_SUFFIXES:
        if lower.endswith(suffix) and len(lower) > len(suffix) + 4:
            return lower[: -len(suffix)] + replacement
    return None


def apply_postprocess_rules(word: str, lang: str, pred_lemma: str, pred_upos: str) -> str:
    word_lower = word.lower()

    if lang == "de":
        if word_lower in DE_CONTRACTIONS:
            return DE_CONTRACTIONS[word_lower]

    elif lang == "en":
        if pred_upos in ("NOUN", "PROPN") and word_lower in EN_IRREGULAR_PLURALS:
            return EN_IRREGULAR_PLURALS[word_lower]
        if pred_upos == "DET" and word_lower in EN_DET_LEMMAS:
            return EN_DET_LEMMAS[word_lower]

    elif lang == "es":
        if pred_upos == "VERB":
            reflexive = es_reflexive_lemma(word_lower)
            if reflexive is not None:
                return reflexive

    return pred_lemma
