"""Generate a1_new_008.conllu — de_a1_train_851 through de_a1_train_900."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

START_ID = 851
BATCH_SIZE = 25

# 2 batches × 25 = 50 A1 sentences (3–8 tokens, present tense)
SENTENCE_BATCHES: list[list[str]] = [
    # 851–875: Shopping & supermarket
    [
        "Ich stehe früh auf.",
        "Wir frühstücken um acht.",
        "Er packt die Brotdose ein.",
        "Sie kauft frisches Obst.",
        "Ich brauche zwei Liter Milch.",
        "Das Brot kostet zwei Euro.",
        "Wir gehen zum Supermarkt.",
        "Er sucht die Tomaten.",
        "Sie nimmt einen Korb.",
        "Ich finde den Gang.",
        "Wir stehen an der Kasse.",
        "Er bezahlt mit Bargeld.",
        "Sie packt alles ein.",
        "Das Obst ist süß.",
        "Ich kaufe drei Bananen.",
        "Er trägt die Tüte.",
        "Sie hat den Beleg.",
        "Wir bringen den Einkauf heim.",
        "Das Gemüse ist billig.",
        "Ich nehme eine Flasche Wasser.",
        "Er kauft Schokolade ein.",
        "Sie wiegt die Äpfel ab.",
        "Wir haben viel eingekauft.",
        "Der Markt öffnet früh.",
        "Ich hole Brot vom Bäcker.",
    ],
    # 876–900: Transport, hobbies & weekend
    [
        "Der Bus fährt langsam.",
        "Wir steigen am Bahnhof ein.",
        "Ich fahre mit dem Zug.",
        "Das Taxi wartet dort.",
        "Er kauft eine Fahrkarte.",
        "Die Straßenbahn kommt gleich.",
        "Wir fahren nach Köln.",
        "Es regnet sehr stark.",
        "Der Himmel ist grau.",
        "Es ist windig draußen.",
        "Ich spiele gern Schach.",
        "Sie fotografiert den Hund.",
        "Wir kochen heute Nudeln.",
        "Er schaut einen Film an.",
        "Sie tanzt mit Freunden.",
        "Ich schreibe eine Postkarte.",
        "Wir besuchen die Großeltern.",
        "Das Schwimmbad ist offen.",
        "Er spielt am Samstag Fußball.",
        "Sie liest am Abend gern.",
        "Wir grillen im Garten.",
        "Das Picknick macht Spaß.",
        "Ich lade dich ein.",
        "Er fährt ans Meer.",
        "Wir machen viele Fotos.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 50, f"Expected 50 sentences, got {len(SENTENCES)}"

MODALS = {"müssen", "können", "wollen", "sollen", "dürfen", "mögen", "möchte", "möchtest", "möchten"}
AUX_FORMS = {
    "ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen",
    "habe", "hat", "hatte", "hätte", "hätten", "haben", "habt", "gehabt", "hast",
    "wird", "wurde", "werden", "würde", "würden", "geworden", "wirst",
    "kann", "kannst", "können", "konnte", "muss", "müssen", "will", "wollen",
    "soll", "sollen", "darf", "dürfen", "möchte", "möchtest", "möchten", "mag",
}
AUX_LEMMAS = {"sein", "haben", "werden", "können", "müssen", "wollen", "sollen", "dürfen", "mögen"}

_CONTRACTION_BASE = {
    "im": ("in", ("in", "dem")),
    "am": ("an", ("an", "dem")),
    "zum": ("zu", ("zu", "dem")),
    "zur": ("zu", ("zu", "der")),
    "vom": ("von", ("von", "dem")),
    "ins": ("in", ("in", "das")),
    "beim": ("bei", ("bei", "dem")),
    "ans": ("an", ("an", "das")),
}

CONTRACTIONS: dict[str, tuple[str, str]] = {}
CONTRACTION_EXPANSIONS: dict[str, tuple[str, ...]] = {}
for _form, (_lemma, _expansion) in _CONTRACTION_BASE.items():
    CONTRACTIONS[_form] = (_lemma, "ADP")
    CONTRACTION_EXPANSIONS[_form] = _expansion
    _cap = _form[0].upper() + _form[1:]
    CONTRACTIONS[_cap] = (_lemma, "ADP")
    CONTRACTION_EXPANSIONS[_cap] = _expansion

MONTHS_SEASONS = {
    "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
    "September", "Oktober", "November", "Dezember",
    "Frühling", "Sommer", "Herbst", "Winter",
}

HYPHENATED_NOUNS = {
    "T-Shirt": "T-Shirt",
    "E-Mail": "E-Mail",
}

SPECIAL_LEMMAS: dict[str, tuple[str, str]] = {
    "mich": ("ich", "PRON"),
    "dich": ("du", "PRON"),
    "sich": ("sich", "PRON"),
    "uns": ("wir", "PRON"),
    "euch": ("ihr", "PRON"),
    "mir": ("ich", "PRON"),
    "dir": ("du", "PRON"),
    "ihm": ("er", "PRON"),
    "ihnen": ("sie", "PRON"),
    "Ihnen": ("Sie", "PRON"),
    "Sie": ("sie", "PRON"),
    "mein": ("mein", "DET"),
    "meine": ("mein", "DET"),
    "meinen": ("mein", "DET"),
    "meinem": ("mein", "DET"),
    "meiner": ("mein", "DET"),
    "meines": ("mein", "DET"),
    "dein": ("dein", "DET"),
    "deine": ("dein", "DET"),
    "deinen": ("dein", "DET"),
    "deinem": ("dein", "DET"),
    "deiner": ("dein", "DET"),
    "deines": ("dein", "DET"),
    "sein": ("sein", "DET"),
    "seine": ("sein", "DET"),
    "seinen": ("sein", "DET"),
    "seinem": ("sein", "DET"),
    "seiner": ("sein", "DET"),
    "seines": ("sein", "DET"),
    "unser": ("unser", "DET"),
    "unsere": ("unser", "DET"),
    "unseren": ("unser", "DET"),
    "unserem": ("unser", "DET"),
    "unserer": ("unser", "DET"),
    "unseres": ("unser", "DET"),
    "ihr": ("ihr", "DET"),
    "ihre": ("ihr", "DET"),
    "ihren": ("ihr", "DET"),
    "ihrem": ("ihr", "DET"),
    "ihrer": ("ihr", "DET"),
    "ihres": ("ihr", "DET"),
    "der": ("der", "DET"),
    "die": ("der", "DET"),
    "das": ("der", "DET"),
    "den": ("der", "DET"),
    "dem": ("der", "DET"),
    "des": ("der", "DET"),
    "ein": ("ein", "DET"),
    "eine": ("ein", "DET"),
    "einen": ("ein", "DET"),
    "einem": ("ein", "DET"),
    "einer": ("ein", "DET"),
    "eines": ("ein", "DET"),
    "zwei": ("zwei", "NUM"),
    "grünes": ("grün", "ADJ"),
    "grüne": ("grün", "ADJ"),
    "braune": ("braun", "ADJ"),
    "blaue": ("blau", "ADJ"),
    "kleine": ("klein", "ADJ"),
    "lange": ("lang", "ADJ"),
    "großen": ("groß", "ADJ"),
    "große": ("groß", "ADJ"),
    "kurze": ("kurz", "ADJ"),
    "neue": ("neu", "ADJ"),
    "neuen": ("neu", "ADJ"),
    "schwarz": ("schwarz", "ADJ"),
    "grau": ("grau", "ADJ"),
    "warm": ("warm", "ADJ"),
    "kalt": ("kalt", "ADJ"),
    "bunt": ("bunt", "ADJ"),
    "schön": ("schön", "ADJ"),
    "riesig": ("riesig", "ADJ"),
    "wild": ("wild", "ADJ"),
    "schlau": ("schlau", "ADJ"),
    "langsam": ("langsam", "ADJ"),
    "süß": ("süß", "ADJ"),
    "festlich": ("festlich", "ADJ"),
    "regnerisch": ("regnerisch", "ADJ"),
    "dunkel": ("dunkel", "ADJ"),
    "nett": ("nett", "ADJ"),
    "glücklich": ("glücklich", "ADJ"),
    "traurig": ("traurig", "ADJ"),
    "müde": ("müde", "ADJ"),
    "aufgeregt": ("aufgeregt", "ADJ"),
    "wütend": ("wütend", "ADJ"),
    "nervös": ("nervös", "ADJ"),
    "stolz": ("stolz", "ADJ"),
    "froh": ("froh", "ADJ"),
    "überrascht": ("überrascht", "ADJ"),
    "gelangweilt": ("gelangweilt", "ADJ"),
    "ängstlich": ("ängstlich", "ADJ"),
    "zufrieden": ("zufrieden", "ADJ"),
    "eifersüchtig": ("eifersüchtig", "ADJ"),
    "hungrig": ("hungrig", "ADJ"),
    "durstig": ("durstig", "ADJ"),
    "krank": ("krank", "ADJ"),
    "gut": ("gut", "ADJ"),
    "laut": ("laut", "ADJ"),
    "früh": ("früh", "ADV"),
    "hoch": ("hoch", "ADV"),
    "weg": ("weg", "ADV"),
    "gern": ("gern", "ADV"),
    "gerne": ("gern", "ADV"),
    "heute": ("heute", "ADV"),
    "jetzt": ("jetzt", "ADV"),
    "morgens": ("morgens", "ADV"),
    "nachts": ("nachts", "ADV"),
    "viel": ("viel", "ADV"),
    "oft": ("oft", "ADV"),
    "sehr": ("sehr", "ADV"),
    "hart": ("hart", "ADV"),
    "schnell": ("schnell", "ADV"),
    "laut": ("laut", "ADV"),
    "draußen": ("draußen", "ADV"),
    "heim": ("heim", "ADV"),
    "Wo": ("wo", "ADV"),
    "wo": ("wo", "ADV"),
    "Wann": ("wann", "ADV"),
    "wann": ("wann", "ADV"),
    "Wie": ("wie", "ADV"),
    "wie": ("wie", "ADV"),
    "Warum": ("warum", "ADV"),
    "warum": ("warum", "ADV"),
    "Wer": ("wer", "PRON"),
    "wer": ("wer", "PRON"),
    "Was": ("was", "PRON"),
    "was": ("was", "PRON"),
    "ein": ("ein", "DET"),
    "bisschen": ("bisschen", "ADV"),
    "letzte": ("letzt", "ADJ"),
    "siebte": ("siebt", "ADJ"),
    "achtundzwanzig": ("achtundzwanzig", "NUM"),
    "Vögel": ("Vogel", "NOUN"),
    "Blumen": ("Blume", "NOUN"),
    "Blätter": ("Blatt", "NOUN"),
    "Äpfel": ("Apfel", "NOUN"),
    "Mäntel": ("Mantel", "NOUN"),
    "Hände": ("Hand", "NOUN"),
    "Füße": ("Fuß", "NOUN"),
    "Haare": ("Haar", "NOUN"),
    "Zähne": ("Zahn", "NOUN"),
    "Finger": ("Finger", "NOUN"),
    "Nägel": ("Nagel", "NOUN"),
    "Stiefel": ("Stiefel", "NOUN"),
    "Hosen": ("Hose", "NOUN"),
    "Schuhe": ("Schuh", "NOUN"),
    "Handschuhe": ("Handschuh", "NOUN"),
    "Tage": ("Tag", "NOUN"),
    "Bücher": ("Buch", "NOUN"),
    "Tiere": ("Tier", "NOUN"),
}

VERB_LEMMA_MAP = {
    "sitzt": "sitzen",
    "frisst": "fressen",
    "läuft": "laufen",
    "sehe": "sehen",
    "schwimmt": "schwimmen",
    "schläft": "schlafen",
    "summt": "summen",
    "kräht": "krähen",
    "rennt": "rennen",
    "hüpft": "hüpfen",
    "füttern": "füttern",
    "streichle": "streicheln",
    "tut": "tun",
    "wäscht": "waschen",
    "putze": "putzen",
    "zeigt": "zeigen",
    "kämme": "kämmen",
    "öffne": "öffnen",
    "schüttelt": "schütteln",
    "hebt": "heben",
    "hebe": "heben",
    "schneidet": "schneiden",
    "spüre": "spüren",
    "trage": "tragen",
    "trägt": "tragen",
    "zieht": "ziehen",
    "ziehe": "ziehen",
    "suche": "suchen",
    "kauft": "kaufen",
    "kaufe": "kaufen",
    "binde": "binden",
    "bindet": "binden",
    "wäscht": "waschen",
    "brauche": "brauchen",
    "schwimmen": "schwimmen",
    "fallen": "fallen",
    "blühen": "blühen",
    "wird": "werden",
    "pflücken": "pflücken",
    "schneit": "schneien",
    "singen": "singen",
    "grillen": "grillen",
    "mache": "machen",
    "machen": "machen",
    "lese": "lesen",
    "bleiben": "bleiben",
    "fahren": "fahren",
    "beginnt": "beginnen",
    "gehe": "gehen",
    "arbeitet": "arbeiten",
    "malt": "malen",
    "fliegt": "fliegen",
    "fühlt": "fühlen",
    "machst": "machen",
    "kommst": "kommen",
    "lachst": "lachen",
    "heißt": "heißen",
    "trägst": "tragen",
    "isst": "essen",
    "gehst": "gehen",
    "findest": "finden",
    "möchtest": "mögen",
    "möchte": "mögen",
    "mag": "mögen",
    "hast": "haben",
    "habe": "haben",
    "hat": "haben",
    "haben": "haben",
    "ist": "sein",
    "sind": "sein",
    "bin": "sein",
    "bist": "sein",
    "werden": "werden",
    "kann": "können",
    "kannst": "können",
    "wohnt": "wohnen",
    "wohnst": "wohnen",
}

NOUN_LEMMA_MAP = {
    "Kaninchen": "Kaninchen",
    "Eichhörnchen": "Eichhörnchen",
    "Schildkröte": "Schildkröte",
    "Lieblingstier": "Lieblingstier",
    "Feuerwehrmann": "Feuerwehrmann",
    "Busfahrer": "Busfahrer",
    "Krankenschwester": "Krankenschwester",
    "Programmierer": "Programmierer",
    "Mechaniker": "Mechaniker",
    "Architektin": "Architektin",
    "Verkäuferin": "Verkäuferin",
    "Lehrerin": "Lehrerin",
    "Ärztin": "Ärztin",
    "Friseurin": "Friseurin",
    "Sekretärin": "Sekretärin",
    "Sängerin": "Sängerin",
    "Köchin": "Köchin",
    "Tänzerin": "Tänzerin",
    "Januar": "Januar",
    "Februar": "Februar",
    "März": "März",
    "April": "April",
    "Mai": "Mai",
    "Juni": "Juni",
    "Juli": "Juli",
    "August": "August",
    "September": "September",
    "Oktober": "Oktober",
    "November": "November",
    "Dezember": "Dezember",
    "Frühling": "Frühling",
    "Sommer": "Sommer",
    "Herbst": "Herbst",
    "Winter": "Winter",
    "Hamster": "Hamster",
    "Giraffe": "Giraffe",
    "Elefant": "Elefant",
    "Pullover": "Pullover",
    "Krawatte": "Krawatte",
    "Handschuh": "Handschuh",
    "Geburtstag": "Geburtstag",
    "Unterricht": "Unterricht",
    "Prüfung": "Prüfung",
    "Geschenke": "Geschenk",
    "Schmerz": "Schmerz",
    "Gesicht": "Gesicht",
    "Rücken": "Rücken",
    "Bauch": "Bauch",
    "Jeans": "Jeans",
    "Weste": "Weste",
    "Bluse": "Bluse",
    "Mütze": "Mütze",
    "Gürtel": "Gürtel",
    "Schal": "Schal",
}


def _strip_trailing_punct(text: str) -> str:
    return re.sub(r"[.,;:!?]+$", "", text)


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    if lemma and _strip_trailing_punct(lemma) != lemma and _strip_trailing_punct(form) == form:
        lemma = _strip_trailing_punct(lemma)

    if form in CONTRACTIONS:
        lemma, upos = CONTRACTIONS[form]
    elif form in SPECIAL_LEMMAS:
        lemma, upos = SPECIAL_LEMMAS[form]

    if form in MONTHS_SEASONS:
        upos = "NOUN"
        lemma = form

    if form.lower() in {"das", "die", "der", "den", "dem", "des"} and upos == "PRON":
        lemma = "der"
        upos = "DET"

    if upos == "NOUN":
        if form in NOUN_LEMMA_MAP:
            lemma = NOUN_LEMMA_MAP[form]
        elif lemma:
            lemma = lemma[0].upper() + lemma[1:]

    lower = form.lower()

    if lower in AUX_FORMS:
        upos = "AUX"
        if lower in {"ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen"}:
            lemma = "sein"
        elif lower in {"habe", "hat", "hatte", "hätte", "hätten", "haben", "habt", "gehabt", "hast"}:
            lemma = "haben"
        elif lower in {"wird", "wurde", "werden", "würde", "würden", "geworden", "wirst"}:
            lemma = "werden"
        elif lower in {"kann", "kannst", "können", "konnte"}:
            lemma = "können"
        elif lower in {"muss", "müssen"}:
            lemma = "müssen"
        elif lower in {"will", "wollen"}:
            lemma = "wollen"
        elif lower in {"soll", "sollen"}:
            lemma = "sollen"
        elif lower in {"darf", "dürfen"}:
            lemma = "dürfen"
        elif lower in {"möchte", "möchtest", "möchten", "mag"}:
            lemma = "mögen"
    elif upos in {"VERB", "AUX"}:
        if lower in MODALS or lemma.lower() in MODALS:
            upos = "AUX"
            if lower in {"möchte", "möchtest", "möchten", "mag"}:
                lemma = "mögen"
            elif lower in {"kann", "kannst", "können"}:
                lemma = "können"
        else:
            upos = "VERB"
            lemma = lemma.lower()
            if form in VERB_LEMMA_MAP:
                lemma = VERB_LEMMA_MAP[form]
            elif lemma in VERB_LEMMA_MAP:
                lemma = VERB_LEMMA_MAP[lemma]

    if upos == "ADJ":
        lemma = lemma.lower()
        if form in SPECIAL_LEMMAS:
            lemma, _ = SPECIAL_LEMMAS[form]

    if upos == "PROPN" and lemma:
        lemma = lemma[0].upper() + lemma[1:]

    if upos == "PUNCT":
        lemma = form

    if lower in {"weil", "dass", "obwohl", "wenn", "ob", "als"}:
        upos = "SCONJ"
        lemma = lower
    elif lower in {"und", "oder", "aber", "sondern"}:
        upos = "CCONJ"
        lemma = lower
    elif lower in {
        "um", "für", "mit", "von", "bei", "nach", "aus", "über", "vor", "unter",
        "durch", "ohne", "gegen", "wegen", "in", "an", "auf", "zu",
    }:
        if not (lower == "zu" and upos == "PART"):
            upos = "ADP"
            lemma = lower
    elif lower in {"nicht", "ja", "nein"}:
        upos = "PART"
        lemma = lower
    elif lower in {"ich", "du", "er", "sie", "es", "wir", "ihr"}:
        upos = "PRON"
        lemma = lower

    if form == "Ihnen":
        upos = "PRON"
        lemma = "Sie"
    elif form == "Sie":
        upos = "PRON"
        lemma = "sie"

    return lemma, upos


def tokenize_text(sentence: str) -> list[str]:
    tokens: list[str] = []
    for word in sentence.split():
        match = re.match(r"^(.+?)([.,;:!?]+)$", word)
        if match:
            tokens.append(match.group(1))
            tokens.extend(list(match.group(2)))
        else:
            tokens.append(word)
    return tokens


def stanza_words_flat(doc) -> list:
    words = []
    for stanza_sent in doc.sentences:
        for word in stanza_sent.words:
            if isinstance(word.id, int):
                words.append(word)
    return words


def match_expansion(form: str, words: list, start: int) -> int | None:
    key = form if form in CONTRACTION_EXPANSIONS else form.lower()
    if key not in CONTRACTION_EXPANSIONS:
        return None
    expansion = CONTRACTION_EXPANSIONS[key]
    if start + len(expansion) > len(words):
        return None
    for i, piece in enumerate(expansion):
        if _strip_trailing_punct(words[start + i].text).lower() != piece.lower():
            return None
    return start + len(expansion)


def match_hyphen(form: str, words: list, start: int) -> int | None:
    if "-" not in form or start >= len(words):
        return None
    built = ""
    idx = start
    while idx < len(words) and len(built) < len(form):
        built += words[idx].text
        idx += 1
        if built == form:
            return idx
    return None


def align_tokens(sentence: str, words: list) -> list[tuple[str, str, str]]:
    aligned: list[tuple[str, str, str]] = []
    text_tokens = tokenize_text(sentence)
    wi = 0

    for form in text_tokens:
        if form in ".,;:!?":
            aligned.append((form, form, "PUNCT"))
            continue

        if wi >= len(words):
            lemma, upos = normalize_token(form, "X", form)
            aligned.append((form, lemma, upos))
            continue

        end = match_expansion(form, words, wi)
        if end is not None:
            head = words[wi]
            lemma, upos = normalize_token(form, head.upos or "X", head.lemma or form)
            aligned.append((form, lemma, upos))
            wi = end
            continue

        end = match_hyphen(form, words, wi)
        if end is not None:
            head = words[wi]
            upos = head.upos or "NOUN"
            if form in HYPHENATED_NOUNS:
                upos = "NOUN"
                lemma = HYPHENATED_NOUNS[form]
            else:
                lemma, upos = normalize_token(form, upos, head.lemma or form)
            aligned.append((form, lemma, upos))
            wi = end
            continue

        head = words[wi]
        stanza_form = _strip_trailing_punct(head.text)
        stanza_lemma = head.lemma or stanza_form
        if stanza_form.lower() == form.lower() or form in MONTHS_SEASONS:
            lemma, upos = normalize_token(form, head.upos or "X", stanza_lemma)
        else:
            lemma, upos = normalize_token(form, "X", form)
        aligned.append((form, lemma, upos))
        wi += 1

    return aligned


SEPARABLE_PARTICLES = frozenset({"auf", "ab", "an", "ein", "aus", "mit", "weg", "heim"})
MANNER_ADVS = frozenset({"langsam", "schnell", "laut", "gern", "gerne"})


def apply_manual_fixes(
    sentence: str,
    aligned: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Apply project-specific lemma/UPOS overrides after Stanza alignment."""
    fixed = list(aligned)
    content = [(i, form, lemma, upos) for i, (form, lemma, upos) in enumerate(fixed) if upos != "PUNCT"]

    if content:
        last_idx, last_form, _, _ = content[-1]
        if last_form in SEPARABLE_PARTICLES:
            fixed[last_idx] = (last_form, last_form, "ADV")

    for i, form, lemma, upos in content:
        if form == "Euro":
            fixed[i] = (form, "Euro", "NOUN")
        elif form in MANNER_ADVS and i > 0:
            prev_form = fixed[i - 1][0].lower()
            if prev_form.endswith(("t", "en", "st")) and upos == "ADJ":
                fixed[i] = (form, form, "ADV")

    return fixed


def main() -> None:
    import stanza

    print("Loading Stanza...")
    nlp = stanza.Pipeline(
        lang="de",
        processors="tokenize,mwt,pos,lemma",
        use_gpu=False,
        verbose=False,
    )

    target_path = project_root / "data/handcraft/de/train/a1_new_008.conllu"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    output_lines: list[str] = []
    global_idx = 0

    for batch_num, batch in enumerate(SENTENCE_BATCHES):
        assert len(batch) == BATCH_SIZE, f"Batch {batch_num} has {len(batch)} sentences"
        print(
            f"Processing batch {batch_num + 1}/2 "
            f"(sentences {START_ID + global_idx}–{START_ID + global_idx + BATCH_SIZE - 1})"
        )
        for sent in batch:
            sent_id = f"de_a1_train_{START_ID + global_idx}"
            doc = nlp(sent)
            words = stanza_words_flat(doc)

            output_lines.append(f"# sent_id = {sent_id}")
            output_lines.append(f"# text = {sent}")

            token_counter = 1
            for form, lemma, upos in apply_manual_fixes(sent, align_tokens(sent, words)):
                cols = [
                    str(token_counter),
                    form,
                    lemma,
                    upos,
                    "_",
                    "_",
                    "_",
                    "_",
                    "_",
                    "_",
                ]
                output_lines.append("\t".join(cols))
                token_counter += 1

            output_lines.append("")
            global_idx += 1

    conllu_text = "\n".join(output_lines) + "\n"
    target_path.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {global_idx} sentences to {target_path}")

    validation_res = validate_text(conllu_text)
    print(f"Validate: count={validation_res.sentence_count}, passed={validation_res.passed}")
    if not validation_res.passed:
        for err in validation_res.errors:
            print(f"  {err}")
        sys.exit(1)

    lemma_res = check_text(conllu_text, lang="de")
    print(f"Lemma check: passed={lemma_res.passed}")
    if not lemma_res.passed:
        for err in lemma_res.errors:
            print(f"  {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()