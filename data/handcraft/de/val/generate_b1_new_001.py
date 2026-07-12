"""Generate b1_new_001.conllu (de_b1_val_016–100) for German B1 validation."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

START_ID = 16
SENT_ID_PREFIX = "de_b1_val"

# 85 B1 validation sentences (8–15 tokens, mixed B1 structures)
SENTENCES: list[str] = [
    # 016–037: daily life and housing
    "Ich würde gern früher aufstehen, wenn ich nicht so müde wäre.",
    "Die Wohnung, die wir gestern besichtigt haben, gefällt uns sehr.",
    "Ohne einen Termin kann man beim Amt oft lange warten.",
    "Er hat sich entschieden, in eine kleinere Stadt umzuziehen.",
    "Wir müssen die Miete bis zum ersten des Monats überweisen.",
    "Das Zimmer wurde von den Nachbarn gestern Abend sehr laut renoviert.",
    "Ich freue mich darauf, meine neue Kollegin kennenzulernen.",
    "Wenn du Hilfe brauchst, ruf mich bitte einfach an.",
    "Trotz des Regens sind wir wie geplant spazieren gegangen.",
    "Sie hat vergessen, den Müll vor der Abholung rauszubringen.",
    "Der Schlüssel, den du mir geliehen hast, liegt auf dem Tisch.",
    "Wir haben uns über die hohen Nebenkosten beim Vermieter beschwert.",
    "Ich könnte heute nicht kommen, weil ich einen Arzttermin habe.",
    "Das Essen wurde von meiner Mutter mit viel Liebe zubereitet.",
    "Obwohl die Wohnung klein ist, fühlen wir uns hier wohl.",
    "Er möchte, dass wir das Treffen auf nächste Woche verschieben.",
    "Wegen der Baustelle mussten wir einen anderen Weg nehmen.",
    "Ich interessiere mich sehr für nachhaltiges Wohnen in der Stadt.",
    "Die Lampe, mit der ich gestern gearbeitet habe, ist kaputt.",
    "Wenn wir früher losfahren, verpassen wir den Zug nicht.",
    "Sie hat das Paket versehentlich an die falsche Adresse geschickt.",
    "Ohne Internetverbindung kann ich meine Hausaufgaben nicht machen.",
    # 038–058: work and education
    "Der Bericht wurde von dem Team bis Freitag fertiggestellt.",
    "Ich weiß nicht, ob ich die Prüfung beim ersten Versuch bestehe.",
    "Während des Unterrichts hat der Lehrer viele Fragen gestellt.",
    "Er hat sich für den Kurs angemeldet, weil er Deutsch verbessern will.",
    "Die Aufgabe, die du mir geschickt hast, war sehr hilfreich.",
    "Wir sollten die Ergebnisse sorgfältig prüfen, bevor wir sie abgeben.",
    "Sie würde gern im Ausland arbeiten, wenn sie die Chance bekäme.",
    "Das Praktikum wurde von vielen Studierenden als sehr lehrreich bezeichnet.",
    "Ich habe mich gestern mit meinem Chef über mein Gehalt unterhalten.",
    "Obwohl die Prüfung schwer war, habe ich sie gut bestanden.",
    "Der Professor erklärte, dass die Klausur nächsten Monat stattfindet.",
    "Wir müssen das Projekt rechtzeitig abschließen, sonst gibt es Probleme.",
    "Die Präsentation, auf die wir uns vorbereitet haben, lief sehr gut.",
    "Er konnte nicht kommen, weil er an einer wichtigen Besprechung teilnahm.",
    "Ich würde dir gern helfen, wenn ich mehr Zeit hätte.",
    "Die Hausaufgaben wurden von den Schülern pünktlich abgegeben.",
    "Sobald der Kurs zu Ende ist, bewerbe ich mich um eine Stelle.",
    "Sie hat sich sehr über die positive Rückmeldung gefreut.",
    "Wir haben vereinbart, das Meeting auf Dienstag zu verschieben.",
    "Der Kollege, mit dem ich zusammenarbeite, ist sehr zuverlässig.",
    "Wenn du Fragen hast, schreib mir bitte eine kurze Nachricht.",
    # 059–079: travel, health, and services
    "Wegen des Streiks fielen viele Züge in der Region aus.",
    "Ich würde gern nach Italien reisen, wenn ich Urlaub bekomme.",
    "Die Fahrkarten wurden an der Automaten leider nicht ausgegeben.",
    "Er hat vergessen, seinen Reisepass für die Reise einzupacken.",
    "Wir sind trotz der Verspätung sicher am Bahnhof angekommen.",
    "Der Arzt sagte, dass ich mich ein paar Tage ausruhen sollte.",
    "Ich habe einen Termin beim Zahnarzt, weil mich etwas stört.",
    "Die Apotheke, die in der Nähe liegt, hat heute länger geöffnet.",
    "Ohne Rezept darf man dieses Medikament in Deutschland nicht kaufen.",
    "Sie fühlt sich besser, obwohl sie noch leichte Schmerzen hat.",
    "Das Rezept wurde von dem Hausarzt gestern Nachmittag ausgestellt.",
    "Wenn du dich schlecht fühlst, solltest du zu Hause bleiben.",
    "Ich möchte einen Tisch für vier Personen am Samstag reservieren.",
    "Der Kellner empfahl uns ein Gericht, das sehr beliebt ist.",
    "Wir haben die Rechnung geteilt, weil niemand allein zahlen wollte.",
    "Die Reservierung, die wir online gemacht haben, wurde bestätigt.",
    "Er konnte das Restaurant nicht finden, weil die Karte ungenau war.",
    "Ich würde gern weniger Zucker essen, wenn es nicht so schwer wäre.",
    "Das Formular wurde von der Mitarbeiterin freundlich erklärt.",
    "Trotz der langen Wartezeit war der Service am Ende sehr gut.",
    "Sie hat sich beim Arztbesuch über die hohen Kosten beschwert.",
    # 080–100: environment, technology, and relationships
    "Wenn wir weniger Plastik benutzen, schützen wir die Natur.",
    "Die Stadt hat beschlossen, mehr Fahrradwege in den Stadtteilen zu bauen.",
    "Ich interessiere mich sehr für erneuerbare Energien und Klimaschutz.",
    "Das alte Gebäude wurde von der Gemeinde sorgfältig renoviert.",
    "Obwohl das Wetter schlecht war, haben wir den Ausflug gemacht.",
    "Er hat mir erklärt, wie man das neue Programm richtig installiert.",
    "Die Datei, die du mir geschickt hast, lässt sich nicht öffnen.",
    "Wir sollten das Passwort ändern, bevor jemand unbefugten Zugriff erhält.",
    "Ich würde gern weniger Zeit am Handy verbringen, wenn ich könnte.",
    "Das Handy wurde von meinem Bruder versehentlich in Wasser fallen gelassen.",
    "Sie hat sich sehr gefreut, als sie die gute Nachricht bekam.",
    "Wir haben uns gestern Abend über unsere Pläne für die Zukunft unterhalten.",
    "Er entschuldigte sich, weil er zu dem Treffen zu spät gekommen war.",
    "Ich hoffe, dass wir uns bald wieder persönlich treffen können.",
    "Die Freundin, mit der ich seit Jahren befreundet bin, wohnt in Berlin.",
    "Wenn du Zeit hast, könnten wir zusammen einen Kaffee trinken.",
    "Trotz des Streits haben sie sich am Ende wieder versöhnt.",
    "Sie würde gern öfter ihre Familie besuchen, wenn die Fahrt kürzer wäre.",
    "Wir haben vereinbart, uns nächsten Monat wieder zu sehen.",
    "Ich bin dankbar für deine Hilfe, obwohl du selbst sehr beschäftigt bist.",
    "Er hat versprochen, dass er sich bald bei mir melden wird.",
]

assert len(SENTENCES) == 85, f"Expected 85 sentences, got {len(SENTENCES)}"

MODALS = {"müssen", "können", "wollen", "sollen", "dürfen", "mögen", "möchte", "möchten"}
AUX_LEMMAS = {"sein", "haben", "werden"}
AUX_FORMS = {
    "ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen",
    "habe", "hat", "hast", "hatte", "hätte", "hätten", "haben", "habt", "gehabt",
    "wird", "wurde", "werden", "würde", "würden", "geworden",
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
    "Sie": ("Sie", "PRON"),
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
    "viele": ("viel", "DET"),
    "vielen": ("viel", "DET"),
    "viel": ("viel", "DET"),
    "mehr": ("mehr", "DET"),
    "heute": ("heute", "ADV"),
    "gestern": ("gestern", "ADV"),
    "morgen": ("morgen", "ADV"),
    "immer": ("immer", "ADV"),
    "oft": ("oft", "ADV"),
    "sehr": ("sehr", "ADV"),
    "gerne": ("gern", "ADV"),
    "gern": ("gern", "ADV"),
    "hier": ("hier", "ADV"),
    "noch": ("noch", "ADV"),
    "wieder": ("wieder", "ADV"),
    "bald": ("bald", "ADV"),
    "einfach": ("einfach", "ADV"),
    "leider": ("leider", "ADV"),
    "pünktlich": ("pünktlich", "ADV"),
    "sorgfältig": ("sorgfältig", "ADV"),
    "freundlich": ("freundlich", "ADV"),
    "versehentlich": ("versehentlich", "ADV"),
    "rechtzeitig": ("rechtzeitig", "ADV"),
    "nächsten": ("nah", "ADJ"),
    "nächste": ("nah", "ADJ"),
    "kleinere": ("klein", "ADJ"),
    "klein": ("klein", "ADJ"),
    "hohen": ("hoch", "ADJ"),
    "falsche": ("falsch", "ADJ"),
    "falschen": ("falsch", "ADJ"),
    "neue": ("neu", "ADJ"),
    "neuen": ("neu", "ADJ"),
    "alte": ("alt", "ADJ"),
    "alten": ("alt", "ADJ"),
    "gute": ("gut", "ADJ"),
    "guten": ("gut", "ADJ"),
    "gut": ("gut", "ADJ"),
    "leichte": ("leicht", "ADJ"),
    "leichter": ("leicht", "ADJ"),
    "weniger": ("wenig", "ADJ"),
    "erste": ("erst", "ADJ"),
    "ersten": ("erst", "ADJ"),
    "erst": ("erst", "ADV"),
    "ans": ("ans", "ADP"),
    "Hause": ("Haus", "NOUN"),
    "Berlin": ("Berlin", "PROPN"),
    "Italien": ("Italien", "PROPN"),
    "Deutschland": ("Deutschland", "PROPN"),
    "Samstag": ("Samstag", "NOUN"),
    "Dienstag": ("Dienstag", "NOUN"),
    "Freitag": ("Freitag", "NOUN"),
    "Automaten": ("Automat", "NOUN"),
}

SCONJS = {
    "weil", "dass", "obwohl", "wenn", "ob", "da", "als", "wie",
    "während", "seit", "bis", "bevor", "deshalb", "falls", "sobald",
}
CCONJS = {"und", "oder", "aber", "sondern", "als"}
ADPS = {
    "um", "für", "mit", "von", "bei", "nach", "aus", "über", "vor", "unter",
    "durch", "ohne", "gegen", "wegen", "in", "an", "auf", "am", "im", "ins",
    "ab", "zur", "zum", "ans", "trotz",
}
PARTS = {"nicht", "ja", "nein", "zu"}

CONTRACTIONS = {
    "im": ("in", "ADP"),
    "am": ("an", "ADP"),
    "zum": ("zu", "ADP"),
    "zur": ("zu", "ADP"),
    "vom": ("von", "ADP"),
    "ins": ("in", "ADP"),
    "beim": ("bei", "ADP"),
    "ans": ("ans", "ADP"),
}

CONTRACTION_EXPANSIONS = {
    "im": ("in", "dem"),
    "am": ("an", "dem"),
    "zum": ("zu", "dem"),
    "zur": ("zu", "der"),
    "beim": ("bei", "dem"),
    "vom": ("von", "dem"),
    "ins": ("in", "das"),
    "ans": ("an", "das"),
}


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    if form in CONTRACTIONS:
        lemma, upos = CONTRACTIONS[form]

    if form.lower() in {"das", "die", "der", "den", "dem", "des"} and upos == "PRON":
        lemma = "der"
        upos = "PRON"
    elif form in SPECIAL_LEMMAS:
        lemma, upos = SPECIAL_LEMMAS[form]

    lower = form.lower()

    if lower in AUX_FORMS:
        upos = "AUX"
        if lower in {"ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen"}:
            lemma = "sein"
        elif lower in {"habe", "hat", "hast", "hatte", "hätte", "hätten", "haben", "habt", "gehabt"}:
            lemma = "haben"
        else:
            lemma = "werden"
    elif upos == "AUX" and lemma not in AUX_LEMMAS:
        upos = "VERB"

    if upos == "VERB" or lemma in MODALS or lower in MODALS:
        upos = "VERB"
        lemma = lemma.lower()
        if not (lemma.endswith("en") or lemma.endswith("n")):
            if form.lower().endswith("en") or form.lower().endswith("n"):
                lemma = form.lower()

    if upos == "NOUN" and lemma:
        lemma = lemma[0].upper() + lemma[1:]

    if upos == "ADJ":
        lemma = lemma.lower()

    if upos == "PROPN" and lemma:
        lemma = lemma[0].upper() + lemma[1:]

    if upos == "PUNCT":
        lemma = form

    if lower in SCONJS:
        if lower not in {"während", "seit", "bis"} or upos != "ADP":
            upos = "SCONJ"
            lemma = lower
    elif lower in CCONJS:
        upos = "CCONJ"
        lemma = lower
    elif lower in ADPS:
        if not (lower == "zu" and upos == "PART"):
            upos = "ADP"
            lemma = lower
    elif lower in PARTS:
        if lower == "zu" and upos == "PART":
            pass
        else:
            upos = "PART"
            lemma = lower
    elif form == "Sie":
        upos = "PRON"
        lemma = "Sie"
    elif lower in {"ich", "du", "er", "sie", "es", "wir", "man"}:
        upos = "PRON"
        lemma = lower

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
    if form not in CONTRACTION_EXPANSIONS:
        return None
    expansion = CONTRACTION_EXPANSIONS[form]
    if start + len(expansion) > len(words):
        return None
    for i, piece in enumerate(expansion):
        if words[start + i].text.lower() != piece.lower():
            return None
    return start + len(expansion)


def align_tokens(sentence: str, words: list) -> list[tuple[str, str, str]]:
    aligned: list[tuple[str, str, str]] = []
    text_tokens = tokenize_text(sentence)
    wi = 0

    for form in text_tokens:
        if form in ".,;:!?":
            aligned.append((form, form, "PUNCT"))
            if wi < len(words) and words[wi].text == form:
                wi += 1
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

        head = words[wi]
        lemma, upos = normalize_token(form, head.upos or "X", head.lemma or form)
        aligned.append((form, lemma, upos))
        wi += 1

    return aligned


def sentence_to_conllu(sent_id: str, sent: str, nlp) -> list[str]:
    lines: list[str] = [f"# sent_id = {sent_id}", f"# text = {sent}"]
    doc = nlp(sent)
    words = stanza_words_flat(doc)
    aligned = align_tokens(sent, words)
    for token_counter, (form, lemma, upos) in enumerate(aligned, start=1):
        cols = [
            str(token_counter),
            form,
            lemma,
            upos,
            "_", "_", "_", "_", "_", "_",
        ]
        lines.append("\t".join(cols))
    lines.append("")
    return lines


def main() -> None:
    import stanza

    print("Loading Stanza...")
    nlp = stanza.Pipeline(
        lang="de",
        processors="tokenize,mwt,pos,lemma",
        use_gpu=False,
        verbose=False,
    )

    target_path = project_root / "data/handcraft/de/val/b1_new_001.conllu"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    all_lines: list[str] = []
    for idx, sent in enumerate(SENTENCES):
        sent_id = f"{SENT_ID_PREFIX}_{START_ID + idx:03d}"
        all_lines.extend(sentence_to_conllu(sent_id, sent, nlp))

    conllu_text = "\n".join(all_lines) + "\n"
    target_path.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {len(SENTENCES)} sentences to {target_path}")

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