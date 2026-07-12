"""Generate b2_new_001.conllu: 85 B2 German validation sentences (de_b2_val_016–100)."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.duplicate_detector import check_cross_file, check_text as check_dup_text
from lemmatizer.data.lemma_checker import check_text

# 6 batches: 15+15+15+15+15+10 = 85
BATCHES: list[list[str]] = [
    # Batch 1 (016–030): Education
    [
        "Die Universität hat beschlossen, die Zulassungskriterien ab dem nächsten Semester zu verschärfen.",
        "Obwohl die Prüfung anspruchsvoll war, haben die meisten Studierenden bestanden.",
        "Es wird erwartet, dass die Nachfrage nach Sprachkursen in den kommenden Jahren weiter steigen werde.",
        "Der Rektor betonte, die Hochschule werde weiterhin auf Exzellenz in Forschung und Lehre setzen.",
        "Die Förderung begabter Schüler erfordert ein abgestimmtes Konzept von Schule, Elternhaus und Politik.",
        "Obschon die Digitalisierung Fortschritte bringt, bleibt der direkte Austausch im Unterricht unverzichtbar.",
        "Es sei wichtig, dass alle Kinder unabhängig von ihrer Herkunft gleiche Bildungschancen erhalten.",
        "Die Reform des Schulsystems wurde von Lehrerverbänden grundsätzlich begrüßt.",
        "Würde man mehr in die Ausbildung von Lehrkräften investieren, könnte die Qualität des Unterrichts steigen.",
        "Die Studie, deren Ergebnisse im Frühjahr veröffentlicht wurden, deckt erhebliche regionale Unterschiede auf.",
        "Es mangelt nicht an guten Ideen, sondern an der finanziellen Ausstattung der Bildungseinrichtungen.",
        "Hätte die Schule früher interveniert, wäre der Leistungsabstand möglicherweise geringer geworden.",
        "Die Einführung bilingualer Unterrichtsangebote soll die Internationalisierung der Campusstandorte voranbringen.",
        "Obwohl die Anmeldungen zunahmen, reichten die Plätze im Masterstudiengang nicht aus.",
        "Es wird davon ausgegangen, dass lebenslanges Lernen in der Arbeitswelt der Zukunft unverzichtbar sein werde.",
    ],
    # Batch 2 (031–045): Health
    [
        "Die Vorsorgeuntersuchung soll helfen, Krankheiten in einem frühen Stadium zu erkennen.",
        "Obwohl die Behandlung erfolgreich verlief, benötigt der Patient noch eine längere Rehabilitationsphase.",
        "Es wird befürchtet, dass die Wartezeiten in den Kliniken in den nächsten Monaten weiter zunehmen werden.",
        "Der Arzt erklärte, die Symptome deuteten auf eine chronische Entzündung der Gelenke hin.",
        "Die Bekämpfung des Rauchens erfordert konsequente Präventionsmaßnahmen und öffentliche Aufklärungskampagnen.",
        "Obschon die Impfquote gestiegen ist, bleibt der Schutz bestimmter Risikogruppen ein zentrales Anliegen.",
        "Es sei ratsam, bei anhaltenden Beschwerden umgehend einen Facharzt aufzusuchen.",
        "Die Patienten, deren Akten unvollständig waren, mussten die Untersuchung verschieben.",
        "Würde man mehr in die Pflege investieren, könnte die Belastung des Personals spürbar sinken.",
        "Die Entwicklung neuer Therapien wird von der medizinischen Fachwelt mit großer Erwartung verfolgt.",
        "Hätte man die Hygienevorschriften strikter befolgt, wäre die Ansteckungsrate wahrscheinlich niedriger ausgefallen.",
        "Es wird angenommen, dass die Lebenserwartung in den Industrieländern weiter ansteigen werde.",
        "Die Versorgung ländlicher Regionen mit medizinischen Dienstleistungen bleibt eine drängende Herausforderung.",
        "Obwohl die Studie vielversprechende Ergebnisse lieferte, mahnten Experten zu weiteren Langzeitbeobachtungen.",
        "Die Krankenkasse kündigte an, die Kostenübernahme für innovative Behandlungsmethoden zu prüfen.",
    ],
    # Batch 3 (046–060): Culture & Media
    [
        "Die Ausstellung, auf die sich der Katalog bezieht, wird bis Ende des Monats geöffnet sein.",
        "Obwohl das Buch kontrovers diskutiert wurde, erreichte es die Spitze der Bestsellerlisten.",
        "Es wird erwartet, dass das neue Filmförderprogramm junge Talente gezielt unterstützen werde.",
        "Der Kritiker betonte, das Werk unterscheide sich deutlich von den bisherigen Produktionen des Regisseurs.",
        "Die Erhaltung historischer Bauwerke erfordert Fachwissen, Geduld und langfristige finanzielle Planung.",
        "Obschon das Publikum begeistert applaudierte, fielen die ersten Kritiken gemischt aus.",
        "Es steht zu hoffen, dass die Kulturveranstaltungen auch in schwierigen Zeiten weiter stattfinden können.",
        "Die Verbreitung von Streamingdiensten hat die traditionellen Fernsehprogramme grundlegend verändert.",
        "Würde man mehr in den öffentlich-rechtlichen Rundfunk investieren, könnte die Programmvielfalt zunehmen.",
        "Die Inszenierung des Klassikers wurde von Theaterliebhabern und Fachkritikern gleichermaßen gelobt.",
        "Hätte der Verlag früher gehandelt, wäre die zweite Auflage bereits im Frühjahr erschienen.",
        "Es mangelt nicht an kreativen Impulsen, sondern an tragfähigen Finanzierungsmodellen für die Kultur.",
        "Obwohl die Besucherzahlen sanken, hielt das Museum an seinem Bildungsauftrag fest.",
        "Die Übersetzung des Romans gelang es, die sprachliche Feinheit des Originals weitgehend zu bewahren.",
        "Es wird davon abgeraten, die kulturelle Vielfalt einer Gesellschaft auf reine Marktgesetze zu reduzieren.",
    ],
    # Batch 4 (061–075): Law & Justice
    [
        "Die Anklage behauptet, der Beschuldigte habe gegen geltende Compliance-Vorschriften verstoßen.",
        "Obwohl die Beweislage dünn ist, hält die Staatsanwaltschaft an der Eröffnung des Verfahrens fest.",
        "Es wird erwartet, dass das Gericht in der kommenden Woche ein Urteil sprechen werde.",
        "Der Anwalt erklärte, seine Mandantschaft werde alle Vorwürfe entschieden zurückweisen.",
        "Die Stärkung des Verbraucherschutzes erfordert klare gesetzliche Regelungen und wirksame Kontrollmechanismen.",
        "Obschon die Revision eingelegt wurde, bleibt das Urteil bis auf weiteres rechtskräftig.",
        "Es sei notwendig, dass jedem Angeklagten ein fairer und unvoreingenommener Prozess garantiert werde.",
        "Die Zeugin, auf deren Aussage die Anklage maßgeblich beruht, erschien nicht zum Termin.",
        "Würde man die Strafgesetze reformieren, könnte die Justiz effizienter arbeiten.",
        "Die Durchsetzung internationaler Abkommen wird durch unterschiedliche Rechtstraditionen oft erschwert.",
        "Hätte die Polizei schneller reagiert, wäre der Schaden für die Betroffenen geringer ausgefallen.",
        "Es wird davon ausgegangen, dass die neue Datenschutzverordnung weitreichende Folgen für Unternehmen haben werde.",
        "Die Kläger fordern, die Verantwortlichen sollen für den entstandenen Schaden haftbar gemacht werden.",
        "Obwohl die Verhandlung langwierig war, einigten sich beide Parteien schließlich außergerichtlich.",
        "Die Einhaltung grundlegender Menschenrechte darf in keinem Rechtsstaat zur Disposition gestellt werden.",
    ],
    # Batch 5 (076–090): Work & Career
    [
        "Die Personalabteilung hat angekündigt, flexible Arbeitsmodelle künftig stärker zu fördern.",
        "Obwohl das Unternehmen Gewinne erzielte, wurden keine zusätzlichen Stellen geschaffen.",
        "Es wird befürchtet, dass die Automatisierung viele mittelqualifizierte Berufe obsolet machen werde.",
        "Der Betriebsrat betonte, die Arbeitnehmer seien auf die Mitbestimmung bei wichtigen Entscheidungen angewiesen.",
        "Die Sicherung qualifizierter Fachkräfte erfordert attraktive Arbeitsbedingungen und gezielte Weiterbildungsangebote.",
        "Obschon die Bewerber zahlreich waren, fand man nur wenige Kandidaten mit der erforderlichen Erfahrung.",
        "Es ist sinnvoll, vor Vertragsabschluss die genauen Konditionen schriftlich festhalten zu lassen.",
        "Die Mitarbeiter, deren Leistungen überdurchschnittlich waren, erhielten eine außerordentliche Prämie.",
        "Würde man Homeoffice dauerhaft ermöglichen, könnte sich die Work-Life-Balance vieler Beschäftigter verbessern.",
        "Die Umstrukturierung des Konzerns führte zu erheblichen Verunsicherungen in den Belegschaften.",
        "Hätte die Geschäftsführung früher kommuniziert, wäre der Streik möglicherweise vermeidbar gewesen.",
        "Es wird erwartet, dass die Nachfrage nach IT-Spezialisten in den nächsten Jahren weiter steigen werde.",
        "Obwohl die Tarifverhandlungen festgefahren waren, erzielten beide Seiten schließlich einen Kompromiss.",
        "Die Förderung von Unternehmergeist soll junge Menschen ermutigen, eigene Geschäftsideen zu verwirklichen.",
        "Es mangelt nicht an Arbeitswilligen, sondern an ausreichend bezahlbaren Wohnmöglichkeiten in der Region.",
    ],
    # Batch 6 (091–100): Environment & Urban
    [
        "Die Stadtverwaltung plant, bis 2030 den Anteil des öffentlichen Nahverkehrs deutlich zu erhöhen.",
        "Obwohl die Bürgerinitiative protestierte, wurde das Bauprojekt im Stadtrat mehrheitlich genehmigt.",
        "Es wird angenommen, dass urbane Grünflächen das Mikroklima in dicht bebauten Vierteln spürbar verbessern.",
        "Der Architekt erklärte, das Gebäude solle höchsten energetischen Standards und ökologischen Ansprüchen genügen.",
        "Die Sanierung der Wasserversorgung erfordert umfangreiche Investitionen und eine langfristige Ingenieurplanung.",
        "Obschon die Lärmbelastung hoch ist, bleibt die Lage an der Hauptverkehrsstraße für viele attraktiv.",
        "Es steht außer Frage, dass der Klimaschutz bei künftigen Stadtentwicklungsprojekten Priorität haben muss.",
        "Die Anwohner, auf deren Widerspruch die Behörde rechnete, akzeptierten die Lärmschutzwand überraschend schnell.",
        "Würde man mehr Fahrradwege ausbauen, könnte der motorisierte Individualverkehr im Stadtzentrum abnehmen.",
        "Die Verdichtung innerstädtischer Flächen soll den Flächenverbrauch begrenzen und die Infrastruktur effizienter nutzen.",
    ],
]

sentences: list[str] = [s for batch in BATCHES for s in batch]
assert len(sentences) == 85, f"Expected 85 sentences, got {len(sentences)}"

import stanza

print("Loading Stanza...")
nlp = stanza.Pipeline(lang="de", processors="tokenize,mwt,pos,lemma", use_gpu=False)

MODALS = {"müssen", "können", "wollen", "sollen", "dürfen", "mögen", "möchte", "möchten"}
AUX_LEMMAS = {"sein", "haben", "werden", "müssen", "können", "wollen", "sollen", "dürfen", "mögen", "lassen"}

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
    "ihr": ("ihr", "DET"),
    "ihre": ("ihr", "DET"),
    "ihren": ("ihr", "DET"),
    "ihrem": ("ihr", "DET"),
    "ihrer": ("ihr", "DET"),
    "ihres": ("ihr", "DET"),
    "unser": ("unser", "DET"),
    "unsere": ("unser", "DET"),
    "unseren": ("unser", "DET"),
    "unserem": ("unser", "DET"),
    "unserer": ("unser", "DET"),
    "unseres": ("unser", "DET"),
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
    "alle": ("all", "DET"),
    "allen": ("all", "DET"),
    "jeder": ("jed", "DET"),
    "jedem": ("jed", "DET"),
    "jeden": ("jed", "DET"),
    "jede": ("jed", "DET"),
    "jedes": ("jed", "DET"),
    "mehr": ("mehr", "DET"),
    "viele": ("viel", "DET"),
    "vielen": ("viel", "DET"),
    "viel": ("viel", "DET"),
    "weniger": ("wenig", "DET"),
    "manche": ("manch", "DET"),
    "manchen": ("manch", "DET"),
    "neue": ("neu", "ADJ"),
    "neuen": ("neu", "ADJ"),
    "neues": ("neu", "ADJ"),
    "neuer": ("neu", "ADJ"),
    "neuem": ("neu", "ADJ"),
    "große": ("groß", "ADJ"),
    "großen": ("groß", "ADJ"),
    "großes": ("groß", "ADJ"),
    "großer": ("groß", "ADJ"),
    "gute": ("gut", "ADJ"),
    "guten": ("gut", "ADJ"),
    "gutes": ("gut", "ADJ"),
    "bessere": ("gut", "ADJ"),
    "besseren": ("gut", "ADJ"),
    "besten": ("gut", "ADJ"),
    "erste": ("erst", "ADJ"),
    "ersten": ("erst", "ADJ"),
    "letzten": ("letzt", "ADJ"),
    "letzte": ("letzt", "ADJ"),
    "weitere": ("weit", "ADJ"),
    "weiteren": ("weit", "ADJ"),
    "deutsche": ("deutsch", "ADJ"),
    "deutschen": ("deutsch", "ADJ"),
    "öffentlichen": ("öffentlich", "ADJ"),
    "öffentliche": ("öffentlich", "ADJ"),
    "sozialen": ("sozial", "ADJ"),
    "soziale": ("sozial", "ADJ"),
    "internationalen": ("international", "ADJ"),
    "internationale": ("international", "ADJ"),
    "heute": ("heute", "ADV"),
    "gestern": ("gestern", "ADV"),
    "morgen": ("morgen", "ADV"),
    "bald": ("bald", "ADV"),
    "noch": ("noch", "ADV"),
    "nur": ("nur", "ADV"),
    "sehr": ("sehr", "ADV"),
    "stets": ("stets", "ADV"),
    "bereits": ("bereits", "ADV"),
    "weiterhin": ("weiterhin", "ADV"),
    "deren": ("der", "DET"),
    "dessen": ("der", "DET"),
    "davon": ("davon", "ADV"),
    "daran": ("daran", "ADV"),
    "darauf": ("darauf", "ADV"),
    "dazu": ("dazu", "ADV"),
    "dabei": ("dabei", "ADV"),
    "dafür": ("dafür", "ADV"),
    "dagegen": ("dagegen", "ADV"),
    "im": ("in", "ADP"),
    "am": ("an", "ADP"),
    "zur": ("zu", "ADP"),
    "zum": ("zu", "ADP"),
    "beim": ("bei", "ADP"),
    "vom": ("von", "ADP"),
    "ins": ("in", "ADP"),
    "ans": ("an", "ADP"),
    "man": ("man", "PRON"),
    "niemand": ("niemand", "PRON"),
    "jemand": ("jemand", "PRON"),
    "etwas": ("etwas", "PRON"),
    "nichts": ("nichts", "PRON"),
}

SEIN_FORMS = {
    "ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen",
    "sei", "seien", "wäre",
}
HABEN_FORMS = {
    "habe", "hat", "hatte", "hätte", "hätten", "haben", "habt", "gehabt",
    "hätte",
}
WERDEN_FORMS = {
    "wird", "wurde", "werden", "würde", "würden", "geworden", "werde", "würde",
}
MODAL_FORMS: dict[str, str] = {
    "muss": "müssen", "musste": "müssen", "müsste": "müssen", "müssen": "müssen",
    "kann": "können", "konnte": "können", "könnte": "können", "können": "können",
    "will": "wollen", "wollte": "wollen", "wollen": "wollen",
    "soll": "sollen", "sollte": "sollen", "solle": "sollen", "sollen": "sollen",
    "darf": "dürfen", "durfte": "dürfen", "dürfte": "dürfen", "dürfen": "dürfen",
    "mag": "mögen", "möchte": "mögen", "möchten": "mögen", "mögen": "mögen",
    "lasse": "lassen", "lässt": "lassen", "lassen": "lassen", "ließ": "lassen",
}

SCONJ_WORDS = {
    "weil", "dass", "obwohl", "obschon", "wenn", "ob", "da", "als", "wie",
    "während", "bevor", "nachdem", "damit", "weshalb", "sodass", "obgleich",
}
CCONJ_WORDS = {"und", "oder", "aber", "sondern", "doch"}
ADP_WORDS = {
    "um", "für", "mit", "von", "bei", "nach", "aus", "über", "vor", "unter",
    "durch", "ohne", "gegen", "wegen", "in", "an", "auf", "zu", "bis", "seit",
    "trotz", "während",
}
PART_WORDS = {"nicht", "ja", "nein", "auch", "noch"}


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    """Apply handcrafted lemma/UPOS rules."""
    if form in SPECIAL_LEMMAS:
        lemma, upos = SPECIAL_LEMMAS[form]

    if form.lower() in {"das", "die", "der", "den", "dem", "des"} and upos == "PRON":
        lemma, upos = "der", "PRON"

    if upos == "NOUN" and lemma:
        lemma = lemma[0].upper() + lemma[1:] if lemma else lemma

    if upos == "ADJ" and lemma:
        lemma = lemma.lower()

    if upos == "PROPN" and lemma:
        lemma = lemma[0].upper() + lemma[1:] if lemma else lemma

    if upos == "PUNCT":
        lemma = form

    fl = form.lower()
    if fl in SEIN_FORMS:
        return "sein", "AUX"
    if fl in HABEN_FORMS:
        return "haben", "AUX"
    if fl in WERDEN_FORMS:
        return "werden", "AUX"
    if fl in MODAL_FORMS:
        return MODAL_FORMS[fl], "AUX"

    if upos == "VERB" or lemma in MODALS or fl in MODALS:
        upos = "VERB"
        if lemma:
            lemma = lemma.lower()
            if not (lemma.endswith("en") or lemma.endswith("n")):
                if form.lower().endswith("en") or form.lower().endswith("n"):
                    lemma = form.lower()

    if form.lower() in SCONJ_WORDS:
        if form.lower() in {"während", "seit", "bis"} and upos == "ADP":
            pass
        else:
            upos = "SCONJ"
            lemma = form.lower()
    elif form.lower() in CCONJ_WORDS:
        upos = "CCONJ"
        lemma = form.lower()
    elif form.lower() in ADP_WORDS:
        if form.lower() == "zu" and upos == "PART":
            pass
        else:
            upos = "ADP"
            lemma = form.lower()
    elif form.lower() in PART_WORDS:
        upos = "PART" if form.lower() != "noch" or upos not in {"ADV"} else upos
        if upos == "PART":
            lemma = form.lower()
    elif form.lower() in {"ich", "du", "er", "sie", "es", "wir", "ihr", "man"}:
        upos = "PRON"
        lemma = form.lower()

    if upos == "AUX" and lemma not in AUX_LEMMAS:
        upos = "VERB"

    return lemma, upos


def _reconstruct_text(forms: list[str]) -> str:
    punct_prefixes = ".,;:!?\"')"
    parts: list[str] = []
    for form in forms:
        if form and form[0] in punct_prefixes and parts:
            parts[-1] = parts[-1] + form
        else:
            parts.append(form)
    return " ".join(parts)


def _word_for_token(token, words_by_id: dict[int, object]):
    ids = token.id if isinstance(token.id, tuple) else (token.id,)
    return words_by_id.get(ids[0])


output_lines: list[str] = []
start_id = 16

for idx, sent in enumerate(sentences):
    sent_id = f"de_b2_val_{start_id + idx:03d}"
    doc = nlp(sent)

    sent_forms: list[str] = []
    sent_rows: list[str] = []
    token_counter = 1

    for stanza_sent in doc.sentences:
        words_by_id = {
            w.id: w for w in stanza_sent.words if isinstance(w.id, int)
        }
        for token in stanza_sent.tokens:
            form = token.text
            word = _word_for_token(token, words_by_id)
            upos = (word.upos if word and word.upos else "X")
            lemma = (word.lemma if word and word.lemma else form)
            lemma, upos = normalize_token(form, upos, lemma)

            cols = [
                str(token_counter),
                form,
                lemma,
                upos,
                "_", "_", "_", "_", "_", "_",
            ]
            sent_rows.append("\t".join(cols))
            sent_forms.append(form)
            token_counter += 1

    output_lines.append(f"# sent_id = {sent_id}")
    output_lines.append(f"# text = {_reconstruct_text(sent_forms)}")
    output_lines.extend(sent_rows)
    output_lines.append("")

output_lines.append("")
conllu_text = "\n".join(output_lines)

target_path = project_root / "data/handcraft/de/val/b2_new_001.conllu"
target_path.write_text(conllu_text, encoding="utf-8")
print(f"Wrote {len(sentences)} sentences to {target_path}")

validation_res = validate_text(conllu_text)
print(f"Validate: count={validation_res.sentence_count}, passed={validation_res.passed}")
if not validation_res.passed:
    for err in validation_res.errors[:15]:
        print("  ", err)
    sys.exit(1)

lemma_res = check_text(conllu_text, lang="de")
print(f"Lemma check: passed={lemma_res.passed}")
if not lemma_res.passed:
    for err in lemma_res.errors[:15]:
        print("  ", err)
    sys.exit(1)

dup_res = check_dup_text(conllu_text)
print(f"Duplicate check (within file): passed={dup_res.passed}")
if not dup_res.passed:
    for err in dup_res.duplicates[:10]:
        print("  ", err)
    sys.exit(1)

existing_val = (project_root / "data/handcraft/de/val/b2.conllu").read_text(encoding="utf-8")
cross_dup = check_dup_text(existing_val + "\n" + conllu_text)
print(f"Duplicate check (vs b2.conllu): passed={cross_dup.passed}")
if not cross_dup.passed:
    for err in cross_dup.duplicates[:10]:
        print("  ", err)
    sys.exit(1)

print("All checks passed.")