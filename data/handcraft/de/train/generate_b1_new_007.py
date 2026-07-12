"""Generate b1_new_007.conllu (de_b1_train_505–704) in sub-batches of 25."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 200 B1 sentences: 8–15 tokens, mixed tenses and structures
SENTENCE_BATCHES: list[list[str]] = [
    # 505–529: Job applications
    [
        "Ich habe meine Bewerbungsunterlagen gestern elektronisch verschickt.",
        "Wenn Sie Fragen haben, können Sie mich gerne anrufen.",
        "Ich würde gerne mehr über die Stelle erfahren.",
        "Mein aktueller Lebenslauf ist als Anhang beigefügt.",
        "Ich habe fünf Jahre in dem Bereich gearbeitet.",
        "Die Firma sucht einen erfahrenen Mitarbeiter für das Team.",
        "Ich freue mich auf Ihre baldige Antwort.",
        "Können Sie mir den genauen Gehaltsrahmen nennen?",
        "Ich wäre bereit, ab nächstem Monat zu beginnen.",
        "Der Personalleiter hat mir ein Gespräch angeboten.",
        "Ich habe gute Referenzen von meinem letzten Chef.",
        "Diese Stelle erfordert sehr gute Deutschkenntnisse von allen.",
        "Ich möchte gerne Teil eines dynamischen Teams werden.",
        "Meine Motivation für diese Arbeit ist sehr groß.",
        "Ich habe mich online auf die Anzeige beworben.",
        "Würden Sie mir bitte die Arbeitszeiten mitteilen?",
        "Er hat die Prüfung bestanden und die Stelle erhalten.",
        "Ich suche eine Vollzeitstelle mit flexiblen Arbeitszeiten.",
        "Meine Fähigkeiten passen gut zu den Anforderungen.",
        "Ich würde mich freuen, wenn wir uns persönlich treffen.",
        "Das Unternehmen bietet seinen Mitarbeitern gute Weiterbildungsmöglichkeiten an.",
        "Ich habe die Stellenanzeige in der Zeitung gelesen.",
        "Könnten Sie mir mehr über das Team erzählen?",
        "Ich bin überzeugt, dass ich die Aufgabe gut erfülle.",
        "Mein Berufsziel ist eine leitende Position in dem Marketing.",
    ],
    # 530–554: Opinions
    [
        "Meiner Meinung nach sollte man mehr für Bildung tun.",
        "Ich finde, dass das eine sehr gute Idee ist.",
        "Viele Leute sind der Ansicht, dass das zu teuer ist.",
        "Ich bin nicht damit einverstanden, so früh aufzustehen.",
        "Was denkst du über den neuen Film?",
        "Ich glaube, dass wir mehr zusammenarbeiten sollten.",
        "Er meint, dass das Wetter morgen besser wird.",
        "Ich halte das für eine vernünftige Entscheidung.",
        "Wir haben an dem Abend lange über Politik diskutiert.",
        "Ich stimme dir in diesem Punkt völlig zu.",
        "Das finde ich persönlich nicht besonders interessant.",
        "Sie sagt, dass Qualität wichtiger als Preis ist.",
        "Ich würde sagen, dass das ein großer Fortschritt ist.",
        "Viele junge Menschen interessieren sich für Nachhaltigkeit.",
        "Ich finde es unfair, dass nicht alle gleich behandelt werden.",
        "Er hat seine Meinung während der Diskussion geändert.",
        "Ich bin der Meinung, dass Musik Leben bereichert.",
        "Das ist meiner Ansicht nach ein wichtiges Thema.",
        "Ich würde lieber zu Hause bleiben als ausgehen.",
        "Sie findet, dass der Service in dem Restaurant schlecht war.",
        "Wir müssen respektieren, dass jeder anders denkt.",
        "Ich habe keine starke Meinung zu diesem Thema.",
        "Er behauptet, dass er die Wahrheit gesagt hat.",
        "Ich finde Reisen die beste Form der Bildung.",
        "Was würdest du an meiner Stelle tun?",
    ],
    # 555–579: Media
    [
        "Ich lese jeden Morgen die Nachrichten online.",
        "Der Artikel über den Klimawandel war sehr informativ.",
        "Hast du das Interview mit dem Minister gesehen?",
        "Die Nachrichten in dem Fernsehen berichten über den Wahlkampf.",
        "Ich folge mehreren Journalisten auf den sozialen Medien.",
        "Das Buch wurde in vielen Zeitungen positiv besprochen.",
        "Wir haben eine Dokumentation über Tierarten geschaut.",
        "Der Bericht zeigt, dass die Wirtschaft wächst.",
        "Ich höre bei dem Frühstück gerne einen Podcast.",
        "Die Schlagzeile hat mich heute Morgen überrascht.",
        "Er schreibt regelmäßig Kommentare unter Artikeln in dem Netz.",
        "Das Foto, das in der Zeitung erschien, war eindrucksvoll.",
        "Ich finde Fake News ein großes Problem in dem Internet.",
        "Wir haben den Podcast über deutsche Geschichte empfohlen.",
        "Die Sendung wurde wegen technischer Probleme abgesetzt.",
        "Ich abonniere eine Wochenzeitung für aktuelle Berichte.",
        "Der Journalist hat mit den Betroffenen gesprochen.",
        "Man sollte Nachrichten aus verschiedenen Quellen vergleichen.",
        "Das Video wurde innerhalb eines Tages millionenfach geteilt.",
        "Ich schaue mir abends oft Nachrichtensendungen an.",
        "Die Kritik in der Rezension war eher negativ.",
        "Wir diskutieren oft über Artikel aus der Tagespresse.",
        "Die Pressefreiheit ist ein wichtiges demokratisches Recht.",
        "Er hat einen Blog über nachhaltigen Konsum gestartet.",
        "Ich vertraue nur seriösen Nachrichtenquellen in dem Netz.",
    ],
    # 580–604: Environment
    [
        "Wir sollten weniger Plastik in dem Alltag verwenden.",
        "Der Wald in unserer Region wird aktiv geschützt.",
        "Ich fahre mit dem Fahrrad, um CO2 zu sparen.",
        "Viele deutsche Städte investieren in erneuerbare Energien.",
        "Das Recyclingsystem in Deutschland funktioniert sehr gut.",
        "Wir haben an einer Müllsammelaktion an dem Fluss teilgenommen.",
        "Der neue Klimabericht warnt vor steigenden Temperaturen.",
        "Ich kaufe bevorzugt Produkte ohne unnötige Verpackung.",
        "Die Artenvielfalt nimmt in vielen Gebieten alarmierend ab.",
        "Wir müssen unsere Kinder für die Natur sensibilisieren.",
        "Der Solarpark liefert Strom für tausend Haushalte.",
        "Ich finde, dass jeder einen Beitrag leisten kann.",
        "Der See wurde durch Industrieabfälle stark verschmutzt.",
        "Wir pflanzen jedes Jahr Bäume in unserem Stadtteil.",
        "Öffentliche Verkehrsmittel sind deutlich umweltfreundlicher als Autos.",
        "Die Regierung hat neue strenge Umweltgesetze verabschiedet.",
        "Ich trage eine Mehrwegflasche immer in meiner Tasche.",
        "Der große Nationalpark beherbergt viele seltene Tierarten.",
        "Wir sollten Energie sparen, wenn wir nicht zu Hause sind.",
        "Die Dürre hat die Ernte der Landwirte stark beeinträchtigt.",
        "Ich engagiere mich in einer lokalen Umweltorganisation.",
        "Der Windpark wurde trotz Protesten der Anwohner gebaut.",
        "Kompostieren hilft, Bioabfall sinnvoll zu nutzen.",
        "Wir haben den Stromverbrauch in dem Haushalt deutlich gesenkt.",
        "Der Naturschutz braucht deutlich mehr politische Unterstützung.",
    ],
    # 605–629: Culture
    [
        "Wir haben gestern ein klassisches Konzert in dem Theater besucht.",
        "Das Museum zeigt eine Ausstellung über mittelalterliche Kunst.",
        "Ich interessiere mich sehr für die deutsche Literatur.",
        "Der Karneval in Köln ist weltberühmt und sehr bunt.",
        "Wir haben traditionelle Gerichte auf dem Fest probiert.",
        "Die Oper, die wir gesehen haben, war wunderschön.",
        "Ich lese gerne Bücher von deutschen Autoren.",
        "Das Fest der Kulturen findet jedes Jahr in dem Park statt.",
        "Wir haben eine Führung durch die historische Altstadt gemacht.",
        "Er spielt in einer Amateurtheatergruppe mit großer Leidenschaft.",
        "Die Weihnachtsmärkte in Deutschland sind besonders gemütlich.",
        "Ich finde Volksmusik und Tanz sehr unterhaltsam.",
        "Das Denkmal erinnert an wichtige Ereignisse der Geschichte.",
        "Wir haben ein neues Theaterstück in dem Stadttheater gesehen.",
        "Die Kunstgalerie zeigt Werke junger zeitgenössischer Künstler.",
        "Ich möchte mehr über die Bräuche in Bayern erfahren.",
        "Das Buchfest zieht jedes Jahr viele Besucher an.",
        "Wir haben ein traditionelles Volksfest in dem Dorf gefeiert.",
        "Die Symphonie, die gestern gespielt wurde, war beeindruckend.",
        "Ich besuche regelmäßig Lesungen in der Stadtbibliothek.",
        "Das Kulturprogramm der Stadt ist dieses Jahr sehr vielfältig.",
        "Wir haben einen Film über deutsche Geschichte in dem Kino gesehen.",
        "Das Volkslied wurde von der ganzen Gruppe mitgesungen.",
        "Ich finde kulturellen Austausch zwischen Ländern sehr wichtig.",
        "Die Ausstellung über impressionistische Malerei hat mich begeistert.",
    ],
    # 630–654: Konjunktiv II
    [
        "Wenn ich mehr Zeit hätte, würde ich öfter reisen.",
        "Ich wünschte, ich könnte besser Deutsch sprechen.",
        "Hättest du Lust, an dem Wochenende mit uns zu wandern?",
        "Er würde gerne in dem Ausland arbeiten, wenn es möglich wäre.",
        "Ich hätte gern ein größeres Zimmer mit Balkon.",
        "Wenn das Wetter schöner wäre, würden wir picknicken.",
        "Sie wünschte sich mehr Unterstützung von ihren Kollegen.",
        "Könntest du mir bitte das Salz reichen?",
        "Ich würde an deiner Stelle die Stelle annehmen.",
        "Wenn ich reich wäre, würde ich ein Haus an dem Meer kaufen.",
        "Er hätte das Projekt früher abschließen sollen.",
        "Ich wünschte, wir hätten mehr Ferien in dem Jahr.",
        "Würdest du mir helfen, wenn ich dich bräuchte?",
        "Sie wäre gern Lehrerin, aber es war nicht möglich.",
        "Hätten Sie einen Moment Zeit für eine kurze Frage?",
        "Ich würde lieber zu Fuß gehen als den Bus nehmen.",
        "Wenn er pünktlicher wäre, würde das Team zufrieden sein.",
        "Ich hätte nie gedacht, dass das so schwierig wird.",
        "Würdest du bitte das offene Fenster schließen?",
        "Sie wünschte, sie hätte die Prüfung bestanden.",
        "Ich wäre froh, wenn du mich anrufen würdest.",
        "Hätte ich das gewusst, wäre ich früher gekommen.",
        "Er würde das Buch lesen, wenn er mehr Zeit hätte.",
        "Ich könnte morgen früher kommen, wenn das hilft.",
        "Wäre es möglich, den Termin zu verschieben?",
    ],
    # 655–679: Passive
    [
        "Das Haus wurde in dem letzten Jahr komplett renoviert.",
        "In Deutschland wird traditionell viel Bier getrunken.",
        "Der Brief wurde gestern an den Kunden geschickt.",
        "Das Essen wird in der Küche frisch zubereitet.",
        "Die alte Brücke wird wegen Reparaturen gesperrt.",
        "Das Problem wurde von unserem Team schnell gelöst.",
        "In der Schule werden viele Sprachen unterrichtet.",
        "Das Buch wurde in zwanzig Sprachen übersetzt.",
        "Die neue Straße wird nächstes Jahr gebaut.",
        "Alle Fenster wurden wegen des Sturms geschlossen.",
        "In diesem Restaurant wird nur frisches Gemüse verwendet.",
        "Die Prüfung wird an dem Ende des Semesters geschrieben.",
        "Das Paket wurde leider an die falsche Adresse geliefert.",
        "Hier wird streng darauf geachtet, dass Regeln eingehalten werden.",
        "Die Entscheidung wurde nach langer Beratung getroffen.",
        "In der Fabrik werden täglich hundert Autos produziert.",
        "Das Museum wird an dem Montag wegen Renovierung geschlossen.",
        "Der Antrag muss bis Freitag eingereicht werden.",
        "Die Nachricht wurde über alle Kanäle verbreitet.",
        "In unserer Stadt wird viel für den Radverkehr getan.",
        "Das Projekt wird von einem erfahrenen Architekten geleitet.",
        "Die Meldung wurde bereits an die Behörden weitergeleitet.",
        "In diesem Land wird Kaffee sehr gerne getrunken.",
        "Die Gebühren werden an dem Monatsende automatisch abgebucht.",
        "Das Gebäude wurde in dem Krieg stark beschädigt.",
    ],
    # 680–704: Relative clauses, weil/dass/wenn
    [
        "Der Mann, der neben mir sitzt, ist mein Nachbar.",
        "Das ist das Buch, das ich dir empfehlen wollte.",
        "Ich bleibe zu Hause, weil ich mich nicht gut fühle.",
        "Er hat gesagt, dass er morgen später kommt.",
        "Wenn es regnet, nehme ich immer einen Regenschirm mit.",
        "Die Frau, mit der ich gesprochen habe, war sehr nett.",
        "Ich glaube, dass er die Prüfung bestehen wird.",
        "Wir gehen in das Kino, wenn der Film gut bewertet ist.",
        "Das Auto, das vor dem Haus steht, gehört meinem Bruder.",
        "Sie ist traurig, weil ihr Freund weggezogen ist.",
        "Ich weiß, dass du viel für die Prüfung gelernt hast.",
        "Wenn du Zeit hast, können wir uns treffen.",
        "Der Film, den wir gestern gesehen haben, war spannend.",
        "Er bleibt in dem Bett, weil er Fieber hat.",
        "Ich hoffe, dass das Wetter an dem Wochenende schön wird.",
        "Wenn ich koche, höre ich gerne leise Musik.",
        "Das Haus, in dem ich aufgewachsen bin, steht an dem See.",
        "Sie lernt Deutsch, weil sie in Berlin studieren möchte.",
        "Er meint, dass die Reise zu teuer geworden ist.",
        "Wenn der Bus Verspätung hat, komme ich zu spät.",
        "Die Stadt, die wir besucht haben, liegt an dem Rhein.",
        "Ich bin froh, dass du mir bei der Arbeit geholfen hast.",
        "Wenn es warm ist, sitzen wir gerne auf der Terrasse.",
        "Das Lied, das sie gerade spielt, kenne ich gut.",
        "Wir fahren nach Hause, weil es schon sehr spät ist.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

START_ID = 505
BATCH_SIZE = 25

MODALS = {"müssen", "können", "wollen", "sollen", "dürfen", "mögen", "möchte", "möchten"}
AUX_LEMMAS = {"sein", "haben", "werden"}
AUX_FORMS = {
    "ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen",
    "habe", "hat", "hatte", "hätte", "hätten", "haben", "habt", "gehabt",
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
    "Ihre": ("ihr", "DET"),
    "Ihren": ("ihr", "DET"),
    "Ihrer": ("ihr", "DET"),
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
    "jeden": ("jed", "DET"),
    "jede": ("jed", "DET"),
    "jedes": ("jed", "DET"),
    "jeder": ("jed", "DET"),
    "jedem": ("jed", "DET"),
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
    "bitte": ("bitten", "ADV"),
    "Köln": ("Köln", "PROPN"),
    "Bayern": ("Bayern", "PROPN"),
    "Berlin": ("Berlin", "PROPN"),
    "Rhein": ("Rhein", "PROPN"),
    "CO2": ("CO2", "NOUN"),
    "Fake": ("Fake", "NOUN"),
    "News": ("News", "NOUN"),
}

SCONJS = {"weil", "dass", "obwohl", "wenn", "ob", "da", "als", "wie", "während", "seit", "bis"}
CCONJS = {"und", "oder", "aber", "sondern"}
ADPS = {
    "um", "für", "mit", "von", "bei", "nach", "aus", "über", "vor", "unter",
    "durch", "ohne", "gegen", "wegen", "in", "an", "auf", "am", "im", "ins",
}
PARTS = {"nicht", "ja", "nein", "zu"}


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    """Apply handcraft lemma/UPOS conventions."""
    if form in SPECIAL_LEMMAS:
        lemma, upos = SPECIAL_LEMMAS[form]

    if form.lower() in {"das", "die", "der", "den", "dem", "des"} and upos == "PRON":
        lemma = "der"

    lower = form.lower()

    if lower in AUX_FORMS:
        upos = "AUX"
        if lower in {"ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen", "wäre", "wären"}:
            lemma = "sein"
        elif lower in {"habe", "hat", "hatte", "hätte", "hätten", "haben", "habt", "gehabt"}:
            lemma = "haben"
        else:
            lemma = "werden"
    elif upos == "AUX" and lemma not in AUX_LEMMAS:
        upos = "VERB"

    if upos == "VERB" or lemma in MODALS or lower in MODALS:
        upos = "VERB"
        lemma = lemma.lower()
        if not (lemma.endswith("en") or lemma.endswith("n")):
            # Stanza lemma fallback: use form if already infinitive-like
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
    elif lower in {"ich", "du", "er", "sie", "es", "wir"}:
        upos = "PRON"
        lemma = lower

    return lemma, upos


def sentence_to_conllu(sent_id: str, sent: str, nlp) -> list[str]:
    lines: list[str] = [f"# sent_id = {sent_id}", f"# text = {sent}"]
    doc = nlp(sent)
    token_counter = 1
    for stanza_sent in doc.sentences:
        for word in stanza_sent.words:
            if not isinstance(word.id, int):
                continue
            form = word.text
            upos = word.upos or "X"
            lemma = word.lemma if word.lemma else form
            lemma, upos = normalize_token(form, upos, lemma)
            cols = [
                str(token_counter),
                form,
                lemma,
                upos,
                "_", "_", "_", "_", "_", "_",
            ]
            lines.append("\t".join(cols))
            token_counter += 1
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

    target_path = project_root / "data/handcraft/de/train/b1_new_007.conllu"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    all_lines: list[str] = []
    global_idx = 0

    for batch_num, batch in enumerate(SENTENCE_BATCHES):
        assert len(batch) == BATCH_SIZE, f"Batch {batch_num} has {len(batch)} sentences"
        print(f"Processing batch {batch_num + 1}/8 (sentences {START_ID + global_idx}–{START_ID + global_idx + BATCH_SIZE - 1})")
        for sent in batch:
            sent_id = f"de_b1_train_{START_ID + global_idx}"
            all_lines.extend(sentence_to_conllu(sent_id, sent, nlp))
            global_idx += 1

    conllu_text = "\n".join(all_lines) + "\n"
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