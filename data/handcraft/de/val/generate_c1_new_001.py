"""Generate C1 German validation CoNLL-U: de_c1_val_016 through de_c1_val_100."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 85 handcrafted C1 validation sentences (academic / scholarly style)
SENTENCES = [
    # Philology & literary studies (016-030)
    "Der von dem Literaturwissenschaftler herausgegebene Textband, dessen Annotationen erhebliche Deutungsarbeit erfordern, erweist sich als unverzichtbare Quelle.",
    "Obgleich die von dem Philologen rekonstruierte Lesart überzeugend argumentiert, bleiben einzelne Textstellen umstritten.",
    "Inwiefern die von dem Übersetzer gewählten Formulierungen dem Ursprungstext gerecht werden, bedarf einer feingliedrigen semantischen Analyse.",
    "Der von dem Dramaturgen entworfene Bühnentext, der mythologische Motive neu kontextualisiert, fand bei der Premiere breite Zustimmung.",
    "Gleichwohl vermag die von dem Kritiker angeführte Interpretation nicht, sämtliche narrative Inkonsistenzen plausibel zu erklären.",
    "Nichtsdestotrotz weist die von dem Rezensenten vorgebrachte Lesart auf bislang unbeachtete symbolische Ebenen hin.",
    "Insofern die von dem Archivar erschlossenen Dokumente authentisch sind, ergeben sich weitreichende Konsequenzen für die Biographieforschung.",
    "Der von dem Herausgeber kommentierte Werkabschnitt, dessen Überlieferungsgeschichte lückenhaft bleibt, bleibt philologisch höchst umstritten.",
    "Obwohl die von dem Autor behauptete Intention nachvollziehbar erscheint, lässt sich ihre textimmanente Belegbarkeit nicht ohne Weiteres feststellen.",
    "Der von dem Kommentator erläuterte Passus, der allegorische Lesarten nahelegt, wurde in der Fachdiskussion intensiv debattiert.",
    "Wenngleich die von dem Forscher aufgestellte These kühn wirkt, vermag sie doch mehrere bislang isolierte Befunde zu verbinden.",
    "Inwiefern das von dem Metaphorologen entwickelte Konzept tragfähig bleibt, hängt von der Reichweite seiner Anwendungsfälle ab.",
    "Der von dem Biographen rekonstruierte Lebensabschnitt, dessen Quellenlage äußerst fragmentarisch ist, bleibt in zentralen Punkten spekulativ.",
    "Gleichwohl erweist sich die von dem Literaturhistoriker gezogene Parallelisierung als methodisch fragwürdig, wenngleich sie heuristisch fruchtbar sein kann.",
    "Der von dem Stilistiker beschriebene Sprachgebrauch, der barocke Elemente aufweist, korreliert mit der datierten Entstehungszeit.",
    # Social sciences (031-045)
    "Nichtsdestotrotz lässt sich die von dem Kritiker erhobene Behauptung, der Roman sei zyklisch angelegt, anhand der Kapitelstruktur belegen.",
    "Insofern die von dem Philosophen zitierten Stellen korrekt wiedergegeben wurden, stützt sich seine Argumentation auf verlässliche Grundlagen.",
    "Der von dem Lexikographen erstellte Wortindex, der seltene Neologismen erfasst, erleichtert die gezielte Textexegese erheblich.",
    "Obgleich die von dem Ästhetiker vorgetragene Theorie eloquent formuliert ist, entzieht sich ihr normativer Anspruch jeder empirischen Prüfung.",
    "Die von dem Musikwissenschaftler transkribierte Partitur, deren Entstehungskontext ungeklärt bleibt, bedarf einer sorgfältigen Quellenkritik.",
    "Gleichwohl vermochte die von dem Kulturhistoriker angebotene Deutung, einen Konsens in der Fachwelt herbeizuführen.",
    "Der von dem Soziologen erhobene Einwand, die Stichprobe sei nicht repräsentativ, lässt sich anhand der Methodendokumentation widerlegen.",
    "Inwiefern die von dem Anthropologen gesammelten Felddaten generalisierbar sind, bedarf einer eingehenden statistischen Überprüfung.",
    "Obgleich die von dem Politologen skizzierte Theorie kohärent wirkt, fehlt ihr bislang eine belastbare empirische Fundierung.",
    "Der von dem Ökonomen verfasste Policy Brief, dessen Empfehlungen weitreichend sind, stieß im Ministerium auf Vorbehalte.",
    "Insofern die von dem Demografen prognostizierten Trends eintreten, erübrigen sich kurzfristige Gegenmaßnahmen, gleichwohl bleibt die Debatte offen.",
    "Nichtsdestotrotz weist die von dem Statistiker durchgeführte Metaanalyse auf signifikante Effekte in mehreren Teilpopulationen hin.",
    "Der von dem Psychologen entwickelte Fragebogen, der implizite Einstellungen erfasst, wurde in drei unabhängigen Studien validiert.",
    "Gleichwohl legt der von dem Pädagogen dargelegte Ansatz dar, dass heterogene Lerngruppen differenzierte Förderkonzepte erfordern.",
    "Inwiefern die von dem Kommunikationswissenschaftler untersuchten Medienformate das politische Urteilsvermögen prägen, bleibt umstritten.",
    # Philosophy & ethics (046-060)
    "Der von dem Ethiker erörterte Gedankenexperiment, das Grenzfälle der Verantwortungszuschreibung beleuchtet, erfuhr vielfältige Rezeption.",
    "Obgleich die von dem Metaphysiker vertretene Position logisch stringent ist, bleibt ihre ontologische Verpflichtung debattabel.",
    "Insofern die von dem Erkenntnistheoretiker formulierten Kriterien gültig sind, ergeben sich weitreichende Konsequenzen für die Wissenschaftstheorie.",
    "Der von dem Moralphilosophen entworfene Kategorienapparat, dessen Reichweite begrenzt erscheint, vermag dennoch zentrale Dilemmata zu klären.",
    "Gleichwohl vermag die von dem Phänomenologen beschriebene Erfahrungsstruktur nicht, sämtliche intentionalen Akte zu erfassen.",
    "Nichtsdestotrotz erweist sich die von dem Logiker angeführte Argumentation als widerlegbar, obgleich sie zunächst zwingend wirkt.",
    "Inwiefern das von dem Rechtsphilosophen geprägte Prinzip der Verhältnismäßigkeit universell anwendbar ist, bedarf weiterer Klärung.",
    "Der von dem Theologen interpretierte Textabschnitt, der exegetische Kontroversen auslöste, bleibt in der Hermeneutik hoch relevant.",
    "Obwohl die von dem Epistemologen behauptete Unterscheidung plausibel erscheint, lässt sich ihre Abgrenzung operational kaum festlegen.",
    "Wenngleich die von dem Pragmatiker vertretene Lesart überzeugend klingt, vermag sie normative Implikationen nur unzureichend zu fassen.",
    "Der von dem Ästhetiker entwickelte Begriff des Erhabenen, der an Kant anknüpft, erfuhr in der Gegenwartsphilosophie neue Deutungen.",
    "Insofern die von dem Politischen Philosophen erhobene Forderung nach Deliberation berechtigt ist, bleibt ihr institutionelles Design prekär.",
    "Gleichwohl weist die von dem Wissenschaftstheoretiker vorgebrachte Kritik auf methodische Blindstellen in der Disziplin hin.",
    "Der von dem Philosophen dargelegte Begriff der Autonomie, der eng mit der praktischen Vernunft verknüpft ist, erfuhr unterschiedliche Deutungen.",
    "Nichtsdestotrotz lässt sich die von dem Kritiker erhobene These, die Theorie sei zirkulär, anhand der Prämissenstruktur entkräften.",
    # History & cultural studies (061-075)
    "Inwiefern die von dem Historiker rekonstruierte Ereigniskette haltbar ist, bedarf einer kritischen Quellenrevision.",
    "Der von dem Archäologen freigelegte Fundkomplex, dessen Datierung lange umstritten war, liefert neue Anhaltspunkte für die Chronologie.",
    "Obgleich die von dem Regionalhistoriker geschilderte Entwicklung stimmig wirkt, bleiben zentrale Akteure dokumentarisch unterbelichtet.",
    "Der von dem Mediävisten edierte Handschriftenfonds, der bisher unzugängliche Texte erschließt, erweist sich als philologische Fundgrube.",
    "Insofern die von dem Militärhistoriker angeführten Quellen authentisch sind, ergeben sich weitreichende Revisionen der bisherigen Darstellung.",
    "Gleichwohl vermag die von dem Kunsthistoriker vorgenommene Ikonographie nicht, sämtliche Bildsymbole eindeutig zu deuten.",
    "Nichtsdestotrotz weist die von dem Numismatiker katalogisierte Münzserie auf weitreichende Handelsverbindungen im Mittelmeerraum hin.",
    "Der von dem Archivar erschlossene Nachlass, dessen Erschließung Jahre in Anspruch nahm, erweist sich als historische Fundgrube.",
    "Inwiefern die von dem Byzantinisten übersetzten Urkunden verlässlich sind, lässt sich erst nach paläographischer Prüfung beurteilen.",
    "Obwohl die von dem Frühgeschichtler rekonstruierte Siedlungsstruktur plausibel erscheint, fehlen bislang belastbare Radiokarbondaten.",
    "Der von dem Historiker verfasste Syntheseband, der kontroverse Deutungen vermittelt, zeichnet sich durch narrative Prägnanz aus.",
    "Wenngleich die von dem Kulturwissenschaftler analysierten Artefakte fragmentarisch sind, vermögen sie dennoch Rückschlüsse auf Praktiken zu erlauben.",
    "Insofern die von dem Zeitgeschichtler befragten Zeitzeugen glaubwürdig berichten, ergeben sich weitreichende Korrekturen des offiziellen Narrativs.",
    "Gleichwohl erweist sich die von dem Diplomatikhistoriker erstellte Edition als unverzichtbar, obgleich einzelne Lesarten umstritten bleiben.",
    "Der von dem Museologen konzipierte Ausstellungskatalog, der kontextualisierende Essays bündelt, fand in der Fachpresse breite Anerkennung.",
    # Linguistics & mixed academic (076-100)
    "Nichtsdestotrotz weist die von dem Sprachwissenschaftler durchgeführte Korpusanalyse auf syntaktische Verschiebungen im neunzehnten Jahrhundert hin.",
    "Inwiefern die von dem Dialektologen erhobenen Daten repräsentativ sind, bedarf einer sorgfältigen statistischen Auswertung.",
    "Der von dem Phonetiker erstellte Atlas, der fein abgestufte Lautunterschiede kartiert, erleichtert vergleichende Studien erheblich.",
    "Obgleich die von dem Lexikographen dokumentierten Anglizismen zahlreich sind, bleibt ihre Integration in den Sprachgebrauch graduell.",
    "Der von dem Pragmatiker untersuchte Diskurs, dessen Höflichkeitsstrategien differieren, liefert wichtige Hinweise für die Gesprächsanalyse.",
    "Insofern die von dem Morphologen beschriebene Flexionsklasse produktiv bleibt, ergeben sich weitreichende Implikationen für die Grammatiktheorie.",
    "Gleichwohl vermag die von dem Korpuslinguisten erhobene Behauptung nicht, sämtliche Ausnahmefälle der Valenztheorie zu erklären.",
    "Der von dem Historischen Linguisten rekonstruierte Lautwandel, der mehrere Subsysteme betrifft, bleibt in der Forschung hoch umstritten.",
    "Nichtsdestotrotz erweist sich die von dem Typologen angeführte Generalisierung als haltbar, wenngleich Gegenbeispiele nicht ausbleiben.",
    "Inwiefern das von dem Kognitiven Linguisten geprägte Konzept der Conceptual Metaphor Theory anwendbar bleibt, bedarf weiterer empirischer Prüfung.",
    "Der von dem Juristen formulierte Rechtsgrundsatz, dessen Reichweite der EuGH präzisierte, erfuhr in der Literatur eingehende Würdigung.",
    "Obwohl die von dem Verwaltungsrechtler erhobene Einwendung formal stichhaltig ist, bleibt ihre praktische Durchsetzbarkeit fraglich.",
    "Der von dem Europarechtler kommentierte Artikel, der Grundfreiheiten absichert, wurde in der aktuellen Rechtsprechung mehrfach herangezogen.",
    "Insofern die von dem Strafrechtler vertretene Auslegung überzeugt, erübrigen sich weitergehende Gesetzesrevisionen, gleichwohl bleibt die Debatte lebendig.",
    "Gleichwohl weist die von dem Verfassungsrechtler vorgebrachte Argumentation auf systematische Spannungen im Grundgesetz hin.",
    "Der von dem Medizinethiker erörterte Grenzfall, der Einwilligungsfähigkeit unter Betäubung betrifft, bleibt in der Klinik hoch relevant.",
    "Nichtsdestotrotz lässt sich die von dem Epidemiologen erhobene Behauptung, die Inzidenz steige, anhand der Meldedaten plausibilisieren.",
    "Inwiefern die von dem Kliniker dokumentierten Verläufe generalisierbar sind, bedarf einer multizentrischen Nachkontrolle.",
    "Der von dem Forscher veröffentlichte Bericht, dessen Methodenteil lückenhaft wirkt, legt dennoch nahe, dass die Intervention wirksam ist.",
    "Obgleich die von dem Gutachter vorgenommene Analyse gravierende methodische Mängel aufweist, vermag sie dennoch einen klaren Trend aufzuzeigen.",
    "Der von dem Naturwissenschaftler präsentierte Mechanismus, der bisherige Modelle ergänzt, erfuhr in Fachzeitschriften breite Diskussion.",
    "Insofern die von dem Theoretischen Physiker entwickelte Gleichung konsistent bleibt, ergeben sich weitreichende Konsequenzen für die Kosmologie.",
    "Gleichwohl erweist sich die von dem Mathematiker bewiesene Behauptung als zentral, wenngleich ihre Anwendbarkeit im Alltag begrenzt bleibt.",
    "Der von dem Informatiker implementierte Algorithmus, dessen Laufzeit polynomial bleibt, eröffnet neue Perspektiven für die Netzwerkanalyse.",
    "Nichtsdestotrotz weist die von dem Klimaforscher modellierte Projektion auf signifikante Unsicherheiten in den Randbedingungen hin.",
]

assert len(SENTENCES) == 85, f"Expected 85 sentences, got {len(SENTENCES)}"

START_ID = 16
ID_PREFIX = "de_c1_val"
TARGET_PATH = project_root / "data/handcraft/de/val/c1_new_001.conllu"

MODAL_FORMS = {
    "kann": "können", "könnte": "können", "konnte": "können", "können": "können", "könnt": "können",
    "muss": "müssen", "müsste": "müssen", "musste": "müssen", "müssen": "müssen", "müsse": "müssen",
    "soll": "sollen", "sollte": "sollen", "sollen": "sollen",
    "darf": "dürfen", "dürfte": "dürfen", "dürfen": "dürfen",
    "will": "wollen", "wollte": "wollen", "wollen": "wollen",
    "mag": "mögen", "möchte": "mögen", "mögen": "mögen",
}

SEIN_FORMS = {"ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen", "sei", "seien"}
HABEN_FORMS = {"habe", "hat", "hatte", "hätte", "hätten", "haben", "habt", "gehabt"}
WERDEN_FORMS = {"wird", "wurde", "werden", "würde", "würden", "geworden", "worden"}

SPECIAL_LEMMAS: dict[str, tuple[str, str]] = {
    "mich": ("ich", "PRON"), "dich": ("du", "PRON"), "sich": ("sich", "PRON"),
    "uns": ("wir", "PRON"), "euch": ("ihr", "PRON"), "mir": ("ich", "PRON"),
    "dir": ("du", "PRON"), "ihm": ("er", "PRON"), "ihnen": ("sie", "PRON"),
    "Ihnen": ("Sie", "PRON"), "Sie": ("Sie", "PRON"),
    "mein": ("mein", "DET"), "meine": ("mein", "DET"), "meinen": ("mein", "DET"),
    "sein": ("sein", "DET"), "seine": ("sein", "DET"), "seinen": ("sein", "DET"),
    "seinem": ("sein", "DET"), "seiner": ("sein", "DET"), "seines": ("sein", "DET"),
    "der": ("der", "DET"), "die": ("der", "DET"), "das": ("der", "DET"),
    "den": ("der", "DET"), "dem": ("der", "DET"), "des": ("der", "DET"),
    "ein": ("ein", "DET"), "eine": ("ein", "DET"), "einen": ("ein", "DET"),
    "einem": ("ein", "DET"), "einer": ("ein", "DET"), "eines": ("ein", "DET"),
    "ihren": ("ihr", "DET"), "ihre": ("ihr", "DET"), "ihrem": ("ihr", "DET"),
    "ihrer": ("ihr", "DET"), "ihr": ("ihr", "DET"), "dessen": ("dessen", "DET"),
    "deren": ("deren", "DET"),
    "zur": ("zu", "ADP"), "zum": ("zu", "ADP"), "beim": ("bei", "ADP"),
    "im": ("in", "ADP"), "am": ("an", "ADP"), "ans": ("an", "ADP"),
    "ins": ("in", "ADP"), "vom": ("von", "ADP"),
    "darauf": ("darauf", "ADV"), "darin": ("darin", "ADV"), "hin": ("hin", "ADV"),
    "dar": ("dar", "ADV"), "dennoch": ("dennoch", "ADV"), "aus": ("aus", "ADP"),
    "gerecht": ("gerecht", "ADJ"), "überzeugt": ("überzeugen", "VERB"),
    "nicht": ("nicht", "PART"), "zu": ("zu", "PART"),
    "bereits": ("bereits", "ADV"), "noch": ("noch", "ADV"), "auch": ("auch", "ADV"),
    "nur": ("nur", "ADV"), "sehr": ("sehr", "ADV"), "deutlich": ("deutlich", "ADV"),
    "erheblich": ("erheblich", "ADV"), "häufig": ("häufig", "ADV"),
    "weitere": ("weit", "ADJ"), "weiterer": ("weit", "ADJ"), "weiteren": ("weit", "ADJ"),
    "weitreichende": ("weitreichend", "ADJ"), "weitreichenden": ("weitreichend", "ADJ"),
    "weitreichender": ("weitreichend", "ADJ"),
    "eingehenden": ("eingehend", "ADJ"), "eingehende": ("eingehend", "ADJ"),
    "herausgegebene": ("herausgegeben", "ADJ"), "rekonstruierte": ("rekonstruiert", "ADJ"),
    "gewählten": ("gewählt", "ADJ"), "entworfene": ("entworfen", "ADJ"),
    "angeführte": ("angeführt", "ADJ"), "vorgebrachte": ("vorgebracht", "ADJ"),
    "erschlossenen": ("erschlossen", "ADJ"), "kommentierte": ("kommentiert", "ADJ"),
    "behauptete": ("behauptet", "ADJ"), "erläuterte": ("erläutert", "ADJ"),
    "aufgestellte": ("aufgestellt", "ADJ"), "entwickelte": ("entwickelt", "ADJ"),
    "rekonstruierte": ("rekonstruiert", "ADJ"), "gezogene": ("gezogen", "ADJ"),
    "beschriebene": ("beschrieben", "ADJ"), "erstellte": ("erstellt", "ADJ"),
    "vorgetragene": ("vorgetragen", "ADJ"), "transkribierte": ("transkribiert", "ADJ"),
    "angebotene": ("angeboten", "ADJ"), "erhobene": ("erhoben", "ADJ"),
    "gesammelten": ("gesammelt", "ADJ"), "skizzierte": ("skizziert", "ADJ"),
    "verfasste": ("verfasst", "ADJ"), "prognostizierten": ("prognostiziert", "ADJ"),
    "durchgeführte": ("durchgeführt", "ADJ"), "entwickelte": ("entwickelt", "ADJ"),
    "dargelegte": ("dargelegt", "ADJ"), "untersuchten": ("untersucht", "ADJ"),
    "erörterte": ("erörtert", "ADJ"), "vertretene": ("vertreten", "ADJ"),
    "formulierten": ("formuliert", "ADJ"), "entworfene": ("entworfen", "ADJ"),
    "beschriebene": ("beschrieben", "ADJ"), "angeführte": ("angeführt", "ADJ"),
    "geprägte": ("geprägt", "ADJ"), "interpretierte": ("interpretiert", "ADJ"),
    "behauptete": ("behauptet", "ADJ"), "entwickelte": ("entwickelt", "ADJ"),
    "erhobene": ("erhoben", "ADJ"), "vorgebrachte": ("vorgebracht", "ADJ"),
    "dargelegte": ("dargelegt", "ADJ"), "erhobene": ("erhoben", "ADJ"),
    "freigelegte": ("freigelegt", "ADJ"),
    "geschilderte": ("geschildert", "ADJ"), "edierte": ("ediert", "ADJ"),
    "angeführten": ("angeführt", "ADJ"), "vorgenommene": ("vorgenommen", "ADJ"),
    "katalogisierte": ("katalogisiert", "ADJ"), "erschlossene": ("erschlossen", "ADJ"),
    "übersetzten": ("übersetzt", "ADJ"), "rekonstruierte": ("rekonstruiert", "ADJ"),
    "verfasste": ("verfasst", "ADJ"), "analysierten": ("analysiert", "ADJ"),
    "befragten": ("befragt", "ADJ"), "erstellte": ("erstellt", "ADJ"),
    "konzipierte": ("konzipiert", "ADJ"), "durchgeführte": ("durchgeführt", "ADJ"),
    "erhobenen": ("erhoben", "ADJ"), "erstellte": ("erstellt", "ADJ"),
    "dokumentierten": ("dokumentiert", "ADJ"), "untersuchte": ("untersucht", "ADJ"),
    "beschriebene": ("beschrieben", "ADJ"), "erhobene": ("erhoben", "ADJ"),
    "rekonstruierte": ("rekonstruiert", "ADJ"), "angeführte": ("angeführt", "ADJ"),
    "geprägte": ("geprägt", "ADJ"), "formulierte": ("formuliert", "ADJ"),
    "erhobene": ("erhoben", "ADJ"), "kommentierte": ("kommentiert", "ADJ"),
    "vertretene": ("vertreten", "ADJ"), "vorgebrachte": ("vorgebracht", "ADJ"),
    "erörterte": ("erörtert", "ADJ"), "erhobene": ("erhoben", "ADJ"),
    "dokumentierten": ("dokumentiert", "ADJ"), "veröffentlichte": ("veröffentlicht", "ADJ"),
    "vorgenommene": ("vorgenommen", "ADJ"), "präsentierte": ("präsentiert", "ADJ"),
    "entwickelte": ("entwickelt", "ADJ"), "bewiesene": ("bewiesen", "ADJ"),
    "implementierte": ("implementiert", "ADJ"), "modellierte": ("modelliert", "ADJ"),
    "philologische": ("philologisch", "ADJ"), "philologischen": ("philologisch", "ADJ"),
    "methodische": ("methodisch", "ADJ"), "methodischen": ("methodisch", "ADJ"),
    "empirischen": ("empirisch", "ADJ"), "empirische": ("empirisch", "ADJ"),
    "statistischen": ("statistisch", "ADJ"), "semantischen": ("semantisch", "ADJ"),
    "narrative": ("narrativ", "ADJ"), "symbolische": ("symbolisch", "ADJ"),
    "allegorische": ("allegorisch", "ADJ"), "barocke": ("barock", "ADJ"),
    "kontextualisierende": ("kontextualisierend", "ADJ"), "vergleichende": ("vergleichend", "ADJ"),
    "zugrunde": ("zugrunde", "ADV"),
    "obgleich": ("obgleich", "SCONJ"), "obwohl": ("obwohl", "SCONJ"),
    "wenngleich": ("wenngleich", "SCONJ"), "insofern": ("insofern", "ADV"),
    "inwiefern": ("inwiefern", "ADV"), "gleichwohl": ("gleichwohl", "ADV"),
    "nichtsdestotrotz": ("nichtsdestotrotz", "ADV"),
    "EuGH": ("EuGH", "PROPN"), "Grundgesetz": ("Grundgesetz", "NOUN"),
    "neunzehnten": ("neunzehnt", "ADJ"),
}

VERB_OVERRIDES: dict[str, str] = {
    "erfordert": "erfordern", "erweist": "erweisen", "argumentiert": "argumentieren",
    "bleibt": "bleiben", "bedarf": "bedürfen", "fand": "finden", "vermag": "vermögen",
    "weist": "hinweisen", "ergeben": "ergeben", "erscheint": "erscheinen", "lässt": "lassen",
    "wurde": "werden", "debattiert": "debattieren", "wirkt": "wirken", "verbinden": "verbinden",
    "hängt": "hängen", "korreliert": "korrelieren", "belegen": "belegen", "stützt": "stützen",
    "erleichtert": "erleichtern", "entzieht": "entziehen", "vermochte": "vermögen",
    "herbeizuführen": "herbeiführen", "widerlegen": "widerlegen", "fehlt": "fehlen",
    "stieß": "stoßen", "erübrigen": "erübrigen", "legt": "darlegen", "prägen": "prägen",
    "erfuhr": "erfahren", "erfassen": "erfassen", "entkräften": "entkräften",
    "liefert": "liefern", "deuten": "deuten", "beurteilen": "beurteilen",
    "zeichnet": "auszeichnen", "erlauben": "erlauben", "plausibilisieren": "plausibilisieren",
    "nahelegt": "nahelegen", "aufweist": "aufweisen", "aufzuzeigen": "aufzeigen",
    "eröffnet": "eröffnen", "kontextualisiert": "kontextualisieren",
    "beleuchtet": "beleuchten", "betrifft": "betreffen", "steige": "steigen",
    "präzisierte": "präzisieren", "herangezogen": "heranziehen", "absichert": "sichern",
    "überzeugt": "überzeugen",
}

SCONJ_WORDS = {
    "dass", "weil", "obwohl", "wenn", "ob", "als", "wie", "damit",
    "obgleich", "wenngleich",
}
CCONJ_WORDS = {"und", "oder", "aber", "sondern"}
ADP_WORDS = {
    "um", "für", "mit", "von", "bei", "nach", "aus", "über", "vor", "unter",
    "durch", "ohne", "gegen", "wegen", "in", "an", "auf", "zu", "trotz",
    "angesichts", "infolge", "gegenüber", "als",
}
PART_WORDS = {"nicht", "ja", "nein"}

CONTRACTION_PAIRS: dict[tuple[str, str], tuple[str, str, str]] = {
    ("in", "dem"): ("im", "in", "ADP"),
    ("an", "dem"): ("am", "an", "ADP"),
    ("zu", "dem"): ("zum", "zu", "ADP"),
    ("zu", "der"): ("zur", "zu", "ADP"),
    ("von", "dem"): ("vom", "von", "ADP"),
    ("bei", "dem"): ("beim", "bei", "ADP"),
    ("in", "das"): ("ins", "in", "ADP"),
    ("an", "das"): ("ans", "an", "ADP"),
}


def simple_tokenize(sentence: str) -> list[str]:
    forms: list[str] = []
    for word in sentence.split():
        match = re.match(r"^(.+?)([.,;:!?]+)$", word)
        if match:
            forms.append(match.group(1))
            forms.extend(list(match.group(2)))
        else:
            forms.append(word)
    return forms


def _merge_hyphen_token(
    expected: str, stanza_words: list, start: int
) -> tuple[str, str, str, int] | None:
    if "-" not in expected:
        return None
    built = ""
    idx = start
    content_words = []
    while idx < len(stanza_words):
        w = stanza_words[idx]
        if w.text == "-":
            built += "-"
            idx += 1
        else:
            built += w.text
            content_words.append(w)
            idx += 1
            if built == expected:
                head = content_words[-1]
                upos = head.upos or "NOUN"
                if upos in {"PROPN", "NOUN", "X"}:
                    upos = "NOUN"
                    lemma = expected
                else:
                    lemma = head.lemma or expected
                return expected, upos, lemma, idx - start
            if len(built) > len(expected):
                return None
    return None


def align_stanza_to_text(
    sentence: str, stanza_words: list
) -> list[tuple[str, str, str]]:
    expected = simple_tokenize(sentence)
    aligned: list[tuple[str, str, str]] = []
    si = 0
    for exp in expected:
        if si >= len(stanza_words):
            raise ValueError(f"Stanza tokens exhausted for '{exp}' in: {sentence}")

        sw = stanza_words[si]
        if sw.text == exp:
            aligned.append((exp, sw.upos or "X", sw.lemma or exp))
            si += 1
            continue

        merged = _merge_hyphen_token(exp, stanza_words, si)
        if merged:
            form, upos, lemma, consumed = merged
            aligned.append((form, upos, lemma))
            si += consumed
            continue

        if si + 1 < len(stanza_words):
            pair = (stanza_words[si].text.lower(), stanza_words[si + 1].text.lower())
            if pair in CONTRACTION_PAIRS:
                form, lemma, upos = CONTRACTION_PAIRS[pair]
                if form == exp:
                    aligned.append((form, upos, lemma))
                    si += 2
                    continue

        aligned.append((exp, sw.upos or "X", sw.lemma or exp))
        si += 1

    if si != len(stanza_words):
        raise ValueError(
            f"Unconsumed Stanza tokens {si}/{len(stanza_words)} in: {sentence}"
        )
    return aligned


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    if form in SPECIAL_LEMMAS:
        return SPECIAL_LEMMAS[form]

    low = form.lower()

    if upos == "PUNCT":
        return form, "PUNCT"

    if low in SEIN_FORMS:
        return "sein", "AUX"
    if low in HABEN_FORMS:
        return "haben", "AUX"
    if low in WERDEN_FORMS:
        return "werden", "AUX"
    if low in MODAL_FORMS:
        return MODAL_FORMS[low], "AUX"

    if form in VERB_OVERRIDES:
        return VERB_OVERRIDES[form], "VERB"
    if lemma in VERB_OVERRIDES:
        return VERB_OVERRIDES[lemma], "VERB"

    if low in SCONJ_WORDS:
        return low, "SCONJ"
    if low in CCONJ_WORDS:
        return low, "CCONJ"
    if low in PART_WORDS:
        return low, "PART"
    if low == "zu" and upos in {"PART", "ADP"}:
        if upos == "PART":
            return "zu", "PART"
        return "zu", "ADP"
    if low in ADP_WORDS or upos == "ADP":
        return low, "ADP"

    if form in {"ich", "du", "er", "sie", "es", "wir", "ihr", "man"}:
        return form.lower(), "PRON"
    if form == "Sie":
        return "Sie", "PRON"
    if form == "Es":
        return "es", "PRON"

    if upos == "DET" or (upos == "PRON" and low in {"der", "die", "das", "den", "dem", "des"}):
        if low in {"der", "die", "das", "den", "dem", "des"}:
            return "der", "DET"
        if low in {"ein", "eine", "einen", "einem", "einer", "eines"}:
            return "ein", "DET"

    if upos == "NOUN":
        lem = lemma[0].upper() + lemma[1:] if lemma else form
        return lem, "NOUN"

    if upos == "VERB":
        lem = lemma.lower()
        if form in VERB_OVERRIDES:
            lem = VERB_OVERRIDES[form]
        return lem, "VERB"

    if upos == "ADJ":
        return lemma.lower(), "ADJ"

    if upos == "PROPN":
        lem = lemma[0].upper() + lemma[1:] if lemma else form
        return lem, "PROPN"

    if upos == "ADV":
        return lemma.lower() if lemma else low, "ADV"

    if upos == "AUX":
        if lemma in {"sein", "haben", "werden"}:
            return lemma, "AUX"
        if low in MODAL_FORMS:
            return MODAL_FORMS[low], "AUX"
        return lemma.lower(), "VERB"

    return lemma, upos


def build_conllu(sentences: list[str], nlp) -> str:
    output_lines: list[str] = []

    for idx, sent in enumerate(sentences):
        sent_num = START_ID + idx
        sent_id = f"{ID_PREFIX}_{sent_num:03d}"
        output_lines.append(f"# sent_id = {sent_id}")
        output_lines.append(f"# text = {sent}")

        doc = nlp(sent)
        stanza_words = [
            w for s in doc.sentences for w in s.words if isinstance(w.id, int)
        ]
        aligned = align_stanza_to_text(sent, stanza_words)

        for token_counter, (form, upos, lemma) in enumerate(aligned, 1):
            lemma, upos = normalize_token(form, upos, lemma)
            cols = [
                str(token_counter), form, lemma, upos,
                "_", "_", "_", "_", "_", "_",
            ]
            output_lines.append("\t".join(cols))

        output_lines.append("")

    return "\n".join(output_lines) + "\n"


def main() -> None:
    import stanza

    print("Loading Stanza...")
    nlp = stanza.Pipeline(lang="de", processors="tokenize,mwt,pos,lemma", use_gpu=False)

    print("Generating CoNLL-U...")
    conllu_text = build_conllu(SENTENCES, nlp)

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    TARGET_PATH.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {TARGET_PATH}")

    val = validate_text(conllu_text)
    lem = check_text(conllu_text, lang="de")
    print(f"COUNT {val.sentence_count} VAL {val.passed} LEM {lem.passed}")
    if val.errors:
        print("VAL ERRORS:")
        for e in val.errors[:30]:
            print(f"  {e}")
    if lem.errors:
        print("LEM ERRORS:")
        for e in lem.errors[:30]:
            print(f"  {e}")

    if not val.passed or not lem.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()