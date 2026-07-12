"""Generate C1 German CoNLL-U training data: de_c1_train_781 through de_c1_train_900."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

BATCH_1 = [
    # Administrative & EU law 781-805
    "Die Verwaltungsgerichtsbarkeit prüft, ob die Eilverfügung der Behörde verhältnismäßig und hinreichend begründet war.",
    "Es bedarf klarer Leitplanken, damit algorithmische Verwaltungsentscheidungen rechtsstaatlich überprüfbar bleiben.",
    "Die Europäische Kommission leitete ein Vertragsverletzungsverfahren ein, weil Mitgliedstaaten EU-Recht systematisch nicht umsetzten.",
    "Obwohl die Anhörung öffentlich verlief, blieben zentrale interne Prüfungsunterlagen der Behörde unzugänglich.",
    "Der Verwaltungsakt muss Rechtsfolgen, Fristen und Rechtsbehelfe für Betroffene ausdrücklich und verständlich benennen.",
    "Es ist umstritten, ob die Rückwirkungsklausel mit dem Vertrauensschutzprinzip vereinbar ist.",
    "Die Subsidiaritätsprüfung soll verhindern, dass die Union Kompetenzen ohne hinreichende Rechtfertigung ausweitet.",
    "Trotz eingehender Stellungnahmen konnte die Kommission keinen einvernehmlichen Kompromiss mit dem Mitgliedstaat erzielen.",
    "Die Akteneinsicht stellt ein zentrales Instrument der verwaltungsrechtlichen Kontrolle staatlicher Entscheidungen dar.",
    "Es lässt sich nicht leugnen, dass digitale Bürgerportale die Transparenz behördlicher Verfahren deutlich erhöhen können.",
    "Der EuGH klärte, dass Datenschutzauflagen auch bei grenzüberschreitenden behördlichen Datenübermittlungen gelten.",
    "Viele Verwaltungsrechtler kritisieren, dass Ermessensentscheidungen in der Praxis zu wenig begründet werden.",
    "Die Begründungspflicht verlangt, dass wesentliche Tatsachen und rechtliche Erwägungen nachvollziehbar dargestellt werden.",
    "Angesichts komplexer Regulierungsketten müssen Zuständigkeiten zwischen Bund, Ländern und Kommunen präzise geklärt werden.",
    "Es ist ratsam, verwaltungsrechtliche Risiken bereits in frühen Phasen der Digitalisierungsprojekte systematisch zu bewerten.",
    "Die verwaltungsgerichtliche Kontrolle beschränkt sich auf Rechtswidrigkeit, nicht auf die sachliche Zweckmäßigkeit von Entscheidungen.",
    "Ungeachtet politischer Dringlichkeit bleibt die Einhaltung formeller Verfahrensrechte ein verfassungsrechtlicher Mindeststandard.",
    "Der Verwaltungsbericht verdeutlicht, dass Bürgerbeschwerden häufig an unzureichender Kommunikation der Behörden scheitern.",
    "Es bleibt offen, ob einheitliche EU-Standards für algorithmische Entscheidungssysteme bis 2028 verbindlich werden.",
    "Die Rechtsmittelbelehrung muss Betroffene über Fristen, Zuständigkeiten und erforderliche Formvorschriften zuverlässig informieren.",
    "Es ist absehbar, dass die Digitalisierung der Verwaltung neue Anforderungen an Beweissicherung und Dokumentation stellt.",
    "Die Verhältnismäßigkeitsprüfung erfordert eine Abwägung zwischen Zweck, Eignung und Erforderlichkeit staatlicher Eingriffe.",
    "Trotz verbesserter Onlinezugänge bleibt der persönliche Ansprechpartner für vulnerable Gruppen unverzichtbar.",
    "Es bedarf verbindlicher Transparenzstandards, damit automatisierte Verwaltungsverfahren für Bürger nachvollziehbar bleiben.",
    "Die unionsrechtliche Staatshaftung kann greifen, wenn nationale Behörden EU-Vorschriften vorsätzlich oder fahrlässig verletzen.",
]

BATCH_2 = [
    # Labor market & digital economy 806-830
    "Die Arbeitsmarktpolitik muss strukturelle Umbrüche durch Qualifizierung und regionale Förderprogramme sozial abfedern.",
    "Es ist erwiesen, dass Plattformarbeit neue Herausforderungen für soziale Absicherung und tarifvertragliche Standards schafft.",
    "Die Tarifparteien verhandelten über flexible Arbeitszeitmodelle, ohne die betriebliche Mitbestimmungsrechte zu schwächen.",
    "Obwohl die Einstellungszahlen stiegen, blieb die Lohnungleichheit zwischen Branchen und Regionen hoch.",
    "Die Digitalisierung der Produktion erfordert Investitionen in Weiterbildung und moderne betriebliche Infrastruktur.",
    "Es bedarf klarer Regeln, damit algorithmische Personalauswahl diskriminierungsfrei und überprüfbar gestaltet wird.",
    "Der Arbeitsmarktbericht zeigt, dass Fachkräftemangel insbesondere technische Berufe und Pflegeberufe betrifft.",
    "Trotz sinkender Arbeitslosigkeit bleibt die Langzeitarbeitslosigkeit in strukturschwachen Regionen ein zentrales Problem.",
    "Die Mindestlohnkommission empfahl eine moderate Anpassung angesichts gestiegener Lebenshaltungskosten und Inflationsraten.",
    "Es lässt sich nicht bestreiten, dass Homeoffice-Regelungen Produktivität und Work-Life-Balance differenziert beeinflussen.",
    "Die betriebliche Transformation erfordert eine enge Abstimmung zwischen Management, Betriebsrat und Beschäftigtenvertretungen.",
    "Viele Ökonomen plädieren für gezielte Aktivierungspolitik statt pauschaler Deregulierung am Arbeitsmarkt.",
    "Die Analyse der Teilzeitquoten deutet auf anhaltende Geschlechterunterschiede in Führungspositionen hin.",
    "Es ist von zentraler Bedeutung, dass Weiterbildungsangebote arbeitsmarktferne Gruppen gezielt erreichen.",
    "Die Plattformökonomie verschiebt Verantwortung für Arbeitsbedingungen zwischen Auftraggebern und digitalen Vermittlern.",
    "Angesichts demografischer Veränderungen müssen Rentensysteme und Erwerbsbiografien langfristig aufeinander abgestimmt werden.",
    "Die Gewerkschaft betonte, dass Outsourcing nicht als Umgehung kollektiver Arbeitnehmerrechte genutzt werden darf.",
    "Es bleibt fraglich, ob kurzfristige Zuwanderungsprogramme den Fachkräftebedarf in Kernbranchen nachhaltig decken.",
    "Die Einführung einer Vier-Tage-Woche erfordert belastbare Evaluierungen von Produktivität und Personalkosten.",
    "Ungeachtet konjunktureller Schwankungen bleibt die soziale Partnerschaft ein stabiler Rahmen der Arbeitsmarktpolitik.",
    "Der Digitalpakt für Arbeit soll kleine Unternehmen bei der Einführung sicherer und produktiver IT-Systeme unterstützen.",
    "Es ist ratsam, Plattformunternehmen stärker zur Mitwirkung an beruflicher Qualifizierung verpflichtend heranzuziehen.",
    "Die Arbeitszeitflexibilisierung darf gesundheitliche Belastungen und Überstunden nicht systematisch in Betrieben normalisieren.",
    "Es ist abzuwarten, ob KI-gestützte Produktivitätsgewinne zu breiteren Lohnsteigerungen in Dienstleistungsberufen führen.",
    "Die regionale Arbeitsmarktbeobachtung identifizierte Engpässe in grünen Technologien und digitalen Schlüsselberufen.",
]

BATCH_3 = [
    # Clinical research & health policy 831-855
    "Die randomisierte kontrollierte Studie untersuchte die Wirksamkeit einer kombinierten medikamentösen und psychotherapeutischen Behandlung.",
    "Es ist erwiesen, dass Früherkennungsprogramme die Mortalität bei bestimmten Krebsarten statistisch signifikant senken können.",
    "Die klinische Leitlinie empfiehlt eine individualisierte Therapieentscheidung auf Basis validierter Biomarker und Komorbiditäten.",
    "Obwohl die Zulassungsstudien vielversprechend waren, bleiben Langzeitnebenwirkungen des neuen Wirkstoffs unzureichend dokumentiert.",
    "Die Gesundheitsökonomie bewertet, ob innovative Therapien im Verhältnis zu ihren Mehrkosten vertretbar sind.",
    "Es bedarf transparenter Register, um Selektionsbias in klinischen Studien frühzeitig zu erkennen und zu korrigieren.",
    "Die Versorgungsforschung analysiert, warum medizinische Standards in ländlichen Regionen langsamer implementiert werden.",
    "Trotz verbesserter Diagnostik gelang es nicht, alle nosokomialen Infektionsketten vollständig zu unterbrechen.",
    "Die evidenzbasierte Medizin verlangt eine systematische Integration von Studiendaten, klinischer Erfahrung und Patientenpräferenzen.",
    "Es lässt sich nicht leugnen, dass soziale Determinanten die Ergebnisse chronischer Erkrankungen erheblich mitprägen.",
    "Die Pharmakovigilanz meldete unerwartete Nebenwirkungen, die eine Neubewertung des Risiko-Nutzen-Profils erforderlich machen.",
    "Viele Kliniker kritisieren, dass administrative Dokumentationspflichten die direkte Patientenversorgung übermäßig belasten.",
    "Die Metaanalyse zeigte moderate Effektstärken, deren klinische Relevanz kontextabhängig sorgfältig interpretiert werden muss.",
    "Es ist von essenzieller Bedeutung, dass Placebokontrollen ethisch vertretbar und wissenschaftlich stringent ausgestaltet sind.",
    "Die Public-Health-Strategie kombiniert Prävention, Früherkennung und strukturierte Versorgungspfade für chronische Erkrankungen.",
    "Angesichts steigender Multimorbidität müssen integrierte Versorgungsmodelle interdisziplinär finanziert und evaluiert werden.",
    "Der Ethikrat betonte, dass genomische Daten besonderen Schutz vor Diskriminierung und unbefugter Weitergabe verdienen.",
    "Es bleibt offen, inwieweit telemedizinische Angebote die Versorgungsqualität in unterversorgten Regionen dauerhaft verbessern.",
    "Die Studienprotokolle legten vorab fest, welche sekundären Endpunkte statistisch und klinisch relevant ausgewertet werden.",
    "Ungeachtet hoher Innovationskosten bleibt der Zugang zu wirksamen Therapien ein zentrales Thema der Gerechtigkeitspolitik.",
    "Die Impfstoffevaluierung berücksichtigte sowohl immunologische Wirksamkeit als auch logistische Anforderungen der Massenverteilung.",
    "Es ist ratsam, klinische Entscheidungsunterstützung transparent zu dokumentieren und regelmäßig auf Bias zu prüfen.",
    "Die Versorgungsstatistik deutet auf regionale Unterschiede bei Wartezeiten für spezialisierte fachärztliche Behandlungen hin.",
    "Es ist absehbar, dass Präzisionsmedizin Diagnose und Therapie zunehmend personalisiert und datenintensiver gestalten wird.",
    "Die Validierung diagnostischer Tests ist Voraussetzung für belastbare Screeningprogramme in der Bevölkerungsmedizin.",
]

BATCH_4 = [
    # International security & diplomacy 856-880
    "Die Sicherheitsstrategie betont, dass hybride Bedrohungen diplomatische, wirtschaftliche und militärische Instrumente kombinieren.",
    "Es ist unbestritten, dass Vertrauensbildung zwischen Konfliktparteien langfristige Verhandlungsprozesse erheblich erleichtert.",
    "Der Friedensvermittler schlug vor, Waffenruhen zunächst in humanitär kritischen Korridoren verlässlich zu überwachen.",
    "Obwohl Sanktionen verschärft wurden, blieb der Zugang zu lebensnotwendigen Gütern für Zivilbevölkerung begrenzt.",
    "Die internationalen Abrüstungsgespräche erfordern verifizierbare Inspektionsmechanismen und transparente Berichterstattung über Rüstungsbestände.",
    "Es bedarf abgestimmter diplomatischer Initiativen, damit regionale Spannungen nicht in offene militärische Konfrontation eskalieren.",
    "Die NATO-Analyse warnt vor Destabilisierungsversuchen durch Desinformation und gezielte Angriffe auf kritische Infrastruktur.",
    "Trotz internationaler Resolutionen gelang es nicht, alle humanitären Zugänge zu belagerten Gebieten dauerhaft zu sichern.",
    "Die Krisendiplomatie verbindet militärische Zurückhaltung mit gezielten wirtschaftlichen Anreizen für Deeskalation.",
    "Es lässt sich nicht leugnen, dass Rüstungsexporte die Sicherheitslage in fragilen Regionen zusätzlich verschärfen können.",
    "Die multilaterale Friedensmission soll zivile Strukturen schützen, ohne die Souveränität der Gaststaaten zu untergraben.",
    "Viele Strategen plädieren für eine stärkere europäische Rolle in präventiver Konfliktmediation und ziviler Krisenreaktion.",
    "Die Geheimdienstkooperation erfordert klare rechtsstaatliche Kontrollen und parlamentarische Nachverfolgung sensibler Operationen.",
    "Es ist abzuwarten, ob neue Sicherheitsabkommen die Rüstungsdynamik in der Region tatsächlich nachhaltig dämpfen.",
    "Die humanitäre Völkerrechtskommission dokumentierte systematische Verstöße gegen den Schutz nichtkombattierender Zivilpersonen.",
    "Angesichts cyberoperativer Bedrohungen müssen Verteidigungsstrategien digitale Resilienz und internationale Normbildung integrieren.",
    "Der Außenminister betonte, dass Entwicklungspolitik und Sicherheitspolitik in fragilen Staaten untrennbar miteinander verknüpft sind.",
    "Ungeachtet taktischer Fortschritte bleibt ein politisches Gesamtkonzept für die Nachkriegsordnung weiterhin umstritten.",
    "Die strategische Kommunikation soll Fehlinformationen entkräften, ohne legitime politische Kritik zu diskreditieren.",
    "Es ist ratsam, Rüstungskontrollregime regelmäßig an technologischen Entwicklungen und neuen Waffensystemen anzupassen.",
    "Die Sicherheitskooperation im Indopazifik erfordert abgestimmte maritime Präsenz und verlässliche Bündnispflichten.",
    "Es bleibt fraglich, ob wirtschaftliche Entspannung ohne verbindliche Demokratisierungsfortschritte langfristig stabilisierend wirkt.",
    "Die Friedensforschung analysiert, wie institutionelle Reformen nach bewaffneten Konflikten Rückfallrisiken verringern können.",
    "Der Lagebericht verdeutlicht, dass Migrationsbewegungen häufig mit klimabedingten und sicherheitspolitischen Faktoren zusammenhängen.",
    "Die nukleare Abschreckungsdebatte erfordert eine nüchterne Abwägung von Risiken, Kosten und strategischer Stabilität.",
]

BATCH_5 = [
    # Research methodology & science policy 881-900
    "Die Forschungsmethodik verlangt, dass Hypothesen präregistriert und Datenanalysen unabhängig reproduzierbar dokumentiert werden.",
    "Es steht außer Frage, dass offene Forschungsdaten die wissenschaftliche Qualität und internationale Zusammenarbeit stärken.",
    "Die Peer-Review-Kultur soll konstruktive Kritik fördern, ohne junge Forschende übermäßig zu demotivieren.",
    "Obwohl die Stichprobe repräsentativ war, blieben methodische Limitationen in der Diskussion transparent zu benennen.",
    "Die nationale Wissenschaftspolitik investiert verstärkt in interdisziplinäre Forschungsinfrastrukturen und langfristige Großgeräteprojekte.",
    "Es bedarf fairer Evaluierungskriterien, damit qualitativ hochwertige Forschung nicht allein an Publikationszahlen gemessen wird.",
    "Die Replikationskrise hat zu strengeren Standards für statistische Poweranalysen und offene Materialien geführt.",
    "Trotz knapper Haushaltsmittel bleibt die Grundlagenforschung ein zentraler Motor technologischer und gesellschaftlicher Innovation.",
    "Die Forschungsethikkommission prüft, ob die Einwilligung der Probanden informiert, freiwillig und dokumentiert erfolgte.",
    "Es lässt sich nicht bestreiten, dass Forschungsfreiheit und gesellschaftliche Verantwortung in Spannung stehen können.",
    "Die Metawissenschaft untersucht, wie Anreizsysteme die Reproduzierbarkeit und Robustheit empirischer Befunde beeinflussen.",
    "Viele Wissenschaftler kritisieren, dass Drittmittelwettbewerbe strukturell kleinere Institute und neue Fachgebiete benachteiligen.",
    "Die Datenmanagementpläne sollen FAIR-Prinzipien umsetzen und langfristige Nachnutzung durch andere Forschende ermöglichen.",
    "Es ist ratsam, KI-gestützte Analyseverfahren methodisch zu validieren und potenzielle Verzerrungen offen zu berichten.",
    "Die Wissenschaftskommunikation muss Unsicherheiten, Grenzen und Vorläufigkeit empirischer Erkenntnisse verständlich vermitteln.",
    "Angesichts globaler Herausforderungen müssen internationale Forschungskooperationen geopolitische Risiken und Sicherheitsinteressen berücksichtigen.",
    "Der Evaluationsbericht verdeutlicht, dass Diversity-Programme die Innovationskraft akademischer Institutionen messbar stärken können.",
    "Es bleibt offen, ob Open-Access-Modelle langfristig tragfähige Finanzierungswege für qualitätsgesicherte Publikationen bieten.",
    "Die Forschungsförderung priorisiert Projekte, die gesellschaftliche Transformation, Klimaschutz und digitale Souveränität verbinden.",
    "Die interdisziplinäre Zusammenarbeit erfordert gemeinsame Methodenstandards und klare Regeln zur Autorenschaft und Datennutzung.",
]

ALL_BATCHES = [BATCH_1, BATCH_2, BATCH_3, BATCH_4, BATCH_5]
SENTENCES = [s for batch in ALL_BATCHES for s in batch]
assert len(SENTENCES) == 120, f"Expected 120 sentences, got {len(SENTENCES)}"
for i, batch in enumerate(ALL_BATCHES, 1):
    assert len(batch) == (25 if i < 5 else 20), f"BATCH_{i} has {len(batch)} sentences"

START_ID = 781
TARGET_PATH = project_root / "data/handcraft/de/train/c1_new_006.conllu"

MODAL_FORMS = {
    "kann": "können", "könnte": "können", "konnte": "können", "können": "können", "könnt": "können",
    "muss": "müssen", "müsste": "müssen", "musste": "müssen", "müssen": "müssen", "müsse": "müssen",
    "soll": "sollen", "sollte": "sollen", "sollen": "sollen",
    "darf": "dürfen", "dürfte": "dürfen", "dürfen": "dürfen",
    "will": "wollen", "wollte": "wollen", "wollen": "wollen",
    "mag": "mögen", "möchte": "mögen", "mögen": "mögen",
}

SEIN_FORMS = {"ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen", "sei", "seien"}
HABEN_FORMS = {"habe", "hat", "hatte", "hätte", "hätten", "haben", "habt", "gehabt", "hätte"}
WERDEN_FORMS = {"wird", "wurde", "werden", "würde", "würden", "geworden", "worden"}

SPECIAL_LEMMAS: dict[str, tuple[str, str]] = {
    "mich": ("ich", "PRON"), "dich": ("du", "PRON"), "sich": ("sich", "PRON"),
    "uns": ("wir", "PRON"), "euch": ("ihr", "PRON"), "mir": ("ich", "PRON"),
    "dir": ("du", "PRON"), "ihm": ("er", "PRON"), "ihnen": ("sie", "PRON"),
    "Ihnen": ("Sie", "PRON"), "Sie": ("Sie", "PRON"),
    "mein": ("mein", "DET"), "meine": ("mein", "DET"), "meinen": ("mein", "DET"),
    "meinem": ("mein", "DET"), "meiner": ("mein", "DET"), "meines": ("mein", "DET"),
    "dein": ("dein", "DET"), "deine": ("dein", "DET"), "deinen": ("dein", "DET"),
    "deinem": ("dein", "DET"), "deiner": ("dein", "DET"), "deines": ("dein", "DET"),
    "sein": ("sein", "DET"), "seine": ("sein", "DET"), "seinen": ("sein", "DET"),
    "seinem": ("sein", "DET"), "seiner": ("sein", "DET"), "seines": ("sein", "DET"),
    "unser": ("unser", "DET"), "unsere": ("unser", "DET"), "unseren": ("unser", "DET"),
    "unserem": ("unser", "DET"), "unserer": ("unser", "DET"), "unseres": ("unser", "DET"),
    "der": ("der", "DET"), "die": ("der", "DET"), "das": ("der", "DET"),
    "den": ("der", "DET"), "dem": ("der", "DET"), "des": ("der", "DET"),
    "ein": ("ein", "DET"), "eine": ("ein", "DET"), "einen": ("ein", "DET"),
    "einem": ("ein", "DET"), "einer": ("ein", "DET"), "eines": ("ein", "DET"),
    "alle": ("alle", "DET"), "viele": ("viel", "DET"), "vielen": ("viel", "DET"),
    "vieler": ("vieler", "DET"), "mehrere": ("mehrere", "DET"), "einige": ("einig", "DET"),
    "ihren": ("ihr", "DET"), "ihre": ("ihr", "DET"), "ihrem": ("ihr", "DET"),
    "ihrer": ("ihr", "DET"), "ihr": ("ihr", "DET"),
    "zur": ("zu", "ADP"), "zum": ("zu", "ADP"), "beim": ("bei", "ADP"),
    "im": ("in", "ADP"), "am": ("an", "ADP"), "ans": ("an", "ADP"),
    "ins": ("in", "ADP"), "vom": ("von", "ADP"),
    "darauf": ("darauf", "ADV"), "daraus": ("daraus", "ADV"), "darin": ("darin", "ADV"),
    "hervor": ("hervor", "ADV"), "zurück": ("zurück", "ADV"), "hin": ("hin", "ADV"),
    "ab": ("ab", "ADP"), "dar": ("dar", "ADV"),
    "nicht": ("nicht", "PART"), "zu": ("zu", "PART"),
    "heute": ("heute", "ADV"), "gestern": ("gestern", "ADV"), "morgen": ("morgen", "ADV"),
    "bereits": ("bereits", "ADV"), "noch": ("noch", "ADV"), "auch": ("auch", "ADV"),
    "nur": ("nur", "ADV"), "sehr": ("sehr", "ADV"), "deutlich": ("deutlich", "ADV"),
    "erheblich": ("erheblich", "ADV"), "spürbar": ("spürbar", "ADV"),
    "häufig": ("häufig", "ADV"), "letztlich": ("letztlich", "ADV"),
    "nachhaltig": ("nachhaltig", "ADV"), "angemessen": ("angemessen", "ADV"),
    "möglich": ("möglich", "ADJ"), "zeitnah": ("zeitnah", "ADV"),
    "strittig": ("strittig", "ADJ"), "verfassungswidrig": ("verfassungswidrig", "ADJ"),
    "neuen": ("neu", "ADJ"), "neue": ("neu", "ADJ"), "neuer": ("neu", "ADJ"),
    "öffentlich": ("öffentlich", "ADV"), "hinreichend": ("hinreichend", "ADV"),
    "systematisch": ("systematisch", "ADV"), "ausdrücklich": ("ausdrücklich", "ADV"),
    "verständlich": ("verständlich", "ADV"), "zentral": ("zentral", "ADJ"),
    "zentrale": ("zentral", "ADJ"), "interne": ("intern", "ADJ"), "internen": ("intern", "ADJ"),
    "algorithmische": ("algorithmisch", "ADJ"), "algorithmischen": ("algorithmisch", "ADJ"),
    "verwaltungsrechtliche": ("verwaltungsrechtlich", "ADJ"),
    "verwaltungsrechtlichen": ("verwaltungsrechtlich", "ADJ"),
    "verwaltungsgerichtliche": ("verwaltungsgerichtlich", "ADJ"),
    "strukturschwachen": ("strukturschwach", "ADJ"),
    "strukturelle": ("strukturell", "ADJ"), "langfristige": ("langfristig", "ADJ"),
    "langfristig": ("langfristig", "ADV"), "langfristigen": ("langfristig", "ADJ"),
    "grenzüberschreitenden": ("grenzüberschreitend", "ADJ"),
    "wesentliche": ("wesentlich", "ADJ"), "komplexer": ("komplex", "ADJ"),
    "verfassungsrechtlicher": ("verfassungsrechtlich", "ADJ"),
    "automatisierte": ("automatisiert", "ADJ"), "automatisierten": ("automatisiert", "ADJ"),
    "unionsrechtliche": ("unionsrechtlich", "ADJ"), "vorsätzlich": ("vorsätzlich", "ADV"),
    "fahrlässig": ("fahrlässig", "ADV"), "sozial": ("sozial", "ADV"),
    "tarifvertragliche": ("tarifvertraglich", "ADJ"), "betriebliche": ("betriebliche", "ADJ"),
    "betrieblichen": ("betriebliche", "ADJ"), "flexible": ("flexibel", "ADJ"),
    "technische": ("technisch", "ADJ"), "technischen": ("technisch", "ADJ"),
    "diskriminierungsfrei": ("diskriminierungsfrei", "ADV"), "moderate": ("moderat", "ADJ"),
    "gestiegener": ("gestiegen", "ADJ"), "anhaltende": ("anhaltend", "ADJ"),
    "arbeitsmarktferne": ("arbeitsmarktfern", "ADJ"), "demografischer": ("demografisch", "ADJ"),
    "kurzfristige": ("kurzfristig", "ADJ"), "grünen": ("grün", "ADJ"),
    "digitalen": ("digital", "ADJ"), "medikamentösen": ("medikamentös", "ADJ"),
    "psychotherapeutischen": ("psychotherapeutisch", "ADJ"), "individualisierte": ("individualisiert", "ADJ"),
    "innovative": ("innovativ", "ADJ"), "ländlichen": ("ländlich", "ADJ"),
    "chronischer": ("chronisch", "ADJ"), "chronische": ("chronisch", "ADJ"),
    "unerwartete": ("unerwartet", "ADJ"), "administrative": ("administrativ", "ADJ"),
    "moderate": ("moderat", "ADJ"), "klinische": ("klinisch", "ADJ"),
    "klinischen": ("klinisch", "ADJ"), "klinischer": ("klinisch", "ADJ"),
    "statistisch": ("statistisch", "ADV"), "ethisch": ("ethisch", "ADV"),
    "integrierte": ("integriert", "ADJ"), "integrierten": ("integriert", "ADJ"),
    "interdisziplinär": ("interdisziplinär", "ADV"), "interdisziplinäre": ("interdisziplinär", "ADJ"),
    "genomische": ("genomisch", "ADJ"), "telemedizinische": ("telemedizinisch", "ADJ"),
    "sekundären": ("sekundär", "ADJ"), "hoher": ("hoch", "ADJ"),
    "spezialisierte": ("spezialisiert", "ADJ"), "fachärztliche": ("fachärztlich", "ADJ"),
    "hybride": ("hybrid", "ADJ"), "militärische": ("militärisch", "ADJ"),
    "militärischen": ("militärisch", "ADJ"), "humanitären": ("humanitär", "ADJ"),
    "humanitärer": ("humanitär", "ADJ"), "humanitäre": ("humanitär", "ADJ"),
    "lebensnotwendigen": ("lebensnotwendig", "ADJ"), "verifizierbare": ("verifizierbar", "ADJ"),
    "internationaler": ("international", "ADJ"), "internationale": ("international", "ADJ"),
    "belagerten": ("belagert", "ADJ"), "zivile": ("zivil", "ADJ"), "zivilen": ("zivil", "ADJ"),
    "europäische": ("europäisch", "ADJ"), "präventiver": ("präventiv", "ADJ"),
    "sensibler": ("sensibel", "ADJ"), "cyberoperativer": ("cyberoperativ", "ADJ"),
    "digitale": ("digital", "ADJ"), "taktischer": ("taktisch", "ADJ"),
    "strategische": ("strategisch", "ADJ"), "strategischen": ("strategisch", "ADJ"),
    "verbindliche": ("verbindlich", "ADJ"), "maritime": ("maritim", "ADJ"),
    "wirtschaftliche": ("wirtschaftlich", "ADJ"), "institutionelle": ("institutionell", "ADJ"),
    "klimabedingten": ("klimabedingt", "ADJ"), "sicherheitspolitischen": ("sicherheitspolitisch", "ADJ"),
    "nukleare": ("nuklear", "ADJ"), "offene": ("offen", "ADJ"), "offenen": ("offen", "ADJ"),
    "repräsentativ": ("repräsentativ", "ADJ"), "interdisziplinäre": ("interdisziplinär", "ADJ"),
    "qualitativ": ("qualitativ", "ADV"), "statistische": ("statistisch", "ADJ"),
    "offene": ("offen", "ADJ"), "gesellschaftliche": ("gesellschaftlich", "ADJ"),
    "empirischer": ("empirisch", "ADJ"), "empirische": ("empirisch", "ADJ"),
    "geopolitische": ("geopolitisch", "ADJ"), "messbar": ("messbar", "ADV"),
    "qualitätsgesicherte": ("qualitätsgesichert", "ADJ"),
    "EU": ("EU", "PROPN"), "EuGH": ("EuGH", "PROPN"), "NATO": ("NATO", "PROPN"),
    "EU-Recht": ("EU-Recht", "NOUN"), "Homeoffice": ("Homeoffice", "NOUN"),
    "KI": ("KI", "NOUN"), "FAIR": ("FAIR", "PROPN"), "Open-Access": ("Open-Access", "NOUN"),
    "Peer-Review": ("Peer-Review", "NOUN"), "Public-Health": ("Public-Health", "NOUN"),
    "Work-Life-Balance": ("Work-Life-Balance", "NOUN"),
}

VERB_OVERRIDES: dict[str, str] = {
    "gilt": "gelten", "erwiesen": "erweisen", "bedarf": "bedürfen",
    "hob": "heben", "wies": "weisen", "ließ": "lassen", "lässt": "lassen",
    "betonte": "betonen", "plädieren": "plädieren", "plädiert": "plädieren",
    "erfordert": "erfordern", "deutet": "deuten", "verdeutlicht": "verdeutlichen",
    "verdeutlichte": "verdeutlichen", "bleibt": "bleiben", "bleiben": "bleiben",
    "erklärte": "erklären", "verletzte": "verletzen", "erzielen": "erzielen",
    "erzielt": "erzielen", "bietet": "bieten", "begrenzen": "begrenzen",
    "begrenzt": "begrenzen", "verpflichtet": "verpflichten", "begründen": "begründen",
    "überarbeitet": "überarbeiten", "stützt": "stützen", "stützten": "stützen",
    "verändert": "verändern", "verändern": "verändern", "einräumen": "einräumen",
    "erstreckt": "erstrecken", "könnten": "können", "prüft": "prüfen",
    "gewährleisten": "gewährleisten", "legte": "legen", "formuliert": "formulieren",
    "unterliegen": "unterliegen", "dämpft": "dämpfen", "senken": "senken",
    "erwarten": "erwarten", "wirken": "wirken", "zeigen": "zeigen",
    "variieren": "variieren", "belasten": "belasten", "beeinflusst": "beeinflussen",
    "beeinflussen": "beeinflussen", "hängt": "hängen", "bewirken": "bewirken",
    "verringert": "verringern", "verzeichnet": "verzeichnen", "überschreiten": "überschreiten",
    "abbaut": "abbauen", "legt": "legen", "empfahl": "empfehlen", "ausrichten": "ausrichten",
    "steigt": "steigen", "entschärft": "entschärfen", "erschwert": "erschweren",
    "dämpfen": "dämpfen", "untersuchte": "untersuchen", "moduliert": "modulieren",
    "erklärt": "erklären", "lieferte": "liefern", "lieferten": "liefern",
    "kritisieren": "kritisieren", "übertraf": "übertreffen", "begünstigt": "begünstigen",
    "identifizierte": "identifizieren", "ausbauen": "ausbauen", "berücksichtigt": "berücksichtigen",
    "übersetzen": "übersetzen", "zeigte": "zeigen", "verbesserte": "verbessern",
    "erfassen": "erfassen", "verzerren": "verzerren", "verweist": "verweisen",
    "erschweren": "erschweren", "verschlechterte": "verschlechtern",
    "blieben": "bleiben", "spielt": "spielen", "profitieren": "profitieren",
    "harmonisiert": "harmonisieren", "wuchs": "wachsen", "reduziert": "reduzieren",
    "destabilisieren": "destabilisieren", "verlangen": "verlangen", "einschränken": "einschränken",
    "bewerten": "bewerten", "sichern": "sichern", "stieg": "steigen",
    "verlängern": "verlängern", "aushandeln": "aushandeln", "entgegenwirken": "entgegenwirken",
    "ersetzen": "ersetzen", "erleichterte": "erleichtern", "verbindet": "verbinden",
    "zeigten": "zeigen", "trennen": "trennen", "betont": "betonen",
    "formulieren": "formulieren", "finanziert": "finanzieren",
    "berücksichtigen": "berücksichtigen", "stellt": "stellen", "fragen": "fragen",
    "ermöglicht": "ermöglichen", "abbilden": "abbilden", "kündigte": "kündigen",
    "erreichen": "erreichen", "ist": "sein", "offenzulegen": "offenlegen",
    "billigte": "billigen", "genießen": "genießen", "betreffen": "betreffen",
    "fordern": "fordern", "zielen": "zielen", "angepasst": "anpassen",
    "stärken": "stärken", "helfen": "helfen", "einigten": "einigen", "stärkt": "stärken",
    "übersetzt": "übersetzen", "beschleunigen": "beschleunigen", "sparen": "sparen",
    "entspricht": "entsprechen", "verlagern": "verlagern",
    "argumentiert": "argumentieren", "dürfen": "dürfen", "steht": "stehen",
    "kennzeichnen": "kennzeichnen", "wies": "weisen", "setzt": "setzen",
    "erarbeiten": "erarbeiten", "beeinträchtigen": "beeinträchtigen",
    "beschleunigt": "beschleunigen", "erfolgen": "erfolgen", "beiträgt": "beitragen",
    "fragmentieren": "fragmentieren", "erhöhen": "erhöhen", "verlangt": "verlangen",
    "informierte": "informieren", "verzahnen": "verzahnen",
    "verstärken": "verstärken", "umstellen": "umstellen",
    "umfasst": "umfassen", "integrieren": "integrieren",
    "auslösen": "auslösen", "verbessert": "verbessern",
    "prägen": "prägen", "vermitteln": "vermitteln", "eröffnet": "eröffnen",
    "abgesichert": "absichern", "geworden": "werden", "identifizieren": "identifizieren",
    "verhindern": "verhindern", "bestätigt": "bestätigen", "antizipieren": "antizipieren",
    "haben": "haben", "bleibe": "bleiben", "darf": "dürfen",
    "müsse": "müssen", "sollen": "sollen", "werden": "werden", "wird": "werden",
    "könnten": "können", "könne": "können", "muss": "müssen",
    "korreliert": "korrelieren", "leitete": "leiten", "umsetzten": "umsetzen",
    "verlief": "verlaufen", "benennen": "benennen", "ausweitet": "ausweiten",
    "erhöhen": "erhöhen", "klärte": "klären", "gelten": "gelten",
    "dargestellt": "darstellen", "geklärt": "klären", "bewerten": "bewerten",
    "beschränkt": "beschränken", "scheitern": "scheitern", "informieren": "informieren",
    "stellt": "stellen", "greifen": "greifen", "verletzen": "verletzen",
    "abfädern": "abfädern", "schafft": "schaffen", "verhandelten": "verhandeln",
    "schwächen": "schwächen", "betrifft": "betreffen", "beeinflussen": "beeinflussen",
    "erreichen": "erreichen", "verschiebt": "verschieben", "abgestimmt": "abstimmen",
    "nutzt": "nutzen", "decken": "decken", "erfordert": "erfordern",
    "unterstützen": "unterstützen", "heranzuziehen": "heranziehen",
    "normalisieren": "normalisieren", "führen": "führen", "identifizierte": "identifizieren",
    "senken": "senken", "empfiehlt": "empfehlen", "dokumentiert": "dokumentieren",
    "implementiert": "implementieren", "unterbrechen": "unterbrechen",
    "mitprägen": "mitprägen", "meldete": "melden", "machen": "machen",
    "interpretiert": "interpretieren", "ausgestaltet": "ausgestalten",
    "evaluiert": "evaluieren", "verbessern": "verbessern", "legten": "legen",
    "ausgewertet": "auswerten", "berücksichtigte": "berücksichtigen",
    "verteilung": "verteilen", "prüfen": "prüfen", "gestalten": "gestalten",
    "kombinieren": "kombinieren", "erleichtert": "erleichtern",
    "schlug": "schlagen", "überwachen": "überwachen", "verschärft": "verschärfen",
    "eskaliieren": "eskalieren", "warnt": "warnen", "sichern": "sichern",
    "verschärfen": "verschärfen", "plädieren": "plädieren",
    "dämpfen": "dämpfen", "dokumentierte": "dokumentieren",
    "integrieren": "integrieren", "verknüpft": "verknüpfen",
    "entkräften": "entkräften", "anzupassen": "anpassen",
    "verringern": "verringern", "zusammenhängen": "zusammenhängen",
    "verlangt": "verlangen", "dokumentiert": "dokumentieren",
    "fördern": "fördern", "investiert": "investieren", "geführt": "führen",
    "erfolgte": "erfolgen", "stehen": "stehen", "benachteiligen": "benachteiligen",
    "umsetzen": "umsetzen", "validieren": "validieren", "berichten": "berichten",
    "berücksichtigen": "berücksichtigen", "stärken": "stärken",
    "bieten": "bieten", "verbinden": "verbinden", "war": "sein",
}

SCONJ_WORDS = {
    "dass", "weil", "obwohl", "wenn", "ob", "als", "wie", "damit", "inwieweit",
}
CCONJ_WORDS = {"und", "oder", "aber", "sondern"}
ADP_WORDS = {
    "um", "für", "mit", "von", "bei", "nach", "aus", "über", "vor", "unter",
    "durch", "ohne", "gegen", "wegen", "in", "an", "auf", "zu", "trotz",
    "angesichts", "ungeachtet", "infolge", "gegenüber", "ab",
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

        if exp.endswith("'") and si + 1 < len(stanza_words):
            base = exp[:-1]
            if stanza_words[si].text == base and stanza_words[si + 1].text == "'":
                aligned.append((exp, stanza_words[si].upos or "PROPN", base))
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
        if not (lem.endswith("en") or lem.endswith("n")):
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


def count_tokens(sentence: str) -> int:
    return len(simple_tokenize(sentence))


def build_conllu(sentences: list[str], nlp) -> str:
    output_lines: list[str] = []
    token_warnings: list[str] = []

    for idx, sent in enumerate(sentences):
        sent_id = f"de_c1_train_{START_ID + idx}"
        tc = count_tokens(sent)
        if tc < 12 or tc > 20:
            token_warnings.append(f"{sent_id}: {tc} tokens — {sent}")

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

    if token_warnings:
        print("TOKEN COUNT WARNINGS:")
        for w in token_warnings:
            print(f"  {w}")
        raise ValueError(f"{len(token_warnings)} sentences outside 12-20 token range")

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
        print("VAL ERRORS:", val.errors[:20])
    if lem.errors:
        print("LEM ERRORS:", lem.errors[:20])

    if not val.passed or not lem.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()