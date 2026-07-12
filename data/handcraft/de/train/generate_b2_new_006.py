"""Generate b2_new_006.conllu: 200 B2 German sentences (de_b2_train_734–933)."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 8 batches × 25 sentences = 200
BATCHES: list[list[str]] = [
    # Batch 1 (734–758): Globalization
    [
        "Die Globalisierung hat die wirtschaftlichen Beziehungen zwischen den Kontinenten grundlegend verändert.",
        "Obwohl der Welthandel wächst, profitieren nicht alle Regionen gleichermaßen von der Öffnung der Märkte.",
        "Es wird befürchtet, dass der Protektionismus die internationale Zusammenarbeit nachhaltig schwächen werde.",
        "Die Verlagerung von Produktionsstätten ins Ausland führt zu Debatten über Arbeitsplätze und Lohnniveau.",
        "Obschon die Lieferketten effizienter geworden sind, sind sie zugleich anfälliger für globale Störungen.",
        "Der Ökonom erklärte, die Abhängigkeit von ausländischen Rohstoffen nehme in vielen Branchen zu.",
        "Die Integration der Weltmärkte erfordert gemeinsame Standards und verlässliche Handelsabkommen.",
        "Es sei wichtig, dass Entwicklungsländer fair am globalen Wirtschaftssystem teilhaben können.",
        "Die Ausweitung des internationalen Finanzverkehrs wurde von Regulierungsbehörden genau beobachtet.",
        "Würde man Handelshemmnisse abbauen, könnte der Austausch von Waren und Dienstleistungen deutlich zunehmen.",
        "Obwohl die Konzerne weltweit agieren, bleiben nationale Steuerregelungen oft unkoordiniert.",
        "Es wird erwartet, dass die nächste Welthandelsrunde neue Regeln für digitale Güter festlegen werde.",
        "Die Verbreitung globaler Marken hat das Konsumverhalten in vielen Gesellschaften nachhaltig geprägt.",
        "Hätte man früher auf faire Handelsbedingungen gedrungen, wäre die Kluft geringer geworden.",
        "Der Bericht dokumentiert, wie sich die Migration von Fachkräften auf lokale Arbeitsmärkte auswirkt.",
        "Die Stärkung multilateraler Institutionen gilt als Voraussetzung für eine stabile Weltwirtschaft.",
        "Obschon die Transportkosten gesunken sind, belasten Umweltauflagen die globalen Logistikketten.",
        "Es mangelt nicht an politischem Willen, sondern an verbindlichen internationalen Durchsetzungsmechanismen.",
        "Die Rolle multinationaler Unternehmen bei der Gestaltung globaler Normen wird kontrovers diskutiert.",
        "Der Außenhandelsminister betonte, offene Märkte dürften nicht mit sozialer Verantwortung in Konflikt stehen.",
        "Die Auswirkungen der Globalisierung auf lokale Kulturen werden in der Forschung intensiv untersucht.",
        "Es werde angenommen, dass regionale Wirtschaftsräume künftig an strategischer Bedeutung gewinnen werden.",
        "Obwohl die Zölle gesenkt wurden, bleiben technische Handelsbarrieren für viele Exporteure bestehen.",
        "Die Förderung fairer Lieferketten soll die Arbeitsbedingungen in Produktionsländern spürbar verbessern.",
        "Die Anpassung nationaler Wirtschaftspolitik an globale Entwicklungen bleibt eine zentrale Herausforderung.",
    ],
    # Batch 2 (759–783): Human Rights
    [
        "Die Menschenrechte gelten als universelles Fundament jeder demokratischen Rechtsordnung.",
        "Obwohl viele Staaten die Konvention unterzeichnet haben, werden Grundrechte weltweit noch immer verletzt.",
        "Es wird erwartet, dass das neue Gesetz den Schutz von Minderheiten deutlich stärken werde.",
        "Die Gewährleistung freier Meinungsäußerung ist ein unverzichtbarer Bestandteil der Menschenwürde.",
        "Obschon die Berichte alarmierend sind, fehlt es oft an wirksamen Sanktionsmöglichkeiten.",
        "Der Beauftragte erklärte, die Lage in Gefängnissen habe sich in einigen Regionen verschlechtert.",
        "Die Bekämpfung von Zwangsarbeit erfordert internationale Kooperation und konsequente Strafverfolgung.",
        "Es sei unerlässlich, dass Flüchtlinge rechtsstaatlich und menschenwürdig behandelt werden.",
        "Die Durchsetzung des Verbots von Folter wurde von Menschenrechtsorganisationen nachdrücklich gefordert.",
        "Würde man Opfer besser schützen, könnte die Hemmschwelle für Straftaten möglicherweise steigen.",
        "Obwohl die Resolution verabschiedet wurde, bleibt ihre praktische Umsetzung in vielen Ländern fraglich.",
        "Es wird befürchtet, dass die Einschränkung von Versammlungsfreiheit demokratische Prozesse nachhaltig schwäche.",
        "Die Stärkung der Rechte von Menschen mit Behinderung ist gesetzlich in vielen Staaten verankert.",
        "Hätte die Staatengemeinschaft früher eingegriffen, wäre das Leid vieler Zivilisten geringer ausgefallen.",
        "Der Bericht weist darauf hin, dass Kinderarbeit in bestimmten Sektoren noch immer verbreitet sei.",
        "Die Überwachung der Menschenrechtslage durch unabhängige Beobachter gilt als unverzichtbar.",
        "Obschon Fortschritte erzielt wurden, bleibt die Gleichstellung der Geschlechter ein drängendes Anliegen.",
        "Es mangelt nicht an internationalen Normen, sondern an der Bereitschaft zur konsequenten Durchsetzung.",
        "Die Schaffung sicherer Korridore für humanitäre Hilfe wurde als dringende Maßnahme bezeichnet.",
        "Der Richter betonte, niemand dürfe willkürlich inhaftiert oder ohne faires Verfahren verurteilt werden.",
        "Die Ausweitung des Asylrechts wird in der öffentlichen Debatte emotional und kontrovers geführt.",
        "Es werde empfohlen, die Rechte indigener Völker stärker in nationale Gesetzgebung aufzunehmen.",
        "Obwohl die Pressefreiheit garantiert ist, geraten unabhängige Journalisten in vielen Ländern unter Druck.",
        "Die Verurteilung diskriminierender Praktiken soll ein klares Signal an alle Beteiligten senden.",
        "Die Wahrung der Menschenwürde bleibt auch in Krisenzeiten eine nicht verhandelbare Verpflichtung des Staates.",
    ],
    # Batch 3 (784–808): Scientific Ethics
    [
        "Die ethische Bewertung neuer Forschungsmethoden erfordert eine sorgfältige Abwägung aller Risiken.",
        "Obwohl die Studie vielversprechend ist, dürfen die Teilnehmer nicht ohne informierte Zustimmung eingebunden werden.",
        "Es wird betont, dass wissenschaftliche Integrität die Grundlage jeder glaubwürdigen Publikation bilde.",
        "Die Durchführung von Tierversuchen muss strengen ethischen und rechtlichen Vorgaben entsprechen.",
        "Obschon die Daten wertvoll sind, stellt die Verwendung genetischer Informationen erhebliche ethische Fragen.",
        "Der Ethikrat erklärte, die Grenzen der Forschung an menschlichen Embryonen seien klar zu ziehen.",
        "Die Vermeidung von Datenfälschung erfordert transparente Verfahren und unabhängige Kontrollmechanismen.",
        "Es sei wichtig, dass Forscherinnen und Forscher mögliche Interessenkonflikte offenlegen und dokumentieren.",
        "Die Anwendung künstlicher Intelligenz in der Medizin wirft Fragen nach Verantwortung und Transparenz auf.",
        "Würde man die Langzeitfolgen ignorieren, könnten unbeabsichtigte Schäden für Patienten entstehen.",
        "Obwohl die Finanzierung gesichert war, wurde das Projekt wegen ethischer Bedenken vorübergehend ausgesetzt.",
        "Es wird erwartet, dass neue Richtlinien für klinische Studien in Kürze veröffentlicht werden.",
        "Die Einhaltung der Datenschutzstandards bei Forschungsprojekten wurde von der Kommission streng geprüft.",
        "Hätte man die Probanden besser aufgeklärt, wäre die Studie ethisch unbedenklicher verlaufen.",
        "Der Bericht empfiehlt, die öffentliche Debatte über Gentechnik sachlich und evidenzbasiert zu führen.",
        "Die Verantwortung der Wissenschaft gegenüber der Gesellschaft wird in Zeiten schneller Innovation diskutiert.",
        "Obschon die Technologie verfügbar ist, bleibt ihre Anwendung am Menschen ethisch höchst umstritten.",
        "Es mangelt nicht an Regeln, sondern an ihrer konsequenten Umsetzung in allen Forschungseinrichtungen.",
        "Die Überprüfung von Publikationen durch unabhängige Gutachter gilt als zentrales Qualitätsmerkmal.",
        "Der Professor betonte, wissenschaftlicher Fortschritt dürfe nicht auf Kosten grundlegender ethischer Prinzipien gehen.",
        "Die Entwicklung von Impfstoffen muss sowohl wirksam als auch gerecht auf der ganzen Welt zugänglich sein.",
        "Es werde angenommen, dass ethische Beratungsgremien künftig eine noch größere Rolle spielen werden.",
        "Obwohl die Ergebnisse beeindruckend sind, mahnt die Fachwelt zu größerer Vorsicht bei der Interpretation.",
        "Die Förderung offener Wissenschaft soll Manipulationen erschweren und die Reproduzierbarkeit von Studien erhöhen.",
        "Die Achtung der Würde von Versuchspersonen bleibt auch bei dringend benötigten medizinischen Durchbrüchen oberste Priorität.",
    ],
    # Batch 4 (809–833): Urbanization
    [
        "Die Urbanisierung verändert das Gesicht vieler Metropolen in einem bisher kaum gekannten Tempo.",
        "Obwohl die Städte wachsen, fehlt es an bezahlbarem Wohnraum für einkommensschwache Haushalte.",
        "Es wird erwartet, dass die Mehrheit der Menschen künftig in urbanen Räumen leben werde.",
        "Die Planung nachhaltiger Verkehrssysteme ist entscheidend für die Lebensqualität in dicht besiedelten Räumen.",
        "Obschon neue Stadtteile entstehen, droht die soziale Segregation in vielen Ballungszentren weiter zuzunehmen.",
        "Der Stadtplaner erklärte, Grünflächen müssten bei jedem Neubauprojekt verbindlich berücksichtigt werden.",
        "Die Sanierung verfallener Industriegebiete bietet Chancen für gemischte Wohn- und Arbeitsnutzungen.",
        "Es sei ratsam, den öffentlichen Nahverkehr auszubauen, bevor weitere Außenbezirke erschlossen werden.",
        "Die Verdichtung innerstädtischer Flächen soll den Flächenverbrauch und lange Pendelstrecken verringern.",
        "Würde man mehr in sozialen Wohnungsbau investieren, könnte die Wohnungsnot spürbar gelindert werden.",
        "Obwohl die Infrastruktur modernisiert wurde, kommt der Ausbau der Digitalnetze in Randlagen nur langsam voran.",
        "Es wird befürchtet, dass steigende Mieten viele langjährige Bewohner aus ihren Vierteln verdrängen werden.",
        "Die Entwicklung fußläufiger Quartiere trägt dazu bei, den Autoverkehr im Zentrum merklich zu reduzieren.",
        "Hätte man früher auf nachhaltige Mobilität gesetzt, wären die Luftbelastungen geringer ausgefallen.",
        "Der Bericht dokumentiert, wie sich die Hitzeinseln in dicht bebauten Gebieten in den Sommermonaten verstärken.",
        "Die Beteiligung der Anwohner an städtebaulichen Entscheidungen wird als Element guter Stadtpolitik angesehen.",
        "Obschon die Bevölkerungszahl steigt, bleibt der Erhalt historischer Stadtkerne eine große Herausforderung.",
        "Es mangelt nicht an Konzepten, sondern an ausreichend finanziellen Mitteln für ihre konsequente Umsetzung.",
        "Die Förderung von Mikromobilität soll kurze Wege im Alltag effizienter und umweltfreundlicher gestalten.",
        "Der Bürgermeister betonte, lebenswerte Städte dürften nicht nur Wirtschaftszentren, sondern müssten Wohnräume sein.",
        "Die Anpassung der Kanalisation an häufigere Starkregenereignisse erfordert langfristige Investitionsplanung.",
        "Es werde empfohlen, leerstehende Geschäftsflächen gezielt in Wohnraum umzuwandeln.",
        "Obwohl neue Satellitenstädte geplant sind, bleibt die Anbindung an das regionale Kernnetz unzureichend.",
        "Die Schaffung öffentlicher Begegnungsräume soll das Zusammenleben in heterogenen Stadtgesellschaften stärken.",
        "Die Bewältigung des Wachstums ohne weitere Zerstörung wertvoller Landschaftsflächen bleibt eine Kernaufgabe.",
    ],
    # Batch 5 (834–858): Demographic Change
    [
        "Der demografische Wandel stellt die Sozialsysteme und den Arbeitsmarkt vor grundlegende Anpassungen.",
        "Obwohl die Lebenserwartung steigt, wächst die Zahl pflegebedürftiger Menschen schneller als erwartet.",
        "Es wird angenommen, dass der Anteil älterer Menschen in den kommenden Jahrzehnten deutlich zunehmen werde.",
        "Die Sicherung des Fachkräfteangebots erfordert gezielte Maßnahmen in Bildung, Zuwanderung und Weiterbildung.",
        "Obschon die Geburtenrate niedrig bleibt, hat die Bevölkerung in einigen Regionen durch Zuwanderung zugenommen.",
        "Der Demograf erklärte, ländliche Gebiete würden stärker von Abwanderung und Überalterung betroffen sein.",
        "Die Vereinbarkeit von Familie und Beruf gilt als Schlüssel zu einer stabilen demografischen Entwicklung.",
        "Es sei wichtig, dass ältere Arbeitnehmer ihre Erfahrung länger und flexibler im Erwerbsleben einbringen können.",
        "Die Anpassung der Rentensysteme an steigende Lebenserwartung wurde von Experten als unumgänglich bezeichnet.",
        "Würde man mehr in Kinderbetreuung investieren, könnte die Vereinbarkeit für junge Eltern deutlich verbessert werden.",
        "Obwohl die Stadt wächst, schrumpfen viele Gemeinden im ländlichen Raum seit Jahren kontinuierlich.",
        "Es wird befürchtet, dass der Pflegenotstand die Lebensqualität vieler Betroffener nachhaltig beeinträchtigen werde.",
        "Die Förderung des generationsübergreifenden Wohnens soll Einsamkeit im Alter wirksamer entgegenwirken.",
        "Hätte man früher auf präventive Gesundheitsprogramme gesetzt, wären Pflegekosten möglicherweise geringer.",
        "Der Bericht zeigt, wie sich die Altersstruktur auf die Nachfrage nach medizinischer Versorgung auswirkt.",
        "Die Stärkung der Silver Economy wird als Chance für Innovation und neue Beschäftigungsfelder gesehen.",
        "Obschon Zuwanderung diskutiert wird, bleibt ihre gesellschaftliche Akzeptanz von vielen Faktoren abhängig.",
        "Es mangelt nicht an politischen Debatten, sondern an langfristig tragfähigen Lösungen für den Arbeitsmarkt.",
        "Die Verbesserung der Arbeitsbedingungen in Pflegeberufen soll den dramatischen Fachkräftemangel mildern.",
        "Der Minister betonte, der demografische Wandel dürfe nicht zur Last alleiniger Generationen werden.",
        "Die Entwicklung altersgerechter Wohnformen erfordert Zusammenarbeit von Kommunen, Bauwirtschaft und Pflegeanbietern.",
        "Es werde erwartet, dass die Erwerbsbeteiligung älterer Menschen in den nächsten Jahren weiter steigen werde.",
        "Obwohl die Sterberate stabil blieb, veränderte die Pandemie die Wahrnehmung von Vorsorge und Risiko.",
        "Die Förderung von Familien mit mehreren Kindern wird als Beitrag zur langfristigen Stabilität diskutiert.",
        "Die Gestaltung einer alterssicheren Gesellschaft bleibt eine zentrale Aufgabe für Politik und Wirtschaft.",
    ],
    # Batch 6 (859–883): Energy Transition
    [
        "Die Energiewende erfordert einen umfassenden Umbau von Erzeugung, Netzen und Verbrauchsmustern.",
        "Obwohl der Ausbau erneuerbarer Energien voranschreitet, hängt die Versorgungssicherheit noch von fossilen Quellen ab.",
        "Es wird erwartet, dass die Strompreise bis zur vollständigen Dekarbonisierung volatil bleiben werden.",
        "Die Reduzierung der Treibhausgasemissionen ist das zentrale Ziel nationaler und europäischer Klimapolitik.",
        "Obschon viele Konzepte vorliegen, verzögert sich der Netzausbau in mehreren Bundesländern erheblich.",
        "Der Energieminister erklärte, der Ausstieg aus der Kohle müsse sozialverträglich und planbar gestaltet werden.",
        "Die Speicherung grüner Energie stellt die Forschung vor technische und wirtschaftliche Herausforderungen.",
        "Es sei notwendig, den Energieverbrauch in Gebäuden durch Sanierung und Effizienzmaßnahmen deutlich zu senken.",
        "Die Förderung von Wasserstoffprojekten wurde im Haushalt mit zusätzlichen Mitteln ausgestattet.",
        "Würde man Genehmigungsverfahren beschleunigen, könnte der Ausbau der Windkraft merklich vorankommen.",
        "Obwohl die Solaranlagen zulegen, fehlt es an ausreichender Kapazität im überregionalen Übertragungsnetz.",
        "Es wird befürchtet, dass Verzögerungen beim Netzausbau die Klimaziele des Landes gefährden werden.",
        "Die Umstellung der Industrie auf klimaneutrale Prozesse erfordert hohe Investitionen und Planungssicherheit.",
        "Hätte man früher in Netzinfrastruktur investiert, wären Engpässe bei der Stromverteilung geringer.",
        "Der Bericht dokumentiert, wie sich der Kohleausstieg auf strukturschwache Regionen auswirken könnte.",
        "Die Stärkung der Bürgerenergiegenossenschaften gilt als Weg zu mehr Akzeptanz vor Ort.",
        "Obschon die Technologie verfügbar ist, bleibt die Finanzierung großer Offshoreprojekte komplex.",
        "Es mangelt nicht an politischem Bekenntnis, sondern an der Geschwindigkeit praktischer Umsetzung.",
        "Die Einführung dynamischer Stromtarife soll den Verbrauch flexibler an das Angebot anpassen.",
        "Der Experte betonte, die Energiewende dürfe die Versorgungssicherheit unter keinen Umständen gefährden.",
        "Die Sanierung alter Heizungsanlagen wird durch Zuschüsse und Beratungsangebote für Haushalte erleichtert.",
        "Es werde angenommen, dass Elektromobilität und erneuerbare Energien künftig eng miteinander verzahnt werden.",
        "Obwohl die Emissionen sinken, bleibt der Verkehrssektor eine der größten Klimaschutzbaustellen.",
        "Die Ausweitung von Geothermie und Biomasse soll die Energiemixvielfalt in Regionen erhöhen.",
        "Die Erreichung der Klimaneutralität bis Mitte des Jahrhunderts erfordert entschlossenes Handeln aller Akteure.",
    ],
    # Batch 7 (884–908): Digital Privacy
    [
        "Der Schutz der digitalen Privatsphäre gewinnt angesichts umfassender Datensammlung zunehmend an Bedeutung.",
        "Obwohl neue Gesetze verabschiedet wurden, bleibt die Kontrolle großer Plattformen eine große Herausforderung.",
        "Es wird erwartet, dass strengere Vorgaben für den Umgang mit personenbezogenen Daten in Kraft treten werden.",
        "Die Transparenz algorithmischer Entscheidungen ist für viele Nutzer ein zentrales Anliegen im digitalen Alltag.",
        "Obschon Verschlüsselung verfügbar ist, speichern viele Dienste sensible Informationen in unzureichend geschützter Form.",
        "Der Datenschutzbeauftragte erklärte, Unternehmen müssten Datenminimierung konsequent in ihre Prozesse integrieren.",
        "Die Einwilligung der Nutzer muss verständlich, freiwillig und jederzeit widerrufbar gewährt werden.",
        "Es sei wichtig, dass Bürger erfahren, welche Profile über sie online erstellt werden.",
        "Die Überwachung des Onlineverhaltens durch Werbenetzwerke wirft erhebliche rechtliche und ethische Fragen auf.",
        "Würde man Tracking standardmäßig einschränken, könnte das Vertrauen in digitale Angebote spürbar steigen.",
        "Obwohl die App beliebt ist, wurde ihre Datenschutzerklärung von Verbraucherschützern als unklar kritisiert.",
        "Es wird befürchtet, dass staatliche Überwachungsinstrumente Grundrechte im Netz nachhaltig einschränken werden.",
        "Die Anonymisierung von Forschungsdaten soll den Schutz sensibler Informationen bei wissenschaftlichen Studien gewährleisten.",
        "Hätte man Sicherheitslücken früher geschlossen, wäre der Datenverlust für Millionen Nutzer vermeidbar gewesen.",
        "Der Bericht empfiehlt, Privacy by Design von Beginn an in die Entwicklung neuer Produkte einzubinden.",
        "Die Durchsetzung der Datenschutzgrundverordnung erfordert ausreichend Personal und wirksame Bußgelder.",
        "Obschon Nutzer mehr Kontrolle wünschen, gestalten sich technische Einstellungen oft zu komplex.",
        "Es mangelt nicht an rechtlichen Regeln, sondern an ihrer gleichmäßigen Durchsetzung im internationalen Raum.",
        "Die Beschränkung des Zugriffs auf biometrische Daten wurde als notwendiger Schritt begrüßt.",
        "Der Jurist betonte, digitale Grundrechte dürften nicht zugunsten kurzfristiger Sicherheitsinteressen eingeschränkt werden.",
        "Die Entwicklung dezentraler Speicherlösungen soll Abhängigkeiten von wenigen Cloud-Anbietern verringern.",
        "Es werde angenommen, dass künftige Geräte standardmäßig stärkere Schutzmechanismen für Nutzerdaten bieten werden.",
        "Obwohl Cookies abgelehnt werden können, bleibt die tatsächliche Kontrolle über Tracking oft begrenzt.",
        "Die Aufklärung über Phishing und Identitätsdiebstahl soll das Risiko finanzieller Schäden im Alltag senken.",
        "Die Wahrung der Privatsphäre im digitalen Zeitalter bleibt eine gemeinsame Verantwortung von Staat, Wirtschaft und Bürgern.",
    ],
    # Batch 8 (909–933): Cross-cutting themes
    [
        "Die Verknüpfung globaler Märkte mit Menschenrechtsschutz wird in internationalen Foren intensiv verhandelt.",
        "Obwohl Städte modernisiert werden, dürfen digitale Überwachungssysteme nicht die Grundrechte der Bewohner verletzen.",
        "Es wird erwartet, dass die Energiewende neue Arbeitsplätze schafft und zugleich alte Strukturen verändern werde.",
        "Die ethische Nutzung von Gesundheitsdaten erfordert klare Grenzen und unabhängige Kontrollinstanzen.",
        "Obschon die Bevölkerung altert, bietet die Digitalisierung Chancen für barrierefreie und sichere Dienstleistungen.",
        "Der Bericht erklärte, globale Lieferketten müssten sowohl ökologischen als auch sozialen Standards genügen.",
        "Die Förderung erneuerbarer Energien in wachsenden Metropolen gilt als Schlüssel zur Klimaverträglichkeit.",
        "Es sei wichtig, dass wissenschaftliche Erkenntnisse nicht ohne Rücksprache gegen die Würde von Personen eingesetzt werden.",
        "Die Verantwortung von Technologieunternehmen für den Schutz persönlicher Daten wird weltweit diskutiert.",
        "Würde man urbane Räume klimaneutral planen, könnten Emissionen und Lebensqualität gleichermaßen verbessert werden.",
        "Obwohl die Menschenrechtslage kritisch ist, setzen einige Staaten auf digitale Überwachung als Sicherheitsinstrument.",
        "Es wird befürchtet, dass demografischer Druck die Finanzierung öffentlicher Infrastruktur langfristig belasten werde.",
        "Die Verbindung von Forschungsethik und Datenschutz gewinnt bei internationalen Kooperationsprojekten an Bedeutung.",
        "Hätte man Datenschutz früher priorisiert, wäre das Vertrauen in digitale Verwaltungsangebote größer geworden.",
        "Der Experte betonte, nachhaltige Stadtentwicklung dürfe nicht auf Kosten sozial schwacher Gruppen erfolgen.",
        "Die Stärkung globaler Menschenrechtsstandards soll auch Unternehmen in internationale Lieferketten einbeziehen.",
        "Obschon die Energiewende teuer ist, werden die Folgen des Zögerns langfristig deutlich höher ausfallen.",
        "Es mangelt nicht an technischen Lösungen, sondern an Regeln für ihren verantwortungsvollen Einsatz.",
        "Die Entwicklung transparenter KI-Systeme soll Diskriminierung verhindern und die digitale Privatsphäre stärken.",
        "Der Kommentator erklärte, der demografische Wandel erfordere Städte, die für alle Generationen lebenswert seien.",
        "Die Auswirkungen der Globalisierung auf lokale Kultur und Identität werden in urbanen Milieus besonders sichtbar.",
        "Es werde empfohlen, ethische Richtlinien für Genomforschung international zu harmonisieren und durchzusetzen.",
        "Obwohl der Datenschutz verbessert wurde, bleibt die Kontrolle über sensible Gesundheitsdaten ein sensibles Thema.",
        "Die Gestaltung einer gerechten Energiewende soll auch einkommensschwache Haushalte vor übermäßigen Belastungen schützen.",
        "Die Balance zwischen technologischem Fortschritt und Grundrechten bleibt die zentrale Frage unserer Zeit.",
    ],
]

sentences: list[str] = [s for batch in BATCHES for s in batch]
assert len(sentences) == 200, f"Expected 200 sentences, got {len(sentences)}"

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
    """Map a surface token (possibly MWT) to its primary Stanza word."""
    ids = token.id if isinstance(token.id, tuple) else (token.id,)
    return words_by_id.get(ids[0])


output_lines: list[str] = []
start_id = 734

for idx, sent in enumerate(sentences):
    sent_id = f"de_b2_train_{start_id + idx}"
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

target_path = project_root / "data/handcraft/de/train/b2_new_006.conllu"
target_path.write_text(conllu_text, encoding="utf-8")
print(f"Wrote {len(sentences)} sentences to {target_path}")

validation_res = validate_text(conllu_text)
print(f"Validate: count={validation_res.sentence_count}, passed={validation_res.passed}")
if not validation_res.passed:
    for err in validation_res.errors[:10]:
        print("  ", err)
    sys.exit(1)

lemma_res = check_text(conllu_text, lang="de")
print(f"Lemma check: passed={lemma_res.passed}")
if not lemma_res.passed:
    for err in lemma_res.errors[:10]:
        print("  ", err)
    sys.exit(1)

print("All checks passed.")