"""Generate C1 German CoNLL-U training data: de_c1_train_381 through de_c1_train_580."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 8 batches of 25 sentences (381-580)
BATCH_1 = [
    # Economics 381-405
    "Die anhaltende Geldpolitik der Zentralbank beeinflusst die Kreditvergabe an mittelständische Unternehmen erheblich.",
    "Angesichts der steigenden Inflation fordern Ökonomen eine striktere Regulierung der Finanzmärkte.",
    "Die Volatilität der Rohstoffpreise stellt Exporteure vor erhebliche Planungsunsicherheiten im globalen Handel.",
    "Es ist absehbar, dass die Zinswende die Immobilienmärkte in Ballungsräumen nachhaltig dämpfen wird.",
    "Der Bericht verdeutlicht, dass strukturelle Reformen für nachhaltiges Wachstum unerlässlich geworden sind.",
    "Trotz konjunktureller Einbrüche bleibt die Arbeitslosenquote in der exportorientierten Industrie bemerkenswert niedrig.",
    "Die Verschuldung öffentlicher Haushalte überschreitet in mehreren Mitgliedstaaten die vereinbarten Stabilitätskriterien deutlich.",
    "Es lässt sich nicht bestreiten, dass der Fachkräftemangel die Wettbewerbsfähigkeit des Standorts gefährdet.",
    "Die Analyse der Konsumdaten deutet auf eine schleichende Veränderung der Ausgabengewohnheiten hin.",
    "Um dem Preisdruck entgegenzuwirken, müssen Unternehmen ihre Lieferketten strategisch diversifizieren.",
    "Die Privatisierung staatlicher Betriebe führte in der Vergangenheit häufig zu sozialen Spannungen.",
    "Es ist von zentraler Bedeutung, dass die Geldpolitik inflationäre Tendenzen frühzeitig erkennt und bekämpft.",
    "Die wachsende Ungleichheit bei der Vermögensverteilung erfordert eine grundlegende politische Diskussion.",
    "Viele erfahrene Analysten prognostizieren für das kommende Geschäftsjahr eine moderate Konjunkturerholung.",
    "Die schrittweise Einführung einer digitalen Zentralbankwährung könnte das Zahlungsverkehrssystem grundlegend transformieren.",
    "Infolge der Handelsbeschränkungen kam es zu erheblichen Verwerfungen in den internationalen Lieferketten.",
    "Der Wirtschaftsminister betonte, dass Investitionen in grüne Technologien prioritär gefördert werden müssten.",
    "Die Rendite staatlicher Anleihen spiegelt die Erwartungen der Märkte an die künftige Fiskalpolitik wider.",
    "Es bleibt abzuwarten, ob die getroffenen Maßnahmen die Inflation dauerhaft unter Kontrolle bringen.",
    "Die Liberalisierung des Energiemarktes hat in einigen Regionen zu deutlich sinkenden Strompreisen geführt.",
    "Ungeachtet der anhaltenden wirtschaftlichen Risiken setzen Investoren verstärkt auf nachhaltige Anlagestrategien.",
    "Die anhaltende Deflation bestimmter Sektoren könnte die gesamtwirtschaftliche Erholung erheblich verzögern.",
    "Es ist ratsam, vor größeren Investitionsentscheidungen die makroökonomischen Rahmenbedingungen sorgfältig zu prüfen.",
    "Die Konzentration des Marktes auf wenige Akteure untergräbt die Effizienz des Wettbewerbsmechanismus.",
    "Die Neuordnung der globalen Handelsbeziehungen stellt etablierte Wirtschaftsmodelle vor fundamentale Herausforderungen.",
]

BATCH_2 = [
    # Law 406-430
    "Die im aktuellen Gesetzentwurf vorgesehenen Regelungen widersprechen eindeutig grundlegenden verfassungsrechtlichen Prinzipien.",
    "Der Angeklagte berief sich auf seine Verfassungsrechte und beantragte die Aussetzung des Verfahrens.",
    "Es ist unbestritten, dass der Schutz persönlicher Daten ein fundamentales Grundrecht darstellt.",
    "Die jüngste Gerichtsentscheidung setzt einen Präzedenzfall für künftige Streitigkeiten im Arbeitsrecht.",
    "Obwohl die Beweislage äußerst komplex ist, plädiert die Staatsanwaltschaft auf eine Verurteilung.",
    "Die Revision des Strafgesetzbuches zielt darauf ab, Cyberkriminalität effektiver zu bekämpfen.",
    "Es bedarf einer umfassenden Reform des Verwaltungsverfahrens, um bürokratische Hemmnisse abzubauen.",
    "Der Kläger macht geltend, dass der Vertrag aufgrund wesentlicher Irrtümer anfechtbar sei.",
    "Die im Urteil angeführten Argumente überzeugen durch ihre stringente juristische Begründung.",
    "Trotz eingehender Verhandlungen konnte zwischen den Parteien kein außergerichtlicher Vergleich erzielt werden.",
    "Es ist von entscheidender Bedeutung, dass richterliche Entscheidungen unabhängig und unparteiisch erfolgen.",
    "Die Haftung des Auftragnehmers erstreckt sich auch auf Schäden, die durch Subunternehmer verursacht wurden.",
    "Viele Juristen kritisieren, dass die geltende Gesetzgebung digitale Geschäftsmodelle unzureichend regelt.",
    "Die Verjährungsfrist für zivilrechtliche Ansprüche beginnt mit Kenntnis des schädigenden Ereignisses.",
    "Es lässt sich nicht leugnen, dass die Durchsetzung von Verbraucherrechten erhebliche Ressourcen erfordert.",
    "Der österreichische Verfassungsgerichtshof erklärte mehrere Bestimmungen des neuen Sicherheitsgesetzes für verfassungswidrig.",
    "Angesichts der rechtlichen Unsicherheit empfiehlt es sich, vor Vertragsabschluss fachkundigen Rat einzuholen.",
    "Die Beweisaufnahme ergab, dass der Beklagte seine vertraglichen Pflichten vorsätzlich verletzt hatte.",
    "Es ist absehbar, dass die Digitalisierung der Justiz die Verfahrensdauer erheblich verkürzen wird.",
    "Die Kanzlei vertritt die Auffassung, dass der Schadensersatzanspruch verjährt ist.",
    "Unter Berücksichtigung aller Umstände erscheint eine Verurteilung wegen fahrlässiger Körperverletzung angemessen.",
    "Die Neuregelung des Mietrechts soll Mieter vor ungerechtfertigten Kündigungen besser schützen.",
    "Es ist zu befürchten, dass die Komplexität des Steuerrechts zu häufigen Fehlinterpretationen führt.",
    "Der Richter wies die Berufung zurück und bestätigte das Urteil des Landgerichts in vollem Umfang.",
    "Die im Vertrag festgelegten Haftungsbeschränkungen sind nach geltender Rechtsprechung grundsätzlich wirksam.",
]

BATCH_3 = [
    # Research 431-455
    "Die im Labor gewonnenen Ergebnisse stellen einen bedeutenden Durchbruch in der Krebsforschung dar.",
    "Es ist erwiesen, dass regelmäßige körperliche Aktivität die kognitive Leistungsfähigkeit im Alter erhält.",
    "Die Studie legt nahe, dass genetische Faktoren die Anfälligkeit für bestimmte Erkrankungen beeinflussen.",
    "Obwohl die Stichprobengröße begrenzt war, liefern die Daten wertvolle Erkenntnisse für die Epidemiologie.",
    "Die Forscher wiesen nach, dass die untersuchten Substanzen eine entzündungshemmende Wirkung aufweisen.",
    "Es bedarf weiterer Langzeitstudien, um die Langzeiteffekte der neuen Therapie zuverlässig zu bewerten.",
    "Die Publikation der Ergebnisse in einer renommierten Fachzeitschrift erhöht ihre wissenschaftliche Anerkennung.",
    "Trotz intensiver Bemühungen gelang es dem Team nicht, die ursprüngliche Hypothese experimentell zu bestätigen.",
    "Die Anwendung maschinellen Lernens revolutioniert die Auswertung großer medizinischer Datensätze erheblich.",
    "Es ist von essenzieller Wichtigkeit, dass Forschungsprojekte ethischen Grundsätzen strikt entsprechen.",
    "Die im Bericht dokumentierten Messwerte weichen in mehreren Punkten von den erwarteten Werten ab.",
    "Viele Wissenschaftler plädieren für eine stärkere Offenlegung von Forschungsdaten und Methoden.",
    "Die Entwicklung neuer Impfstoffe erfordert eine enge Zusammenarbeit zwischen Universitäten und Industrie.",
    "Es lässt sich nicht bestreiten, dass die Replikationskrise das Vertrauen in die Wissenschaft erschüttert hat.",
    "Die Analyse der Probandendaten ergab statistisch signifikante Unterschiede zwischen den Versuchsgruppen.",
    "Angesichts der begrenzten Finanzierung müssen Forschungseinrichtungen ihre Prioritäten strategisch neu setzen.",
    "Die im Experiment beobachteten Phänomene lassen sich mit den bisherigen theoretischen Modellen nicht erklären.",
    "Es ist ratsam, vor der Veröffentlichung die statistische Auswertung von unabhängigen Experten prüfen zu lassen.",
    "Die Erforschung der Quantenmechanik eröffnet völlig neue Perspektiven für die Materialwissenschaft.",
    "Ungeachtet der methodischen Einschränkungen bietet die Studie wichtige Ansatzpunkte für Folgeuntersuchungen.",
    "Die im Feld gesammelten Daten belegen einen deutlichen Rückgang der Artenvielfalt in urbanen Gebieten.",
    "Es ist absehbar, dass die Genomforschung personalisierte Medizin in naher Zukunft ermöglichen wird.",
    "Die Peer-Review-Verfahren müssen reformiert werden, um die Qualität wissenschaftlicher Publikationen zu sichern.",
    "Der Forschungsbericht verweist darauf, dass interdisziplinäre Ansätze besonders fruchtbare Ergebnisse liefern.",
    "Die Validierung der Messinstrumente ist Voraussetzung für die Aussagekraft jeder empirischen Untersuchung.",
]

BATCH_4 = [
    # Policy 456-480
    "Die Regierung kündigte an, dass die Klimaschutzmaßnahmen bis 2030 deutlich verschärft werden sollen.",
    "Es steht außer Frage, dass die Energiewende ohne massive Investitionen in erneuerbare Energien scheitern wird.",
    "Der Gesetzentwurf sieht vor, dass Unternehmen ihre CO2-Emissionen transparent offenlegen müssen.",
    "Obwohl erhebliche Bedenken geäußert wurden, stimmte das Parlament dem Reformpaket letztlich zu.",
    "Die im Koalitionsvertrag vereinbarten Ziele erfordern eine konsequente Umsetzung auf allen politischen Ebenen.",
    "Es ist von großer Bedeutung, dass Bildungspolitik Chancengleichheit für alle gesellschaftlichen Schichten gewährleistet.",
    "Die Verschärfung der Asylpolitik stieß bei Menschenrechtsorganisationen auf scharfe internationale Kritik.",
    "Trotz des Widerstands der Opposition wurde das Gesundheitsreformgesetz mit knapper Mehrheit verabschiedet.",
    "Es bedarf einer kohärenten Migrationspolitik, die wirtschaftliche Interessen und humanitäre Verpflichtungen vereint.",
    "Der Ministerpräsident betonte, dass die Digitalisierung der Verwaltung oberste politische Priorität genieße.",
    "Die im Haushaltsplan vorgesehenen Kürzungen betreffen vor allem den Bereich der sozialen Sicherungssysteme.",
    "Viele Kommunalpolitiker fordern eine stärkere finanzielle Unterstützung durch den Bund und die Länder.",
    "Die Einführung des Mindestlohns hat die Einkommenssituation zahlreicher Beschäftigter spürbar verbessert.",
    "Es lässt sich nicht leugnen, dass die politische Polarisierung die Gesetzgebungsfähigkeit des Parlaments beeinträchtigt.",
    "Die Neuausrichtung der Außenpolitik zielt darauf ab, internationale Partnerschaften strategisch zu stärken.",
    "Angesichts der beschleunigten demografischen Entwicklung müssen die Rentensysteme grundlegend reformiert werden.",
    "Die im Bericht empfohlenen Maßnahmen sollen die Integration von Zuwanderern nachhaltig verbessern.",
    "Es ist zu hoffen, dass die internationalen Klimaverhandlungen zu verbindlichen Reduktionszielen führen.",
    "Die Förderung von Start-ups ist ein zentraler Bestandteil der nationalen Innovationsstrategie.",
    "Ungeachtet der parteipolitischen Differenzen einigten sich die Fraktionen auf einen Kompromissentwurf.",
    "Die Verschärfung der Datenschutzbestimmungen stärkt die Rechte der Bürger gegenüber digitalen Konzernen.",
    "Es ist abzuwarten, inwieweit die angekündigten Reformen tatsächlich in die Praxis umgesetzt werden.",
    "Die kommunale Regionalpolitik soll strukturschwache Gebiete durch gezielte Investitionsprogramme stärker fördern.",
    "Der Bericht verdeutlicht, dass präventive Gesundheitspolitik langfristig erhebliche Kosten einsparen kann.",
    "Die im Koalitionsvertrag festgeschriebenen Klimaziele erfordern einen radikalen Umbau der Energieinfrastruktur.",
]

BATCH_5 = [
    # Philosophy 481-505
    "Die Frage nach dem Wesen des Bewusstseins beschäftigt Philosophen seit der Antike auf vielfältige Weise.",
    "Es ist umstritten, ob moralische Urteile auf universellen Prinzipien oder kulturellen Konventionen beruhen.",
    "Der Essay argumentiert, dass die Freiheit des Willens mit deterministischen Naturgesetzen vereinbar sei.",
    "Obwohl die Argumentation stringent ist, bleiben einige zentrale Prämissen der Theorie unbelegt.",
    "Die Unterscheidung zwischen Sein und Schein bildet einen Grundpfeiler der phänomenologischen Philosophie.",
    "Es bedarf einer grundlegenden Klärung des Begriffs der Gerechtigkeit im Kontext globaler Ungleichheit.",
    "Der Philosoph wies nach, dass ethische Verantwortung nicht ohne Autonomie des handelnden Subjekts denkbar ist.",
    "Trotz divergierender Positionen einigten sich die Teilnehmer auf die Notwendigkeit eines interkulturellen Dialogs.",
    "Es lässt sich nicht leugnen, dass technologischer Fortschritt neue ethische Dilemmata hervorruft.",
    "Die Kritik am Utilitarismus betont, dass individuelle Rechte nicht utilitaristisch aufgewogen werden dürfen.",
    "Angesichts der Komplexität des Problems empfiehlt sich eine differenzierte Betrachtung der verschiedenen Positionen.",
    "Die im Werk entwickelte Theorie des Guten beeinflusste die gesamte nachkantische Moralphilosophie nachhaltig.",
    "Es ist von zentraler Bedeutung, dass philosophische Argumente logisch kohärent und widerspruchsfrei sind.",
    "Viele Denker kritisieren, dass der postmoderne Relativismus jede normative Grundlage untergräbt.",
    "Die Frage nach der Erkennbarkeit der Wahrheit trennt Realisten und Konstruktivisten seit Jahrhunderten.",
    "Es ist ratsam, vor der Bildung einer festen Meinung die zugrunde liegenden Annahmen kritisch zu hinterfragen.",
    "Die Debatte über künstliche Intelligenz wirft grundlegende Fragen nach der Natur des menschlichen Geistes auf.",
    "Ungeachtet der historischen Distanz bleiben die Werke Platons für die aktuelle Ethikdebatte hoch relevant.",
    "Der Vortrag verdeutlichte, dass Verantwortungsethik eine Alternative zur klassischen Pflichtenethik darstellt.",
    "Es ist absehbar, dass die Neurowissenschaften traditionelle philosophische Konzepte des Selbst herausfordern werden.",
    "Die im Text angeführten Einwände gegen den Determinismus überzeugen durch ihre systematische Schärfe.",
    "Die Unterscheidung zwischen analytischer und kontinentaler Philosophie prägt das Fach bis heute erheblich.",
    "Es bleibt offen, ob eine letztgültige Begründung moralischer Normen überhaupt möglich ist.",
    "Die Auseinandersetzung mit dem Nihilismus führte im neunzehnten Jahrhundert zu tiefgreifenden philosophischen Krisen.",
    "Der Autor plädiert für eine Synthese von Vernunft und Erfahrung als Grundlage epistemologischer Erkenntnis.",
]

BATCH_6 = [
    # Economics/Law 506-530
    "Die kartellrechtliche Prüfung des Zusammenschlusses könnte den geplanten Börsengang erheblich verzögern.",
    "Es ist unbestritten, dass Insiderhandel den Grundsätzen eines fairen Kapitalmarktes widerspricht.",
    "Der Wirtschaftsprüfer stellte fest, dass die Bilanzierung bestimmter Derivategeschäfte fehlerhaft erfolgt war.",
    "Obwohl die Insolvenzanfechtung eingelegt wurde, konnten die Gläubiger nur einen Bruchteil ihrer Forderungen durchsetzen.",
    "Die im Schiedsverfahren getroffene Entscheidung ist für beide Vertragsparteien bindend und nicht anfechtbar.",
    "Es bedarf strengerer Regulierung, um systemische Risiken im internationalen Finanzsektor wirksam zu begrenzen.",
    "Die Compliance-Abteilung prüft, ob die Geschäftspraktiken den geltenden Wettbewerbsrechtlichen Vorgaben entsprechen.",
    "Trotz der wirtschaftlichen Einbußen hielt das Gericht die Vertragsstrafe für angemessen und rechtmäßig.",
    "Es ist von entscheidender Bedeutung, dass Bilanzskandale konsequent aufgeklärt und geahndet werden.",
    "Die Verstaatlichung der Bank während der Finanzkrise stellte einen beispiellosen Eingriff in den Markt dar.",
    "Viele Ökonomen warnen, dass protektionistische Maßnahmen den Welthandel langfristig schädigen.",
    "Die im Gesetz verankerte Publizitätspflicht zwingt börsennotierte Unternehmen zur Offenlegung ihrer Geschäftszahlen.",
    "Es lässt sich nicht bestreiten, dass Steueroasen die nationale Steuerbasis erheblich untergraben.",
    "Der Schiedsrichter wies die Klage ab und verurteilte den Beklagten zur Zahlung der Prozesskosten.",
    "Angesichts der umfassenden regulatorischen Änderungen müssen Finanzinstitute ihre Risikomodelle umfassend überarbeiten.",
    "Die Fusion der beiden Konzerne bedarf der vorherigen Genehmigung durch die europäische Wettbewerbsbehörde.",
    "Es ist ratsam, vor grenzüberschreitenden Transaktionen die jeweiligen steuerrechtlichen Implikationen zu prüfen.",
    "Die komplexe Bewertung immaterieller Vermögenswerte stellt Wirtschaftsprüfer vor erhebliche methodische Herausforderungen.",
    "Ungeachtet der juristischen Komplexität einigten sich die Parteien auf einen gütlichen Vergleich.",
    "Die Durchsetzung von Patentrechten in Entwicklungsländern erfordert eine Anpassung internationaler Abkommen.",
    "Es ist zu befürchten, dass die Verschärfung der Bankenregulierung die Kreditvergabe unnötig einschränkt.",
    "Der Wirtschaftsethiker argumentiert, dass Unternehmensverantwortung über die rechtliche Mindestanforderung hinausgehen muss.",
    "Die im Urteil festgestellte Kartellbildung führte zu empfindlichen Geldbußen für die beteiligten Unternehmen.",
    "Es bleibt abzuwarten, ob die neue EU-Verordnung den digitalen Binnenmarkt tatsächlich vereinheitlichen wird.",
    "Die Anerkennung ausländischer Schiedssprüche erfordert die Einhaltung internationaler Abkommen und Konventionen.",
]

BATCH_7 = [
    # Research/Policy 531-555
    "Die moderne evidenzbasierte Politikgestaltung erfordert eine systematische Auswertung aktueller wissenschaftlicher Studienergebnisse.",
    "Es ist erwiesen, dass frühkindliche Bildung die späteren Bildungschancen nachhaltig und messbar verbessert.",
    "Die im Forschungsprojekt entwickelten Algorithmen sollen die Früherkennung von Krankheiten ermöglichen.",
    "Obwohl die Datenlage unvollständig ist, stützen die Befunde die Wirksamkeit der präventiven Maßnahmen.",
    "Die Regierung beauftragte ein unabhängiges Institut mit der Evaluation der Bildungsreform.",
    "Es bedarf einer stärkeren Verzahnung von Wissenschaft und Politik, um gesellschaftliche Herausforderungen zu bewältigen.",
    "Die im Weißbuch empfohlenen Forschungsschwerpunkte orientieren sich an den Nachhaltigkeitszielen der Vereinten Nationen.",
    "Trotz knapper Haushaltsmittel erhöhte der Bund die Förderung grundlagenorientierter Forschung deutlich.",
    "Es ist von essenzieller Wichtigkeit, dass politische Entscheidungen auf verlässlichen wissenschaftlichen Erkenntnissen basieren.",
    "Die detaillierte Analyse epidemiologischer Daten lieferte entscheidende Erkenntnisse für die Pandemiebekämpfung.",
    "Viele Experten kritisieren, dass die Klimapolitik die neuesten Forschungsergebnisse unzureichend berücksichtigt.",
    "Die Entwicklung wirksamer Impfstoffe innerhalb eines Jahres stellte einen beispiellosen wissenschaftlichen Erfolg dar.",
    "Es lässt sich nicht leugnen, dass die Finanzierung der Grundlagenforschung in den letzten Jahren stagnierte.",
    "Die im Policy Brief dargelegten Empfehlungen zielen auf eine Reduktion der Treibhausgasemissionen ab.",
    "Angesichts der Beschleunigung des Klimawandels müssen politische Maßnahmen deutlich verschärft werden.",
    "Die im Labor entwickelte Technologie könnte die Energieeffizienz industrieller Produktionsprozesse erheblich steigern.",
    "Es ist ratsam, politische Programme regelmäßig durch unabhängige wissenschaftliche Evaluationen überprüfen zu lassen.",
    "Die enge interdisziplinäre Zusammenarbeit zwischen Medizinern und Informatikern revolutioniert die medizinische Diagnostik.",
    "Ungeachtet der politischen Widerstände setzte die Regierung das ambitionierte Forschungsförderprogramm fort.",
    "Die im Gutachten dokumentierten Befunde stützen die Einführung eines flächendeckenden Mindestlohns.",
    "Es ist absehbar, dass die Gentechnik die landwirtschaftliche Ertragssteigerung in Zukunft maßgeblich beeinflussen wird.",
    "Der Forschungsrat empfahl, die Mittel für die Erforschung erneuerbarer Energien prioritär aufzustocken.",
    "Die im Koalitionsvertrag verankerte Forschungsquote soll bis 2030 auf drei Prozent des BIP steigen.",
    "Es bleibt abzuwarten, ob die im Gesetz verankerten Klimaziele tatsächlich erreicht werden können.",
    "Die Translation von Forschungsergebnissen in die klinische Praxis erfordert strukturelle Veränderungen im Gesundheitssystem.",
]

BATCH_8 = [
    # Philosophy/Mixed 556-580
    "Die Frage nach der Verantwortung für autonome Systeme stellt Rechtsphilosophie und Technikethik vor neue Herausforderungen.",
    "Es ist umstritten, ob algorithmische Entscheidungen mit demokratischen Grundprinzipien vereinbar sind.",
    "Der Diskurs über Gerechtigkeit im Gesundheitswesen verbindet ethische Überlegungen mit ökonomischen Analysen.",
    "Obwohl die Theorie elegant ist, lassen sich ihre zentralen Annahmen empirisch nur schwer überprüfen.",
    "Die im Seminar erörterten Texte illustrieren die Vielfalt philosophischer Zugänge zum Freiheitsbegriff.",
    "Es bedarf einer Neubewertung des Verhältnisses von individueller Freiheit und kollektiver Verantwortung.",
    "Die Debatte über Gerechtigkeit als Fairness prägt die politische Philosophie seit Rawls' einflussreichem Hauptwerk.",
    "Trotz philosophischer Differenzen erkennen die meisten Denker die Bedeutung der Menschenwürde an.",
    "Es lässt sich nicht bestreiten, dass ökonomische Ungleichheit ethische Fragen der Verteilungsgerechtigkeit aufwirft.",
    "Die Kritik am anthropozentrischen Weltbild fordert eine grundlegende Neubewertung des Mensch-Natur-Verhältnisses.",
    "Angesichts der technologischen Entwicklung müssen ethische Rahmenbedingungen für die KI-Forschung geschaffen werden.",
    "Die im Traktat entwickelte Tugendethik bietet eine Alternative zu regelbasierter Moralphilosophie.",
    "Es ist von zentraler Bedeutung, dass wissenschaftliche Erkenntnis und ethische Reflexion Hand in Hand gehen.",
    "Viele Philosophen argumentieren, dass der Klimawandel eine neue Kategorie globaler moralischer Verpflichtungen begründet.",
    "Die Frage nach der Legitimität staatlicher Eingriffe in die Wirtschaft berührt Grundfragen der politischen Philosophie.",
    "Es ist ratsam, bei der Entwicklung von Technologien potenzielle gesellschaftliche Folgen von Anfang an mitzudenken.",
    "Die Auseinandersetzung mit dem Utilitarismus wirft die Frage nach der Abwägung individueller und kollektiver Interessen auf.",
    "Ungeachtet kultureller Unterschiede gibt es universelle moralische Intuitionen, die jede Ethik berücksichtigen muss.",
    "Der Vortrag verdeutlichte, dass ökonomische Rationalität und moralische Verantwortung nicht zwangsläufig im Widerspruch stehen.",
    "Es ist absehbar, dass die Neurowissenschaften das traditionelle Verständnis von Bewusstsein grundlegend verändern werden.",
    "Die im Essay angeführten Argumente für einen kosmopolitischen Ansatz gewinnen angesichts globaler Krisen an Überzeugungskraft.",
    "Die Verknüpfung von Recht und Moral bildet seit der Aufklärung einen zentralen Gegenstand juristischer Theoriebildung.",
    "Es bleibt offen, ob eine rein säkulare Ethik ausreichend normative Orientierung für komplexe Gesellschaften bieten kann.",
    "Die philosophische Reflexion über den technologischen Fortschritt ist Voraussetzung für eine verantwortungsvolle Innovationspolitik.",
    "Der Autor plädiert für eine integrative Perspektive, die ökonomische Effizienz und soziale Gerechtigkeit miteinander verbindet.",
]

ALL_BATCHES = [BATCH_1, BATCH_2, BATCH_3, BATCH_4, BATCH_5, BATCH_6, BATCH_7, BATCH_8]
SENTENCES = [s for batch in ALL_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

START_ID = 381
TARGET_PATH = project_root / "data/handcraft/de/train/c1_new_004.conllu"

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
    "ins": ("in", "ADP"), "vom": ("von", "ADP"), "beim": ("bei", "ADP"),
    "darauf": ("darauf", "ADV"), "daraus": ("daraus", "ADV"), "darin": ("darin", "ADV"),
    "hervor": ("hervor", "ADV"), "zurück": ("zurück", "ADV"), "hin": ("hin", "ADV"),
    "ab": ("ab", "ADP"), "dar": ("dar", "ADV"), "geltend": ("geltend", "ADV"),
    "im": ("in", "ADP"), "am": ("an", "ADP"), "zum": ("zu", "ADP"), "zur": ("zu", "ADP"),
    "vom": ("von", "ADP"), "beim": ("bei", "ADP"), "ins": ("in", "ADP"), "ans": ("an", "ADP"),
    "nicht": ("nicht", "PART"), "zu": ("zu", "PART"),
    "heute": ("heute", "ADV"), "gestern": ("gestern", "ADV"), "morgen": ("morgen", "ADV"),
    "bereits": ("bereits", "ADV"), "noch": ("noch", "ADV"), "auch": ("auch", "ADV"),
    "nur": ("nur", "ADV"), "sehr": ("sehr", "ADV"), "deutlich": ("deutlich", "ADV"),
    "erheblich": ("erheblich", "ADV"), "spürbar": ("spürbar", "ADV"),
    "häufig": ("häufig", "ADV"), "letztlich": ("letztlich", "ADV"),
    "grundlegend": ("grundlegend", "ADJ"), "grundlegende": ("grundlegend", "ADJ"),
    "grundlegenden": ("grundlegend", "ADJ"), "erhebliche": ("erheblich", "ADJ"),
    "erheblichen": ("erheblich", "ADJ"), "erheblicher": ("erheblich", "ADJ"),
    "wesentliche": ("wesentlich", "ADJ"), "wesentlichen": ("wesentlich", "ADJ"),
    "verschiedenen": ("verschieden", "ADJ"), "verschiedene": ("verschieden", "ADJ"),
    "neue": ("neu", "ADJ"), "neuen": ("neu", "ADJ"), "neuer": ("neu", "ADJ"),
    "neues": ("neu", "ADJ"), "neuem": ("neu", "ADJ"),
    "bestimmten": ("bestimmt", "ADJ"), "bestimmte": ("bestimmt", "ADJ"),
    "bestimmter": ("bestimmt", "ADJ"), "geltenden": ("geltend", "ADJ"),
    "geltende": ("geltend", "ADJ"), "geltender": ("geltend", "ADJ"),
    "angeführten": ("angeführt", "ADJ"), "angeführte": ("angeführt", "ADJ"),
    "getroffenen": ("getroffen", "ADJ"), "getroffene": ("getroffen", "ADJ"),
    "gewonnenen": ("gewonnen", "ADJ"), "gewonnene": ("gewonnen", "ADJ"),
    "vorgesehenen": ("vorgesehen", "ADJ"), "festgelegten": ("festgelegt", "ADJ"),
    "festgeschriebenen": ("festgeschrieben", "ADJ"), "vereinbarten": ("vereinbart", "ADJ"),
    "dokumentierten": ("dokumentiert", "ADJ"), "empfohlenen": ("empfohlen", "ADJ"),
    "entwickelten": ("entwickelt", "ADJ"), "entwickelte": ("entwickelt", "ADJ"),
    "beobachteten": ("beobachtet", "ADJ"), "untersuchten": ("untersucht", "ADJ"),
    "belegten": ("belegt", "ADJ"), "gesammelten": ("gesammelt", "ADJ"),
    "steigenden": ("steigend", "ADJ"), "wachsenden": ("wachsend", "ADJ"),
    "wachsende": ("wachsend", "ADJ"), "stagnierenden": ("stagnierend", "ADJ"),
    "anhaltenden": ("anhaltend", "ADJ"), "zunehmenden": ("zunehmend", "ADJ"),
    "kommenden": ("kommend", "ADJ"), "künftigen": ("künftig", "ADJ"),
    "künftige": ("künftig", "ADJ"), "strukturelle": ("strukturell", "ADJ"),
    "strukturellen": ("strukturell", "ADJ"), "finanzielle": ("finanziell", "ADJ"),
    "finanziellen": ("finanziell", "ADJ"), "wirtschaftliche": ("wirtschaftlich", "ADJ"),
    "wirtschaftlichen": ("wirtschaftlich", "ADJ"), "politische": ("politisch", "ADJ"),
    "politischen": ("politisch", "ADJ"), "soziale": ("sozial", "ADJ"),
    "sozialen": ("sozial", "ADJ"), "internationale": ("international", "ADJ"),
    "internationalen": ("international", "ADJ"), "nationale": ("national", "ADJ"),
    "nationalen": ("national", "ADJ"), "öffentlichen": ("öffentlich", "ADJ"),
    "öffentliche": ("öffentlich", "ADJ"), "effizienterer": ("effizienter", "ADJ"),
    "effizientere": ("effizienter", "ADJ"), "bessere": ("gut", "ADJ"),
    "besseren": ("gut", "ADJ"), "höhere": ("hoch", "ADJ"), "höheren": ("hoch", "ADJ"),
    "stärkere": ("stark", "ADJ"), "stärkeren": ("stark", "ADJ"),
    "deutlicher": ("deutlich", "ADJ"), "deutlichere": ("deutlich", "ADJ"),
    "moderate": ("moderat", "ADJ"), "moderaten": ("moderat", "ADJ"),
    "ambitionierten": ("ambitioniert", "ADJ"), "ambitionierte": ("ambitioniert", "ADJ"),
    "unabhängigen": ("unabhängig", "ADJ"), "unabhängige": ("unabhängig", "ADJ"),
    "verbindlichen": ("verbindlich", "ADJ"), "verbindliche": ("verbindlich", "ADJ"),
    "wirksamer": ("wirksam", "ADJ"), "wirksamere": ("wirksam", "ADJ"),
    "nachhaltige": ("nachhaltig", "ADJ"), "nachhaltigen": ("nachhaltig", "ADJ"),
    "nachhaltig": ("nachhaltig", "ADV"),
    "entscheidende": ("entscheidend", "ADJ"), "entscheidenden": ("entscheidend", "ADJ"),
    "zentraler": ("zentral", "ADJ"), "zentrale": ("zentral", "ADJ"),
    "fundamentale": ("fundamental", "ADJ"), "fundamentalen": ("fundamental", "ADJ"),
    "universellen": ("universell", "ADJ"), "universelle": ("universell", "ADJ"),
    "stringente": ("stringent", "ADJ"), "stringenten": ("stringent", "ADJ"),
    "logisch": ("logisch", "ADJ"), "logische": ("logisch", "ADJ"),
    "statistisch": ("statistisch", "ADJ"), "statistische": ("statistisch", "ADJ"),
    "empirisch": ("empirisch", "ADJ"), "empirische": ("empirisch", "ADJ"),
    "interdisziplinäre": ("interdisziplinär", "ADJ"), "interdisziplinären": ("interdisziplinär", "ADJ"),
    "interkulturellen": ("interkulturell", "ADJ"), "flächendeckenden": ("flächendeckend", "ADJ"),
    "grenzüberschreitenden": ("grenzüberschreitend", "ADJ"),
    "grundlagenorientierter": ("grundlagenorientiert", "ADJ"),
    "außergerichtlicher": ("außergerichtlich", "ADJ"),
    "verfassungsrechtlichen": ("verfassungsrechtlich", "ADJ"),
    "verfassungswidrig": ("verfassungswidrig", "ADJ"),
    "entscheidend": ("entscheidend", "ADJ"),
    "ratsam": ("ratsam", "ADJ"), "angemessen": ("angemessen", "ADJ"),
    "umstritten": ("umstritten", "ADJ"), "unbestritten": ("unbestritten", "ADJ"),
    "offensichtlich": ("offensichtlich", "ADV"), "fraglich": ("fraglich", "ADJ"),
    "unabdingbar": ("unabdingbar", "ADV"),
    "EU": ("EU", "PROPN"), "UN": ("UN", "PROPN"), "BIP": ("BIP", "NOUN"),
    "CO2": ("CO2", "NOUN"), "KI": ("KI", "NOUN"), "Rawls": ("Rawls", "PROPN"),
    "Platons": ("Plato", "PROPN"),
}

# Verb form -> infinitive overrides
VERB_OVERRIDES: dict[str, str] = {
    "gilt": "gelten", "erwiesen": "erweisen", "bedarf": "bedürfen",
    "hob": "heben", "wies": "weisen", "stimmte": "stimmen",
    "ließ": "lassen", "lässt": "lassen", "schließen": "schließen",
    "schließt": "schließen", "gelang": "gelingen", "macht": "machen",
    "betonte": "betonen", "äußerten": "äußern", "geäußert": "äußern",
    "plädieren": "plädieren", "plädiert": "plädieren", "konnte": "können",
    "erzielt": "erzielen", "fordern": "fordern", "fördert": "fördern",
    "fördern": "fördern", "fördert": "fördern", "gewährleistet": "gewährleisten",
    "beeinflusst": "beeinflussen", "beeinflussen": "beeinflussen",
    "stellt": "stellen", "stellen": "stellen", "dämpfen": "dämpfen",
    "dämpft": "dämpfen", "verdeutlicht": "verdeutlichen",
    "verdeutlichte": "verdeutlichen", "bleibt": "bleiben", "bleiben": "bleiben",
    "überschreitet": "überschreiten", "gefährdet": "gefährden",
    "deutet": "deuten", "diversifizieren": "diversifizieren",
    "führte": "führen", "erkennt": "erkennen", "bekämpft": "bekämpfen",
    "erfordert": "erfordern", "prognostizizieren": "prognostizieren",
    "prognostizieren": "prognostizieren", "transformieren": "transformieren",
    "könnte": "können", "kam": "kommen", "müssten": "müssen", "müsse": "müssen",
    "spiegelt": "spiegeln", "bringt": "bringen", "bringt": "bringen",
    "geführt": "führen", "setzen": "setzen", "verzögern": "verzögern",
    "prüfen": "prüfen", "untergräbt": "untergraben",
    "widersprechen": "widersprechen", "berief": "berufen",
    "beantragte": "beantragen", "darstellt": "darstellen",
    "setzt": "setzen", "plädiert": "plädieren", "zielt": "zielen",
    "abzubauen": "abbauen", "macht": "machen",
    "überzeugen": "überzeugen", "erzielt": "erzielen", "erfolgen": "erfolgen",
    "erstreckt": "erstrecken", "verursacht": "verursachen",
    "kritisieren": "kritisieren", "beginnt": "beginnen",
    "erfordert": "erfordern", "erklärte": "erklären",
    "empfiehlt": "empfehlen", "ergab": "ergeben", "verletzt": "verletzen",
    "verkürzen": "verkürzen", "vertritt": "vertreten", "verjährt": "verjähren",
    "erscheint": "erscheinen", "schützen": "schützen", "führt": "führen",
    "wies": "weisen", "bestätigte": "bestätigen", "sind": "sein",
    "stellen": "stellen", "erhält": "erhalten", "legt": "legen",
    "liefert": "liefern", "lieferten": "liefern", "wiesen": "weisen",
    "aufweisen": "aufweisen", "bewerten": "bewerten", "erhöht": "erhöhen",
    "bestätigen": "bestätigen", "revolutioniert": "revolutionieren",
    "erschüttert": "erschüttern", "ergab": "ergeben", "erklären": "erklären",
    "eröffnet": "eröffnen", "bietet": "bieten", "ermöglichen": "ermöglichen",
    "sichern": "sichern", "verweist": "verweisen", "ist": "sein",
    "kündigte": "kündigen", "scheitern": "scheitern", "sieht": "sehen",
    "offenlegen": "offenlegen", "stimmte": "stimmen", "erfordern": "erfordern",
    "genieße": "genießen", "betreffen": "betreffen", "verbessert": "verbessern",
    "beeinträchtigt": "beeinträchtigen", "stärken": "stärken",
    "verbessern": "verbessern", "führen": "führen", "reformiert": "reformieren",
    "einigten": "einigen", "stärkt": "stärken", "umgesetzt": "umsetzen",
    "fördern": "fördern", "einsparen": "einsparen", "erfordern": "erfordern",
    "beschäftigt": "beschäftigen", "beruhen": "beruhen", "argumentiert": "argumentieren",
    "bleiben": "bleiben", "bildet": "bilden", "denkbar": "denken",
    "hervorruft": "hervorrufen", "betont": "betonen", "empfiehlt": "empfehlen",
    "beeinflusste": "beeinflussen", "untergräbt": "untergraben",
    "trennt": "trennen", "hinterfragen": "hinterfragen", "wirft": "werfen",
    "bleiben": "bleiben", "stellt": "stellen", "herausfordern": "herausfordern",
    "überzeugen": "überzeugen", "prägt": "prägen", "führte": "führen",
    "plädiert": "plädieren", "verzögern": "verzögern", "widerspricht": "widersprechen",
    "stellte": "stellen", "eingelegt": "einlegen", "durchsetzen": "durchsetzen",
    "begrenzen": "begrenzen", "prüft": "prüfen", "hielt": "halten",
    "warnen": "warnen", "schädigen": "schädigen", "zwingt": "zwingen",
    "untergraben": "untergraben", "verurteilte": "verurteilen",
    "überarbeiten": "überarbeiten", "bedarf": "bedürfen", "erfordert": "erfordern",
    "einigten": "einigen", "erfordert": "erfordern", "einschränkt": "einschränken",
    "argumentiert": "argumentieren", "führte": "führen", "vereinheitlichen": "vereinheitlichen",
    "erfordert": "erfordern", "erfordert": "erfordern", "verbessert": "verbessern",
    "entwickelten": "entwickeln", "stützen": "stützen", "beauftragte": "beauftragen",
    "bewältigen": "bewältigen", "orientieren": "orientieren", "erhöhte": "erhöhen",
    "basieren": "basieren", "lieferte": "liefern", "berücksichtigt": "berücksichtigen",
    "stellte": "stellen", "stagnierte": "stagnieren", "zielen": "zielen",
    "verschärft": "verschärfen", "steigern": "steigern", "überprüfen": "überprüfen",
    "revolutioniert": "revolutionieren", "setzte": "setzen", "stützen": "stützen",
    "beeinflussen": "beeinflussen", "empfahl": "empfehlen", "aufstocken": "aufstocken",
    "steigen": "steigen", "erreicht": "erreichen", "erfordert": "erfordern",
    "verbindet": "verbinden", "überprüfen": "überprüfen", "illustrieren": "illustrieren",
    "erkennen": "erkennen", "prägt": "prägen", "anerkennen": "anerkennen",
    "aufwirft": "aufwirfen", "fordert": "fordern", "geschaffen": "schaffen",
    "bietet": "bieten", "gehen": "gehen", "begründet": "begründen",
    "berührt": "berühren", "mitzudenken": "mitdenken", "wirft": "werfen",
    "gibt": "geben", "stehen": "stehen", "verändern": "verändern",
    "gewinnen": "gewinnen", "bildet": "bilden", "bieten": "bieten",
    "verbindet": "verbinden",
    "entgegenzuwirken": "entgegenwirken", "abzuschließen": "abschließen",
    "einzuholen": "einholen", "prüfen": "prüfen", "offenlegen": "offenlegen",
    "durchsetzen": "durchsetzen", "überprüfen": "überprüfen",
    "abzuwarten": "abwarten", "zu": "zu",
}

SCONJ_WORDS = {
    "dass", "weil", "obwohl", "wenn", "ob", "als", "wie", "damit", "inwieweit",
}
CCONJ_WORDS = {"und", "oder", "aber", "sondern"}
ADP_WORDS = {
    "um", "für", "mit", "von", "bei", "nach", "aus", "über", "vor", "unter",
    "durch", "ohne", "gegen", "wegen", "in", "an", "auf", "zu", "trotz",
    "angesichts", "ungeachtet", "infolge", "unter", "gegenüber",
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
    """Tokenize to match conllu_validator text reconstruction."""
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
    """Align Stanza tokens to surface forms in # text."""
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

        # Possessive apostrophe: Rawls' -> Rawls + '
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
    """Apply C1 handcraft lemma/UPOS normalization."""
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
        # zu-infinitive particle vs zu+dat preposition - keep stanza's choice
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
            elif lem.endswith("t"):
                # past participle used as verb - try to fix common cases
                pass
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