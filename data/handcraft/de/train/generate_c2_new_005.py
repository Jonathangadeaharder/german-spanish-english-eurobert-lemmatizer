"""Generate 200 handcrafted German C2 CoNLL-U sentences (de_c2_train_361–560)."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 8 sub-batches of 25 sentences each (361–560)
BATCH_1 = [
    "Obschon die Erkenntnistheorie seit Kant umfassend diskutiert wurde, bleibt die Frage nach dem Ding an sich ungelöst.",
    "Die Transzendentalphilosophie beansprucht, die Bedingungen der Möglichkeit von Erfahrung überhaupt zu ergründen.",
    "Es ist unabdingbar, die aporetische Struktur des Selbstbewusstseins philosophisch zu durchdringen.",
    "Wenngleich der Idealismus die Einheit von Subjekt und Objekt behauptet, vermag er die Pluralität der Erscheinungen kaum zu erklären.",
    "Die Hermeneutik insistiert darauf, dass jedes Verstehen stets schon von einem Vorverständnis geleitet wird.",
    "Gleichwohl lässt sich die Geschichtlichkeit des menschlichen Daseins nicht auf bloße Faktizität reduzieren.",
    "In Anbetracht der Komplexität metaphysischer Probleme erscheint ein endgültiger Beweis der Seele als schier unmöglich.",
    "Die Phänomenologie fordert uns auf, zu der Sache selbst zurückzukehren und vorgefasste Meinungen auszuklammern.",
    "Es bedarf einer grundlegenden Revision unseres ontologischen Selbstverständnisses, um der Moderne gerecht zu werden.",
    "Der Skeptizismus untergräbt jede Gewissheit, ohne jedoch den Anspruch auf Wahrheit vollständig aufgeben zu können.",
    "Die Dialektik offenbart, dass der Widerspruch nicht bloß zu tilgen, sondern in höherer Einheit zu vermitteln ist.",
    "Es gilt, die kategoriale Struktur des Denkens von ihren empirischen Verunreinigungen methodisch zu reinigen.",
    "Die Frage nach dem Sinn des Seins entzieht sich jeder rein begrifflichen Festlegung und bleibt stets offen.",
    "Ungeachtet aller analytischen Präzision bleibt die Bedeutungstheorie mit dem Problem der Referenz konfrontiert.",
    "Die Intentionalität des Bewusstseins verweist stets auf etwas, das ihr als Korrelat gegenübersteht.",
    "Es ist von eminenter Bedeutung, die Grenzen der Vernunft von ihren transzendentalen Möglichkeiten zu unterscheiden.",
    "Die Epoché der phänomenologischen Reduktion setzt die natürliche Einstellung vorläufig außer Kraft.",
    "Wiewohl die Erkenntnislehre den Wahrheitsanspruch der Wissenschaft verteidigt, bleibt das Kriterium der Wahrheit umstritten.",
    "Die Aporie der Selbsterkenntnis besteht darin, dass das Erkennende zugleich Erkanntes werden muss.",
    "Es bleibt abzuwarten, ob die Neuauflage des Rationalismus eine überzeugende Antwort auf den Relativismus bietet.",
    "Die Kontingenz der geschichtlichen Entwicklung widerlegt jede teleologische Deutung der Weltgeschichte.",
    "Mit akribischer Genauigkeit analysierte der Philosoph die Struktur des moralischen Urteils.",
    "Die Unhintergehbarkeit der Lebenswelt bildet den Ausgangspunkt jeder ernsthaften sozialphilosophischen Reflexion.",
    "Es vermag niemand, die letztgültige Begründung ethischer Normen allein aus der Vernunft abzuleiten.",
    "Die Selbstreferenz des Bewusstseins stellt die klassische Dichotomie von Subjekt und Objekt grundsätzlich in Frage.",
]

BATCH_2 = [
    "Die Poetik des Spätmittelalters zeichnet sich durch eine außergewöhnliche Dichte allegorischer Bildlichkeit aus.",
    "Obschon der Roman experimentelle Erzählformen aufgreift, bewahrt er dennoch eine stringent geschlossene Handlung.",
    "Die Melancholie der Landschaft spiegelt die innere Zerrissenheit der tragischen Heldin wider.",
    "Es ist unverkennbar, dass die Lyrik dieser Epoche von einer tiefen Sehnsucht nach Transzendenz durchdrungen ist.",
    "Der Dichter entwirft in seinem Spätwerk ein düsteres Panorama der entfremdeten modernen Großstadt.",
    "Die Intertextualität des Romans verweist auf zahlreiche literarische Vorbilder der europäischen Klassik.",
    "Mit bewundernswerter Finesse führt der Autor mehrere Erzählebenen kunstvoll ineinander und übersteigt damit konventionelle Erzählmuster.",
    "Die Ästhetik des Hässlichen durchbricht die harmonischen Konventionen bürgerlicher Kunstauffassungen und eröffnet neue ästhetische Erfahrungsräume.",
    "Es bedarf eines sensiblen Lesers, um die subtilen Ironien dieser prosimetrischen Erzählung zu erfassen.",
    "Die Rhetorik der Rede entfaltet eine suggestive Kraft, die das Publikum schier unentrinnbar gefangen nimmt.",
    "Trotz aller philologischen Bemühungen bleibt die Autorschaft dieses anonym überlieferten Textes umstritten.",
    "Die Metafiktion durchkreuzt die naive Illusion eines unmittelbaren Zugangs zu der erzählten Wirklichkeit.",
    "Es gilt, die symbolische Tiefendimension des Dramas von seiner bloßen Handlungsfülle zu unterscheiden.",
    "Die Prosodie des Verses erzeugt durch geschickte Klangfiguren eine eindringliche musikalische Wirkung.",
    "Der Kritiker erkannte in dem Werk den vorläufigen Höhepunkt des literarischen Symbolismus.",
    "Die Parodie nimmt die konventionellen Erwartungen des Lesers ironisch in das Visier und unterläuft sie.",
    "Es ist hinlänglich bekannt, dass die Gattungsgrenzen in der Moderne zunehmend porös geworden sind.",
    "Die Ekphrase des Gemäldes verleiht der stillen Szenerie eine fast hörbar werdende bewegte Lebendigkeit.",
    "Mit unverhohlener Vehemenz verteidigte die Schriftstellerin die Unabhängigkeit der künstlerischen Form.",
    "Die Ambiguität der Metapher erschließt einen Reichtum an Bedeutungsschichten, der keiner Paraphrase vermag.",
    "Es bleibt abzuwarten, ob die neue Übersetzung der Odyssee den sprachlichen Zauber des Originals einfängt.",
    "Die Narratologie untersucht die Erzählstrukturen mit methodischer Strenge und begrifflicher Schärfe.",
    "Der Essay verknüpft persönliche Reflexion mit kulturhistorischer Analyse auf meisterhafte Weise.",
    "Die Katharsis des Tragischen beruht auf der paradoxen Lust an dem Schrecken und an dem Mitleid.",
    "Es vermag die Kritik nicht, die suggestive Wirkung dieser pathetischen Redewendungen völlig zu entkräften.",
]

BATCH_3 = [
    "Die Verfassungsgerichtsbarkeit sichert die unveräußerlichen Grundrechte der Bürger gegenüber staatlicher Willkür.",
    "Obschon die Regierung umfassende Reformen ankündigte, stießen diese in dem Parlament auf erbitterten Widerstand.",
    "Die Souveränität des Volkes bildet das normative Fundament jeder demokratischen Legitimation.",
    "Es ist unabdingbar, die Gewaltenteilung konsequent zu wahren, um tyrannische Tendenzen zu verhindern.",
    "Die Amtsimmunität des Abgeordneten schützt die parlamentarische Redefreiheit vor gerichtlicher Verfolgung.",
    "Trotz vehementer Proteste der Opposition hielt die Koalition an dem umstrittenen Gesetzentwurf fest.",
    "Die Rechtsstaatlichkeit verlangt, dass staatliches Handeln an allgemeine, vorher verkündete Gesetze gebunden ist.",
    "Es bedarf einer grundlegenden Novellierung des Asylrechts, um den humanitären Verpflichtungen gerecht zu werden.",
    "Die Diplomatie bemühte sich, den eskalierenden Konflikt durch vermittelnde Verhandlungen zu entschärfen.",
    "Der Staatsanwalt erhob Anklage wegen schwerer Untreue und Korruption in dem höchsten politischen Amt.",
    "Die Präambel der Verfassung verankert die Unantastbarkeit der Menschenwürde als obersten Verfassungswert.",
    "Es gilt, die extraterritoriale Geltung internationalen Rechts gegen souveränitätsfixierte Einwände zu behaupten.",
    "Die Sanktionen zielten darauf ab, das autoritäre Regime wirtschaftlich und politisch zu isolieren.",
    "Mit nachdrücklicher Entschiedenheit forderte die Opposition die sofortige Einberufung eines Untersuchungsausschusses.",
    "Die Referendarin vertrat ihre Mandantin vor Gericht mit beeindruckender juristischer Argumentationskunst.",
    "Es ist von eminenter Bedeutung, die Rechtsfolgen des Vertragsbruchs präzise zu bestimmen.",
    "Die Legislative hat das Recht, die Regierung durch Misstrauensvoten zu der Rechenschaft zu ziehen.",
    "Ungeachtet internationaler Empfehlungen weigerte sich das Land, die Todesstrafe rechtlich abzuschaffen.",
    "Die Gerichtsbarkeit prüfte die Verfassungsmäßigkeit des Gesetzes mit größter sorgfältiger Gewissenhaftigkeit.",
    "Es bleibt abzuwarten, ob die Friedensverhandlungen zu einem tragfähigen Kompromiss führen werden.",
    "Die Immunität des Staates gegen zivilrechtliche Klagen stößt in der Rechtsdogmatik auf erhebliche Kritik.",
    "Der Völkerrechtler betonte die zwingende Natur der humanitären Grundsätze in bewaffneten Konflikten.",
    "Die Subsidiarität begrenzt das Handeln übergeordneter Instanzen zugunsten lokaler Entscheidungskompetenzen und stärkt damit die demokratische Selbstverwaltung.",
    "Es vermag keine Partei, die wachsende Politikverdrossenheit allein durch rhetorische Appelle zu überwinden.",
    "Die Plebiszitäre Demokratie unterliegt der ständigen Gefahr populistischer Manipulation der öffentlichen Meinung.",
]

BATCH_4 = [
    "Die Geldpolitik der Zentralbank zielt auf die Stabilisierung der Inflationserwartungen in der Eurozone.",
    "Obschon die Konjunktur Anzeichen einer Erholung zeigt, bleibt die Arbeitslosenquote auf besorgniserregend hohem Niveau.",
    "Die Marktkapitalisierung des Unternehmens übertrifft die gesamte Wirtschaftsleistung mehrerer kleiner Staaten.",
    "Es ist unabdingbar, die systemischen Risiken des Finanzsektors durch wirksame Regulierung zu begrenzen.",
    "Die Deflationstendenz gefährdet die Investitionsbereitschaft der Unternehmen und hemmt das Wirtschaftswachstum.",
    "Trotz massiver Subventionen gelang es nicht, die angeschlagene Industrie dauerhaft wettbewerbsfähig zu machen.",
    "Die Terms of Trade verschlechterten sich für die exportorientierte Volkswirtschaft in dem vergangenen Geschäftsjahr erheblich.",
    "Es bedarf einer koordinierten Fiskalpolitik, um die Schuldenkrise der öffentlichen Haushalte nachhaltig zu bewältigen.",
    "Die Privatisierung staatlicher Betriebe soll die Effizienz der Märkte steigern und Haushaltslücken schließen.",
    "Der Ökonom warnte vor den destabilisierenden Folgen unkontrollierter Kapitalströme in Schwellenländern.",
    "Die Handelsbilanz weist erstmals seit Jahren ein signifikantes Überschussdefizit gegenüber den Hauptpartnern aus.",
    "Es gilt, die externen Effekte der Produktion in die volkswirtschaftliche Kostenrechnung angemessen einzubeziehen.",
    "Die Liquiditätskrise der Bank zwang die Aufsichtsbehörde zu einem außerordentlichen Eingreifen.",
    "Mit bemerkenswerter Umsicht diversifizierte der Investor sein Portfolio über verschiedene Anlageklassen hinweg.",
    "Die Stagflation der siebziger Jahre gilt als warnendes Beispiel geldpolitischer Fehlsteuerung.",
    "Es ist von eminenter Bedeutung, die Nachhaltigkeit der Staatsfinanzen langfristig zu sichern.",
    "Die Monopolstellung des Konzerns rechtfertigt eine besonders strenge kartellrechtliche Überprüfung durch die zuständigen Wettbewerbsbehörden.",
    "Ungeachtet günstiger Rohstoffpreise stagnierte die industrielle Produktion in dem vierten Quartal überraschenderweise.",
    "Die Aktienrallye an den Börsen steht in krassem Gegensatz zu der schwachen Realwirtschaft.",
    "Es bleibt abzuwarten, ob die Zinserhöhung die Inflation wirksam dämpfen wird.",
    "Die Arbeitsproduktivität stieg trotz digitaler Innovationen in vielen Branchen nur marginal an.",
    "Der Finanzminister kündigte eine umfassende Reform des progressiven Steuersystems für das kommende Jahr an.",
    "Die Opportunitätskosten der Ressourcenallokation werden in klassischen ökonomischen Modellen häufig systematisch unterschätzt.",
    "Es vermag die Geldpolitik allein nicht, strukturelle Probleme des Arbeitsmarktes dauerhaft zu lösen.",
    "Die Oligopolstruktur des Marktes begünstigt Preisabsprachen und schadet letztlich den Verbraucherinteressen.",
]

BATCH_5 = [
    "Die klinische Studie lieferte erstmals belastbare Evidenz für die Wirksamkeit des neuartigen Impfstoffs.",
    "Obschon die Symptome zunächst unspezifisch erschienen, deuteten Laborwerte auf eine seltene Autoimmunerkrankung hin.",
    "Die Genomsequenzierung revolutioniert die personalisierte Medizin und eröffnet vielversprechende Therapieansätze für zahlreiche seltene Erkrankungen.",
    "Es ist unabdingbar, die ethischen Grenzen der Keimbahnintervention in einer breiten gesellschaftlichen Debatte zu klären.",
    "Die Epidemiologie belegt einen signifikanten Zusammenhang zwischen Luftverschmutzung und kardiovaskulären Erkrankungen.",
    "Trotz intensiver Forschungsanstrengungen bleibt die Pathogenese dieser neurodegenerativen Erkrankung weitgehend ungeklärt.",
    "Die Pharmakokinetik des Wirkstoffs erlaubt eine einmal tägliche Verabreichung bei gleichbleibender therapeutischer Konzentration.",
    "Es bedarf einer interdisziplinären Zusammenarbeit, um die komplexen Mechanismen des Mikrobioms zu entschlüsseln.",
    "Die Diagnosestellung erfordert eine sorgfältige Differenzierung ähnlicher klinischer Syndrome nach aktuellen Leitlinien.",
    "Der Chirurg führte den Eingriff mit beeindruckender Präzision und minimalinvasiver Operationstechnik durch.",
    "Die Placebokontrollierte Studie bestätigte die statistisch signifikante Überlegenheit der neuen Therapie.",
    "Es gilt, die Prinzipien der informierten Einwilligung bei klinischen Versuchen konsequent einzuhalten.",
    "Die Resistenzbildung gegen Antibiotika stellt eine der größten Herausforderungen der modernen Medizin dar.",
    "Mit akribischer Sorgfalt dokumentierte die Ärztin den Verlauf der postoperativen Genesung.",
    "Die Metastasierung des Tumors erforderte eine sofortige Anpassung des multimodalen Behandlungsplans.",
    "Es ist von eminenter Bedeutung, die Reproduzierbarkeit wissenschaftlicher Ergebnisse durch offene Daten zu gewährleisten.",
    "Die Immuntherapie aktiviert die körpereigenen Abwehrkräfte zu der gezielten Bekämpfung maligner Zellen.",
    "Ungeachtet vielversprechender Frühstudien bleibt die klinische Anwendung des Verfahrens weiterhin umstritten.",
    "Die Biomarker ermöglichen eine frühzeitige Erkennung des Krankheitsverlaufs mit hoher diagnostischer Spezifität.",
    "Es bleibt abzuwarten, ob die Gentherapie langfristig schwerwiegende Nebenwirkungen zu der Folge haben wird.",
    "Die Nosokomiale Infektionsrate sank nach Einführung strenger Hygienemaßnahmen auf ein historisches Minimum.",
    "Der Epidemiologe warnte vor einer unterschätzten Ausbreitung resistenter Erreger in Gesundheitseinrichtungen.",
    "Die Palliativmedizin verfolgt das Ziel, die Lebensqualität schwerstkranker Patienten würdevoll zu erhalten.",
    "Es vermag die Schulmedizin allein nicht, die Wirksamkeit komplementärer Verfahren pauschal zu verwerfen.",
    "Die Proteomforschung entschlüsselt zunehmend die molekularen Grundlagen komplexer chronischer Erkrankungen und erweitert damit diagnostische Möglichkeiten.",
]

BATCH_6 = [
    "Die Archäologen entdeckten in der Ausgrabungsstätte Überreste einer bislang unbekannten bronzezeitlichen Siedlung.",
    "Obschon die Quellenlage äußerst fragmentarisch ist, lässt sich der politische Umbruch dennoch annähernd rekonstruieren.",
    "Die Geschichtswissenschaft hinterfragt zunehmend die teleologischen Narrative nationalstaatlicher Gründungsmythen und dekonstruiert deren ideologische Funktionen.",
    "Es ist unabdingbar, die eurozentrische Perspektive der bisherigen Geschichtsschreibung kritisch zu überwinden.",
    "Die Paläographie des Manuskripts erschließt erst nach jahrelanger Arbeit die Handschrift des unbekannten Schreibers.",
    "Trotz eingehender Recherchen blieb die genaue Datierung des Kunstwerks weiterhin Gegenstand hitziger Debatten.",
    "Die Rezeption der Antike prägte das Selbstverständnis der europäischen Intellektuellen in dem achtzehnten Jahrhundert.",
    "Es bedarf einer nuancierten Betrachtung, um koloniale Gewaltstrukturen nicht zu verharmlosen oder zu überzeichnen.",
    "Die Ikonographie des Freskos verweist auf religiöse Symbolsysteme der spätmittelalterlichen Ordenskultur.",
    "Der Historiker rekonstruierte die Ereignisse jener schicksalhaften Nacht mit akribischer quellenkritischer Genauigkeit.",
    "Die Oral History bewahrt Erinnerungen unterdrückter Gemeinschaften, die in offiziellen Archiven fehlen.",
    "Es gilt, die Materialität historischer Artefakte als eigenständige Quelle der Erkenntnis zu erschließen.",
    "Die Annalistik des Klosters dokumentiert den Alltag mittelalterlicher Mönche mit erstaunlicher Detailgenauigkeit.",
    "Mit eindringlichem Pathos schilderte der Chronist den Untergang der einst mächtigen Handelsrepublik.",
    "Die Musealisierung des Kulturerbes wirft grundlegende Fragen nach Authentizität und Repräsentation auf.",
    "Es ist von eminenter Bedeutung, die Völker des Globalen Südens in die Geschichtserzählung einzubeziehen.",
    "Die Epigraphik lieferte entscheidende Hinweise auf die Verwaltungsstrukturen des römischen Grenzprovinzgebietes.",
    "Ungeachtet politischer Widerstände gelang es dem Museum, die umstrittene Ausstellung dennoch zu eröffnen.",
    "Die Mikrogeschichte beleuchtet das Alltagsleben gewöhnlicher Menschen jenseits großer politischer Ereignisse.",
    "Es bleibt abzuwarten, ob die neu gefundenen Dokumente das etablierte Geschichtsbild grundlegend revidieren.",
    "Die Restaurierung des Gemäldes legte verborgene Pentimenti frei, die die Entstehungsgeschichte erhellen.",
    "Der Kulturwissenschaftler analysierte die performativen Praktiken der frühneuzeitlichen Festkultur und legte deren symbolische Bedeutung offen.",
    "Die Mnemotechnik der Gedenkstätten prägt das kollektive Erinnern an traumatische historische Ereignisse.",
    "Es vermag keine einzelne Disziplin, die Vielschichtigkeit historischer Prozesse erschöpfend zu erfassen.",
    "Die Historiographie der Gegenwart reflektiert zunehmend ihre eigene methodische Voreingenommenheit und thematisiert epistemologische Grenzen.",
]

BATCH_7 = [
    "Die künstliche Intelligenz durchdringt zunehmend sensible Bereiche der medizinischen Diagnostik und Entscheidungsfindung.",
    "Obschon die Algorithmen beeindruckende Leistungen erbringen, bleibt ihre Entscheidungslogik häufig intransparent.",
    "Die Digitalisierung der Verwaltung verspricht Effizienzgewinne, birgt jedoch erhebliche Risiken für den Datenschutz.",
    "Es ist unabdingbar, die ethischen Implikationen autonomer Waffensysteme auf internationaler Ebene zu regulieren.",
    "Die Kybernetische Sicherheit erfordert permanente Wachsamkeit gegen ausgeklügelte Angriffe auf kritische Infrastrukturen.",
    "Trotz umfangreicher Investitionen gelang es nicht, die Skalierungsprobleme des Quantencomputings zu überwinden.",
    "Die Plattformökonomie konzentriert marktbeherrschende Macht in den Händen weniger global agierender Konzerne.",
    "Es bedarf einer transparenten Governance, um den Einsatz prädiktiver Überwachungsalgorithmen demokratisch zu kontrollieren.",
    "Die Biometrische Überwachung untergräbt fundamentale Freiheitsrechte und verdient eine strenge rechtliche Einhegung.",
    "Der Informatiker entwickelte ein Verfahren zu der fehlertoleranten Verschlüsselung sensibler Kommunikationsdaten.",
    "Die Automatisierung routinemäßiger Tätigkeiten verändert die Struktur des Arbeitsmarktes in grundlegender Weise.",
    "Es gilt, die algorithmische Verzerrung bei Kreditvergabe und Personalentscheidungen systematisch zu identifizieren.",
    "Die Blockchaintechnologie ermöglicht dezentrale Transaktionen ohne vermittelnde Intermediäre und verändert damit etablierte Finanzarchitekturen grundlegend.",
    "Mit bemerkenswerter Weitsicht prognostizierte die Expertin die gesellschaftlichen Folgen der Datenökonomie.",
    "Die Singularitätshypothese bleibt spekulativ, obwohl die Rechenleistung exponentiell weiter zunimmt.",
    "Es ist von eminenter Bedeutung, die Nachvollziehbarkeit maschineller Entscheidungen rechtlich zu verankern.",
    "Die digitale Souveränität der Staaten gerät angesichts globaler Cloudinfrastrukturen zunehmend unter Druck.",
    "Ungeachtet technischer Gegenmaßnahmen verbreiten sich Desinformationskampagnen in sozialen Netzwerken rasant und untergraben demokratische Diskurse.",
    "Die Interoperabilität verschiedener Systeme stellt einen entscheidenden Faktor erfolgreicher Digitalisierungsstrategien dar.",
    "Es bleibt abzuwarten, ob die neue Regulierung den Missbrauch generativer Modelle wirksam eindämmen wird.",
    "Die Technikfolgenabschätzung soll unbeabsichtigte gesellschaftliche Nebenwirkungen innovativer Verfahren frühzeitig antizipieren und politisch bewerten.",
    "Der Ethikrat empfahl eine vorsichtige Einführung autonomer Fahrzeuge in dicht besiedelten urbanen Räumen.",
    "Die Postprivacydebatte hinterfragt, ob informationelle Selbstbestimmung in dem digitalen Zeitalter noch durchsetzbar ist.",
    "Es vermag keine einzelne Technologie, die komplexen Herausforderungen der Klimakrise allein zu bewältigen.",
    "Die Virtualisierung der Arbeitswelt erzwingt eine grundlegende Neubewertung traditioneller Organisationsmodelle und verändert Führungskulturen nachhaltig.",
]

BATCH_8 = [
    "Die Pragmatik der Sprache untersucht, wie Äußerungen in konkreten Kommunikationssituationen ihre Wirkung entfalten.",
    "Obschon die Grammatik scheinbar starren Regeln folgt, durchbrechen lebendige Redeformen diese Normen ständig.",
    "Die Semantik des Begriffs bleibt umstritten, solange keine einvernehmliche Definition in dem Fachdiskurs vorliegt.",
    "Es ist unabdingbar, die diskursiven Machtstrukturen hinter scheinbar neutralen sprachlichen Formulierungen offenzulegen.",
    "Die Rhetorik des Redners verband scharfsinnige Argumentation mit einer berückenden suggestiven Klanggestaltung.",
    "Trotz eingehender Analyse gelang es den Linguisten nicht, die Herkunft jenes Lehnworts zweifelsfrei zu klären.",
    "Die Morphologie des Wortes deutet auf eine komplexe historische Entwicklung über mehrere Sprachstufen hin.",
    "Es bedarf einer interdisziplinären Herangehensweise, um Metaphern in politischen Debatten angemessen zu deuten.",
    "Die Phonologie erklärt, warum bestimmte Lautkombinationen in einer Sprache als unmarkiert gelten.",
    "Der Philologe edierte den Text nach strengen textkritischen Grundsätzen und erschloss seine überlieferten Varianten.",
    "Die Soziolinguistik beschreibt die Korrelation zwischen sozialer Herkunft und spezifischen Sprachmustern.",
    "Es gilt, die Performative Kraft institutioneller Sprechakte von bloßen konstativen Aussagen zu unterscheiden.",
    "Die Etymologie des Ausdrucks führt über mittelhochdeutsche Formen bis in das frühe germanische Sprachgebiet zurück.",
    "Mit souveräner Stilsicherheit meisterte die Dolmetscherin die idiomatischen Feinheiten beider Ausgangssprachen.",
    "Die Polysemie des Lexems erschwert eine eindeutige Übersetzung in zahlreiche Zielsprachen erheblich.",
    "Es ist von eminenter Bedeutung, die kulturellen Konnotationen sprachlicher Zeichen bei der Übersetzung zu beachten.",
    "Die Textlinguistik analysiert Kohäsionsmittel, die den Zusammenhang zusammenhängender Äußerungen herstellen.",
    "Ungeachtet standardisierter Rechtschreibung bleiben regionale Varianten in der gesprochenen Umgangssprache lebendig.",
    "Die Aphasie des Patienten beeinträchtigte vor allem die Fähigkeit zu der Bildung grammatisch korrekter Sätze.",
    "Es bleibt abzuwarten, ob die neue Rechtschreibreform die Anglizismen in dem Deutschen langfristig begrenzen wird.",
    "Die Korpuslinguistik ermöglicht die empirische Untersuchung großer Textmengen mit statistischen Verfahren.",
    "Der Sprachphilosoph betonte die untrennbare Verflechtung von Sprache, Denken und gesellschaftlicher Wirklichkeit.",
    "Die Idiomatik der Wendung entzieht sich einer wörtlichen Übersetzung und verlangt kulturspezifisches Verständnis.",
    "Es vermag die Normative Grammatik allein nicht, die Dynamik lebendiger Sprachwandelprozesse angemessen zu erfassen.",
    "Die Prosodie der Äußerung vermittelt subtile emotionale Nuancen, die der rein schriftliche Text nicht trägt.",
]

BATCHES = [BATCH_1, BATCH_2, BATCH_3, BATCH_4, BATCH_5, BATCH_6, BATCH_7, BATCH_8]
SENTENCES = [s for batch in BATCHES for s in batch]

assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

import stanza

print("Loading Stanza...")
nlp = stanza.Pipeline(lang="de", processors="tokenize,mwt,pos,lemma", use_gpu=False)

MODALS = {"müssen", "können", "wollen", "sollen", "dürfen", "mögen", "möchte"}
AUX_LEMMAS = {"sein", "haben", "werden"}

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
    "jegliche": ("jeglicher", "DET"),
    "jeglicher": ("jeglicher", "DET"),
    "jegliches": ("jeglicher", "DET"),
    "jeglichen": ("jeglicher", "DET"),
    "jeglichem": ("jeglicher", "DET"),
    "dieser": ("dieser", "DET"),
    "diese": ("dieser", "DET"),
    "dieses": ("dieser", "DET"),
    "diesen": ("dieser", "DET"),
    "diesem": ("dieser", "DET"),
    "jener": ("jener", "DET"),
    "jene": ("jener", "DET"),
    "jenes": ("jener", "DET"),
    "jenen": ("jener", "DET"),
    "jenem": ("jener", "DET"),
    "Dessen": ("der", "PRON"),
    "dessen": ("der", "PRON"),
    "deren": ("der", "PRON"),
    "ins": ("in", "ADP"),
    "im": ("in", "ADP"),
    "am": ("an", "ADP"),
    "zum": ("zu", "ADP"),
    "zur": ("zu", "ADP"),
    "beim": ("bei", "ADP"),
    "vom": ("von", "ADP"),
    "Obschon": ("Obschon", "SCONJ"),
    "obschon": ("Obschon", "SCONJ"),
    "Wenngleich": ("wenngleich", "SCONJ"),
    "wenngleich": ("wenngleich", "SCONJ"),
    "Gleichwohl": ("gleichwohl", "ADV"),
    "gleichwohl": ("gleichwohl", "ADV"),
    "Wiewohl": ("wiewohl", "SCONJ"),
    "wiewohl": ("wiewohl", "SCONJ"),
    "Ungeachtet": ("ungeachtet", "ADP"),
    "ungeachtet": ("ungeachtet", "ADP"),
}

AUX_FORMS = {
    "ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen", "sei", "seien",
    "habe", "hat", "hatte", "hätte", "hätten", "haben", "habt", "gehabt", "hätt",
    "wird", "wurde", "werden", "würde", "würden", "geworden", "worden",
}

SCONJ_FORMS = {
    "weil", "dass", "obwohl", "wenn", "ob", "da", "als", "wie", "während", "sofern", "falls",
    "bevor", "nachdem", "sodass", "damit", "indem",
}

CCONJ_FORMS = {"und", "oder", "aber", "sondern", "sowie"}
ADP_FORMS = {
    "um", "für", "mit", "von", "bei", "nach", "aus", "über", "vor", "unter", "durch",
    "ohne", "gegen", "wegen", "in", "an", "auf", "trotz", "gegenüber", "entlang",
}
PART_FORMS = {"nicht", "ja", "nein", "doch", "nur", "auch", "schon", "noch", "bloß", "allein"}
PRON_FORMS = {"ich", "du", "er", "sie", "es", "wir", "man", "wer", "was", "etwas", "nichts"}

VERB_IRREGULAR: dict[str, str] = {
    "vermag": "vermögen",
    "vermögen": "vermögen",
    "vermochte": "vermögen",
    "vermochten": "vermögen",
    "bedarf": "bedürfen",
    "bedürfen": "bedürfen",
    "gilt": "gelten",
    "gelten": "gelten",
    "galt": "gelten",
    "galten": "gelten",
    "entspricht": "entsprechen",
    "entsprechen": "entsprechen",
    "entsprach": "entsprechen",
    "stieß": "stoßen",
    "stießen": "stoßen",
    "traf": "treffen",
    "trafen": "treffen",
    "gab": "geben",
    "gaben": "geben",
    "blieb": "bleiben",
    "blieben": "bleiben",
    "zog": "ziehen",
    "zogen": "ziehen",
    "hielt": "halten",
    "hielten": "halten",
    "ließ": "lassen",
    "ließen": "lassen",
    "wies": "weisen",
    "wiesen": "weisen",
    "führt": "führen",
    "führte": "führen",
    "führten": "führen",
    "steht": "stehen",
    "stand": "stehen",
    "standen": "stehen",
    "fällt": "fallen",
    "fiel": "fallen",
    "fielen": "fallen",
    "lässt": "lassen",
    "läßt": "lassen",
    "zieht": "ziehen",
    "ziehen": "ziehen",
    "schließt": "schließen",
    "schloss": "schließen",
    "schlossen": "schließen",
    "spricht": "sprechen",
    "sprach": "sprechen",
    "sprachen": "sprechen",
    "bricht": "brechen",
    "brach": "brechen",
    "brachen": "brechen",
    "hilft": "helfen",
    "half": "helfen",
    "halfen": "helfen",
    "nimmt": "nehmen",
    "nahm": "nehmen",
    "nahmen": "nehmen",
    "kommt": "kommen",
    "kam": "kommen",
    "kamen": "kommen",
    "findet": "finden",
    "fand": "finden",
    "fanden": "finden",
}


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    if form in SPECIAL_LEMMAS:
        return SPECIAL_LEMMAS[form]

    if form.lower() == "das" and upos == "PRON":
        return "der", "PRON"
    if form.lower() in {"die", "der", "den", "dem", "des"} and upos == "PRON":
        return "der", "PRON"

    if form == "Erkanntes":
        return "Erkanntes", "NOUN"

    if upos == "NOUN":
        lemma = lemma[0].upper() + lemma[1:] if lemma else lemma

    if upos == "VERB" or lemma in MODALS or form.lower() in MODALS:
        upos = "VERB"
        lemma = lemma.lower() if lemma else lemma
        if form in VERB_IRREGULAR:
            lemma = VERB_IRREGULAR[form]
        elif lemma in VERB_IRREGULAR:
            lemma = VERB_IRREGULAR[lemma]

    if form.lower() in AUX_FORMS:
        upos = "AUX"
        if form.lower() in {"ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen", "sei", "seien"}:
            lemma = "sein"
        elif form.lower() in {"habe", "hat", "hatte", "hätte", "hätten", "haben", "habt", "gehabt"}:
            lemma = "haben"
        else:
            lemma = "werden"
    elif upos == "AUX" and lemma not in AUX_LEMMAS:
        upos = "VERB"

    if upos == "ADJ":
        lemma = lemma.lower() if lemma else lemma

    if upos == "PROPN":
        lemma = lemma[0].upper() + lemma[1:] if lemma else lemma

    if upos == "PUNCT":
        lemma = form

    low = form.lower()
    if low in SCONJ_FORMS:
        upos = "SCONJ"
        lemma = low
    elif low in CCONJ_FORMS:
        upos = "CCONJ"
        lemma = low
    elif low in ADP_FORMS:
        upos = "ADP"
        lemma = low
    elif low in PART_FORMS:
        upos = "PART"
        lemma = low
    elif low in PRON_FORMS:
        upos = "PRON"
        lemma = "Sie" if form == "Sie" else low

    if form in SPECIAL_LEMMAS:
        return SPECIAL_LEMMAS[form]

    return lemma, upos


def build_conllu(sentences: list[str], start_id: int) -> str:
    output_lines: list[str] = []

    for idx, sent in enumerate(sentences):
        sent_id = f"de_c2_train_{start_id + idx}"
        doc = nlp(sent)
        output_lines.append(f"# sent_id = {sent_id}")
        output_lines.append(f"# text = {sent}")

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
                output_lines.append("\t".join(cols))
                token_counter += 1
        output_lines.append("")

    output_lines.append("")
    return "\n".join(output_lines)


def main() -> None:
    start_id = 361
    conllu_text = build_conllu(SENTENCES, start_id)

    target_path = project_root / "data/handcraft/de/train/c2_new_005.conllu"
    target_path.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {len(SENTENCES)} sentences to {target_path}")

    validation_res = validate_text(conllu_text)
    print(f"Validate: count={validation_res.sentence_count}, passed={validation_res.passed}")
    if not validation_res.passed:
        for err in validation_res.errors[:30]:
            print("  ", err)
        sys.exit(1)

    lemma_res = check_text(conllu_text, lang="de")
    print(f"Lemma check: passed={lemma_res.passed}")
    if not lemma_res.passed:
        for err in lemma_res.errors[:30]:
            print("  ", err)
        sys.exit(1)


if __name__ == "__main__":
    main()