"""Generate C1 German CoNLL-U training data: de_c1_train_581 through de_c1_train_780."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

BATCH_1 = [
    # Constitutional law 581-605
    "Das Bundesverfassungsgericht prüft, ob die neuen Sicherheitsgesetze den Schutz der Grundrechte angemessen gewährleisten.",
    "Die verfassungsrechtliche Abwägung zwischen Freiheit und Sicherheit erfordert eine präzise juristische Differenzierung.",
    "Es ist umstritten, ob das parlamentarische Informationsrecht auch geheime Sicherheitsakten umfassen darf.",
    "Der Grundgesetzgeber legte fest, dass die Menschenwürde unantastbar und nicht relativierbar ist.",
    "Obwohl die Norm klar formuliert ist, bleibt ihre praktische Auslegung in Einzelfällen strittig.",
    "Die Kompetenzverteilung zwischen Bund und Ländern bildet ein zentrales Thema der Verfassungsreformdebatte.",
    "Es bedarf einer Klarstellung, inwieweit digitale Grundrechte verfassungsrechtlich ausdrücklich abgesichert werden.",
    "Das Gericht erklärte die Regelung für verfassungswidrig, weil sie den Gleichheitssatz systematisch verletzte.",
    "Trotz eingehender Anhörungen konnte der Ausschuss keinen Konsens über die Verfassungsänderung erzielen.",
    "Die verfassungsrechtliche Beschwerde bietet Bürgern einen wirkungsvollen Rechtsbehelf gegen staatliche Eingriffe.",
    "Es lässt sich nicht leugnen, dass Ewigkeitsklauseln die verfassungsrechtliche Entwicklungsfähigkeit des Staates begrenzen.",
    "Der Rechtsstaatsgrundsatz verpflichtet Behörden, Verwaltungsentscheidungen ausführlich und nachvollziehbar zu begründen.",
    "Angesichts der neuen technologischen Möglichkeiten müssen verfassungsrechtliche Schutzkonzepte zeitnah überarbeitet werden.",
    "Die im Urteil angeführte Begründung stützt sich auf etablierte verfassungsgerichtliche Rechtsprechung.",
    "Es ist absehbar, dass die Digitalisierung des Staates verfassungsrechtliche Kontrollmechanismen grundlegend verändert.",
    "Viele Verfassungsrechtler kritisieren, dass Notstandsvorschriften zu weitreichende exekutive Befugnisse einräumen.",
    "Die Auslegung der Religionsfreiheit erfordert eine sensible Abwägung kollektiver und individueller Belange.",
    "Ungeachtet politischer Auseinandersetzungen bleibt der Schutz der Minderheitenrechte im Grundgesetz verfassungsrechtlich verankert.",
    "Der Verfassungskommentar verdeutlicht, dass Grundrechte auch auf private Machtstrukturen erstreckt werden können.",
    "Es bleibt offen, ob ein verfassungskonformes Vorgehen ohne Gesetzesänderung überhaupt möglich ist.",
    "Die parlamentarische Kontrolle der Nachrichtendienste stellt ein verfassungsrechtliches Kernproblem der Sicherheitspolitik dar.",
    "Der Grundsatz der Verhältnismäßigkeit begrenzt staatliche Eingriffe in Freiheitsrechte auf das rechtlich Erforderliche.",
    "Es ist ratsam, verfassungsrechtliche Risiken bereits in frühen Phasen des Gesetzgebungsverfahrens zu identifizieren.",
    "Die verfassungsrechtliche Gewaltenteilung soll konzentrierte Machtbildung in einzelnen Staatsorganen wirksam verhindern.",
    "Das Urteil bestätigt, dass legislative Eilentscheidungen verfassungsrechtlich besonders strengen Anforderungen unterliegen.",
]

BATCH_2 = [
    # Macroeconomics 606-630
    "Die geldpolitische Straffung dämpft die Investitionsbereitschaft, ohne die Kerninflation kurzfristig nachhaltig zu senken.",
    "Angesichts steigender Realzinsen erwarten viele Analysten eine merkliche Verlangsamung des Wirtschaftswachstums.",
    "Es ist erwiesen, dass expansive Fiskalpolitik in Krisenzeiten wirksam gegen konjunkturelle Einbrüche wirken kann.",
    "Die aggregierten Nachfragedaten deuten auf eine schleichende Abkühlung des privaten Konsums hin.",
    "Obwohl die Arbeitsmärkte robust bleiben, zeigen Frühindikatoren erste Anzeichen einer konjunkturellen Schwäche.",
    "Die makroökonomische Stabilisierung erfordert eine abgestimmte Koordination von Geldpolitik und Fiskalpolitik.",
    "Es bedarf transparenter Kommunikation, damit Marktteilnehmer geldpolitische Entscheidungen verlässlich antizipieren können.",
    "Der fiskalische Multiplikator staatlicher Investitionen variiert erheblich je nach konjunkturellem Umfeld.",
    "Trotz moderater Lohnzuwächse belasten hohe Energiepreise die Realeinkommen der Haushalte deutlich.",
    "Es lässt sich nicht bestreiten, dass globale Lieferkettenstörungen die Inflationsdynamik nachhaltig beeinflusst haben.",
    "Die Zentralbank betonte, dass Preisstabilität weiterhin das übergeordnete Ziel der Geldpolitik bleibe.",
    "Viele Ökonomen plädieren für eine strukturelle Reform der Schuldenbremse angesichts veränderter demografischer Rahmenbedingungen.",
    "Die Erholung des Außenhandels hängt entscheidend von der wirtschaftlichen Entwicklung in Schwellenländern ab.",
    "Es ist abzuwarten, ob die konjunkturellen Stützmaßnahmen eine nachhaltige Erholung der Binnennachfrage bewirken.",
    "Die Verschlechterung der Handelsbedingungen verringert die verfügbaren Einkommen in rohstoffimportierenden Volkswirtschaften erheblich.",
    "Ungeachtet restriktiver Finanzierungsbedingungen verzeichnet der Dienstleistungssektor weiterhin robuste und stabile Zuwächse.",
    "Die Budgetdefizite mehrerer Industrieländer überschreiten die langfristig als finanzpolitisch tragfähig geltenden Schwellenwerte.",
    "Es ist von zentraler Bedeutung, dass Wirtschaftspolitik strukturelle Produktivitätshemmnisse systematisch abbaut.",
    "Die Analyse der Geldmengenaggregate legt nahe, dass die Inflationsrisiken mittelfristig abnehmen könnten.",
    "Der IWF empfahl, fiskalpolitische Prioritäten stärker auf wachstumsfördernde öffentliche Investitionen auszurichten.",
    "Angesichts volatiler Finanzmärkte müssen makroprudenielle Instrumente systemische Risiken frühzeitig und wirksam begrenzen.",
    "Die Neuverschuldung der öffentlichen Hand steigt trotz konjunktureller Belebung überraschend stark an.",
    "Es bleibt fraglich, ob eine Entkopplung von Lohnentwicklung und Preisentwicklung die Inflation dauerhaft entschärft.",
    "Die konjunkturelle Divergenz zwischen Eurozone und Nordamerika erschwert eine abgestimmte geldpolitische Reaktion.",
    "Der Wirtschaftsbericht verdeutlicht, dass sinkende Unternehmensgewinne die Investitionsdynamik mittelfristig dämpfen werden.",
]

BATCH_3 = [
    # Epidemiology 631-655
    "Die retrospektive Kohortenstudie untersuchte den Zusammenhang zwischen Luftverschmutzung und kardiovaskulären Erkrankungen.",
    "Es ist erwiesen, dass Impfprogramme die Inzidenz schwerer Infektionsverläufe in Risikogruppen deutlich senken.",
    "Die epidemiologische Überwachung erfordert standardisierte Meldesysteme und verlässliche Laboruntersuchungen in allen Regionen.",
    "Obwohl die Fallzahlen zurückgingen, bleibt die Übertragungsdynamik in dicht besiedelten Regionen hoch.",
    "Die Schätzung der effektiven Reproduktionszahl hängt von der Qualität der zugrunde liegenden Daten ab.",
    "Es bedarf weiterer prospektiver Studien, um Langzeitfolgen postakuter Infektionssyndrome belastbar zu bewerten.",
    "Die Kontaktverfolgung stellte während der Pandemie ein zentrales Instrument der öffentlichen Gesundheitsvorsorge dar.",
    "Trotz verbesserter Diagnostik gelang es nicht, alle nosokomialen Ausbrüche frühzeitig zu identifizieren.",
    "Die Analyse der Seroprävalenz lieferte wichtige Hinweise auf eine unterschätzte Verbreitung des Erregers.",
    "Es ist von essenzieller Bedeutung, dass epidemiologische Modelle Unsicherheiten transparent kommunizieren.",
    "Viele Epidemiologen kritisieren, dass Präventionsstrategien soziale Determinanten von Gesundheit unzureichend berücksichtigen.",
    "Die Inzidenzdichte in urbanen Ballungsräumen übertraf die Werte ländlicher Regionen um ein Vielfaches.",
    "Es lässt sich nicht bestreiten, dass Gesundheitsungleichheit die Ausbreitung infektiöser Krankheiten begünstigt.",
    "Die kontaktbasierte Kohortenstudie identifizierte Superspreaderereignisse als wesentliche treibende Faktoren lokaler Ausbrüche.",
    "Angesichts neuer Virusvarianten müssen epidemiologische Meldesysteme ihre Sequenzierungskapazitäten deutlich und schnell ausbauen.",
    "Die adjustierte Mortalitätsrate berücksichtigt relevante demografische Unterschiede zwischen den untersuchten Populationen.",
    "Es ist ratsam, epidemiologische Evidenz regelmäßig in evidenzbasierte Gesundheitsleitlinien zu übersetzen.",
    "Die räumliche Clusteranalyse zeigte signifikante Hotspots in sozioökonomisch stark benachteiligten Stadtvierteln.",
    "Ungeachtet sinkender Hospitalisierungsraten bleibt die Belastung der Intensivstationen in einigen Regionen hoch.",
    "Die Impfdurchimpfung der vulnerablen Gruppen verbesserte den populationsspezifischen Schutz vor schweren Verläufen.",
    "Es ist absehbar, dass integrierte Gesundheitsansätze zoonotische Risiken systematischer erfassen werden.",
    "Der Epidemiologiebericht verweist darauf, dass Verzögerungen bei Meldungen die Modellprognosen verzerren.",
    "Die Berechnung verlorener Lebensjahre ermöglicht einen differenzierten und belastbaren Vergleich pandemischer Todesursachen.",
    "Es bleibt offen, inwieweit saisonale Schwankungen die Transmission neuartiger Erreger nachhaltig beeinflussen.",
    "Die Validierung diagnostischer Sensitivität ist Voraussetzung für belastbare Schätzungen der Attack Rate.",
]

BATCH_4 = [
    # International trade 656-680
    "Die neuen Handelsbeschränkungen erschweren Exporteuren den Zugang zu strategisch wichtigen Vorprodukten erheblich.",
    "Es ist unbestritten, dass multilaterale Handelsregeln die Vorhersehbarkeit grenzüberschreitender Lieferketten stärken.",
    "Die Handelsbilanz des Landes verschlechterte sich infolge steigender Importpreise für Energie und Rohstoffe.",
    "Obwohl Zölle gesenkt wurden, blieben nichttarifäre Handelshemmnisse für kleine Exporteure bestehen.",
    "Die WTO-Streitbeilegung spielt eine zentrale Rolle bei der Klärung handelspolitischer Konflikte.",
    "Es bedarf fairer Wettbewerbsbedingungen, damit Entwicklungsländer vom Welthandel nachhaltig profitieren können.",
    "Der Freihandelsvertrag sieht vor, dass technische Standards schrittweise harmonisiert werden sollen.",
    "Trotz protektionistischer Tendenzen wuchs das Handelsvolumen zwischen den Partnerregionen moderat an.",
    "Die Diversifizierung der Bezugsquellen reduziert die Abhängigkeit von einzelnen geopolitisch instabilen Märkten.",
    "Es lässt sich nicht leugnen, dass Handelskonflikte die globale Wertschöpfung erheblich destabilisieren können.",
    "Die Ursprungsregeln des Abkommens verlangen eine nachweisbare Wertschöpfung innerhalb der Vertragsparteien.",
    "Viele Handelsexperten plädieren für eine grundlegende Reform der Streitbeilegungsmechanismen der Welthandelsorganisation.",
    "Die Terms of Trade verschlechterten sich für Importeure industrieller Vorprodukte aus Schwellenländern deutlich.",
    "Es ist abzuwarten, ob regionale Lieferketten globale Handelsströme langfristig teilweise ersetzen werden.",
    "Die Subventionsprüfung der Kommission könnte den Marktzugang ausländischer Stahlproduzenten deutlich einschränken.",
    "Angesichts geopolitischer Spannungen müssen exportierende Unternehmen ihre Exportstrategien umfassend neu bewerten.",
    "Die Handelspolitik zielt darauf ab, kritische Güter innerhalb verlässlicher Partnerschaften zu sichern.",
    "Ungeachtet administrativer Hürden stieg der grenzüberschreitende Dienstleistungshandel in digitalen Wirtschaftssektoren deutlich.",
    "Der Außenhandelsbericht verdeutlicht, dass Logistikengpässe die Lieferzeiten internationaler Warenströme verlängern.",
    "Es ist ratsam, internationale Handelsverträge unter Berücksichtigung künftiger Klimaschutzauflagen auszuhandeln.",
    "Die Antidumpingmaßnahmen sollen unlauteren Wettbewerb auf dem europäischen Binnenmarkt wirksam entgegenwirken.",
    "Es bleibt fraglich, ob bilaterale Abkommen multilaterale Handelsliberalisierung tatsächlich ergänzen oder ersetzen.",
    "Die Zollunion erleichterte kleinen und mittleren Unternehmen den Zugang zu benachbarten Absatzmärkten.",
    "Der Bericht betont, dass Handelsfinanzierung für Exporteure in Entwicklungsländern weiterhin knapp bleibt.",
    "Die Neuordnung globaler Handelsrouten verändert die Wettbewerbsvorteile traditioneller maritimer Logistikdrehkreuze nachhaltig.",
]

BATCH_5 = [
    # Cognitive science 681-705
    "Die Studie untersuchte, wie Arbeitsgedächtniskapazität die Leistung bei komplexen Problemlöseaufgaben beeinflusst.",
    "Es ist erwiesen, dass multisensorische Integration die Aufmerksamkeitssteuerung im Gehirn erheblich moduliert.",
    "Die kognitive Neurowissenschaft verbindet experimentelle Verhaltensdaten mit modernen hochauflösenden bildgebenden Verfahren.",
    "Obwohl die Stichprobe homogen war, zeigten die Befunde robuste Effekte über verschiedene Paradigmen hinweg.",
    "Das Modell der dualen Prozesse erklärt intuitive und analytische Entscheidungen als interagierende Systeme.",
    "Es bedarf präziser Versuchsdesigns, um kognitive Verzerrungen experimentell zuverlässig von Heuristiken zu trennen.",
    "Die Messung der Reaktionszeiten lieferte Hinweise auf unbewusste Priming-Effekte in Lexikonerkennungsaufgaben.",
    "Trotz methodischer Fortschritte bleibt die Kausalinterpretation korrelativer Bildgebungsdaten wissenschaftlich äußerst anspruchsvoll.",
    "Die Theorie des verteilten Kognierens betont die enge Kopplung von Gehirn, Körper und Umwelt.",
    "Es ist von zentraler Bedeutung, dass kognitive Modelle Vorhersagen empirisch falsifizierbar formulieren.",
    "Viele Forscher kritisieren, dass Replikationsstudien in der Kognitionspsychologie unzureichend finanziert werden.",
    "Die Analyse der Fehlerprofile deutet auf systematische Strategiewechsel während längerer Lernphasen hin.",
    "Es lässt sich nicht bestreiten, dass Schlafqualität die Konsolidierung deklarativer Gedächtnisinhalte beeinflusst.",
    "Die neurokognitive Untersuchung identifizierte frontale Areale als zentral für exekutive Kontrollprozesse.",
    "Angesichts komplexer Datenlagen müssen statistische Modelle individuelle Unterschiede zwischen Probanden explizit berücksichtigen.",
    "Die Hypothese der embodied cognition stellt klassische repräsentationalistische Annahmen grundlegend in Frage.",
    "Es ist ratsam, Verhaltensdaten und Neurodaten durch präregistrierte Analysen transparent miteinander zu verknüpfen.",
    "Die Interferenz zwischen parallelen Aufgaben offenbarte begrenzte Kapazitäten des zentralen Aufmerksamkeitssystems.",
    "Ungeachtet theoretischer Kontroversen liefert die Evidenz für Prädiktionskodierung im visuellen System.",
    "Die Entwicklung kindlicher Mentalisierungsfähigkeit korreliert eng mit der Reifung sozialkognitiver Netzwerke im Gehirn.",
    "Es ist absehbar, dass computergestützte Modelle kognitive Prozesse zunehmend mechanistisch approximieren werden.",
    "Der Forschungsbericht verweist darauf, dass kulturelle Praktiken kognitive Entwicklung mitprägen.",
    "Die Messung der kognitiven Belastung ermöglicht eine feinere Evaluation benutzerfreundlicher Interfaces.",
    "Es bleibt offen, inwieweit große Sprachmodelle menschliche Sprachverarbeitung tatsächlich abbilden.",
    "Die Validierung psychometrischer Instrumente ist Voraussetzung für vergleichbare und belastbare kognitive Leistungsmessungen.",
]

BATCH_6 = [
    # Sustainability policy 706-730
    "Die Regierung kündigte an, dass die Emissionsmärkte bis 2030 deutlich verschärft und ausgeweitet werden sollen.",
    "Es steht außer Frage, dass Klimaneutralität ohne massive Investitionen in Netzinfrastruktur nicht erreichbar ist.",
    "Der Gesetzentwurf verpflichtet Unternehmen, ihre indirekten Emissionen transparent und jährlich offenzulegen.",
    "Obwohl Widerstand aus der Industrie laut wurde, billigte das Parlament das Klimaschutzpaket letztlich.",
    "Die im Aktionsplan verankerten Ziele erfordern eine konsequente Umsetzung auf kommunaler und regionaler Ebene.",
    "Es ist von großer Bedeutung, dass Nachhaltigkeitspolitik soziale Verteilungswirkungen von Anfang an berücksichtigt.",
    "Die Verschärfung der Energieeffizienzstandards soll den Endenergieverbrauch im Gebäudesektor merklich senken.",
    "Trotz günstiger Ausbauzahlen erneuerbarer Energien bleibt die Netzintegration ein zentrales politisches Hemmnis.",
    "Es bedarf kohärenter Förderinstrumente, die ökologische Ziele und wirtschaftliche Wettbewerbsfähigkeit miteinander verbinden.",
    "Der Umweltminister betonte, dass Kreislaufwirtschaft oberste Priorität der Ressourcenpolitik genießen müsse.",
    "Die im Haushalt vorgesehenen Mittel für den Kohleausstieg betreffen vor allem strukturschwache Bergbauregionen.",
    "Viele Kommunen fordern verlässliche Finanzierungszusagen für den langfristigen Ausbau klimaneutraler Nahverkehrssysteme.",
    "Die Einführung einer CO2-Bepreisung hat Investitionsentscheidungen in emissionsarme Technologien spürbar verändert.",
    "Es lässt sich nicht leugnen, dass politische Kurzlaufzeiten langfristige Transformationsprojekte erheblich erschweren.",
    "Die Neuausrichtung der Agrarpolitik zielt auf eine Reduktion stickstoffbedingter Umweltbelastungen ab.",
    "Angesichts zunehmender Dürreperioden müssen regionale Wasserwirtschaftskonzepte angepasst und nachhaltig finanziert werden.",
    "Die im Gutachten empfohlenen Maßnahmen sollen die Biodiversität in geschützten Landschaftsräumen stärken.",
    "Es ist zu hoffen, dass internationale Klimafinanzierungszusagen Entwicklungsländern bei der Dekarbonisierung helfen.",
    "Die Förderung grüner Wasserstoffprojekte ist ein zentraler Bestandteil der nationalen Industriestrategie.",
    "Ungeachtet parteipolitischer Differenzen einigten sich die Fraktionen auf besonders ambitionierte Emissionsreduktionsziele.",
    "Die Verschärfung der Lieferkettensorgfaltspflicht stärkt den Schutz von Mensch und Umwelt im globalen Handel.",
    "Es bleibt abzuwarten, inwieweit die angekündigten Klimaziele in konkrete Sektorprogramme übersetzt werden.",
    "Die kommunale Wärmeplanung soll den Umstieg von fossilen Heizsystemen auf erneuerbare Energien beschleunigen.",
    "Der Bericht verdeutlicht, dass Präventionsmaßnahmen gegen Extremwetterereignisse langfristig erhebliche Kosten sparen.",
    "Die im Koalitionsvertrag festgeschriebenen Ausbauziele erfordern einen beschleunigten Ausbau der Stromnetze.",
]

BATCH_7 = [
    # Media ethics 731-755
    "Die Redaktion prüft, ob die Veröffentlichung anonymisierter Chatprotokolle dem Pressekodex entspricht.",
    "Es ist umstritten, ob algorithmische Empfehlungssysteme journalistische Verantwortung in Plattformbetriebe verlagern.",
    "Der Medienethiker argumentiert, dass Sensationen nicht über den Schutz betroffener Personen gestellt werden dürfen.",
    "Obwohl die Quelle geschützt ist, bleibt die Überprüfbarkeit der zugrunde liegenden Dokumente begrenzt.",
    "Die Informationspflicht gegenüber der Öffentlichkeit steht häufig im Spannungsfeld zum Persönlichkeitsrecht.",
    "Es bedarf klarer redaktioneller Standards, um Desinformation in sozialen Netzwerken wirksam zu kennzeichnen.",
    "Die Berichterstattung über Minderjährige erfordert besondere redaktionelle Sorgfalt und strikte Anonymisierungsregeln.",
    "Trotz öffentlichen Informationsinteresses wies das Gericht die Herausgabe sensibler Akten zurück.",
    "Die Transparenz über Finanzierungsquellen stärkt das Vertrauen in unabhängigen investigativen Journalismus.",
    "Es lässt sich nicht bestreiten, dass Klickmaximierung Anreize für verzerrende Nachrichtenselektion setzt.",
    "Die Kennzeichnung von Meinungsbeiträgen und Werbung ist ein zentrales Element medienethischer Sorgfaltspflicht.",
    "Viele Journalistinnen kritisieren, dass Plattformmonopole den Zugang zu relevanten öffentlichen Informationen begrenzen.",
    "Die Verbreitung manipulierter Audiodateien stellt Redaktionen vor neue Herausforderungen der Quellenprüfung.",
    "Es ist ratsam, ethische Leitlinien für den Einsatz generativer KI in Nachrichtenredaktionen frühzeitig zu erarbeiten.",
    "Die Doppelrolle von Kommentatoren und Moderatoren kann die wahrgenommene Unparteilichkeit öffentlich-rechtlicher Angebote beeinträchtigen.",
    "Angesichts beschleunigter Nachrichtenzyklen müssen redaktionelle Faktenchecks ohne Qualitätsverlust deutlich beschleunigt werden.",
    "Die Berichterstattung über Suizide soll nach medienethischen Vorgaben zurückhaltend und sensibilisierend erfolgen.",
    "Ungeachtet wirtschaftlicher Zwänge bleibt die Wahrhaftigkeitspflicht ein unverzichtbares Fundament journalistischer Arbeit.",
    "Der Medienrat verdeutlichte, dass Bildauswahl erheblich zur emotionalen Rahmung politischer Ereignisse beiträgt.",
    "Es ist absehbar, dass personalisierte Nachrichtenströme den gesellschaftlichen Diskurs stärker fragmentieren werden.",
    "Die Veröffentlichung von Rohdaten kann Transparenz erhöhen, birgt jedoch Risiken für Persönlichkeitsrechte Dritter.",
    "Es bleibt offen, ob freiwillige Selbstverpflichtungen der Plattformen ausreichend gegen Hassrede wirken.",
    "Die journalistische Sorgfaltspflicht verlangt die unabhängige Verifikation von Informationen vor der Verbreitung.",
    "Der Bericht betont, dass Medienkompetenzprogramme Desinformationsresilienz in der Bevölkerung stärken können.",
    "Die ethische Bewertung von Undercover-Recherchen erfordert eine strenge Abwägung von Rechtfertigung und Schaden.",
]

BATCH_8 = [
    # Mixed advanced topics 756-780
    "Die verfassungsrechtliche Prüfung des Klimaschutzgesetzes verbindet Grundrechte mit zentralen ökologischen Staatszielbestimmungen.",
    "Es ist erwiesen, dass makroprudenzielle Regulierung systemische Risiken im Bankensektor wirksam begrenzen kann.",
    "Die epidemiologische Modellierung informierte sowohl Handelsrestriktionen als auch Reiserestriktionen während globaler Gesundheitskrisen.",
    "Obwohl die Exportquoten stiegen, blieben nichttarifäre Barrieren für digitale Dienstleistungen bestehen.",
    "Die kognitive Belastung bei multitaskingbedingter Mediennutzung beeinflusst Aufmerksamkeit und Erinnerungsleistung messbar.",
    "Es bedarf integrierter Nachhaltigkeitsstrategien, die Klimapolitik, Handel und soziale Gerechtigkeit verzahnen.",
    "Die Medienethikkommission empfahl strengere Regeln für den Umgang mit geleakten Regierungsdokumenten.",
    "Trotz günstiger Konjunkturdaten bleibt die Inflation in importabhängigen Sektoren beharrlich erhöht.",
    "Die Grundrechtsdogmatik liefert Kriterien für die Bewertung staatlicher Eingriffe in digitale Kommunikationsräume.",
    "Es lässt sich nicht leugnen, dass globale Handelsketten epidemiologische Risiken über Grenzen hinweg verstärken.",
    "Die neurokognitive Forschung erklärt, warum Menschen komplexe Klimarisiken häufig systematisch unterschätzen.",
    "Angesichts verschärfter Emissionsauflagen müssen exportorientierte Industrien ihre industriellen Produktionsprozesse umfassend umstellen.",
    "Die journalistische Verantwortung bei der Berichterstattung über Pandemien umfasst transparente Unsicherheitskommunikation.",
    "Es ist ratsam, verfassungsrechtliche, ökonomische und ethische Kriterien in Politikberatung systematisch zu integrieren.",
    "Die Analyse internationaler Handelsströme zeigt Abhängigkeiten, die Nachhaltigkeitspolitik strategisch berücksichtigen muss.",
    "Ungeachtet theoretischer Debatten liefert die Evidenz für wirksame Verhaltensinterventionen in der Prävention.",
    "Der Expertenrat verdeutlichte, dass Medienpluralität demokratische Kontrolle staatlicher Machtausübung stützt.",
    "Es bleibt fraglich, ob fiskalische Entlastungen allein die konjunkturelle Erholung in Kernindustrien auslösen.",
    "Die verfassungskonforme Ausgestaltung von CO2-Preisinstrumenten erfordert eine sorgfältig geregelte und transparente Einnahmenverwendung.",
    "Die kognitive Verarbeitung statistischer Risikoinformationen verbessert sich durch gut gestaltete evidenzbasierte Visualisierungsformate.",
    "Es ist absehbar, dass Lieferkettengesetze internationale Handelspraktiken und Medienberichterstattung gleichermaßen prägen.",
    "Die öffentlich-rechtliche Berichterstattung über Wirtschaftspolitik muss komplexe politische Sachverhalte verständlich vermitteln.",
    "Der Policybrief verbindet epidemiologische Daten mit Empfehlungen für resilientere globale Handelsarchitekturen.",
    "Die interdisziplinäre Forschung zu Medien, Kognition und Nachhaltigkeit eröffnet neue Ansätze demokratischer Aufklärung.",
    "Es bleibt offen, ob verfassungsrechtliche, ökonomische und medienethische Standards international harmonisiert werden können.",
]

ALL_BATCHES = [BATCH_1, BATCH_2, BATCH_3, BATCH_4, BATCH_5, BATCH_6, BATCH_7, BATCH_8]
SENTENCES = [s for batch in ALL_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

START_ID = 581
TARGET_PATH = project_root / "data/handcraft/de/train/c1_new_005.conllu"

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
    "angeführte": ("angeführt", "ADJ"), "angeführten": ("angeführt", "ADJ"),
    "veränderten": ("verändert", "ADJ"), "veränderte": ("verändert", "ADJ"),
    "steigender": ("steigend", "ADJ"), "steigenden": ("steigend", "ADJ"),
    "moderater": ("moderat", "ADJ"), "moderaten": ("moderat", "ADJ"),
    "restriktiver": ("restriktiv", "ADJ"), "restriktiven": ("restriktiv", "ADJ"),
    "wachstumsfördernde": ("wachstumsfördernd", "ADJ"), "volatiler": ("volatil", "ADJ"),
    "dicht": ("dicht", "ADJ"), "dichten": ("dicht", "ADJ"),
    "sozioökonomisch": ("sozioökonomisch", "ADJ"), "sozioökonomische": ("sozioökonomisch", "ADJ"),
    "benachteiligten": ("benachteiligte", "ADJ"), "vulnerablen": ("vulnerabel", "ADJ"),
    "neuartiger": ("neuartig", "ADJ"), "strategisch": ("strategisch", "ADJ"),
    "strategische": ("strategisch", "ADJ"), "strategischen": ("strategisch", "ADJ"),
    "geopolitisch": ("geopolitisch", "ADJ"), "geopolitischer": ("geopolitisch", "ADJ"),
    "grenzüberschreitender": ("grenzüberschreitend", "ADJ"), "grenzüberschreitenden": ("grenzüberschreitend", "ADJ"),
    "protektionistischer": ("protektionistisch", "ADJ"), "protektionistischen": ("protektionistisch", "ADJ"),
    "hochauflösenden": ("hochauflösend", "ADJ"), "homogen": ("homogen", "ADJ"),
    "unbewusste": ("unbewusst", "ADJ"), "kulturelle": ("kulturell", "ADJ"),
    "kulturellen": ("kulturell", "ADJ"), "emissionsarme": ("emissionsarm", "ADJ"),
    "strukturschwachen": ("strukturschwach", "ADJ"), "klimaneutraler": ("klimaneutral", "ADJ"),
    "klimaneutralen": ("klimaneutral", "ADJ"), "ambitionierte": ("ambitioniert", "ADJ"),
    "ambitionierten": ("ambitionierten", "ADJ"), "anonymisierter": ("anonymisiert", "ADJ"),
    "anonymisierungsregeln": ("Anonymisierungsregel", "NOUN"),
    "öffentlich-rechtlicher": ("öffentlich-rechtlich", "ADJ"),
    "öffentlich-rechtliche": ("öffentlich-rechtlich", "ADJ"),
    "personalisierte": ("personalisiert", "ADJ"), "importabhängigen": ("importabhängig", "ADJ"),
    "verfassungskonforme": ("verfassungskonform", "ADJ"), "verfassungskonformen": ("verfassungskonform", "ADJ"),
    "exportorientierte": ("exportorientiert", "ADJ"), "exportorientierten": ("exportorientiert", "ADJ"),
    "interdisziplinäre": ("interdisziplinär", "ADJ"),
    "EU": ("EU", "PROPN"), "WTO": ("WTO", "PROPN"), "IWF": ("IWF", "PROPN"),
    "CO2": ("CO2", "NOUN"), "Scope-3": ("Scope-3", "NOUN"), "KI": ("KI", "NOUN"),
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
    "überarbeitet": "überarbeiten", "stützt": "stützen", "stützen": "stützen",
    "verändert": "verändern", "verändern": "verändern", "einräumen": "einräumen",
    "erstreckt": "erstrecken", "könnten": "können",
    "unterliegen": "unterliegen", "dämpft": "dämpfen", "senken": "senken",
    "erwarten": "erwarten", "wirken": "wirken", "zeigen": "zeigen",
    "variieren": "variieren", "belasten": "belasten", "beeinflusst": "beeinflussen",
    "beeinflussen": "beeinflussen", "hängt": "hängen", "bewirken": "bewirken",
    "verringert": "verringern", "verzeichnet": "verzeichnen", "überschreiten": "überschreiten",
    "abbaut": "abbauen", "legt": "legen", "empfahl": "empfehlen", "ausrichten": "ausrichten",
    "steigt": "steigen", "entschärft": "entschärfen", "erschwert": "erschweren",
    "dämpfen": "dämpfen", "untersuchte": "untersuchen", "senken": "senken",
    "moduliert": "modulieren", "erklärt": "erklären", "lieferte": "liefern",
    "lieferten": "liefern", "bleibt": "bleiben", "formulieren": "formulieren",
    "kritisieren": "kritisieren", "übertraf": "übertreffen", "begünstigt": "begünstigen",
    "identifizierte": "identifizieren", "ausbauen": "ausbauen", "berücksichtigt": "berücksichtigen",
    "übersetzen": "übersetzen", "zeigte": "zeigen", "verbesserte": "verbessern",
    "erfassen": "erfassen", "verzerren": "verzerren", "verweist": "verweisen",
    "beeinflussen": "beeinflussen", "erschweren": "erschweren", "verschlechterte": "verschlechtern",
    "blieben": "bleiben", "spielt": "spielen", "profitieren": "profitieren",
    "harmonisiert": "harmonisieren", "wuchs": "wachsen", "reduziert": "reduzieren",
    "destabilisieren": "destabilisieren", "verlangen": "verlangen", "einschränken": "einschränken",
    "bewerten": "bewerten", "sichern": "sichern", "stieg": "steigen",
    "verlängern": "verlängern", "aushandeln": "aushandeln", "entgegenwirken": "entgegenwirken",
    "ersetzen": "ersetzen", "erleichterte": "erleichtern", "bleibt": "bleiben",
    "verändert": "verändern", "beeinflusst": "beeinflussen", "verbindet": "verbinden",
    "zeigten": "zeigen", "trennen": "trennen", "bleibt": "bleiben",
    "betont": "betonen", "formulieren": "formulieren", "finanziert": "finanzieren",
    "deutet": "deuten", "beeinflusst": "beeinflussen", "identifizierte": "identifizieren",
    "berücksichtigen": "berücksichtigen", "stellt": "stellen", "fragen": "fragen",
    "ermöglicht": "ermöglichen", "abbilden": "abbilden", "kündigte": "kündigen",
    "erreichen": "erreichen", "ist": "sein", "verpflichtet": "verpflichten",
    "offenzulegen": "offenlegen", "billigte": "billigen", "erfordern": "erfordern",
    "berücksichtigt": "berücksichtigen", "senken": "senken", "bleibt": "bleiben",
    "verbinden": "verbinden", "genießen": "genießen", "betreffen": "betreffen",
    "fordern": "fordern", "verändert": "verändern", "erschweren": "erschweren",
    "zielen": "zielen", "angepasst": "anpassen", "stärken": "stärken",
    "helfen": "helfen", "einigten": "einigen", "stärkt": "stärken",
    "übersetzt": "übersetzen", "beschleunigen": "beschleunigen", "sparen": "sparen",
    "erfordern": "erfordern", "entspricht": "entsprechen", "verlagern": "verlagern",
    "argumentiert": "argumentieren", "dürfen": "dürfen", "steht": "stehen",
    "kennzeichnen": "kennzeichnen", "erfordert": "erfordern", "wies": "weisen",
    "stärkt": "stärken", "setzt": "setzen", "begrenzen": "begrenzen",
    "kritisieren": "kritisieren", "stellt": "stellen", "erarbeiten": "erarbeiten",
    "beeinträchtigen": "beeinträchtigen", "beschleunigt": "beschleunigen",
    "erfolgen": "erfolgen", "bleibt": "bleiben", "beiträgt": "beitragen",
    "fragmentieren": "fragmentieren", "erhöhen": "erhöhen", "wirken": "wirken",
    "verlangt": "verlangen", "stärken": "stärken", "erfordert": "erfordern",
    "verbindet": "verbinden", "begrenzen": "begrenzen", "informierte": "informieren",
    "blieben": "bleiben", "beeinflusst": "beeinflussen", "verzahnen": "verzahnen",
    "empfahl": "empfehlen", "bleibt": "bleiben", "liefert": "liefern",
    "verstärken": "verstärken", "erklärt": "erklären", "umstellen": "umstellen",
    "umfasst": "umfassen", "integrieren": "integrieren", "zeigt": "zeigen",
    "berücksichtigen": "berücksichtigen", "liefert": "liefern", "stützt": "stützen",
    "auslösen": "auslösen", "erfordert": "erfordern", "verbessert": "verbessern",
    "prägen": "prägen", "vermitteln": "vermitteln", "verbindet": "verbinden",
    "eröffnet": "eröffnen", "prüft": "prüfen", "gewährleisten": "gewährleisten",
    "legte": "legen", "formuliert": "formulieren", "bildet": "bilden",
    "abgesichert": "absichern", "geworden": "werden", "identifizieren": "identifizieren",
    "verhindern": "verhindern", "bestätigt": "bestätigen", "antizipieren": "antizipieren",
    "haben": "haben", "bleibe": "bleiben", "darf": "dürfen", "ist": "sein",
    "müsse": "müssen", "sollen": "sollen", "werden": "werden", "wird": "werden",
    "könnten": "können", "könne": "können", "muss": "müssen",
    "korreliert": "korrelieren",
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