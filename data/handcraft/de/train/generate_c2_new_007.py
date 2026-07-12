"""Generate 140 handcrafted German C2 CoNLL-U sentences (de_c2_train_761–900)."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 6 sub-batches: 25+25+25+25+25+15 = 140 (761–900)
BATCH_1 = [
    "Die soziologische Analyse postmoderner Identitäten zeigt, dass Selbstbeschreibungen zunehmend von fluiden und kontextabhängigen Zugehörigkeiten geprägt sind.",
    "Obschon die Globalisierung wirtschaftliche Vernetzung beschleunigt, verstärkt sie dennoch lokale Ungleichheiten und kulturelle Aneignungskonflikte in peripheren Regionen.",
    "Die Theorie der symbolischen Gewalt erklärt, wie dominante Klassifikationsschemata unsichtbar soziale Hierarchien reproduzieren und als selbstverständlich erscheinen lassen.",
    "Es bedarf einer differenzierten Hermeneutik, um kollektives Gedächtnis nicht vorschnell auf bloße offizielle Gedenkrituale zu reduzieren.",
    "Die Ethnografie urbaner Mikroräume erschließt Praktiken alltäglicher Resistenz, die in aggregierten statistischen Erhebungen systematisch unsichtbar bleiben.",
    "Trotz normativer Universalansprüche bleibt die Menschenrechtsdebatte von geopolitischen Machtinteressen und selektiver moralischer Empörung durchzogen.",
    "Die Dekonstruktion des biologischen Geschlechtsbegriffs legt die normative Konstruiertheit scheinbar natürlicher Kategorien eindringlich offen.",
    "Es gilt, strukturelle Diskriminierung von individuellen Vorurteilen zu scheiden, ohne dabei persönliche Verantwortung vollständig zu suspendieren.",
    "Die Sozialisationstheorie betont, dass Subjektwerdung untrennbar an institutionelle Anerkennungsprozesse und intersubjektive Erwartungen gebunden ist.",
    "Mit bewundernswerter theoretischer Stringenz rekonstruierte die Soziologin die Mechanismen sozialer Exklusion in postindustriellen Dienstleistungsgesellschaften.",
    "Die Kritik des methodologischen Individualismus behauptet, gesellschaftliche Phänomene seien nicht auf aggregierte Einzelentscheidungen zurückführbar.",
    "Es ist von eminenter Bedeutung, empirische Befunde und normative Urteile in der Sozialforschung methodisch sorgfältig voneinander zu trennen.",
    "Die Intersektionalitätsperspektive analysiert, wie Kategorien wie Klasse, Geschlecht und Ethnizität sich gegenseitig modulieren und verstärken.",
    "Ungeachtet digitaler Vernetzung persistieren soziale Milieus, die durch symbolische Grenzziehungen und kulturelle Distinktionsstrategien reproduziert werden.",
    "Die Theorie kommunikativen Handelns verknüpft demokratische Legitimität mit der Möglichkeit unverzerrter und diskursiver Willensbildung in der Öffentlichkeit.",
    "Es bleibt abzuwarten, ob die neue Sozialstudie überarbeitete Konzepte sozialer Kohäsion in fragmentierten Gesellschaften überzeugend operationalisiert.",
    "Die Phänomenologie der Fremdheit beschreibt, wie das Antlitz des Anderen eine ethische Forderung ausübt, die jede instrumentelle Objektivierung untergräbt.",
    "Der Gesellschaftstheoretiker analysierte die Dialektik von Inklusion und Ausgrenzung in supranationalen Migrationsregimen mit großer analytischer Schärfe.",
    "Die Soziologie des Wissens untersucht, unter welchen Bedingungen bestimmte Deutungsmuster als gesellschaftlich verbindlich gelten und andere marginalisiert werden.",
    "Es vermag die rein deskriptive Statistik nicht, die normativen Voraussetzungen zu erfassen, die gesellschaftliche Ungleichheit erst sichtbar machen.",
    "Die Kritik der Ökonomisierung des Sozialen warnt vor der Reduktion demokratischer Teilhabe auf marktähnliche Wettbewerbslogiken und Effizienzkriterien.",
    "Die Bourdieusche Feldanalyse modelliert soziale Räume als Arenen, in denen symbolisches Kapital gegen ökonomische Ressourcen ausgetauscht wird.",
    "Mit unnachgiebiger begrifflicher Klarheit argumentierte sie für die Unverzichtbarkeit reflexiver Distanz in der ethnografischen Feldforschungspraxis.",
    "Die Sozialpathologie moderner Gesellschaften diagnostiziert Einsamkeit, Anomie und Entfremdung als Symptome defizitärer Anerkennungsstrukturen.",
    "Die normative Kraft des Faktischen droht, etablierte Ungleichheitsverhältnisse durch ihre schiere soziale Selbstverständlichkeit zu legitimieren.",
]

BATCH_2 = [
    "Die theologische Hermeneutik der Offenbarung unterscheidet sorgfältig zwischen geschichtlicher Botschaft, metaphysischer Aussage und liturgischer Praxis.",
    "Obschon die Exegese philologische Kriterien anwendet, bleibt die Frage nach dem normativen Anspruch heiliger Texte theologisch unabgeschlossen.",
    "Die negative Theologie behauptet, das Absolute entziehe sich jeder positiven begrifflichen Bestimmung und fordere apophatische Sprechweisen.",
    "Es bedarf einer sorgfältigen Differenzierung, um religiöse Erfahrung nicht vorschnell auf subjektive Psychologie oder soziale Funktion zu reduzieren.",
    "Die Eschatologie der Hoffnung verbindet die Kritik gegenwärtiger Leiden mit der Verheißung einer letztgültigen Versöhnung aller Geschöpflichkeit.",
    "Trotz säkularer Gesellschaftsstrukturen behaupten theologische Debatten nach wie vor Relevanz für Fragen von Schuld, Vergebung und menschlicher Würde.",
    "Die Religionsphilosophie fragt, ob Transzendenz überhaupt ohne Rückgriff auf metaphysische Ontologie sinnvoll behauptet werden kann.",
    "Es gilt, säkulare Moralphilosophie und theologische Ethik in einem produktiven und respektvollen Dialog über gemeinsame Menschlichkeit zu versöhnen.",
    "Die Kritik des Theodizeeproblems untergräbt naive Glaubensgewissheit und zwingt zu der Auseinandersetzung mit dem Leid Unschuldiger.",
    "Mit eindringlicher exegetischer Feinheit erschloss der Theologe die paulinische Rechtfertigungslehre in ihrem historisch kritischen und systematischen Zusammenhang.",
    "Die Sakramentaltheologie versteht sichtbare Zeichen als mediale Vermittlung unsichtbarer Gnade in der konkreten Gemeinschaft der Gläubigen.",
    "Es ist von eminenter Bedeutung, religiöse Pluralität als Chance demokratischer Toleranz und nicht als Bedrohung kultureller Identität zu begreifen.",
    "Die Mystik der Einkehr betont stille Kontemplation als Gegenentwurf zu der hektischen Instrumentalisierung spiritueller Erfahrung in der Gegenwartskultur.",
    "Ungeachtet innerkirchlicher Kontroversen bewahrte die Konferenz den Anspruch, universale ethische Imperative über Konfessionsgrenzen hinweg zu formulieren.",
    "Die Hermeneutik des Glaubens betont, dass Verstehen heiliger Schriften untrennbar an lebensweltliche Vorausverständnisse und Gemeinschaftstraditionen gebunden ist.",
    "Es bleibt abzuwarten, ob die reformierte Dogmatik überarbeitete Trinitätslehren für pluralistische Gesellschaften neu formulieren wird.",
    "Die Theologie der Befreiung verknüpft christliche Hoffnung mit der solidarischen Parteinahme für Unterdrückte und strukturell Ausgeschlossene.",
    "Der Religionshistoriker rekonstruierte die Genese monotheistischer Vorstellungen in dem Kontext antiker nahöstlicher und hellenistischer Religionskontakte.",
    "Die Apophatik des Unsagbaren verweigert jedem positiven Gottesbegriff letzte Ausschöpfung und hält den Horizont des Geheimnisses offen.",
    "Es vermag die rein historische Religionswissenschaft nicht, die existenzielle Bedeutung religiöser Praxis für gläubige Subjekte angemessen zu erfassen.",
    "Die säkulare Kritik der Religion warnt vor der Politisierung heiliger Symbole und der Instrumentalisierung des Transzendenten für ideologische Zwecke.",
    "Die analogia entis behauptet eine proportionale Ähnlichkeit zwischen endlichem Sein und absoluter Transzendenz ohne ontologische Gleichsetzung.",
    "Mit nachdrücklicher pastoraler Sensibilität erörterte der Bischof die Spannungen zwischen kirchlicher Lehre und gesellschaftlichen Pluralisierungsprozessen.",
    "Die Religionsfreiheit als Grundrecht schützt individuelle Gewissensentscheidungen, ohne jedoch kollektive Praktiken von staatlicher Regulierung auszunehmen.",
    "Die theologische Anthropologie versteht den Menschen als relational geschaffenes Wesen, dessen Identität in der Begegnung mit dem Du gründet.",
]

BATCH_3 = [
    "Die makroökonomische Stabilisierungspolitik muss Inflationsbekämpfung, Beschäftigungssicherung und fiskalische Nachhaltigkeit in einem fragilen Gleichgewicht halten.",
    "Obschon die Zentralbank expansiv agierte, blieb die Kreditvergabe an kleine Unternehmen in strukturschwachen Regionen dennoch stark eingeschränkt.",
    "Die Theorie effizienter Märkte behauptet, dass Preise sämtliche verfügbare Information instantan und weitgehend fehlerfrei in sich aufnehmen.",
    "Es bedarf einer differenzierten Regulierung, um systemische Risiken des Finanzsektors zu begrenzen, ohne dabei notwendige Innovationsimpulse zu ersticken.",
    "Die Verteilungswirkung progressiver Besteuerung zielt darauf ab, extreme Vermögenskonzentrationen ohne grundlegende Leistungsanreize zu untergraben.",
    "Trotz robuster Konjunkturdaten bleibt die Arbeitsmarktintegration langzeitarbeitsloser Personen in stark transformierten Branchen hartnäckig problematisch.",
    "Die Ökonomie des Gemeinwohls kritisiert die Reduktion gesellschaftlichen Wohlstands auf das Bruttoinlandsprodukt als verengtes Messinstrument.",
    "Es gilt, kurzfristige Konjunkturprogramme von langfristigen Strukturreformen zu unterscheiden, ohne politische Handlungsfähigkeit zu blockieren.",
    "Die monetäre Transmission geldpolitischer Entscheidungen verzögert sich häufig, weil Erwartungsbildung und Kreditkanäle komplex intervenierende Variablen darstellen.",
    "Mit souveräner analytischer Präzision bewertete der Ökonom die Tragfähigkeit der Staatsverschuldung unter veränderten Zins und Wachstumsannahmen.",
    "Die Externalitäten internalisierender Umweltpolitik verlangt, dass ökologische Kosten in Marktpreisen sichtbar werden und Verursacher belastet werden.",
    "Es ist von eminenter Bedeutung, wettbewerbsrechtliche Rahmenbedingungen so zu gestalten, dass marktbeherrschende Stellungen nicht dauerhaft zementiert werden.",
    "Die Phillipskurve beschreibt historisch eine inverse Beziehung zwischen Inflation und Arbeitslosigkeit, deren Stabilität in der Gegenwart umstritten ist.",
    "Ungeachtet optimistischer Prognosen zeigte die Handelsbilanz weiterhin deutliche strukturelle Ungleichgewichte in zentralen bilateralen Wirtschaftsbeziehungen.",
    "Die ordnungspolitische Verantwortung des Staates begrenzt wettbewerbsverzerrende Subventionen und sichert gleiche Chancen für marktwirtschaftliche Akteure.",
    "Es bleibt abzuwarten, ob die geplante Steuerreform tatsächlich mittelständische Investitionen stimuliert und gleichzeitig fiskalische Spielräume wahrt.",
    "Die Theorie komparativer Kostenvorteile erklärt den internationalen Handel durch unterschiedliche Produktionsbedingungen und spezialisierte Arbeitsteilung.",
    "Der Finanzwissenschaftler modellierte die Volatilität derivativer Instrumente unter extremen Marktstressszenarien mit bemerkenswerter rechnerischer Genauigkeit.",
    "Die soziale Marktwirtschaft verbindet wettbewerbliche Effizienz mit solidarischen Ausgleichsmechanismen und staatlicher Rahmensetzung für fairen Wettbewerb.",
    "Es vermag die rein aggregierte Konjunkturanalyse nicht, die mikroökonomischen Ursachen regionaler Strukturschwäche und industrieller Abwanderung zu erklären.",
    "Die Schuldenbremse zwingt die Haushaltspolitik zu mittelfristiger Ausgabendisziplin und begrenzt die Finanzierung dauerhafter Leistungsausweitungen.",
    "Die Investitionslenkung durch grüne Förderprogramme soll Dekarbonisierung beschleunigen, ohne marktverzerrende Dauersubventionen zu institutionalisieren.",
    "Mit unnachgiebiger methodischer Strenge widerlegte sie die Behauptung, Inflationserwartungen seien dauerhaft ankerlos und unkontrollierbar geworden.",
    "Die Geldpolitik der Nullzinsphase erschwert konventionelle Stimulierungsinstrumente und erzwingt unkonventionelle Ankaufprogramme zu der Liquiditätsversorgung.",
    "Die intergenerationelle Gerechtigkeit verlangt, dass heutige staatliche Haushaltsentscheidungen künftige Belastungsspielräume nicht unverhältnismäßig einschränken.",
]

BATCH_4 = [
    "Die Bioethik der Gentechnik muss therapeutischen Nutzen, Würdegrenzen und unbekannte Langzeitfolgen genetischer Eingriffe sorgfältig gegeneinander abwägen.",
    "Obschon Algorithmen Entscheidungen beschleunigen, reproduzieren sie dennoch oft verborgene Vorurteile der zugrunde liegenden Trainingsdatensätze.",
    "Die Theorie der technologischen Singularität behauptet, künstliche Intelligenz könnte menschliche Kognitionsfähigkeit in absehbarer Zeit grundlegend übertreffen.",
    "Es bedarf transparenter Governancestrukturen, um autonome Systeme verantwortlich zu entwickeln, ohne Innovationsprozesse unnötig zu lähmen.",
    "Die Datenschutzgrundverordnung verankert informationelle Selbstbestimmung als grundlegendes europäisches Grundrecht gegen kommerzielle und staatliche Datenmacht.",
    "Trotz beeindruckender medizinischer Fortschritte bleibt die Zulässigkeit pränataler Selektion ethisch wie rechtlich hochgradig umstritten.",
    "Die Kritik des digitalen Überwachungskapitalismus warnt vor der Kommodifizierung subjektiver Erfahrungen durch allgegenwärtige Verhaltensanalyse und Profilbildung.",
    "Es gilt, technische Machbarkeit von normativer Zulässigkeit zu trennen, bevor gesellschaftlich folgenreiche Automatisierungsentscheidungen getroffen werden.",
    "Die Cybersicherheit strategisch kritischer Infrastrukturen erfordert resiliente Architekturen gegen staatlich geführte und kriminell motivierte Angriffe.",
    "Mit akribischer regulatorischer Weitsicht entwarf die Kommission einen Rahmen für vertrauenswürdige und erklärbare Entscheidungssysteme künstlicher Intelligenz.",
    "Die Debatte über Enhancement fragt, ob pharmakologische oder technische Leistungssteigerung den Begriff menschlicher Natürlichkeit grundlegend in Frage stellt.",
    "Es ist von eminenter Bedeutung, digitale Teilhabe als Voraussetzung demokratischer Gleichheit und nicht als bloßes Konsumgut zu verstehen.",
    "Die Blockchaintechnologie verspricht dezentralisierte Vertrauensprotokolle, ohne jedoch zentrale Verantwortlichkeiten vollständig ersetzen zu können.",
    "Ungeachtet technischer Eleganz scheiterten mehrere ehrgeizige Projekte smarter Städte an mangelnder Bürgerbeteiligung und unzureichenden Datenschutzkonzepten.",
    "Die Robotikethik untersucht, welche autonomen Handlungen Maschinen moralisch zugerechnet werden können und wo menschliche Aufsicht unverzichtbar bleibt.",
    "Es bleibt abzuwarten, ob die neue Verordnung über KI tatsächlich Hochrisikoanwendungen wirksam kontrolliert und gleichzeitig Forschungsfreiräume bewahrt.",
    "Die posthumanistische Perspektive hinterfragt anthropozentrische Vorstellungen und eröffnet Debatten über Rechte nicht menschlicher Agenten und Ökosysteme.",
    "Der Technikphilosoph analysierte die Verschränkung von Medizintechnik und Selbstverständnis chronisch kranker Patienten in digitalisierten Versorgungsketten.",
    "Die digitale Souveränität eines Staates hängt davon ab, kritische Technologien nicht vollständig externen Plattform und Lieferkettenabhängigkeiten auszuliefern.",
    "Es vermag die rein ingenieurwissenschaftliche Perspektive nicht, die gesellschaftlichen Machteffekte algorithmischer Sortierung und Bewertung angemessen zu erfassen.",
    "Die Präzisionsmedizin verspricht individualisierte Therapien, wirft jedoch Fragen nach gerechter Verteilung teurer Behandlungen in solidarischen Systemen auf.",
    "Die Kritik der These des Technikdeterminismus betont, dass gesellschaftliche Institutionen die Entwicklung und Nutzung von Technologien aktiv mitgestalten.",
    "Mit nachdrücklicher ethischer Stringenz lehnte der Ethikrat die uneingeschränkte kommerzielle Nutzung genetischer Risikoprofile ohne informierte Einwilligung ab.",
    "Die Interoperabilität digitaler Gesundheitsdaten erfordert einheitliche Standards, ohne patientennahe Vertraulichkeit und ärztliche Schweigepflicht aufzuweichen.",
    "Die Verantwortung des Ingenieurs endet nicht mit technischer Funktionsfähigkeit, sondern umfasst auch vorhersehbare Missbrauchs und Folgeschäden.",
]

BATCH_5 = [
    "Die komparatistische Literaturwissenschaft untersucht, wie Übersetzungen kulturelle Bedeutungen transformieren und nationale Kanonbildungen transnational durchbrechen.",
    "Obschon die Oper traditionsbewusst inszeniert, experimentiert sie dennoch mit multimedialen Bühnenformen und dekonstruierten Aufführungskonventionen.",
    "Die Musiksemiotik analysiert, wie Leitmotive narrative Spannungsbögen strukturieren und emotionale Erwartungen bei dem aufmerksamen Hörer aktivieren.",
    "Es bedarf einer nuancierten Analyse, um ästhetische Autonomie des Kunstwerks von kommerziellen Vermarktungslogiken sorgfältig zu unterscheiden.",
    "Die Architekturtheorie der Nachhaltigkeit verbindet ökologische Materialwahl mit räumlicher Lebensqualität und demokratischer Nutzbarkeit öffentlicher Bauten.",
    "Trotz digitaler Reproduktion behauptet die Liveperformance nach wie vor eine unersetzliche Präsenz, die jede technische Mediation nur annähernd vermittelt.",
    "Die Filmsemiotik untersucht, wie Schnittrhythmus, Kameraperspektive und Licht dramaturgische Bedeutung unabhängig von expliziter Dialogebene erzeugen.",
    "Es gilt, populäre Kultur nicht pauschal von Hochkultur zu trennen, ohne historisch gewachsene Hierarchien kritisch zu reflektieren.",
    "Die Rezeptionsästhetik des Theaters betont die kokreative Rolle des Publikums bei der Erschließung polyvalenter Bühnenbedeutungen.",
    "Mit bewundernswerter choreografischer Präzision verband die Tänzerin expressiven Körperausdruck mit konzeptueller Kritik normativer Geschlechterrollen.",
    "Die Museologie der Gegenwart hinterfragt koloniale Provenienzen und fordert restitutive Praktiken gegenüber enteigneten Kulturgütern.",
    "Es ist von eminenter Bedeutung, kulturelle Vielfalt als dynamischen Prozess zu begreifen und nicht als statisches Inventar zu versteinern.",
    "Die Ästhetik des Bruchs charakterisiert avantgardistische Kompositionen, die konventionelle Harmonien bewusst verweigern und neue Hörgewohnheiten erzwingen.",
    "Ungeachtet kommerzieller Erfolge hielt das Festival an seinem Anspruch fest, experimentellen Künstlern unabhängige und risikofreudige Produktionsbedingungen zu bieten.",
    "Die Dramaturgie des dokumentarischen Theaters vermischt Zeugnis, Fiktion und Archivmaterial zu einer kritischen Auseinandersetzung mit gegenwärtiger Politik.",
    "Es bleibt abzuwarten, ob die neue Ausstellung überarbeitete Narrative postkolonialer Kunstgeschichte für breitere Öffentlichkeiten zugänglich machen wird.",
    "Die Klanginstallation verwandelt den Ausstellungsraum in ein immersives Feld, das wahrnehmbare und imaginative Grenzen des Hörerlebnisses verschiebt.",
    "Der Filmhistoriker rekonstruierte die Entstehung des Neorealismus als ästhetische und politische Antwort auf faschistische Propagandacinematografie.",
    "Die Intermedialität der Moderne bricht Gattungsgrenzen und verweigert eindeutige Zuordnungen zwischen bildender Kunst, Literatur und performativen Praktiken.",
    "Es vermag die rein kommerzielle Kritik nicht, die symbolische Komplexität experimenteller Kurzfilme für ein aufmerksames Publikum angemessen zu erschließen.",
    "Die Kulturpolitik öffentlicher Institutionen muss Zugänglichkeit, künstlerische Freiheit und gesellschaftliche Repräsentation in ausgewogenem Verhältnis wahren.",
    "Die Poetik der Erinnerung verknüpft autobiographische Schreibweisen mit kollektiven Traumata und hinterfragt naive Authentizitätsansprüche des Zeugnisses.",
    "Mit unnachgiebiger kuratorischer Konsequenz setzte sie unterrepräsentierten Künstlerinnen in einer revisionistischen Neuordnung des Saisonprogramms sichtbare Akzente.",
    "Die Opernregie der Gegenwart dekonstruiert historisierende Bühnenbilder und eröffnet ambivalente Deutungen klassischer Werke für neue Zuschauerschaften.",
    "Die kulturwissenschaftliche Analyse von Alltagsästhetik hebt scheinbar banale Praktiken in den Fokus gesellschaftlicher Sinnstiftung und Identitätsbildung.",
]

BATCH_6 = [
    "Die integrationstheoretische Synthese verbindet erkenntnistheoretische, rechtsphilosophische und ästhetische Perspektiven zu einer kohärenten Gesellschaftsanalyse.",
    "Obschon die Disziplinen getrennte Methoden pflegen, konvergieren sie dennoch in der Frage nach Legitimität normativer Urteile unter pluralen Bedingungen.",
    "Die transnationale Rechtsordnung erschwert klassische Souveränitätsvorstellungen und erzwingt neue Modelle demokratischer Kontrolle supranationaler Institutionen.",
    "Es bedarf einer reflexiven Wissenschaftskultur, die Eigenvorurteile offenlegt und epistemische Vielfalt institutionell anerkennt.",
    "Die philosophische Anthropologie der Technik hinterfragt, ob menschliche Freiheit in zunehmend automatisierten Lebenswelten überhaupt noch tragfähig bleibt.",
    "Trotz divergierender Fachkulturen teilen Geistes und Naturwissenschaften die Herausforderung, komplexe Phänomene ohne unzulässige Reduktion zu erklären.",
    "Die Kritik des Eurozentrismus in der Wissenschaftsgeschichte legt verborgene Machtasymmetrien in scheinbar universalen Erkenntnisansprüchen eindringlich offen.",
    "Es gilt, akademische Exzellenz und gesellschaftliche Verantwortung als sich wechselseitig bedingende und nicht als antagonistische Zielsetzungen zu verstehen.",
    "Die Hermeneutik der Moderne beschreibt Subjektivität als historisch gewordene und zugleich normativ beanspruchte Form selbstreflexiven Bewusstseins.",
    "Mit souveräner intellektueller Breite verknüpfte der Gelehrte rechtsphilosophische Argumente mit kulturhistorischen und erkenntnistheoretischen Befunden.",
    "Die interdisziplinäre Forschungspraxis erfordert Übersetzungsleistungen zwischen Fachsprachen, ohne methodische Standards beliebig zu relativieren.",
    "Es ist von eminenter Bedeutung, universitäre Bildung nicht auf berufliche Anpassung zu reduzieren, sondern als Ort kritischer Urteilsbildung zu wahren.",
    "Die Zukunft demokratischer Öffentlichkeiten hängt davon ab, ob komplexe Sachverhalte verständlich vermittelt werden, ohne sie intellektuell zu verflachen.",
    "Ungeachtet zunehmender Spezialisierung bleibt die Fähigkeit zu der synthetischen Urteilsbildung eine unverzichtbare Voraussetzung verantwortlichen Handelns.",
    "Die humanistische Tradition behauptet, dass Bildung den Menschen zu der Selbstbestimmung befähigt und nicht bloß zu funktionaler Anpassung an Märkte erzieht.",
]

BATCHES = [BATCH_1, BATCH_2, BATCH_3, BATCH_4, BATCH_5, BATCH_6]
SENTENCES = [s for batch in BATCHES for s in batch]

assert len(SENTENCES) == 140, f"Expected 140 sentences, got {len(SENTENCES)}"

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
    "Obgleich": ("obgleich", "SCONJ"),
    "obgleich": ("obgleich", "SCONJ"),
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
    "bevor", "nachdem", "sodass", "damit", "indem", "bevor",
}

CCONJ_FORMS = {"und", "oder", "aber", "sondern", "sowie"}
ADP_FORMS = {
    "um", "für", "mit", "von", "bei", "nach", "aus", "über", "vor", "unter", "durch",
    "ohne", "gegen", "wegen", "in", "an", "auf", "trotz", "gegenüber", "entlang",
}
PART_FORMS = {"nicht", "ja", "nein", "doch", "nur", "auch", "schon", "noch", "bloß", "allein", "dennoch"}
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
    "entrinnen": "entrinnen",
    "überließen": "überlassen",
}


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    if form in SPECIAL_LEMMAS:
        return SPECIAL_LEMMAS[form]

    if form.lower() == "das" and upos == "PRON":
        return "der", "PRON"
    if form.lower() in {"die", "der", "den", "dem", "des"} and upos == "PRON":
        return "der", "PRON"

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
    start_id = 761
    conllu_text = build_conllu(SENTENCES, start_id)

    target_path = project_root / "data/handcraft/de/train/c2_new_007.conllu"
    target_path.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {len(SENTENCES)} sentences to {target_path}")

    validation_res = validate_text(conllu_text)
    print(f"Validate: count={validation_res.sentence_count}, passed={validation_res.passed}")
    if not validation_res.passed:
        for err in validation_res.errors[:50]:
            print("  ", err)
        sys.exit(1)

    lemma_res = check_text(conllu_text, lang="de")
    print(f"Lemma check: passed={lemma_res.passed}")
    if not lemma_res.passed:
        for err in lemma_res.errors[:50]:
            print("  ", err)
        sys.exit(1)

    import re

    blocks = conllu_text.strip().split("\n\n")
    bad: list[tuple[str, int]] = []
    for block in blocks:
        if not block.startswith("#"):
            continue
        sid_m = re.search(r"sent_id = (\S+)", block)
        if not sid_m:
            continue
        sid = sid_m.group(1)
        n = sum(1 for line in block.split("\n") if line and not line.startswith("#"))
        if not (15 <= n <= 25):
            bad.append((sid, n))
    if bad:
        print(f"Token count violations: {len(bad)}")
        for sid, cnt in bad[:20]:
            print(f"  {sid}: {cnt} tokens")
        sys.exit(1)
    print("Token counts: all 15-25")

    first_id = 761
    last_id = 900
    print(f"Sent_ids: de_c2_train_{first_id} – de_c2_train_{last_id}")
    print("Status: OK")


if __name__ == "__main__":
    main()