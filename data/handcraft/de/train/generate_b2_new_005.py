"""Generate b2_new_005.conllu: 200 B2 German sentences (de_b2_train_534–733)."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 8 batches × 25 sentences = 200
BATCHES: list[list[str]] = [
    # Batch 1 (534–558): Politics & Democracy
    [
        "Obwohl die Opposition Bedenken äußerte, wurde das Reformgesetz gestern verabschiedet.",
        "Der Minister betonte, die Regierung werde die Bürgerrechte nach wie vor schützen.",
        "Es wird erwartet, dass die Wahlbeteiligung in diesem Jahr deutlich steigen wird.",
        "Die Demokratie lebt davon, dass unterschiedliche Meinungen respektvoll diskutiert werden.",
        "Obschon die Kritik laut wurde, hielt die Kommission an ihrem Beschluss fest.",
        "Die Verfassung garantiert jedem Bürger das Recht auf freie Meinungsäußerung.",
        "Würden mehr Frauen im Parlament sitzen, könnte die Politik ausgewogener werden.",
        "Der Bericht legt nahe, dass Korruption noch immer ein ernstes Problem sei.",
        "Die Bürgerinitiative setzte sich für eine transparente Verwaltung der öffentlichen Mittel ein.",
        "Nach der Debatte wurde beschlossen, die Steuerreform im Frühjahr umzusetzen.",
        "Obwohl die Umfragen schwankten, blieb die Regierungspartei in der Führung.",
        "Es sei wichtig, dass alle Stimmen im demokratischen Prozess gehört werden.",
        "Die Opposition forderte, der Premierminister solle sich der Verantwortung stellen.",
        "Die Einführung des neuen Wahlrechts wurde von Experten gründlich geprüft.",
        "Hätte man früher gehandelt, wäre die Krise möglicherweise vermeidbar gewesen.",
        "Die Parteien einigten sich darauf, gemeinsam an einer Klimastrategie zu arbeiten.",
        "Der Sprecher erklärte, die Lage unterscheide sich grundlegend von der Vergangenheit.",
        "Es wird davon ausgegangen, dass die Verhandlungen noch mehrere Wochen dauern.",
        "Obwohl die Proteste andauerten, verweigerte das Parlament jede Kompromissbereitschaft.",
        "Die Verwaltung muss die Forderung nach mehr Bürgerbeteiligung ernst nehmen.",
        "Wäre die Koalition geschlossen aufgetreten, hätte man schneller Fortschritte erzielen können.",
        "Der zuständige Ausschuss empfahl, die staatlichen Subventionen schrittweise abzubauen.",
        "Die Presse berichtete, der Staat habe seine Versprechen nur teilweise eingelöst.",
        "Es mangelt nicht an Ideen, sondern an der politischen Durchsetzungskraft.",
        "Die Reform soll gewährleisten, dass alle Regionen gleichberechtigt berücksichtigt werden.",
    ],
    # Batch 2 (559–583): Science & Research
    [
        "Die Hypothese wurde durch mehrere unabhängige Studien eindrucksvoll bestätigt.",
        "Der Forscher erklärte, das Experiment lasse sich nur unter Laborbedingungen wiederholen.",
        "Obwohl die Daten unvollständig waren, lieferte die Analyse wertvolle Erkenntnisse.",
        "Es wird angenommen, dass die Entdeckung weitreichende Folgen für die Medizin haben werde.",
        "Die Veröffentlichung der Ergebnisse erfolgte erst nach einer strengen fachlichen Begutachtung.",
        "Obschon die Theorie umstritten ist, findet sie in der Fachwelt breite Unterstützung.",
        "Die Messung der Temperatur wurde mit modernsten Sensoren durchgeführt.",
        "Würde man die Proben anders behandeln, könnten die Resultate deutlich abweichen.",
        "Der Professor betonte, die Forschung sei noch lange nicht abgeschlossen.",
        "Es ist ratsam, die statistischen Auswertungen vor der Publikation nochmals zu überprüfen.",
        "Die Entwicklung neuer Impfstoffe erfordert jahrelange klinische Tests und Zulassungsverfahren.",
        "Nach der Konferenz wurde beschlossen, die Zusammenarbeit zwischen den Instituten zu vertiefen.",
        "Obwohl die Finanzierung knapp war, gelang es dem Team, das Projekt fortzuführen.",
        "Die Auswirkungen des Klimawandels auf die Artenvielfalt werden intensiv untersucht.",
        "Es werde erwartet, dass die nächste Generation von Batterien deutlich effizienter sei.",
        "Die Durchführung des Experiments wurde wegen technischer Probleme vorübergehend unterbrochen.",
        "Hätte man die Variable berücksichtigt, wäre das Ergebnis wahrscheinlich anders ausgefallen.",
        "Der Bericht weist darauf hin, dass weitere Forschung dringend notwendig sei.",
        "Die Anwendung der neuen Methode führte zu einer erheblichen Verbesserung der Genauigkeit.",
        "Obschon die Ergebnisse vielversprechend sind, mahnt die Wissenschaft zu großer Vorsicht.",
        "Es wird davon abgeraten, die ersten Befunde voreilig zu verallgemeinern.",
        "Die Erforschung des Weltraums erfordert internationale Kooperation und enorme finanzielle Mittel.",
        "Der Autor argumentiert, die bisherige Erklärung des Phänomens sei unzureichend.",
        "Die Validierung der Messwerte erfolgte mithilfe eines unabhängigen Referenzverfahrens.",
        "Wären die Rahmenbedingungen günstiger gewesen, hätte die Studie schneller abgeschlossen werden können.",
    ],
    # Batch 3 (584–608): Society & Social Issues
    [
        "Die Integration von Migranten wird in vielen Ländern kontrovers diskutiert.",
        "Obwohl die Arbeitslosigkeit gesunken ist, bleibt die soziale Ungleichheit bestehen.",
        "Es wird befürchtet, dass die Altersarmut in den kommenden Jahren zunehmen werde.",
        "Die Gleichstellung der Geschlechter ist ein zentrales Anliegen der modernen Gesellschaft.",
        "Obschon viele Fortschritte erzielt wurden, bestehen im Alltag noch immer Vorurteile.",
        "Der Soziologe erklärte, die Struktur der Familie habe sich grundlegend verändert.",
        "Die Bekämpfung der Kinderarmut erfordert gezielte politische Maßnahmen und gesellschaftliches Engagement.",
        "Es sei wichtig, dass benachteiligte Gruppen besser am gesellschaftlichen Leben teilhaben.",
        "Die Verbreitung von Desinformation gefährdet das Vertrauen in demokratische Institutionen.",
        "Würde man mehr in Bildung investieren, könnte die Chancengleichheit deutlich verbessert werden.",
        "Die Senkung der Wohnsubventionen wurde von Wohlfahrtsverbänden scharf kritisiert.",
        "Obwohl die Stadt wächst, fehlt es an bezahlbarem Wohnraum für junge Familien.",
        "Es wird erwartet, dass die Urbanisierung in den nächsten Jahrzehnten weiter zunehmen wird.",
        "Die Stärkung des bürgerschaftlichen Engagements wird als Schlüssel zu dieser Lösung angesehen.",
        "Hätte die Politik früher reagiert, wäre die Wohnungsnot weniger dramatisch geworden.",
        "Der Bericht dokumentiert, wie sich die Lebensbedingungen in ländlichen Regionen verschlechtert haben.",
        "Die Förderung der Inklusion behinderter Menschen ist gesetzlich verankert worden.",
        "Obschon die Debatte emotional geführt wird, sind sachliche Argumente unerlässlich.",
        "Es mangelt nicht an gutem Willen, sondern an konkreten Umsetzungsstrategien.",
        "Die Entwicklung sozialer Wohnprojekte soll die Segregation in den Städten verringern.",
        "Der Experte betonte, die Solidarität in der Gesellschaft dürfe nicht nachlassen.",
        "Die Auswirkungen der Digitalisierung auf den Arbeitsmarkt werden kontrovers bewertet.",
        "Es wird empfohlen, die sozialen Sicherungssysteme an die demografische Entwicklung anzupassen.",
        "Obwohl die Kriminalitätsrate gesunken ist, fühlen sich viele Bürger unsicher.",
        "Die Schaffung neuer Arbeitsplätze in strukturschwachen Regionen bleibt eine große Herausforderung.",
    ],
    # Batch 4 (609–633): Technology & Digitalization
    [
        "Die Digitalisierung der Verwaltung soll den Bürgern den Zugang zu Dienstleistungen erleichtern.",
        "Obwohl die Technologie fortschrittlich ist, werden Datenschutzbedenken immer lauter.",
        "Es wird prognostiziert, dass künstliche Intelligenz viele Berufsbilder grundlegend verändern werde.",
        "Die Entwicklung autonomer Fahrzeuge erfordert neue gesetzliche Rahmenbedingungen und Sicherheitsstandards.",
        "Obschon die Software leistungsfähig ist, mangelt es an benutzerfreundlichen Oberflächen.",
        "Der Ingenieur erklärte, das System lasse sich nur schwer in bestehende Netzwerke integrieren.",
        "Die Speicherung sensibler Daten in der Cloud wirft erhebliche Sicherheitsfragen auf.",
        "Es sei ratsam, vor der Einführung neuer Technologien eine gründliche Risikoanalyse durchzuführen.",
        "Die Verbreitung von Smartphones hat die Art der Kommunikation nachhaltig verändert.",
        "Würde man die Algorithmen transparenter gestalten, könnte das Vertrauen der Nutzer steigen.",
        "Die Modernisierung der digitalen Infrastruktur wurde mit erheblichen Kosten und Zeitaufwand verbunden.",
        "Obwohl die Bandbreite ausreichend war, traten bei der Videokonferenz technische Störungen auf.",
        "Es wird erwartet, dass die nächste Softwareversion zahlreiche Fehler beheben werde.",
        "Die Anwendung von Blockchaintechnologie in der Finanzbranche wird intensiv erforscht.",
        "Hätte man die Sicherheitslücken früher entdeckt, wäre der Datenverlust vermeidbar gewesen.",
        "Der Bericht empfiehlt, die Cybersicherheit als strategische Priorität zu behandeln.",
        "Die Automatisierung von Produktionsprozessen führt zu einer Steigerung der Wirtschaftlichkeit.",
        "Obschon die Innovation vielversprechend klingt, sind die langfristigen Folgen ungewiss.",
        "Es wird davon ausgegangen, dass die Digitalisierung die Bildung grundlegend transformieren werde.",
        "Die Einrichtung eines landesweiten Glasfasernetzes erfordert massive öffentliche Investitionen.",
        "Der Experte warnte, die Abhängigkeit von ausländischen Technologieanbietern nehme stetig zu.",
        "Die Entwicklung nachhaltiger Energietechnologien ist für die Bekämpfung des Klimawandels unerlässlich.",
        "Es werde betont, dass der Schutz der Privatsphäre im digitalen Zeitalter oberste Priorität habe.",
        "Obwohl die Plattform beliebt ist, wird ihre Marktmacht zunehmend kritisch hinterfragt.",
        "Die Einführung des neuen Verschlüsselungsverfahrens wurde von Sicherheitsexperten positiv bewertet.",
    ],
    # Batch 5 (634–658): Economy & Environment
    [
        "Die Inflation hat in den letzten Monaten die Kaufkraft der Verbraucher spürbar geschwächt.",
        "Obwohl die Wirtschaft wächst, bleibt die Zahl der prekären Beschäftigungsverhältnisse hoch.",
        "Es wird befürchtet, dass die nächste Rezession die Staatshaushalte erheblich belasten werde.",
        "Die Reduzierung der Treibhausgasemissionen erfordert einen grundlegenden Umbau der Industrie.",
        "Obschon die Energiewende voranschreitet, hängt der Ausstieg aus der Kohle noch ab.",
        "Der Ökonom erklärte, die Zinsen würden voraussichtlich in den kommenden Quartalen steigen.",
        "Die Förderung erneuerbarer Energien wurde im Haushalt mit zusätzlichen Mitteln ausgestattet.",
        "Es sei notwendig, die Nachhaltigkeit der Lieferketten strenger zu überwachen und zu kontrollieren.",
        "Die Verschärfung der Umweltauflagen führte zu Protesten in der verarbeitenden Industrie.",
        "Würde man die Subventionen umleiten, könnte der Ausbau der Windkraft deutlich beschleunigt werden.",
        "Die Stabilisierung der Finanzmärkte hängt von der Geldpolitik der Zentralbank ab.",
        "Obwohl die Exporte zulegen, belasten steigende Rohstoffpreise die Gewinnmargen der Unternehmen.",
        "Es wird erwartet, dass die Arbeitslosigkeit im kommenden Jahr leicht zurückgehen werde.",
        "Die Bewirtschaftung der natürlichen Ressourcen muss langfristig und verantwortungsvoll erfolgen.",
        "Hätte man früher in den Klimaschutz investiert, wären die Anpassungskosten geringer ausgefallen.",
        "Der Bericht dokumentiert, wie sich die Erderwärmung auf die Landwirtschaft auswirkt.",
        "Die Einführung einer CO2 Abgabe wurde von Umweltschützern als längst überfällig bezeichnet.",
        "Obschon die Konjunktur robust wirkt, warnen Analysten vor versteckten systemischen Risiken.",
        "Es mangelt nicht an Konzepten, sondern an der Bereitschaft zu konsequentem politischem Handeln.",
        "Die Sanierung der maroden Infrastruktur erfordert Koordination zwischen Bund, Ländern und Kommunen.",
        "Der Minister betonte, die Energieversorgungssicherheit dürfe unter keinen Umständen gefährdet werden.",
        "Die Ausweitung des Naturschutzgebietes wurde von Umweltverbänden ausdrücklich begrüßt.",
        "Es wird angenommen, dass die Kreislaufwirtschaft in Zukunft an Bedeutung gewinnen werde.",
        "Obwohl die Renditen attraktiv sind, bergen spekulative Investitionen erhebliche Verlustrisiken.",
        "Die Umstellung auf klimaneutrale Produktionsverfahren stellt viele mittelständische Betriebe vor Herausforderungen.",
    ],
    # Batch 6 (659–683): Education & Culture
    [
        "Die Reform des Bildungssystems soll die Chancengleichheit für alle Schüler verbessern.",
        "Obwohl die Universitäten überfüllt sind, fehlt es an qualifizierten Lehrkräften.",
        "Es wird erwartet, dass die Digitalisierung des Unterrichts neue Lernmethoden ermöglichen werde.",
        "Die Förderung des bilingualen Unterrichts wird als Investition in die Zukunft der Jugend gesehen.",
        "Obschon die Bibliothek renoviert wurde, reichen die Öffnungszeiten für viele Studierende nicht aus.",
        "Der Rektor erklärte, die Qualität der Lehre müsse an erster Stelle stehen.",
        "Die Einführung von Kompetenznachweisen soll die Anerkennung beruflicher Qualifikationen erleichtern.",
        "Es sei wichtig, dass kulturelle Vielfalt in Schulen aktiv gefördert und gewürdigt werde.",
        "Die Finanzierung der Museen hängt zunehmend von privaten Stiftungen und Sponsoren ab.",
        "Würde man mehr in die Lehrerausbildung investieren, könnte der Fachkräftemangel gelindert werden.",
        "Die Durchführung des Austauschprogramms wurde wegen der Pandemie vorübergehend ausgesetzt.",
        "Obwohl die Nachfrage groß ist, sind die Plätze in den Integrationskursen begrenzt.",
        "Es wird empfohlen, die Curricula regelmäßig an die Anforderungen des Arbeitsmarktes anzupassen.",
        "Die Bewahrung des kulturellen Erbes erfordert kontinuierliche Restaurierungsarbeiten und finanzielle Mittel.",
        "Hätte die Schule früher interveniert, wäre der Leistungsrückstand möglicherweise vermieden worden.",
        "Der Kulturminister betonte, die Kunstfreiheit sei ein unverzichtbares Gut der Demokratie.",
        "Die Stärkung des Leseförderprogramms zeigt bereits erste positive Wirkungen in den Grundschulen.",
        "Obschon die Theaterkassen leiden, finden weiterhin ehrgeizige und innovative Produktionen statt.",
        "Es wird davon ausgegangen, dass lebenslanges Lernen in Zukunft zur Norm werden werde.",
        "Die Ausstellung zeitgenössischer Kunst zog in den ersten Wochen zahlreiche Besucher an.",
        "Der Professor argumentierte, die akademische Freiheit dürfe nicht durch politischen Druck eingeschränkt werden.",
        "Die Verbesserung der Studienbedingungen wurde im Hochschulvertrag verbindlich festgeschrieben.",
        "Es werde betont, dass interkulturelle Kompetenz heute eine Schlüsselqualifikation im Berufsleben sei.",
        "Obwohl die Prüfung anspruchsvoll war, bestanden die meisten Teilnehmer mit guten Ergebnissen.",
        "Die Etablierung neuer Forschungszentren soll die internationale Wettbewerbsfähigkeit der Hochschulen stärken.",
    ],
    # Batch 7 (684–708): Law, Justice & Health
    [
        "Die Reform des Strafgesetzbuches wurde nach langen Beratungen endlich verabschiedet.",
        "Obwohl die Beweise belastend waren, sprach die Jury den Angeklagten frei.",
        "Es wird erwartet, dass das neue Gesetz zum Schutz von Whistleblowern bald in Kraft trete.",
        "Die Gewährleistung eines fairen Verfahrens ist ein Grundpfeiler des Rechtsstaats.",
        "Obschon die Klage eingereicht wurde, dauert die gerichtliche Verhandlung noch Monate an.",
        "Der Anwalt erklärte, sein Mandant habe das Recht auf ein unabhängiges Gericht.",
        "Die Bekämpfung der organisierten Kriminalität erfordert internationale Zusammenarbeit und Datenaustausch.",
        "Es sei unerlässlich, dass die Rechte der Angeklagten in jedem Verfahren gewahrt bleiben.",
        "Die Einführung von Videoverhandlungen wurde als effiziente Lösung für überlastete Gerichte begrüßt.",
        "Würde man die Strafen verschärfen, könnte die Abschreckungswirkung möglicherweise steigen.",
        "Die Prävention von Krankheiten durch Impfungen hat die Lebenserwartung weltweit deutlich erhöht.",
        "Obwohl die Behandlung wirksam war, traten bei einigen Patienten unerwartete Nebenwirkungen auf.",
        "Es wird befürchtet, dass die nächste Grippewelle das Gesundheitssystem erneut stark belasten werde.",
        "Die Digitalisierung der Patientenakten soll die Koordination zwischen den Fachärzten verbessern.",
        "Obschon die Studie vielversprechend ist, sind weitere klinische Tests unbedingt erforderlich.",
        "Der Arzt betonte, die Prävention sei wirksamer als die Behandlung bereits fortgeschrittener Erkrankungen.",
        "Die Finanzierung der Krankenhäuser wird in Zeiten steigender Kosten zunehmend zur Herausforderung.",
        "Es werde empfohlen, den Zugang zur psychiatrischen Versorgung flächendeckend auszubauen.",
        "Die Durchsetzung des Verbots diskriminierender Praktiken wurde von Menschenrechtsorganisationen gefordert.",
        "Hätte die Behörde schneller reagiert, wäre der Schaden für die Betroffenen geringer ausgefallen.",
        "Der Richter erklärte, die Entscheidung beruhe auf einer sorgfältigen Abwägung aller Umstände.",
        "Die Verbesserung der Pflegebedingungen in Altenheimen bleibt eine dringende gesellschaftliche Aufgabe.",
        "Es wird angenommen, dass personalisierte Medizin die Behandlung vieler Krankheiten revolutionieren werde.",
        "Obwohl die Pandemie abgeklungen ist, bleiben die Folgen für die psychische Gesundheit spürbar.",
        "Die Stärkung der Rechte von Opfern schwerer Straftaten wurde im Gesetzgebungsverfahren verankert.",
    ],
    # Batch 8 (709–733): Media & International Relations
    [
        "Die Berichterstattung über den Konflikt wurde von beiden Seiten als einseitig kritisiert.",
        "Obwohl die Sanktionen verschärft wurden, änderte sich die Haltung des Regimes nicht.",
        "Es wird erwartet, dass die Friedensverhandlungen in den kommenden Wochen wieder aufgenommen werden.",
        "Die Stärkung der multilateralen Zusammenarbeit gilt als Antwort auf globale Herausforderungen.",
        "Obschon die diplomatischen Bemühungen intensiviert wurden, bleibt eine Einigung fraglich.",
        "Der Außenminister erklärte, sein Land werde die humanitäre Hilfe nicht einschränken.",
        "Die Verbreitung von Nachrichten über soziale Medien verändert die öffentliche Meinungsbildung grundlegend.",
        "Es sei wichtig, dass die Pressefreiheit auch in Krisenzeiten uneingeschränkt gewährleistet bleibe.",
        "Die Aufnahme von Flüchtlingen wurde von der internationalen Gemeinschaft unterschiedlich bewertet.",
        "Würde man die Handelsbarrieren senken, könnte der wirtschaftliche Austausch deutlich zunehmen.",
        "Die Überprüfung des Atomabkommens wurde von den Unterzeichnerstaaten als dringend erachtet.",
        "Obwohl die Beziehungen angespannt sind, finden weiterhin kulturelle Austauschprogramme statt.",
        "Es wird befürchtet, dass der Protektionismus die globale Wirtschaftslage weiter verschlechtern werde.",
        "Die Rolle internationaler Organisationen bei der Konfliktlösung wird kontrovers diskutiert.",
        "Hätte die Diplomatie früher eingegriffen, wäre die Eskalation möglicherweise verhindert worden.",
        "Der Korrespondent berichtete, die Lage an der Grenze habe sich in letzter Zeit verschärft.",
        "Die Förderung des interkulturellen Dialogs soll Vorurteile abbauen und gegenseitiges Verständnis stärken.",
        "Obschon die Resolution verabschiedet wurde, fehlt es an wirksamen Durchsetzungsmechanismen.",
        "Es mangelt nicht an Absichtserklärungen, sondern an verbindlichen und durchsetzbaren Vereinbarungen.",
        "Die Einrichtung von Pufferzonen wurde als mögliche Deeskalationsmaßnahme in Betracht gezogen.",
        "Der Kommentator betonte, die Solidarität zwischen den Partnern dürfe nicht nachlassen.",
        "Die Auswirkungen des Handelskrieges auf die Verbraucherpreise werden mit Sorge beobachtet.",
        "Es werde erwartet, dass die nächste Gipfelkonferenz neue Klimaziele für alle Staaten festlege.",
        "Obwohl die Hilfsgelder angekündigt wurden, erreichen sie die betroffenen Regionen nur langsam.",
        "Die Wahrung der territorialen Integrität wurde von allen Beteiligten als nicht verhandelbar bezeichnet.",
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
                # Try Stanza's lemma; if still not infinitive, use form if infinitive-like
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
start_id = 534

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

target_path = project_root / "data/handcraft/de/train/b2_new_005.conllu"
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