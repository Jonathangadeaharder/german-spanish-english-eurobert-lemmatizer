"""Generate 200 handcrafted German C2 CoNLL-U sentences (de_c2_train_561–760)."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 8 sub-batches of 25 sentences each (561–760)
BATCH_1 = [
    "Die subjektive Ästhetik des achtzehnten Jahrhunderts erklärt das Schöne aus der unmittelbaren Empfindung des betrachtenden Subjekts heraus.",
    "Obschon die Kunst autonom beansprucht, bleibt sie dennoch untrennbar mit gesellschaftlichen Machtstrukturen und symbolischen Ordnungen verwoben.",
    "Die mimetische Theorie behauptet, die Dichtung bilde die Welt nach, ohne jemals ihre sinnliche Fülle vollständig zu erschöpfen.",
    "Es bedarf einer feinsinnigen Analyse, um die ästhetische Autonomie des Kunstwerks von seiner ideologischen Funktion sorgfältig zu unterscheiden.",
    "Die Grazie der Proportionen verleiht der klassizistischen Skulptur eine zeitlose Harmonie, die den Betrachter schier innezuhalten zwingt.",
    "Trotz aller formalistischer Reduktion bleibt die expressiv gestaltete Farbfläche Träger tiefen psychologischen und symbolischen Gehalts.",
    "Die Dekadenz des Fin de Siècle feiert die ästhetische Selbstgenügsamkeit und grenzt sich demonstrativ von bürgerlichen Moralvorstellungen ab.",
    "Es gilt, das Erhabene als Grenzerfahrung der Darstellbarkeit von bloßer Größe und rhetorischer Übertreibung methodisch zu scheiden.",
    "Die Ikonizität des Bildes erschließt Bedeutungsschichten, die sich jeder linearen narrativen Paraphrase grundsätzlich entziehen.",
    "Mit bewundernswerter methodischer Präzision analysierte die Kunsthistorikerin die Kompositionsgesetze der venezianischen Hochrenaissance und ihrer Meisterateliers.",
    "Die Dissoziation von Form und Inhalt charakterisiert die abstrakte Malerei der Moderne und fordert neue Sehgewohnheiten.",
    "Es ist von eminenter Bedeutung, die ästhetische Erfahrung nicht auf subjektiven Geschmack reduzieren zu wollen.",
    "Die Rezeptionsästhetik betont die aktive Rolle des Lesers bei der Konstitution des sinnträchtigen Kunstwerks.",
    "Ungeachtet theoretischer Einwände behauptet die klassische Ästhetik nach wie vor die Universalität gewisser harmonischer Proportionen in der bildenden Kunst.",
    "Die Kitschästhetik simuliert Bedeutungstiefe durch sentimentale Klischees und täuscht dem ungeübten Betrachter unverfängliche Oberflächenwirkung vor.",
    "Es bleibt abzuwarten, ob die gegenwärtige Biennale tatsächlich neue Maßstäbe für installative Kunst setzen wird.",
    "Die Ästhetik des Alltäglichen hebt banale Gegenstände in den Rang des Kunstwerks und dekonstruiert tradierte Genrevorstellungen.",
    "Der Kritiker erkannte in dem Gemälde eine subtile Auseinandersetzung mit barocken Konventionen von Licht und Schatten.",
    "Die Performanz des Körpers verwischt die Grenze zwischen künstlerischer Geste und profaner sozialer Inszenierung.",
    "Es vermag die rein quantitative Bewertung von Kunstwerken deren qualitative singuläre Wirkung niemals angemessen zu erfassen.",
    "Die Synästhesie der Farbtöne evoziert während des Hinhörens musikalische Assoziationen, die über das Sichtbare hinausreichen.",
    "Die Ästhetisierung der Politik birgt die stille Gefahr, demokratische Auseinandersetzungen in bloße Inszenierung zu verwandeln.",
    "Mit unnachgiebiger Konsequenz verteidigte der Philosoph die These von der kognitiven Relevanz ästhetischer Urteile.",
    "Die Aura des Originals schwindet in der technischen Reproduzierbarkeit, wie die Kulturkritik der Moderne eindringlich betont.",
    "Die Grenzüberschreitung provokanter Kunstwerke zwingt das Publikum, tabuisierte gesellschaftliche Normen kritisch zu hinterfragen.",
]

BATCH_2 = [
    "Die Erkenntnistheorie muss klären, unter welchen Bedingungen ein Glaube überhaupt als gerechtfertigte Überzeugung gelten darf.",
    "Obschon der Empirismus die Sinnesdaten privilegiert, bleibt die Frage nach ihrer begrifflichen Verarbeitung weiterhin offen.",
    "Die Gettierproblematik untergräbt die klassische Analyse des Wissens als gerechtfertigten wahren Glaubens und zwingt zu neuer Begriffsbestimmung.",
    "Es bedarf einer kohärentistischen Theorie, um den Zusammenhang einzelner Überzeugungen in einem Netzwerk zu erklären.",
    "Die fallibilistische Haltung gesteht ein, dass selbst bestbegründete wissenschaftliche Theorien prinzipiell widerlegbar bleiben.",
    "Trotz methodischer Strenge vermag der Rationalismus die transzendentale Herkunft apriorischer Erkenntnisurteile nicht hinreichend zu begründen.",
    "Die Unterbestimmtheit der Theorie durch die Erfahrung eröffnet konkurrierenden wissenschaftlichen Erklärungsansätzen gleichermaßen plausible und methodisch vertretbare Interpretationsspielräume.",
    "Es gilt, die konstruktivistische These von der relativen Natur allen Wissens von skeptischem Relativismus sorgfältig zu scheiden.",
    "Die Intuition als Erkenntnisquelle bleibt umstritten, solange keine zuverlässigen Kriterien ihrer epistemischen Verlässlichkeit vorliegen.",
    "Mit akribischer und nachvollziehbarer Argumentationsführung verteidigte der Philosoph die interne Kohärenz eines naturalistischen Erkenntnisprogramms eindringlich.",
    "Die epistemische Ungerechtigkeit betrifft vor allem marginalisierte Gruppen, deren Erfahrungswissen institutionell systematisch abgewertet wird.",
    "Es ist von eminenter Bedeutung, die Grenzen des menschlichen Erkenntnisvermögens von bloßen vorläufigen Kontingenzgrenzen zu unterscheiden.",
    "Die Verlässlichkeit der Wahrnehmung setzt voraus, dass sinnliche Gegebenheiten nicht durch unbekannte Täuschungsmechanismen verfälscht werden.",
    "Ungeachtet pragmatischer Erfolge und technischer Anwendbarkeit bleibt die Frage nach der Wahrheitsapproximation wissenschaftlicher Hypothesen philosophisch unentschieden.",
    "Die kontextualistische Erkenntnistheorie erklärt scheinbare Widersprüche durch unterschiedliche epistemische Standards in verschiedenen kommunikativen und sozialen Situationen.",
    "Es bleibt abzuwarten, ob die Neuformulierung des Externalismus überzeugende Antworten auf das Problem des privaten Sprachgebrauchs bietet.",
    "Die hermeneutische Epistemologie betont, dass jedes Verstehen bereits in sprachlichen und historischen Horizonten verankert ist.",
    "Der Wissenschaftstheoretiker rekonstruierte in seiner bahnbrechenden Studie die Struktur wissenschaftlicher Revolutionen als diskontinuierlichen und konflikthaften Paradigmenwechsel.",
    "Die testimoniale Erkenntnis beruht auf dem berechtigten Vertrauen in die Aussagen kompetenter und redlicher Informationsquellen.",
    "Es vermag der radikale Skeptizismus nicht, den alltäglichen Glauben an eine mindestens teilweise verlässliche Welt vollends zu suspendieren.",
    "Die unterdeterminierte Evidenz zwingt Forscher, zusätzliche theoretische Kriterien wie Einfachheit oder Reichweite heranzuziehen.",
    "Die Frage nach dem Zugriff auf mentale Zustände anderer bleibt das zentrale Problem jeder sozialen Erkenntnistheorie.",
    "Mit nachdrücklicher argumentative Klarheit argumentierte sie für die Unverzichtbarkeit intersubjektiver Verifikation und öffentlicher Prüfung wissenschaftlicher Ergebnisse.",
    "Die Theorie der kontrafaktischen Bedingungen präzisiert kausale Zusammenhänge durch gedankliche Variation hypothetischer Welten und erleichtert damit die Modellierung.",
    "Die Selbstreferenz des erkennenden Bewusstseins stellt jede naiv realistische Ontologie epistemologisch auf den Prüfstand.",
]

BATCH_3 = [
    "Die Verfassungsidentität eines demokratischen Staates gründet auf unveräußerlichen Menschenrechten und der Anerkennung pluraler Lebensformen.",
    "Obschon das Parlament umfassende Gesetzesreformen beschloss, hielt das Bundesverfassungsgericht an den Grundrechten fest.",
    "Die Gewaltenteilung verlangt, dass legislative, exekutive und judikative Funktionen institutionell strikt voneinander getrennt bleiben.",
    "Es bedarf einer sorgfältigen Abwägung zwischen Sicherheitsinteressen und den Freiheitsrechten der Bürger in der digitalen Überwachung.",
    "Die verfassungsrechtliche Prüfung eines Gesetzes folgt dem Grundsatz der Verhältnismäßigkeit auf drei konsekutiven Stufen.",
    "Trotz parlamentarischer Mehrheiten darf die Minderheit ihre Grundrechte nicht durch willkürliche Diskriminierung verwirkt haben.",
    "Die Föderalismusreform zielte darauf ab, die Kompetenzverteilung zwischen Bund und Ländern nachhaltig neu zu ordnen.",
    "Es gilt, die Staatszielbestimmungen als objektivrechtliche Wertentscheidungen gegen partikulare und kurzfristige politische Interessen konsequent durchzusetzen.",
    "Die weitreichende Immunität des Staatsoberhaupts stößt in modernen republikanischen Verfassungsordnungen zunehmend auf grundsätzliche rechtspolitische Einwände.",
    "Mit beeindruckender juristischer Schärfe und systematischer Begründung erklärte die Richterin die Verfassungswidrigkeit des umstrittenen Überwachungsgesetzes.",
    "Die Prinzipien der Rechtsstaatlichkeit binden staatliches Handeln an vorher verkündete Gesetze und unabhängige gerichtliche Kontrolle.",
    "Es ist von eminenter Bedeutung, die Suprematie des Grundgesetzes gegenüber einfachem Gesetzesrecht konsequent zu wahren.",
    "Die verfassungsgebende Gewalt des Volkes setzt demokratische Legitimation und politische Selbstbestimmung gegenüber traditionellen Herrschaftsansprüchen entschieden durch.",
    "Ungeachtet politischer Druckausübung bewahrte das Gericht seine Unabhängigkeit und entschied strikt nach dem Maßstab der Verfassung.",
    "Die streitbare Demokratie erlaubt Einschränkungen bestimmter Grundrechte zu dem Schutz der freiheitlich demokratischen Verfassungsgrundsätze.",
    "Es bleibt abzuwarten, ob die geplante Verfassungsänderung die erforderlichen qualifizierten Mehrheiten in beiden Kammern erreichen wird.",
    "Die Verfassungsinterpretation berücksichtigt sowohl den Wortlaut der Norm als auch ihren systematischen und historischen Sinnzusammenhang.",
    "Der Verfassungsrechtler analysierte die Auswirkungen supranationaler Rechtsakte auf die nationale Souveränität mit großer Genauigkeit.",
    "Die Bindung der Verwaltung an das Gesetz verhindert exekutive Willkür und sichert vorhersehbare rechtsstaatliche Entscheidungsprozesse.",
    "Es vermag keine Regierung, Grundrechte dauerhaft außer Kraft zu setzen, ohne die verfassungsrechtliche Legitimation zu gefährden.",
    "Die Notstandsklausel erlaubt unter engen Voraussetzungen temporäre Grundrechtseingriffe zu dem Schutz der freiheitlichen Verfassungsordnung.",
    "Die Parteienfinanzierung unterliegt verfassungsrechtlichen Transparenzgeboten, um verdeckte Einflussnahme auf den politischen Willensbildungsprozess zu verhindern.",
    "Mit unbeirrbarer Konsequenz verteidigte die Opposition das Parlament als Ort offener demokratischer Willensbildung und Kontrolle.",
    "Die europäische Verfassungsarchitektur verknüpft nationale Identitäten mit gemeinsamen rechtsstaatlichen Wertprinzipien und institutionellen Kontrollmechanismen auf überstaatlicher Ebene.",
    "Die Verhältnismäßigkeit staatlicher Eingriffe verlangt, dass Mittel und Zweck einer Maßnahme in einem angemessenen Verhältnis zueinander stehen.",
]

BATCH_4 = [
    "Die Dekonstruktion literarischer Texte legt die instabilen Bedeutungsdifferenzen offen, die jede scheinbar geschlossene Interpretation unterlaufen.",
    "Obschon der Roman linear erzählt, durchkreuzt seine ironische Erzählperspektive konsequent die naive Autoritätsillusion des traditionellen Erzählens.",
    "Die Intertextualität des Gedichts verweist auf antike Vorbilder und transformiert sie in eine eigenständige moderne Bildsprache.",
    "Es bedarf einer nuancierten Lektüre, um die allegorische Tiefenstruktur des Dramas von bloßer Handlungsebene zu unterscheiden.",
    "Die Poetik der Romantik feiert die Autonomie der Imagination und grenzt sich von dem utilitaristischen Zweckdenken der Aufklärung ab.",
    "Trotz philologischer Sorgfalt blieb die Datierung des Manuskripts Gegenstand kontroverser Debatten in der Editionswissenschaft.",
    "Die Metalepse des Erzählers durchbricht die narrative Illusion und macht die Konstruiertheit der fiktionalen Welt sichtbar.",
    "Es gilt, die performative Kraft des Prosatextes von bloßen referentiellen Aussagen methodisch sorgfältig zu scheiden.",
    "Die Rezeptionsgeschichte des Werkes zeigt, wie literarische Bedeutungen historisch und kulturell stets neu verhandelt werden.",
    "Mit bewundernswerter interpretatorischer Finesse erschloss die Kritikerin die latenten mythischen Strukturen des epischen Gedichts.",
    "Die Ästhetik der Bruchstückhaftigkeit charakterisiert fragmentarische Erzählformen und verweigert dem Leser tröstliche narrative Geschlossenheit.",
    "Es ist von eminenter Bedeutung, die Autorschaft postmoderner Texte nicht auf biographische Intentionen zu reduzieren.",
    "Die narratologische Analyse unterscheidet zwischen erzählender Instanz, fiktionaler Welt und der Rezeption durch den implizierten Leser.",
    "Ungeachtet kontroverser Rezensionen hielt der Verlag an der ambitionierten Publikation des experimentellen Prosabandes fest.",
    "Die Tragödie des Helden beruht auf einem unauflösbaren Konflikt zwischen individueller Freiheit und gesellschaftlicher Pflichterfüllung.",
    "Es bleibt abzuwarten, ob die neue Gesamtausgabe lang unterdrückter Briefe das Werk des Dichters grundlegend neu beleuchten wird.",
    "Die hermeneutische Leseart betont den dialogischen Charakter des Verstehens zwischen Text, Autor und interpretierendem Subjekt.",
    "Der Literaturwissenschaftler rekonstruierte den rhetorischen Aufbau der Festrede mit bemerkenswerter textanalytischer Präzision und historischer Kontextualisierung.",
    "Die Polyphonie des Romans vereint konkurrierende Stimmen, ohne eine eindeutige ideologische Autorität hierarchisch festzulegen.",
    "Es vermag die reine Motivpsychologie nicht, die symbolische Komplexität hochliterarischer Metaphern angemessen zu erklären.",
    "Die Kanonbildung reflektiert gesellschaftliche Machtverhältnisse und privilegiert bestimmte Ästhetiken auf Kosten marginalisierter und kolonial verdrängter Traditionen.",
    "Die Parodie des Genres nimmt konventionelle Erzählmuster ironisch in das Visier und dekonstruiert naive Leseerwartungen systematisch.",
    "Mit unnachgiebiger Strenge bewertete die Jury die Originalität und sprachliche Virtuosität der eingereichten Lyrikbände.",
    "Die Ekphrase des Gedichts verleiht dem beschriebenen Gemälde eine fast hörbar werdende stille gegenwärtige Präsenz.",
    "Die Ironie der Erzählerfigur untergräbt jede vermeintlich objektive Schilderung und zwingt den Leser zu der kritischen Distanzierung.",
]

BATCH_5 = [
    "Die diplomatische Note verlangte unmissverständlich die Achtung der territorialen Integrität und der völkerrechtlich garantierten Souveränität.",
    "Obschon die Verhandlungen monatelang stockten, einigten sich die Delegationen dennoch auf einen vorläufigen Waffenstillstandsvertrag.",
    "Die Klugheit der Außenpolitik besteht darin, harte Interessenlagen durch kalkulierte Kompromissformeln ohne Gesichtsverlust zu vermitteln.",
    "Es bedarf eines sensiblen Taktgefühls, um in der diplomatischen Korrespondenz Kritik zu äußern, ohne die Verhandlungspartner vor den Kopf zu stoßen.",
    "Die Vermittlung des Generalsekretärs führte zu einer vorläufigen Deeskalation des bewaffneten Konflikts an der umkämpften Grenze.",
    "Trotz öffentlicher Rhetorik setzten die Regierungen hinter verschlossenen Türen auf eine schrittweise Normalisierung der bilateralen Beziehungen.",
    "Die Garantiemächte verpflichteten sich, die Einhaltung des Friedensabkommens durch regelmäßige multilaterale Inspektionen zu überwachen.",
    "Es gilt, die diplomatische Immunität von strafrechtlicher Verfolgung von krimineller Straflosigkeit grundsätzlich zu unterscheiden.",
    "Die Abrüstungsgespräche erfordern gegenseitiges Vertrauen, das sich nur durch transparente Verifikationsmechanismen schrittweise herstellen lässt.",
    "Mit souveräner Verhandlungskunst brachte die Botschafterin die verhärteten Positionen beider Seiten auf eine gemeinsame Verhandlungsbasis.",
    "Die Zwangslage der Sanktionen zielte darauf ab, das Regime durch wirtschaftlichen Druck zu substantiellen politischen Reformen zu bewegen.",
    "Es ist von eminenter Bedeutung, die humanitären Gespräche von den strategischen Machtinteressen der konfliktierenden Parteien zu trennen.",
    "Die protokollarischen Förmlichkeiten rahmen das diplomatische Treffen ein und signalisieren dennoch den Willen zu konstruktiven Gesprächen.",
    "Ungeachtet provokanter Äußerungen in den Medien hielten beide Außenminister an dem geplanten bilateralen Gipfeltreffen fest.",
    "Die Guten Dienste des neutralen Staates eröffneten einen diskreten Kanal für vertrauliche Vorverhandlungen über den Friedensprozess.",
    "Es bleibt abzuwarten, ob die Friedenskonferenz zu einem dauerhaften Sicherheitsregime in der krisengeschüttelten Region führen wird.",
    "Die Rhetorik der Entschlossenheit verbirgt häufig die Bereitschaft, bei entscheidenden Punkten substantielle Zugeständnisse einzugehen.",
    "Der Gesandte übermittelte die Note verbale mit der gebührenden Förmlichkeit und ohne jede unbotmäßige emotional gefärbte Polemik.",
    "Die multilateralen Organisationen bieten institutionelle Rahmen, in denen divergierende Staatsinteressen regelbasiert ausgehandelt werden.",
    "Es vermag eine einzelne diplomatische Geste nicht, jahrelang angehäufte Misstrauenskrisen zwischen benachbarten Staaten vollends aufzulösen.",
    "Die Festlegung des Status quo wurde in dem geheimen Protokoll festgeschrieben und soll vorläufig weiteren territorialen Ansprüchen entgegenwirken.",
    "Die Verhandlungsführer betonten die Unabdingbarkeit verbindlicher Sicherheitsgarantien für alle an dem Abkommen beteiligten Staaten.",
    "Mit nachdrücklicher Gelassenheit wies der Minister die unbegründeten Vorwürfe in seiner feierlichen Rede vor der Generalversammlung zurück.",
    "Die diplomatische Korrespondenz der Epoche dokumentiert die feinen graduellen Verschiebungen des europäischen Machtgleichgewichts eindringlich.",
    "Die Deeskalationsstrategie setzt auf sequenzielle Vertrauensbildungsmaßnahmen, bevor Kernfragen der Sicherheitsarchitektur abschließend verhandelt werden.",
]

BATCH_6 = [
    "Die Neurophilosophie untersucht, inwieweit bewusste Erlebnisse auf kausalen Prozessen in dem zentralen Nervensystem reduzierbar sind.",
    "Obschon Daten der funktionellen Magnetresonanztomographie korrelierte Aktivierungsmuster zeigen, bleibt die Frage nach dem qualitativen Charakter subjektiven Erlebens offen.",
    "Das Problem von Leib und Seele stellt die Annahme infrage, dass mentale Zustände von neuronalen Zuständen grundsätzlich verschieden sind.",
    "Es bedarf einer sorgfältigen Theorie, um intentionale Gehalte nicht vorschnell auf bloße synaptische Übertragungsprozesse zu reduzieren.",
    "Die Phänomenologie des Schmerzes widerlegt jede naive Behauptung, Qualia ließen sich ohne Rest physiologisch vollständig erklären.",
    "Trotz beeindruckender empirischer Fortschritte der Neurowissenschaft bleibt die Ontologie des Bewusstseins in der Philosophie hochgradig umstritten.",
    "Die funktionale Bildgebung legt nahe, dass moralische Urteile mit spezifischen kortikalen Netzwerken in enger Wechselwirkung stehen.",
    "Es gilt, die emergentistische These von höheren mentalen Eigenschaften von eliminativen Reduktionismus methodisch zu scheiden.",
    "Die kognitive Architektur des Gehirns modelliert Informationsverarbeitung durch hierarchisch organisierte und rekurrierende neuronale Schaltkreise.",
    "Mit akribischer Begriffsklarheit argumentierte der Philosoph gegen eine naive Identifikation von Geist und Gehirn.",
    "Die Illusion des freien Willens entsteht möglicherweise aus der retrospektiven Rationalisierung bereits feststehender neuronaler Entscheidungsprozesse.",
    "Es ist von eminenter Bedeutung, die normativen Implikationen neurobiologischer Erkenntnisse für Fragen der strafrechtlichen Zurechnung zu klären.",
    "Die neurophänomenologische Methode verbindet erstpersonale Selbstbeschreibung subjektiven Erlebens mit drittpersonaler empirischer und experimenteller Gehirnforschung.",
    "Ungeachtet detaillierter Neurokarten bleibt unklar, wie neuronale Feuerungsmuster überhaupt Bedeutung oder Intentionalität konstituieren.",
    "Die Theorie der erweiterten Kognition behauptet, mentale Prozesse überließen regelmäßig die Grenzen des einzelnen Organismus hinaus.",
    "Es bleibt abzuwarten, ob die neue Hypothese zu der globalen neuronalen Arbeitsraumfrequenz das Bewusstseinsproblem überzeugend löst.",
    "Die Qualiaphilosophie insistiert darauf, dass subjektive Erlebnisqualitäten in jeder naturalistischen Beschreibung prinzipiell unterbestimmt bleiben.",
    "Der Neurowissenschaftler rekonstruierte die synaptische Plastizität als neurobiologische Grundlage des Lernens und der Gedächtnisbildung.",
    "Die epistemische Privatheit innerer Zustände erschwert jede objektivierende Messung des Erlebens durch externe Beobachtungsmethoden grundlegend.",
    "Es vermag die rein reduktionistische Neurowissenschaft nicht, die normative Struktur rationaler Argumentation angemessen zu erklären.",
    "Die neuronalen Korrelate des Bewusstseins liefern empirische Hinweise, ohne dennoch die Erklärungslücke qualitativer Erfahrung zu schließen.",
    "Die Interaktionismusdebatte fragt, ob mentale Kausation in einem physikalisch geschlossenen Weltbild überhaupt konsistent denkbar bleibt.",
    "Mit nachdrücklicher begrifflicher Präzision analysierte sie die Implikationen des Spiegelneuronensystems für intersubjektive und soziale Verstehensprozesse.",
    "Die descendente Kontrolle kognitiver Prozesse durch präfrontale Areale widerlegt naive Stimulationsmodelle rein passiver Wahrnehmungsverarbeitung.",
    "Die Neuroethik untersucht, welche therapeutischen Eingriffe in das Gehirn moralisch zulässig sind und wo fundamentale Würdegrenzen verletzt werden.",
]

BATCH_7 = [
    "Die ästhetische Urteilskraft beansprucht Allgemeinheit, obgleich sie auf subjektiven Gefühlszuständen des reflektierenden Bewusstseins zu beruhen scheint.",
    "Obschon die epistemische Rechtfertigung scheinbar objektiv ist, hängt sie dennoch von kontextabhängigen Standards verlässlicher Überzeugungsbildung ab.",
    "Die verfassungsrechtliche Demokratieklausel schützt die freiheitliche Grundordnung gegen parteipolitische Bestrebungen, die Menschenwürde systematisch in Frage stellen.",
    "Es bedarf einer interdisziplinären Perspektive, um ästhetische Wahrnehmung, kognitive Erkenntnis und politische Normsetzung sinnvoll zu verbinden.",
    "Die Erkenntnistheorie der Wissenschaften betont, dass Beobachtungssätze stets theoriegeladen sind und keine rein neutrale Basis bieten.",
    "Trotz formaler Schönheit der Verfassungsnorm bleibt ihre lebendige Auslegung von gesellschaftlichen Konflikten und Machtverhältnissen geprägt.",
    "Die Sublimierung ästhetischer Erfahrung verweist auf Grenzen der Darstellbarkeit, die zugleich epistemische und politische Implikationen entfalten.",
    "Es gilt, die Rechte der Kunstfreiheit gegen verfassungsrechtliche Einschränkungen abzuwägen, wenn ästhetische Provokation gesellschaftliche Grundwerte berührt.",
    "Die infinitesimale Nuance der Farbgebung erschließt dem wachen Blick Bedeutungsschichten, die jeder begrifflichen Paraphrase entrinnen.",
    "Mit bewundernswerter analytischer Schärfe verband die Philosophin erkenntnistheoretische Reflexion mit einer feinsinnigen Ästhetik des Alltäglichen.",
    "Die Legitimität verfassungsgerichtlicher Entscheidungen beruht auf der Anerkennung ihrer Autorität durch demokratisch legitimierte Staatsorgane und die Öffentlichkeit.",
    "Es ist von eminenter Bedeutung, epistemische Gerechtigkeit als Grundlage inklusiver wissenschaftlicher und ästhetischer Urteilsbildung institutionell zu verankern.",
    "Die Wahrheitskondition analytischer Urteile bleibt strittig, solange keine allgemein akzeptierte Theorie des Wahrheitsbegriffs vorliegt.",
    "Ungeachtet ästhetischer Autonomie und künstlerischer Freiheit unterliegen öffentlich geförderte Kunstwerke verfassungsrechtlichen und demokratischen Rechenschaftsanforderungen.",
    "Die kontemplative Betrachtung des Kunstwerks suspendiert vorübergehend praktisches Handeln und eröffnet einen Raum reiner ästhetischer Erkenntnis.",
    "Es bleibt abzuwarten, ob die neue Theorie der epistemischen Tugenden ein überzeugendes Gegenmodell zu rein reliabilistischen Rechtfertigungskonzeptionen bietet.",
    "Die Garantie der Kunstfreiheit in dem Grundgesetz schützt experimentelle Formen, ohne jedoch den Schutz der Menschenwürde aufzuheben.",
    "Der Verfassungsrechtler erörterte die Spannungen zwischen ästhetischer Freiheit und staatlicher Förderpolitik in öffentlichen Kulturbetrieben.",
    "Die skeptische Herausforderung behauptet, weder ästhetische noch moralische Urteile seien rational in dem Sinne epistemisch rechtfertigbar.",
    "Es vermag die rein legislative Perspektive nicht, die normative Kraft verfassungsidentitärer Wertentscheidungen angemessen zu erfassen.",
    "Die ästhetische Differenz kennzeichnet das Kunstwerk als etwas, das sich der alltäglichen Zweckrationalität demonstrativ entzieht.",
    "Die Grundlagenforschung erfordert epistemische Toleranz gegenüber vorläufigen Hypothesen, die sich erst langfristig empirisch bestätigen lassen.",
    "Mit unnachgiebiger juristischer Präzision begründete das Gericht die Schutzpflicht des Staates für unabhängige akademische Erkenntnisgewinnung.",
    "Die symbolische Kraft öffentlicher Denkmäler wirft verfassungsrechtliche Fragen nach Inklusion, Repräsentation und kollektivem Erinnern auf.",
    "Die Integration ästhetischer Bildung in demokratische Bildungskonzepte soll Urteilskraft und epistemische Urteilsfähigkeit der Bürger gleichermaßen stärken.",
]

BATCH_8 = [
    "Die neurophilosophische Lesart des Romans fragt, inwieweit narrative Identitätskonstruktionen auf neuronale Selbstmuster zurückgeführt werden können.",
    "Obschon die diplomatische Sprache formal nüchtern bleibt, transportiert sie dennoch subtile Machtasymmetrien und strategische Positionierungen.",
    "Die literaturkritische Analyse der Metapher zeigt, wie sprachliche Bilder kognitive Schemata aktivieren und Verstehensprozesse lenken.",
    "Es bedarf einer vorsichtigen Hermeneutik, um Textbedeutung nicht vorschnell auf bloße neurobiologische Korrelationsbefunde zu reduzieren.",
    "Die Friedensverhandlung in dem Roman spiegelt die fragile Rhetorik internationaler Diplomatie und deren Abhängigkeit von wechselseitigem Vertrauen.",
    "Trotz präziser Neurowissenschaften bleibt die literarische Darstellung innerer Monologe philosophisch wie ästhetisch unersetzlich komplex.",
    "Die Botschaft übermittelte ihre Forderungen in klassisch diplomatisierter Formulierung, ohne die Kritik an Menschenrechtsverletzungen zu beschönigen.",
    "Es gilt, die intentionalistische Literaturinterpretation von externalistischen Ansätzen zu unterscheiden, die Texte in historischen Kontexten verorten.",
    "Die Spiegelneuronenforschung eröffnet neue Perspektiven auf narrative Empathie und die kognitive Resonanz literarischer Figurenkonstellationen.",
    "Mit souveräner rhetorischer Kontrolle führte der Gesandte die Verhandlungen durch einen Wendepunkt drohenden politischen Bruchs hindurch.",
    "Die Dekonstruktion des autobiographischen Romans legt die Kluft zwischen erinnerter Erfahrung und nachträglicher narrativer Fiktionalisierung offen.",
    "Es ist von eminenter Bedeutung, neurophilosophische Befunde nicht für deterministische Deutungen literarischer Handlungsfreiheit zu instrumentalisieren.",
    "Die diplomatische Memoirenliteratur dokumentiert hinter den Kulissen verhandelte Kompromisse, die in offiziellen Protokollen nur angedeutet erscheinen.",
    "Ungeachtet faszinierender Neurobilder bleibt die qualitative Dimension des Bewusstseins für die Literaturtheorie konstitutiv unabweisbar.",
    "Die Kritik des Naturalismus wies auf die naive Illusion hin, soziale Wirklichkeit lasse sich unverfälscht in erzählerischer Abbildung einfangen.",
    "Es bleibt abzuwarten, ob die multilateralen Gespräche eine tragfähige Sicherheitsordnung jenseits kurzfristiger bilateralen Interessenentgegenkommen schaffen.",
    "Die neuronale Plastizität liefert Metaphern für die literarische Überarbeitung von Erinnerungstexten und deren ständige semantische Umformung.",
    "Der Literaturhistoriker verglich die Rhetorik diplomatischer Verträge mit der komplexen Dialogstruktur des politischen Romans der Nachkriegszeit.",
    "Die Theorie der verteilten Kognition erweitert literaturwissenschaftliche Modelle, indem sie mentale Prozesse über Text, Körper und Umwelt verteilt.",
    "Es vermag keine reine Textanalyse, die psychologischen und diplomatischen Implikationen eines Briefromans erschöpfend zu entschlüsseln.",
    "Die diplomatische Zurückhaltung in der Pressekonferenz signalisierte Gesprächsbereitschaft, ohne bereits substanzielle Zugeständnisse vorwegzunehmen.",
    "Die Neuroästhetik untersucht, warum bestimmte literarische Rhythmen und Klangfiguren in dem Gehirn charakteristische emotionale Aktivierungsmuster hervorrufen.",
    "Mit eindringlicher Feinheit interpretierte die Kritikerin die Traumsequenz als neurophilosophische Allegorie des fragmentierten Bewusstseins.",
    "Die vertraulichen Gesprächskanäle der Diplomatie ermöglichen Eskalationskontrolle, wo öffentliche Rhetorik konfrontative Festlegungen erzwingen würde.",
    "Die literarische Moderne und die neurophilosophische Forschung teilen die Einsicht, dass Subjektivität weder transparent noch endgültig objektivierbar ist.",
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
    start_id = 561
    conllu_text = build_conllu(SENTENCES, start_id)

    target_path = project_root / "data/handcraft/de/train/c2_new_006.conllu"
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

    # Token count check (15-25 per sentence)
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


if __name__ == "__main__":
    main()