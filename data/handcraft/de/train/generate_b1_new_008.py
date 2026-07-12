"""Generate b1_new_008.conllu (de_b1_train_705–904) in sub-batches of 25."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 200 B1 sentences: 8–15 tokens, mixed tenses and structures
SENTENCE_BATCHES: list[list[str]] = [
    # 705–729: Immigration
    [
        "Viele Menschen wandern heute aus wirtschaftlichen Gründen aus ihrer Heimat aus.",
        "Der Aufenthaltstitel muss vor dem Ablauf in der Ausländerbehörde verlängert werden.",
        "Ich habe letzte Woche meinen Antrag auf eine Aufenthaltserlaubnis eingereicht.",
        "Die Integration in die deutsche Gesellschaft dauert oft mehrere Jahre.",
        "Ohne gültige Papiere kann man in Deutschland keine reguläre Arbeit aufnehmen.",
        "Mein Nachbar ist vor drei Jahren aus Syrien nach Deutschland gekommen.",
        "In dem Integrationskurs lerne ich sowohl Sprache als auch Kultur kennen.",
        "Die Behörde verlangt einen offiziellen Nachweis über meine Wohnadresse.",
        "Viele Geflüchtete suchen nach einem sicheren Ort für ihre Familien.",
        "Ich muss mein Visum unbedingt vor der Reise beantragen.",
        "Die Familienzusammenführung kann leider oft viele Monate in Anspruch nehmen.",
        "Er hat seine Heimat verlassen, weil er dort keine Perspektive sah.",
        "Der Dolmetscher hilft den Neuankömmlingen bei wichtigen Gesprächen mit Behörden.",
        "In Deutschland haben Ausländer grundsätzlich die gleichen Rechte wie Einheimische.",
        "Sie musste viele Dokumente übersetzen lassen, um den Antrag abzuschließen.",
        "Nach dem Sprachkurs hat er die Prüfung mit sehr gutem Ergebnis bestanden.",
        "Die Einbürgerung erfordert normalerweise mindestens acht Jahre legalen Aufenthalt.",
        "Wir haben in dem Rathaus nach Informationen über Wohnsitzanmeldung gefragt.",
        "Der Arbeitsmarkt bietet qualifizierten Migranten mittlerweile viele neue Chancen.",
        "Ich fühle mich hier wohl, obwohl mir die Sprache noch schwerfällt.",
        "Die Residenzpflicht wurde in einigen Bundesländern vor Kurzem abgeschafft.",
        "Er arbeitet ehrenamtlich als Berater für neu zugezogene Familien in dem Viertel.",
        "Das Asylverfahren dauert je nach Fall unterschiedlich lange in Deutschland.",
        "Man braucht für die Anmeldung einen gültigen Reisepass und aktuelle Fotos.",
        "Viele junge Menschen lernen Deutsch, um hier studieren oder arbeiten zu können.",
    ],
    # 730–754: Education system
    [
        "In Deutschland ist die Schulpflicht für alle Kinder zwischen sechs und achtzehn.",
        "Mein Sohn besucht die Grundschule und ist sehr gut in Mathematik.",
        "Das duale Ausbildungssystem verbindet praktische Arbeit mit theoretischem Unterricht.",
        "Nach der Grundschule wählen die Eltern oft die passende weiterführende Schule.",
        "Die Universität bietet viele Studiengänge auf Deutsch und Englisch an.",
        "Ich muss für die Klausur nächste Woche viel Material durchlesen.",
        "Die Lehrerin erklärt den Schülern den neuen Stoff sehr geduldig.",
        "Er hat sein Abitur mit guter Note an einem Gymnasium gemacht.",
        "Das BAföG hilft vielen Studierenden, die Ausbildung finanziell zu schaffen.",
        "In der Berufsschule lernen die Azubis zwei Tage pro Woche Theorie.",
        "Die Prüfung war schwieriger, als wir alle erwartet hatten.",
        "Kinder mit Migrationshintergrund besuchen oft zusätzliche Nachhilfekurse nach der Schule.",
        "Die Schüler sollen bis Freitag ihre Hausaufgaben fertig abgeben.",
        "An der Hochschule gibt es viele internationale Austauschprogramme für Studenten.",
        "Meine Tochter möchte später Medizin studieren und Ärztin werden.",
        "Der Schulweg sollte für junge Kinder möglichst sicher und kurz sein.",
        "Wir haben gestern über die Reform des Bildungssystems in der Klasse diskutiert.",
        "Die Bibliothek öffnet wochentags früh und bleibt bis zwanzig Uhr geöffnet.",
        "Ohne ausreichende Deutschkenntnisse fällt das Lernen in der Schule schwer.",
        "Der Professor hat uns eine längere Liste mit Literatur empfohlen.",
        "Viele Schulen setzen heute digitale Tablets in dem täglichen Unterricht ein.",
        "Die Abschlussprüfung entscheidet oft über den weiteren beruflichen Werdegang.",
        "Ich finde, dass Inklusion an Schulen für alle Beteiligten wichtig ist.",
        "Die Ferien beginnen in Bayern meist etwas später als in Norddeutschland.",
        "Er hat seine Ausbildung zu dem Mechatroniker in nur drei Jahren abgeschlossen.",
    ],
    # 755–779: Consumer rights
    [
        "Ich habe das defekte Produkt innerhalb von zwei Wochen zurückgegeben.",
        "Der Kunde hat das Recht, fehlerhafte Ware innerhalb von vierzehn Tagen zu reklamieren.",
        "Die Gewährleistung gilt für neue Geräte in der Regel zwei Jahre lang.",
        "Sie sollten immer die Quittung aufbewahren, falls später Probleme auftreten.",
        "Bei dem Onlinekauf hat man oft ein gesetzliches Widerrufsrecht von vierzehn Tagen.",
        "Die Verbraucherzentrale hat mir bei der Beschwerde gegen den Anbieter geholfen.",
        "Das Gerät war bei dem Auspacken beschädigt, deshalb habe ich sofort reklamiert.",
        "Versteckte Kosten in dem Vertrag sind nach deutschem Recht nicht zulässig.",
        "Ich habe den Kaufvertrag sorgfältig gelesen, bevor ich unterschrieben habe.",
        "Der Händler weigert sich leider, das kaputte Handy zu reparieren oder zu ersetzen.",
        "Bei irreführender Werbung kann man den Kaufvertrag unter Umständen anfechten.",
        "Wir haben uns über die Preiserhöhung direkt bei der Verbraucherschutzstelle beschwert.",
        "Der Preis auf dem Etikett muss mit dem Betrag an der Kasse übereinstimmen.",
        "Ich fordere eine schriftliche Bestätigung über die Rücksendung des Pakets.",
        "Der Lieferant hat die bestellte Ware trotz mehrfacher Erinnerung nicht geliefert.",
        "Abgemahnte Verträge können unter bestimmten Bedingungen ohne Kosten gekündigt werden.",
        "Sie dürfen keine persönlichen Daten ohne Ihre Zustimmung an Dritte weitergeben.",
        "Ich habe den Vertrag gekündigt, weil die Leistung nicht vertraglich erbracht wurde.",
        "Das Gesetz schützt Verbraucher vor unfairen Klauseln in langfristigen Verträgen.",
        "Man sollte den Zustand der Ware direkt nach der Lieferung sorgfältig prüfen.",
        "Die Heizkostenabrechnung enthielt mehrere Fehler, die ich schriftlich beanstandet habe.",
        "In dem Laden gilt: erst informieren, dann ohne Druck entscheiden.",
        "Der Kundendienst hat mein Anliegen innerhalb von drei Werktagen bearbeitet.",
        "Ich rate davon ab, Verträge zu unterschreiben, die man nicht versteht.",
        "Für digitale Inhalte gelten besondere Regeln bei dem Widerruf nach dem Kauf.",
    ],
    # 780–804: Health insurance
    [
        "In Deutschland ist eine Krankenversicherung für alle Einwohner vorgeschrieben.",
        "Ich bin bei einer gesetzlichen Krankenkasse und zahle den Mitgliedsbeitrag monatlich.",
        "Die Zuzahlung für Medikamente beträgt maximal zehn Euro pro Rezept.",
        "Mein Arzt hat mir eine Überweisung zu dem Facharzt für Orthopädie ausgestellt.",
        "Die Krankenkasse übernimmt die Kosten für Vorsorgeuntersuchungen meist vollständig.",
        "Ich habe gestern eine elektronische Gesundheitskarte von meiner Kasse erhalten.",
        "Ohne Versicherungsnachweis darf man keine medizinische Behandlung in Anspruch nehmen.",
        "Die private Krankenversicherung ist oft teurer, bietet aber mehr Leistungen.",
        "Termine bei dem Hausarzt sind in vielen Städten schwer kurzfristig zu bekommen.",
        "Er musste wegen der Operation mehrere Wochen in dem Krankenhaus bleiben.",
        "Die Familienversicherung gilt für Ehepartner und Kinder ohne zusätzlichen Beitrag.",
        "Ich habe mich bei der Krankenkasse über Zusatzleistungen für Physiotherapie informiert.",
        "Bei einem Notfall sollte man die Nummer einhunderteinzwölf sofort anrufen.",
        "Die Kasse hat die Rechnung für die ambulante Behandlung innerhalb von zwei Wochen bezahlt.",
        "Man muss Krankmeldungen von dem Arzt spätestens an dem dritten Tag vorlegen.",
        "Viele Versicherte nutzen mittlerweile Online - Termine über Apps ihrer Krankenkassen.",
        "Die Gesundheitskarte muss bei jedem Arztbesuch zu der Legitimation mitgebracht werden.",
        "Ich finde die Wartezeiten in Praxen oft zu lang und sehr ärgerlich.",
        "Der Arbeitgeber zahlt während der Krankheit sechs Wochen lang das Gehalt weiter.",
        "Die Versicherung übernimmt nur Kosten, die medizinisch wirklich notwendig sind.",
        "Wir haben in dem Krankenhaus eine mehrsprachige Beratung zu Pflegeleistungen erhalten.",
        "Für Zahnbehandlungen gibt es besondere Regeln zu der Kostenübernahme durch die Kasse.",
        "Ich habe mich nach einem Unfall sofort in der nächsten Notaufnahme gemeldet.",
        "Die Beiträge zu der Krankenversicherung steigen in den letzten Jahren stetig an.",
        "Jeder Versicherte hat das Recht, seine Krankenkasse einmal in dem Jahr zu wechseln.",
    ],
    # 805–829: Urban planning
    [
        "Die Stadt plant, in dem Zentrum mehr Fahrradwege und Grünflächen anzulegen.",
        "In unserem Viertel wird gerade ein neues Wohnprojekt mit günstigen Mietwohnungen gebaut.",
        "Der Bau einer Straßenbahnlinie soll den Verkehr in der Innenstadt spürbar entlasten.",
        "Viele Bürger haben auf der öffentlichen Sitzung über die Verkehrsplanung diskutiert.",
        "Die Fußgängerzone wurde letztes Jahr erweitert, um mehr Platz für Cafés zu schaffen.",
        "Hohe Mieten sind in Großstädten ein drängendes Problem der Stadtentwicklung.",
        "Wir brauchen mehr bezahlbaren Wohnraum für Familien und junge Paare.",
        "Der Stadtrat hat beschlossen, einen weiteren Park in dem Stadtteil zu eröffnen.",
        "Die Lärmbelastung an der Hauptstraße belastet viele Anwohner seit Jahren stark.",
        "Das neue Stadtviertel hat Schulen, Läden und eine gute Anbindung an den ÖPNV.",
        "Die Sanierung der alten Fabrikhalle dauert voraussichtlich noch zwei Jahre.",
        "Ich finde, dass mehr Bäume in den Straßen das Stadtklima deutlich verbessern.",
        "Der Abriss des leerstehenden Gebäudes wurde wegen Denkmalschutz zunächst gestoppt.",
        "Die Stadtverwaltung fördert energetische Sanierungen bei alten Wohngebäuden aktiv.",
        "Auf dem ehemaligen Industriegelände entstehen neue Wohnungen und kleine Läden.",
        "Wir setzen uns für eine Tempo - dreißig - Zone vor der Schule ein.",
        "Die Bebauungspläne für das Gewerbegebiet liegen ab Montag in dem Rathaus aus.",
        "Der Mangel an Parkplätzen führt in der Innenstadt regelmäßig zu langen Staus.",
        "In vielen Vierteln werden leerstehende Wohnungen in Wohnungen für Studierende umgewandelt.",
        "Die Stadt investiert jährlich Millionen in den Ausbau des öffentlichen Nahverkehrs.",
        "Ich wohne gern in einem lebendigen Viertel mit kurzen Wegen zu allen Geschäften.",
        "Der neue Stadtplan sieht vor, mehr öffentlichen Raum für Märkte zu schaffen.",
        "Viele Menschen wünschen sich weniger Autoverkehr und mehr sichere Radwege.",
        "Die Wohnungsbauprämie soll Familien bei dem Kauf einer eigenen Wohnung unterstützen.",
        "Ohne eine gute Infrastruktur verliert ein Stadtteil langfristig an Lebensqualität.",
    ],
    # 830–854: Social media
    [
        "Ich poste abends oft Fotos von meinen Reisen auf Instagram.",
        "Viele Jugendliche verbringen täglich mehrere Stunden in sozialen Netzwerken.",
        "Man sollte vorsichtig sein, welche persönlichen Daten man online teilt.",
        "Er hat mir gestern eine interessante Nachricht über WhatsApp geschickt.",
        "Die Influencerin bewirbt auf ihrem Kanal regelmäßig neue Beauty - Produkte.",
        "Ich folge mehreren deutschen Nachrichtenseiten auch auf Facebook und Twitter.",
        "Cybermobbing ist leider ein ernstes Problem an vielen Schulen geworden.",
        "Wir haben ein privates Album mit Familienfotos nur für Verwandte erstellt.",
        "Der Beitrag wurde innerhalb einer Stunde tausende Male geteilt und kommentiert.",
        "Ich lösche regelmäßig alte Posts, die ich nicht mehr online haben möchte.",
        "Viele Nutzer lesen nur Schlagzeilen, ohne den ganzen Artikel zu öffnen.",
        "Sie hat ihr Passwort geändert, weil ihr Account gehackt worden war.",
        "Auf TikTok lernen manche Jugendliche Sprachen mit kurzen unterhaltsamen Videos.",
        "Die Plattform hat neue Regeln gegen Hasskommentare und falsche Informationen eingeführt.",
        "Ich finde, dass zu viel Bildschirmzeit die Konzentration in der Schule stört.",
        "Er streamt jeden Freitag live und spricht dabei mit seinen Followern.",
        "Man kann in Foren oft hilfreiche Tipps von anderen Nutzern bekommen.",
        "Wir haben eine Gruppe für Nachbarn gegründet, um lokale Infos auszutauschen.",
        "Der Algorithmus zeigt mir häufig Inhalte, die meinen bisherigen Interessen ähneln.",
        "Ich schalte Benachrichtigungen ab, damit mich keine ständigen Nachrichten ablenken.",
        "Viele Unternehmen nutzen soziale Medien heute gezielt für Marketing und Werbung.",
        "Sie hat ein Video hochgeladen, das in wenigen Tagen viral gegangen ist.",
        "Datenschutz ist wichtig, wenn man sich auf neuen Plattformen registriert.",
        "Ich versuche, in dem Internet nur Informationen aus vertrauenswürdigen Quellen zu teilen.",
        "Der Influencer rät seinen Followern, kritisch mit Online - Werbung umzugehen.",
    ],
    # 855–879: Volunteering
    [
        "Ich engagiere mich ehrenamtlich in einer Tafel und verteile Lebensmittel an Bedürftige.",
        "Viele Freiwillige helfen in Asylheimen bei dem Sprachunterricht und bei Freizeitangeboten.",
        "Ohne ehrenamtliche Helfer könnten viele soziale Projekte gar nicht funktionieren.",
        "Sie spendet jeden Samstag zwei Stunden ihrer Zeit für die Stadtbibliothek.",
        "In dem Tierheim suchen sie dringend Freiwillige für die Pflege der Hunde.",
        "Bei dem Stadtfest haben wir Kuchen verkauft und Geld für Vereine gesammelt.",
        "Ehrenamtliche Arbeit bringt nicht nur anderen, sondern auch einem selbst viel Freude.",
        "Er betreut regelmäßig ältere Menschen und besucht sie einmal pro Woche.",
        "Die Organisation bietet Schulungen für alle neuen Freiwilligen in dem Frühjahr an.",
        "Ich finde es wichtig, etwas zurückzugeben und mich in der Gemeinde zu engagieren.",
        "Viele Jugendliche machen ein freiwilliges soziales Jahr nach dem Schulabschluss.",
        "Sie hat an einer Müllsammelaktion an dem Fluss mit anderen Freiwilligen teilgenommen.",
        "Der Verein sucht Helfer, die bei Veranstaltungen Tickets und Getränke verkaufen.",
        "Wir unterrichten Geflüchtete in Deutsch, obwohl wir selbst keine Lehrer sind.",
        "Ehrenamt bedeutet, ohne Bezahlung Zeit und Fähigkeiten für andere einzusetzen.",
        "In dem Nachbarschaftszentrum koordinieren Freiwillige Kurse für Kinder und Eltern.",
        "Er hat bei dem Aufbau des neuen Spielplatzes in dem Viertel kräftig mitgeholfen.",
        "Ich möchte meine Erfahrung als Jurist in dem Beratungszentrum freiwillig einbringen.",
        "Die Stadt vergibt Urkunden an besonders engagierte Freiwillige jedes Jahr in dem Dezember.",
        "Viele Rettungsdienste sind auf das Engagement von ehrenamtlichen Helfern angewiesen.",
        "Wir haben für Obdachlose warme Kleidung gesammelt und an eine Beratungsstelle geliefert.",
        "Sie engagiert sich in einem Projekt, das Lesen in Grundschulen fördert.",
        "Durch Freiwilligenarbeit lernt man neue Menschen kennen und gewinnt wertvolle Erfahrungen.",
        "Der Sportverein lebt vor allem von den vielen unbezahlten Trainern und Organisatoren.",
        "Ich plane, nach der Rente noch mehr Zeit für soziales Engagement zu haben.",
    ],
    # 880–904: EU topics
    [
        "Als EU - Bürger darf man in jedem Mitgliedstaat leben und arbeiten.",
        "Die Europäische Union fördert den kulturellen Austausch zwischen den Mitgliedsländern.",
        "Ich habe mit Erasmus ein Semester in Spanien studiert und viel gelernt.",
        "Der Grenzübertritt innerhalb des Schengen - Raums ist für Reisende meist unkompliziert.",
        "Viele junge Menschen nutzen Erasmus, um in dem Ausland Erfahrungen zu sammeln.",
        "Die EU verabschiedet gemeinsame Regeln für Umweltschutz und Klimaschutz in Europa.",
        "Wir haben auf dem Europatag in dem Rathaus über die Geschichte der Union gesprochen.",
        "Der Euro erleichtert Reisen und Einkäufe in vielen europäischen Ländern deutlich.",
        "Die Europäische Kommission schlägt neue Gesetze vor, die das Parlament berät.",
        "Ich finde, dass die EU für Frieden und Zusammenarbeit in Europa wichtig ist.",
        "Freizügigkeit ist eines der wichtigsten Rechte für Bürger innerhalb der Europäischen Union.",
        "Viele Regionen erhalten EU - Fördermittel für Infrastruktur und wirtschaftliche Entwicklung.",
        "Der Europäische Gerichtshof entscheidet Streitigkeiten über die Auslegung von EU - Recht.",
        "Wir diskutieren in der Schule über Vor - und Nachteile der europäischen Integration.",
        "Die Grenzkontrollen wurden nach dem Beitritt vieler Länder in dem Schengen - Raum abgeschafft.",
        "Ich reise gern durch Europa, weil viele Länder eine gemeinsame Währung nutzen.",
        "Das Europäische Parlament wird alle fünf Jahre direkt von den Bürgern gewählt.",
        "Die EU unterstützt Forschungsprojekte an Universitäten in verschiedenen Mitgliedstaaten.",
        "Viele Verbraucher profitieren von gemeinsamen Standards für Produkte auf dem Binnenmarkt.",
        "Er arbeitet in Brüssel und berichtet über Entscheidungen der europäischen Institutionen.",
        "Die Mitgliedstaaten verhandeln gemeinsam über Handelsabkommen mit Drittstaaten.",
        "Ich habe an einem EU - Programm teilgenommen, das Jugendaustausch in Europa fördert.",
        "Die Europäische Union setzt sich für Menschenrechte und Demokratie in Europa ein.",
        "Als EU - Bürger kann man in vielen Ländern ohne Visum reisen und wohnen.",
        "Wir haben in dem Unterricht über die Erweiterung der Union um neue Mitgliedsländer gesprochen.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

START_ID = 705
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
    "Syrien": ("Syrien", "PROPN"),
    "Spanien": ("Spanien", "PROPN"),
    "Brüssel": ("Brüssel", "PROPN"),
    "Deutschland": ("Deutschland", "PROPN"),
    "Norddeutschland": ("Norddeutschland", "PROPN"),
    "Europa": ("Europa", "PROPN"),
    "Instagram": ("Instagram", "PROPN"),
    "WhatsApp": ("WhatsApp", "PROPN"),
    "Facebook": ("Facebook", "PROPN"),
    "Twitter": ("Twitter", "PROPN"),
    "TikTok": ("TikTok", "PROPN"),
    "Erasmus": ("Erasmus", "PROPN"),
    "BAföG": ("BAföG", "PROPN"),
    "ÖPNV": ("ÖPNV", "PROPN"),
    "EU": ("EU", "PROPN"),
    "EU-Bürger": ("EU-Bürger", "PROPN"),
    "EU-Fördermittel": ("EU-Fördermittel", "PROPN"),
    "EU-Programm": ("EU-Programm", "PROPN"),
    "EU-Recht": ("EU-Recht", "PROPN"),
    "Schengen-Raum": ("Schengen-Raum", "PROPN"),
    "Schengen-Raums": ("Schengen-Raum", "PROPN"),
    "Europatag": ("Europatag", "PROPN"),
    "Euro": ("Euro", "PROPN"),
    "Online-Termine": ("Online-Termin", "NOUN"),
    "Beauty-Produkte": ("Beauty-Produkt", "NOUN"),
}

SCONJS = {"weil", "dass", "obwohl", "wenn", "ob", "da", "als", "wie", "während", "seit", "bis", "bevor", "deshalb", "falls"}
CCONJS = {"und", "oder", "aber", "sondern", "als"}
ADPS = {
    "um", "für", "mit", "von", "bei", "nach", "aus", "über", "vor", "unter",
    "durch", "ohne", "gegen", "wegen", "in", "an", "auf", "am", "im", "ins",
    "ab", "zur", "zum", "zur", "gegenüber",
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
        if lower in {"ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen"}:
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
    elif lower in {"ich", "du", "er", "sie", "es", "wir", "man"}:
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

    target_path = project_root / "data/handcraft/de/train/b1_new_008.conllu"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    all_lines: list[str] = []
    global_idx = 0

    for batch_num, batch in enumerate(SENTENCE_BATCHES):
        assert len(batch) == BATCH_SIZE, f"Batch {batch_num} has {len(batch)} sentences"
        print(
            f"Processing batch {batch_num + 1}/8 "
            f"(sentences {START_ID + global_idx}–{START_ID + global_idx + BATCH_SIZE - 1})"
        )
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