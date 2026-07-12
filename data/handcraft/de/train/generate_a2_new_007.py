"""Generate a2_new_007.conllu — de_a2_train_776 through de_a2_train_975."""

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 8 batches × 25 sentences = 200
BATCHES = [
    # 776–800: Sports
    [
        "Wir spielen jeden Samstag Fußball im Park.",
        "Er läuft jeden Morgen drei Kilometer.",
        "Sie schwimmt gern im neuen Schwimmbad.",
        "Ich gehe zweimal pro Woche joggen.",
        "Der Verein trainiert am Dienstag im Stadion.",
        "Wir haben das Spiel gestern gewonnen.",
        "Er spielt Tennis mit seinem Bruder.",
        "Sie macht Yoga am frühen Morgen.",
        "Ich fahre gern mit dem Fahrrad.",
        "Wir schauen das Fußballspiel im Fernsehen.",
        "Er ist sehr gut im Basketball.",
        "Sie hat beim Laufen den Rekord gebrochen.",
        "Wir treffen uns zum Sport um acht Uhr.",
        "Ich brauche neue Schuhe für den Sport.",
        "Der Trainer erklärt die Übungen sehr gut.",
        "Er trainiert jeden Tag im Fitnessstudio.",
        "Wir haben gestern Volleyball im Park gespielt.",
        "Sie möchte im Sommer Ski fahren lernen.",
        "Ich finde Radfahren sehr entspannend und gesund.",
        "Er holt sich vor dem Sport Wasser.",
        "Wir sind nach dem Training sehr müde.",
        "Sie gewinnt oft bei den kleinen Wettkämpfen.",
        "Ich ziehe mir schnell die Sportsachen an.",
        "Er schaut gern Live-Spiele am Abend.",
        "Wir feiern nach dem Sieg mit der Mannschaft.",
    ],
    # 801–825: Festivals
    [
        "Das Stadtfest beginnt am Samstag um zehn Uhr.",
        "Wir gehen gern auf das Oktoberfest in München.",
        "Es gibt viele Lichter auf dem Weihnachtsmarkt.",
        "Sie tanzt mit Freunden auf dem Fest.",
        "Ich kaufe süße Äpfel auf dem Markt.",
        "Wir hören Live-Musik auf dem Platz.",
        "Er isst Bratwurst beim Sommerfest im Dorf.",
        "Sie trägt ein schönes Kostüm zum Karneval.",
        "Das Feuerwerk war gestern Abend sehr schön.",
        "Wir treffen uns vor dem großen Festzelt.",
        "Ich habe viele Fotos vom Stadtfest gemacht.",
        "Sie lacht gern auf dem lustigen Fest.",
        "Wir bringen Kuchen zur Party mit.",
        "Wir feiern Ostern gern mit der Familie.",
        "Ich finde das Musikfest sehr interessant.",
        "Sie kauft ein Geschenk auf dem Weihnachtsmarkt.",
        "Er tanzt Polka auf dem Volksfest.",
        "Wir essen zusammen auf dem Straßenfest.",
        "Ich habe neue Freunde beim Fest kennengelernt.",
        "Sie wartet auf das große Festfeuerwerk.",
        "Der Markt ist voller bunter Blumen und Lichter.",
        "Wir gehen am Abend gern auf Festivals.",
        "Er spielt Gitarre beim kleinen Straßenfest.",
        "Sie findet das Stadtfest besser als letztes Jahr.",
        "Ich möchte nächstes Jahr wieder zum Fest gehen.",
    ],
    # 826–850: Phone calls
    [
        "Ich rufe dich heute Abend an.",
        "Kannst du mich bitte zurückrufen?",
        "Sie telefoniert gern mit ihrer Mutter.",
        "Er hat gestern lange mit Anna telefoniert.",
        "Wir sprechen gleich am Telefon darüber.",
        "Ich habe deine Nachricht auf dem Handy gelesen.",
        "Sie ruft den Arzt wegen dem Termin an.",
        "Er hebt das Telefon sofort ab.",
        "Wir müssen den Kunden noch einmal anrufen.",
        "Ich höre dich am Telefon sehr schlecht.",
        "Sie legt das Gespräch nach fünf Minuten auf.",
        "Er fragt nach deiner Nummer am Telefon.",
        "Wir telefonieren jeden Sonntag mit den Eltern.",
        "Ich kann dich leider gerade nicht anrufen.",
        "Sie wartet auf deinen Anruf seit gestern.",
        "Er hat vergessen, seine Mutter anzurufen.",
        "Wir reden später noch einmal am Telefon.",
        "Ich gebe dir meine Nummer per SMS.",
        "Sie spricht leise am Telefon im Büro.",
        "Er ruft mich immer am Morgen an.",
        "Wir haben das Problem am Telefon besprochen.",
        "Ich antworte dir gleich auf deine Nachricht.",
        "Sie ist gerade nicht erreichbar am Telefon.",
        "Er möchte dich heute zum Essen einladen.",
        "Wir verabreden uns schnell am Telefon.",
    ],
    # 851–875: Bank / Post office
    [
        "Ich gehe heute zur Bank am Markt.",
        "Sie möchte Geld vom Konto abheben.",
        "Er schickt einen Brief auf der Post.",
        "Wir brauchen Briefmarken für drei Briefe.",
        "Ich habe ein Paket auf der Post abgeholt.",
        "Sie eröffnet ein neues Konto bei der Bank.",
        "Er bezahlt die Rechnung mit seiner Karte.",
        "Wir warten in der Schlange an der Post.",
        "Ich möchte hundert Euro wechseln bitte.",
        "Sie fragt nach dem Kontostand an der Bank.",
        "Er liefert das Paket zur Post am Bahnhof.",
        "Wir haben die Überweisung gestern gemacht.",
        "Ich hebe am Automaten Bargeld ab.",
        "Sie schickt das Formular per Post nach Hause.",
        "Er unterschreibt den Vertrag in der Bank.",
        "Wir kaufen Briefmarken am Schalter in der Post.",
        "Ich habe meine PIN am Automaten vergessen.",
        "Sie möchte ein kleines Paket verschicken.",
        "Er zahlt die Miete jeden Monat ein.",
        "Wir fragen nach den Öffnungszeiten der Bank.",
        "Ich bekomme meine Karte heute von der Bank.",
        "Sie zahlt das Geld auf ihr Sparkonto ein.",
        "Er wartet auf den Brief von der Bank.",
        "Wir schicken die Postkarte an unsere Freunde.",
        "Ich brauche eine Quittung von der Bank.",
    ],
    # 876–900: Comparisons (als / wie)
    [
        "Mein Bruder ist größer als ich.",
        "Das Auto ist schneller als das Fahrrad.",
        "Sie läuft so schnell wie er.",
        "Der Tee ist heißer als das Wasser.",
        "Ich bin genauso groß wie meine Schwester.",
        "Das Fest ist schöner als der Film.",
        "Er ist jünger als seine Freundin.",
        "Die Stadt ist größer als unser Dorf.",
        "Sie singt besser als ich.",
        "Das Hotel ist teurer als das Hostel.",
        "Ich arbeite so viel wie du.",
        "Der Winter ist kälter als der Herbst.",
        "Er ist so stark wie sein Vater.",
        "Das Spiel war spannender als der Film.",
        "Sie ist genauso freundlich wie ihre Mutter.",
        "Mein Zimmer ist kleiner als deins.",
        "Der Zug ist schneller als der Bus.",
        "Ich esse weniger als mein Bruder.",
        "Das Wetter ist heute schöner als gestern.",
        "Er lernt so schnell wie sie.",
        "Die Suppe ist besser als der Salat.",
        "Sie ist älter als ich dachte.",
        "Das Buch ist interessanter als der Film.",
        "Wir sind genauso müde wie ihr.",
        "Der Kaffee ist stärker als der Tee.",
    ],
    # 901–925: Reflexive verbs
    [
        "Ich wasche mich jeden Morgen schnell.",
        "Sie freut sich auf das große Fest.",
        "Er interessiert sich für Fußball und Tennis.",
        "Wir treffen uns um acht vor dem Café.",
        "Du musst dich vor dem Sport umziehen.",
        "Sie erinnert sich an den schönen Urlaub.",
        "Er entschuldigt sich bei seinem Freund.",
        "Ich konzentriere mich auf die Hausaufgaben.",
        "Wir setzen uns auf die Bank im Park.",
        "Sie wundert sich über das große Geschenk.",
        "Er zieht sich warme Kleidung an.",
        "Ich erhole mich nach dem langen Training.",
        "Sie ärgert sich über den kleinen Fehler.",
        "Wir amüsieren uns gern auf dem Fest.",
        "Er duscht sich jeden Abend vor dem Schlaf.",
        "Ich beeile mich, weil der Zug kommt.",
        "Sie kämmt sich die Haare vor dem Spiegel.",
        "Er ruht sich nach dem Spiel kurz aus.",
        "Wir informieren uns über das neue Programm.",
        "Sie schämt sich ein bisschen vor allen.",
        "Ich wasche mir die Hände vor dem Essen.",
        "Er wünscht sich ein neues Fahrrad.",
        "Wir freuen uns auf das Wochenende.",
        "Sie setzt sich neben mich auf die Bank.",
        "Ich entspanne mich gern nach der Arbeit.",
    ],
    # 926–950: Separable verbs
    [
        "Ich stehe jeden Tag um sieben auf.",
        "Er schaltet das Licht im Zimmer aus.",
        "Sie kommt heute Nachmittag mit.",
        "Wir fahren morgen früh nach Berlin ab.",
        "Ich räume mein Zimmer am Samstag auf.",
        "Wir geben das Paket an der Post ab.",
        "Wir holen Lisa vom Bahnhof ab.",
        "Ich ziehe mir schnell die Jacke an.",
        "Er nimmt den Anruf sofort an.",
        "Ich kaufe im Supermarkt für uns ein.",
        "Wir fangen um acht Uhr mit Sport an.",
        "Ich schalte abends gern den Fernseher ein.",
        "Wir bringen die Briefe zur Post mit.",
        "Sie ruft ihre Freundin gleich an.",
        "Wir machen das Fest am Samstag zu.",
        "Ich komme heute leider nicht mit.",
        "Er schreibt die Nummer schnell auf.",
        "Sie lädt ihre Freunde zum Fest ein.",
        "Der Kurs hört um zehn mit dem Training auf.",
        "Ich mache den Computer nach der Arbeit aus.",
        "Er fährt um acht Uhr von zu Hause ab.",
        "Sie schließt das Fenster wegen der Kälte.",
        "Ich steige an der nächsten Haltestelle aus.",
        "Wir geben die Formulare in der Bank ab.",
        "Er ruft seine Mutter jeden Abend an.",
    ],
    # 951–975: Mixed review
    [
        "Ich gehe nach dem Sport schnell nach Hause.",
        "Sie ruft die Bank wegen dem Konto an.",
        "Das Fußballspiel war spannender als erwartet.",
        "Wir feiern das Fest und treffen uns früh.",
        "Er hebt am Automaten zweihundert Euro ab.",
        "Sie freut sich auf das Stadtfest am Samstag.",
        "Ich telefoniere gleich mit meinem Trainer.",
        "Der Brief kommt morgen mit der Post.",
        "Er ist so schnell wie die beste Läuferin.",
        "Wir kaufen Briefmarken und schicken das Paket.",
        "Sie zieht sich die Sportschuhe schnell an.",
        "Ich rufe dich nach dem Training an.",
        "Das Oktoberfest ist größer als unser Markt.",
        "Er entschuldigt sich am Telefon bei ihr.",
        "Wir warten an der Post auf das Paket.",
        "Sie läuft jeden Tag so weit wie er.",
        "Ich lege nach dem Gespräch das Handy auf.",
        "Er zahlt die Rechnung direkt auf dem Konto ein.",
        "Wir machen das Sportfest am Sonntag zu.",
        "Sie interessiert sich für den neuen Tanzkurs.",
        "Das Festzelt ist voller Musik und lauter Stimmen.",
        "Ich hole das Paket heute von der Post ab.",
        "Er ist stärker als ich, aber langsamer.",
        "Wir treffen uns vor dem Stadion zum Spiel.",
        "Sie schickt die Postkarte vom Fest an uns.",
    ],
]

sentences = [s for batch in BATCHES for s in batch]
assert len(sentences) == 200, f"Expected 200 sentences, got {len(sentences)}"

import stanza

print("Loading Stanza...")
nlp = stanza.Pipeline(lang="de", processors="tokenize,mwt,pos,lemma", use_gpu=False)

output_lines: list[str] = []
start_id = 776

MODALS = {"müssen", "können", "wollen", "sollen", "dürfen", "mögen", "möchte", "möchtest", "möchten"}
AUX_FORMS = {
    "ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen",
    "habe", "hat", "hatte", "hätte", "hätten", "haben", "habt", "gehabt",
    "wird", "wurde", "werden", "würde", "würden", "geworden", "wirst",
}
AUX_LEMMAS = {"sein", "haben", "werden"}

CONTRACTIONS = {
    "im": ("in", "ADP"),
    "am": ("an", "ADP"),
    "zum": ("zu", "ADP"),
    "zur": ("zu", "ADP"),
    "vom": ("von", "ADP"),
    "ins": ("in", "ADP"),
    "beim": ("bei", "ADP"),
    "ans": ("an", "ADP"),
}

CONTRACTION_EXPANSIONS = {
    "im": ("in", "dem"),
    "am": ("an", "dem"),
    "zum": ("zu", "dem"),
    "zur": ("zu", "der"),
    "beim": ("bei", "dem"),
    "vom": ("von", "dem"),
    "ins": ("in", "das"),
    "ans": ("an", "das"),
}

HYPHENATED_NOUNS = {
    "U-Bahn": "U-Bahn",
    "E-Mail": "E-Mail",
    "Live-Musik": "Live-Musik",
    "Live-Spiele": "Live-Spiel",
    "Fußballspiel": "Fußballspiel",
}

SEPARABLE_PREFIXES = {
    "an", "auf", "ab", "aus", "ein", "mit", "vor", "zu", "weg", "her", "hin",
    "los", "fest", "weiter", "zurück", "runter", "rauf", "raus", "rein",
}

# verb form + sentence-final particle (particle tagging only; verb lemma stays consistent)
SEPARABLE_VERB_PAIRS: dict[tuple[str, str], str] = {
    ("rufe", "an"): "anrufen",
    ("ruft", "an"): "anrufen",
    ("legt", "auf"): "auflegen",
    ("lege", "auf"): "auflegen",
    ("hebt", "ab"): "abheben",
    ("hole", "ab"): "abheben",
    ("zieht", "an"): "anziehen",
    ("ziehe", "an"): "anziehen",
    ("nimmt", "an"): "annehmen",
    ("fange", "an"): "anfangen",
    ("fangen", "an"): "anfangen",
    ("schalte", "ein"): "einschalten",
    ("bringen", "mit"): "mitbringen",
    ("schreibt", "auf"): "aufschreiben",
    ("lädt", "ein"): "einladen",
    ("hört", "auf"): "aufhören",
    ("schaltet", "aus"): "ausmachen",
    ("fährt", "ab"): "abfahren",
    ("stehe", "auf"): "aufstehen",
    ("kommt", "mit"): "mitkommen",
    ("komme", "mit"): "mitkommen",
    ("fahren", "ab"): "abfahren",
    ("räume", "auf"): "aufräumen",
    ("steige", "aus"): "aussteigen",
    ("geben", "ab"): "abgeben",
    ("holen", "ab"): "abholen",
    ("kaufe", "ein"): "einkaufen",
    ("hebe", "ab"): "abheben",
    ("ruht", "aus"): "ausruhen",
    ("zahlt", "ein"): "einzahlen",
    ("mache", "zu"): "zumachen",
    ("machen", "zu"): "zumachen",
}

COMPARISON_WIE_TRIGGERS = {"so", "genauso", "viel", "weit"}

SPECIAL_LEMMAS = {
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
    "deins": ("dein", "PRON"),
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
    "jeden": ("jeder", "DET"),
    "jede": ("jeder", "DET"),
    "jedes": ("jeder", "DET"),
    "jeder": ("jeder", "DET"),
    "jedem": ("jeder", "DET"),
    "neue": ("neu", "ADJ"),
    "neuen": ("neu", "ADJ"),
    "neues": ("neu", "ADJ"),
    "neuer": ("neu", "ADJ"),
    "neuem": ("neu", "ADJ"),
    "großen": ("groß", "ADJ"),
    "große": ("groß", "ADJ"),
    "großes": ("groß", "ADJ"),
    "groß": ("groß", "ADJ"),
    "größer": ("groß", "ADJ"),
    "größere": ("groß", "ADJ"),
    "kleinen": ("klein", "ADJ"),
    "kleines": ("klein", "ADJ"),
    "kleine": ("klein", "ADJ"),
    "kleinem": ("klein", "ADJ"),
    "kleiner": ("klein", "ADJ"),
    "guten": ("gut", "ADJ"),
    "gute": ("gut", "ADJ"),
    "gutes": ("gut", "ADJ"),
    "gutem": ("gut", "ADJ"),
    "guter": ("gut", "ADJ"),
    "besser": ("gut", "ADJ"),
    "bessere": ("gut", "ADJ"),
    "netten": ("nett", "ADJ"),
    "nette": ("nett", "ADJ"),
    "schönen": ("schön", "ADJ"),
    "schön": ("schön", "ADJ"),
    "schöner": ("schön", "ADJ"),
    "schönes": ("schön", "ADJ"),
    "schöne": ("schön", "ADJ"),
    "frischen": ("frisch", "ADJ"),
    "frisches": ("frisch", "ADJ"),
    "frische": ("frisch", "ADJ"),
    "warmen": ("warm", "ADJ"),
    "warme": ("warm", "ADJ"),
    "heiße": ("heiß", "ADJ"),
    "heißen": ("heiß", "ADJ"),
    "heißer": ("heiß", "ADJ"),
    "scharf": ("scharf", "ADJ"),
    "starke": ("stark", "ADJ"),
    "stärker": ("stark", "ADJ"),
    "schnellen": ("schnell", "ADJ"),
    "schnelle": ("schnell", "ADJ"),
    "schneller": ("schnell", "ADJ"),
    "langsamer": ("langsam", "ADJ"),
    "teurer": ("teuer", "ADJ"),
    "kälter": ("kalt", "ADJ"),
    "jünger": ("jung", "ADJ"),
    "älter": ("alt", "ADJ"),
    "kleiner": ("klein", "ADJ"),
    "weniger": ("wenig", "ADV"),
    "spannender": ("spannend", "ADJ"),
    "spannendes": ("spannend", "ADJ"),
    "interessanter": ("interessant", "ADJ"),
    "interessant": ("interessant", "ADJ"),
    "freundlichen": ("freundlich", "ADJ"),
    "freundlich": ("freundlich", "ADJ"),
    "freundliches": ("freundlich", "ADJ"),
    "lustigen": ("lustig", "ADJ"),
    "bunten": ("bunt", "ADJ"),
    "bunte": ("bunt", "ADJ"),
    "süße": ("süß", "ADJ"),
    "süßen": ("süß", "ADJ"),
    "lauter": ("laut", "ADJ"),
    "bitte": ("bitte", "ADV"),
    "heute": ("heute", "ADV"),
    "gestern": ("gestern", "ADV"),
    "morgen": ("morgen", "ADV"),
    "bald": ("bald", "ADV"),
    "immer": ("immer", "ADV"),
    "oft": ("oft", "ADV"),
    "schon": ("schon", "ADV"),
    "noch": ("noch", "ADV"),
    "nur": ("nur", "ADV"),
    "sehr": ("sehr", "ADV"),
    "etwas": ("etwas", "ADV"),
    "gern": ("gern", "ADV"),
    "gerne": ("gern", "ADV"),
    "hier": ("hier", "ADV"),
    "dort": ("dort", "ADV"),
    "dann": ("dann", "ADV"),
    "gleich": ("gleich", "ADV"),
    "pünktlich": ("pünktlich", "ADV"),
    "leider": ("leider", "ADV"),
    "leise": ("leise", "ADV"),
    "schnell": ("schnell", "ADV"),
    "so": ("so", "ADV"),
    "genauso": ("genauso", "ADV"),
    "zweimal": ("zweimal", "ADV"),
    "dreimal": ("dreimal", "ADV"),
    "pro": ("pro", "ADP"),
    "per": ("per", "ADP"),
    "statt": ("statt", "ADP"),
    "wegen": ("wegen", "ADP"),
}

VERB_LEMMA_MAP = {
    "gegessen": "essen",
    "bestellt": "bestellen",
    "geschlossen": "schließen",
    "bezahlt": "bezahlen",
    "gelesen": "lesen",
    "verschrieben": "verschreiben",
    "vergessen": "vergessen",
    "verletzt": "verletzen",
    "angekommen": "ankommen",
    "verlaufen": "verfahren",
    "abgegeben": "abgeben",
    "verpasst": "verpassen",
    "gefahren": "fahren",
    "verschoben": "verschieben",
    "abgesagt": "absagen",
    "verabredet": "verabreden",
    "notiert": "notieren",
    "gebucht": "buchen",
    "gearbeitet": "arbeiten",
    "begonnen": "beginnen",
    "angefangen": "anfangen",
    "gespielt": "spielen",
    "gekauft": "kaufen",
    "erledigt": "erledigen",
    "gemacht": "machen",
    "gesehen": "sehen",
    "gekommen": "kommen",
    "geholfen": "helfen",
    "aufgeräumt": "aufräumen",
    "gewartet": "warten",
    "verbrannt": "verbrennen",
    "gewonnen": "gewinnen",
    "gebrochen": "brechen",
    "trainiert": "trainieren",
    "läuft": "laufen",
    "schwimmt": "schwimmen",
    "spielt": "spielen",
    "spielen": "spielen",
    "joggen": "joggen",
    "macht": "machen",
    "fahre": "fahren",
    "schauen": "schauen",
    "schaut": "schauen",
    "ist": "sein",
    "sind": "sein",
    "bin": "sein",
    "war": "sein",
    "habe": "haben",
    "hat": "haben",
    "haben": "haben",
    "hast": "haben",
    "wird": "werden",
    "werden": "werden",
    "möchte": "mögen",
    "möchtest": "mögen",
    "möchten": "mögen",
    "kann": "können",
    "kannst": "können",
    "können": "können",
    "muss": "müssen",
    "müssen": "müssen",
    "will": "wollen",
    "wollen": "wollen",
    "soll": "sollen",
    "sollst": "sollen",
    "darf": "dürfen",
    "gehe": "gehen",
    "geht": "gehen",
    "fährt": "fahren",
    "lernt": "lernen",
    "findet": "finden",
    "finden": "finden",
    "ziehe": "ziehen",
    "zieht": "ziehen",
    "holt": "holen",
    "gewinnt": "gewinnen",
    "feiern": "feiern",
    "beginnt": "beginnen",
    "tanzt": "tanzen",
    "kaufe": "kaufen",
    "kauft": "kaufen",
    "hören": "hören",
    "isst": "essen",
    "trägt": "tragen",
    "lacht": "lachen",
    "bringen": "bringen",
    "kennengelernt": "kennenlernen",
    "rufe": "anrufen",
    "ruft": "anrufen",
    "zurückrufen": "zurückrufen",
    "telefoniert": "telefonieren",
    "telefoniere": "telefonieren",
    "telefonieren": "telefonieren",
    "sprechen": "sprechen",
    "spricht": "sprechen",
    "nimmt": "nehmen",
    "anrufen": "anrufen",
    "anzurufen": "anrufen",
    "legt": "auflegen",
    "fragt": "fragen",
    "frage": "fragen",
    "besprochen": "besprechen",
    "antworte": "antworten",
    "einladen": "einladen",
    "lädt": "einladen",
    "schickt": "schicken",
    "schicken": "schicken",
    "abgeholt": "abholen",
    "eröffnet": "eröffnen",
    "warten": "warten",
    "wartet": "warten",
    "wechseln": "wechseln",
    "überweisung": "überweisen",
    "hebe": "abheben",
    "vergessen": "vergessen",
    "verschicken": "verschicken",
    "zahlt": "einzahlen",
    "bekomme": "bekommen",
    "wasche": "waschen",
    "freut": "freuen",
    "interessiert": "interessieren",
    "treffen": "treffen",
    "trifft": "treffen",
    "umziehen": "umziehen",
    "erinnert": "erinnern",
    "entschuldigt": "entschuldigen",
    "konzentriere": "konzentrieren",
    "setzen": "setzen",
    "setzt": "setzen",
    "wundert": "wundern",
    "erhole": "erholen",
    "ärgert": "ärgern",
    "amüsieren": "amüsieren",
    "duscht": "duschen",
    "beeile": "beeilen",
    "kämmt": "kämmen",
    "ruht": "ausruhen",
    "informieren": "informieren",
    "schämt": "schämen",
    "wünscht": "wünschen",
    "entspanne": "entspannen",
    "stehe": "aufstehen",
    "steigen": "aussteigen",
    "steige": "aussteigen",
    "mache": "machen",
    "kommt": "mitkommen",
    "komme": "mitkommen",
    "fahren": "abfahren",
    "räume": "aufräumen",
    "geben": "geben",
    "holen": "abholen",
    "kaufe": "einkaufen",
    "schalte": "einschalten",
    "schaltet": "ausmachen",
    "schließt": "schließen",
    "liefert": "liefern",
    "schreibt": "aufschreiben",
    "höre": "hören",
    "hört": "aufhören",
    "hebt": "abheben",
    "lege": "auflegen",
    "erklärt": "erklären",
    "unterschreibt": "unterschreiben",
    "singt": "singen",
    "esse": "essen",
    "dachte": "denken",
    "arbeite": "arbeiten",
    "gibt": "geben",
    "gibst": "geben",
    "reden": "reden",
    "gebe": "geben",
    "verabreden": "verabreden",
}

NOUN_LEMMA_MAP = {
    "Kinder": "Kind",
    "Hände": "Hand",
    "Haare": "Haar",
    "Äpfel": "Apfel",
    "Blumen": "Blume",
    "Lichter": "Licht",
    "Freunde": "Freund",
    "Briefe": "Brief",
    "Briefmarken": "Briefmarke",
    "Formulare": "Formular",
    "Sportsachen": "Sportsache",
    "Übungen": "Übung",
    "Wettkämpfen": "Wettkampf",
    "Mannschaft": "Mannschaft",
    "Stadtfest": "Stadtfest",
    "Oktoberfest": "Oktoberfest",
    "Weihnachtsmarkt": "Weihnachtsmarkt",
    "Sommerfest": "Sommerfest",
    "Karneval": "Karneval",
    "Feuerwerk": "Feuerwerk",
    "Festzelt": "Festzelt",
    "Fotos": "Foto",
    "Musikfest": "Musikfest",
    "Volksfest": "Volksfest",
    "Straßenfest": "Straßenfest",
    "Festfeuerwerk": "Festfeuerwerk",
    "Festivals": "Festival",
    "Nachricht": "Nachricht",
    "Anruf": "Anruf",
    "Gespräch": "Gespräch",
    "Nummer": "Nummer",
    "Eltern": "Eltern",
    "Konto": "Konto",
    "Kontostand": "Kontostand",
    "Überweisung": "Überweisung",
    "Automaten": "Automat",
    "Formular": "Formular",
    "Vertrag": "Vertrag",
    "Schalter": "Schalter",
    "PIN": "PIN",
    "Sparkonto": "Sparkonto",
    "Postkarte": "Postkarte",
    "Quittung": "Quittung",
    "Paket": "Paket",
    "Bank": "Bank",
    "Post": "Post",
    "Karte": "Karte",
    "Rechnung": "Rechnung",
    "Öffnungszeiten": "Öffnungszeit",
    "Verein": "Verein",
    "Stadion": "Stadion",
    "Fitnessstudio": "Fitnessstudio",
    "Volleyball": "Volleyball",
    "Basketball": "Basketball",
    "Tennis": "Tennis",
    "Fußball": "Fußball",
    "Trainer": "Trainer",
    "Rekord": "Rekord",
    "Training": "Training",
    "Sieg": "Sieg",
    "Kostüm": "Kostüm",
    "Bratwurst": "Bratwurst",
    "Geschenk": "Geschenk",
    "Polka": "Polka",
    "Gitarre": "Gitarre",
    "Handy": "Handy",
    "SMS": "SMS",
    "Kilometer": "Kilometer",
    "Schwimmbad": "Schwimmbad",
    "Fahrrad": "Fahrrad",
    "Schuhe": "Schuh",
    "Jacke": "Jacke",
    "Kleidung": "Kleidung",
    "Hausaufgaben": "Hausaufgabe",
    "Spiegel": "Spiegel",
    "Programm": "Programm",
    "Fernseher": "Fernseher",
    "Haltestelle": "Haltestelle",
    "Sportfest": "Sportfest",
    "Tanzkurs": "Tanzkurs",
    "Stimmen": "Stimme",
    "Läuferin": "Läuferin",
    "Hostel": "Hostel",
    "Herbst": "Herbst",
    "Winter": "Winter",
    "Salat": "Salat",
    "Suppe": "Suppe",
    "Kaffee": "Kaffee",
    "Tee": "Tee",
    "Wasser": "Wasser",
    "Auto": "Auto",
    "Bus": "Bus",
    "Zug": "Zug",
    "Hotel": "Hotel",
    "Film": "Film",
    "Buch": "Buch",
    "Spiel": "Spiel",
    "Wetter": "Wetter",
    "Markt": "Markt",
    "Dorf": "Dorf",
    "Stadt": "Stadt",
    "Zimmer": "Zimmer",
    "Licht": "Licht",
    "Fenster": "Fenster",
    "Computer": "Computer",
    "Kälte": "Kälte",
    "Kunde": "Kunde",
    "Arzt": "Arzt",
    "Termin": "Termin",
    "Miete": "Miete",
    "Bargeld": "Bargeld",
    "Euro": "Euro",
    "Party": "Party",
    "Platz": "Platz",
    "Abend": "Abend",
    "Morgen": "Morgen",
    "Wochenende": "Wochenende",
    "Urlaub": "Urlaub",
    "Fehler": "Fehler",
    "Schlaf": "Schlaf",
    "Arbeit": "Arbeit",
    "Sport": "Sport",
    "Telefon": "Telefon",
    "Bahnhof": "Bahnhof",
    "Supermarkt": "Supermarkt",
    "Schlange": "Schlange",
    "Café": "Café",
    "Park": "Park",
    "Familie": "Familie",
    "Mutter": "Mutter",
    "Bruder": "Bruder",
    "Schwester": "Schwester",
    "Vater": "Vater",
    "Freundin": "Freundin",
    "Freund": "Freund",
}


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    if form in CONTRACTIONS:
        lemma, upos = CONTRACTIONS[form]
    elif form in SPECIAL_LEMMAS:
        lemma, upos = SPECIAL_LEMMAS[form]

    if form.lower() in {"das", "die", "der", "den", "dem", "des"} and upos == "PRON":
        lemma = "der"
        upos = "DET"

    if upos == "NOUN":
        if form in NOUN_LEMMA_MAP:
            lemma = NOUN_LEMMA_MAP[form]
        elif lemma:
            lemma = lemma[0].upper() + lemma[1:]

    if upos in {"VERB", "AUX"} or form.lower() in MODALS or lemma.lower() in MODALS:
        if form.lower() in {"möchte", "möchtest", "möchten"}:
            upos = "VERB"
            lemma = "mögen"
        elif form.lower() in MODALS or lemma.lower() in MODALS:
            upos = "VERB"
            lemma = lemma.lower()
            if form in VERB_LEMMA_MAP:
                lemma = VERB_LEMMA_MAP[form]
            elif lemma in VERB_LEMMA_MAP:
                lemma = VERB_LEMMA_MAP[lemma]
        else:
            upos = "VERB"
            lemma = lemma.lower()
            if form in VERB_LEMMA_MAP:
                lemma = VERB_LEMMA_MAP[form]
            elif lemma in VERB_LEMMA_MAP:
                lemma = VERB_LEMMA_MAP[lemma]

    if form.lower() in AUX_FORMS:
        upos = "AUX"
        if form.lower() in {"ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen"}:
            lemma = "sein"
        elif form.lower() in {"habe", "hat", "hatte", "hätte", "hätten", "haben", "habt", "gehabt", "hast"}:
            lemma = "haben"
        elif form.lower() in {"wird", "wurde", "werden", "würde", "würden", "geworden", "wirst"}:
            lemma = "werden"

    if upos == "ADJ":
        lemma = lemma.lower()
        if form in SPECIAL_LEMMAS:
            lemma, _ = SPECIAL_LEMMAS[form]

    if upos == "PROPN" and lemma:
        lemma = lemma[0].upper() + lemma[1:]

    if upos == "PUNCT":
        lemma = form

    if form.lower() in {
        "weil", "dass", "obwohl", "wenn", "ob", "während", "seit", "bis",
    }:
        upos = "SCONJ"
        lemma = form.lower()
    elif form.lower() in {"und", "oder", "aber", "sondern"}:
        upos = "CCONJ"
        lemma = form.lower()
    elif form.lower() in {
        "um", "für", "mit", "von", "bei", "nach", "aus", "über", "vor", "unter",
        "durch", "ohne", "gegen", "wegen", "in", "an", "auf", "zu", "statt",
    }:
        if form.lower() == "zu" and upos == "PART":
            pass
        else:
            upos = "ADP"
            lemma = form.lower()
    elif form.lower() in {"nicht", "ja", "nein"}:
        upos = "PART"
        lemma = form.lower()
    elif form.lower() in {"ich", "du", "er", "sie", "es", "wir", "ihr"}:
        upos = "PRON"
        lemma = form.lower()

    if form == "Sie" or form == "Ihnen":
        upos = "PRON"
        lemma = "Sie"

    return lemma, upos


def post_process_tokens(aligned: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    verb_forms = {
        i: aligned[i][0].lower()
        for i in range(len(aligned))
        if aligned[i][2] == "VERB"
    }

    for i, (form, lemma, upos) in enumerate(aligned):
        lower = form.lower()

        if lower in SEPARABLE_PREFIXES:
            if i > 0 and aligned[i - 1][2] == "VERB":
                upos = "PART"
                lemma = lower
            elif i + 1 < len(aligned) and aligned[i + 1][0] in ".,;:!?":
                for vi, vform in verb_forms.items():
                    pair = (vform, lower)
                    if pair in SEPARABLE_VERB_PAIRS:
                        upos = "PART"
                        lemma = lower
                        break

        if lower == "als" and i > 0 and aligned[i - 1][2] == "ADJ":
            upos = "CCONJ"
            lemma = "als"

        if lower == "wie" and i > 0:
            prev_form, _, prev_upos = aligned[i - 1]
            if (
                prev_upos in {"ADJ", "ADV"}
                or prev_form.lower() in COMPARISON_WIE_TRIGGERS
            ):
                upos = "CCONJ"
                lemma = "wie"

        result.append((form, lemma, upos))
    return result


def tokenize_text(sentence: str) -> list[str]:
    tokens: list[str] = []
    for word in sentence.split():
        match = re.match(r"^(.+?)([.,;:!?]+)$", word)
        if match:
            tokens.append(match.group(1))
            tokens.extend(list(match.group(2)))
        else:
            tokens.append(word)
    return tokens


def stanza_words_flat(doc) -> list:
    words = []
    for stanza_sent in doc.sentences:
        for word in stanza_sent.words:
            if isinstance(word.id, int):
                words.append(word)
    return words


def match_expansion(form: str, words: list, start: int) -> int | None:
    if form not in CONTRACTION_EXPANSIONS:
        return None
    expansion = CONTRACTION_EXPANSIONS[form]
    if start + len(expansion) > len(words):
        return None
    for i, piece in enumerate(expansion):
        if words[start + i].text.lower() != piece.lower():
            return None
    return start + len(expansion)


def match_hyphen(form: str, words: list, start: int) -> int | None:
    if "-" not in form or start >= len(words):
        return None
    built = ""
    idx = start
    while idx < len(words) and len(built) < len(form):
        built += words[idx].text
        idx += 1
        if built == form:
            return idx
    return None


def align_tokens(sentence: str, words: list) -> list[tuple[str, str, str]]:
    aligned: list[tuple[str, str, str]] = []
    text_tokens = tokenize_text(sentence)
    wi = 0

    for form in text_tokens:
        if form in ".,;:!?":
            aligned.append((form, form, "PUNCT"))
            continue

        if wi >= len(words):
            lemma, upos = normalize_token(form, "X", form)
            aligned.append((form, lemma, upos))
            continue

        end = match_expansion(form, words, wi)
        if end is not None:
            head = words[wi]
            lemma, upos = normalize_token(form, head.upos or "X", head.lemma or form)
            aligned.append((form, lemma, upos))
            wi = end
            continue

        end = match_hyphen(form, words, wi)
        if end is not None:
            head = words[wi]
            upos = head.upos or "NOUN"
            if form in HYPHENATED_NOUNS:
                upos = "NOUN"
                lemma = HYPHENATED_NOUNS[form]
            else:
                lemma, upos = normalize_token(form, upos, head.lemma or form)
            aligned.append((form, lemma, upos))
            wi = end
            continue

        head = words[wi]
        lemma, upos = normalize_token(form, head.upos or "X", head.lemma or form)
        aligned.append((form, lemma, upos))
        wi += 1

    return post_process_tokens(aligned)


for idx, sent in enumerate(sentences):
    sent_id = f"de_a2_train_{start_id + idx}"
    doc = nlp(sent)
    words = stanza_words_flat(doc)

    output_lines.append(f"# sent_id = {sent_id}")
    output_lines.append(f"# text = {sent}")

    token_counter = 1
    for form, lemma, upos in align_tokens(sent, words):
        cols = [
            str(token_counter),
            form,
            lemma,
            upos,
            "_",
            "_",
            "_",
            "_",
            "_",
            "_",
        ]
        output_lines.append("\t".join(cols))
        token_counter += 1

    output_lines.append("")

output_lines.append("")

conllu_text = "\n".join(output_lines)
target_path = project_root / "data/handcraft/de/train/a2_new_007.conllu"
target_path.write_text(conllu_text, encoding="utf-8")
print(f"Wrote {target_path}")

validation_res = validate_text(conllu_text)
print("Validate:", validation_res.passed, "sentences:", validation_res.sentence_count)
if not validation_res.passed:
    for err in validation_res.errors[:30]:
        print(" ", err)

lemma_res = check_text(conllu_text, lang="de")
print("Lemma check:", lemma_res.passed)
if not lemma_res.passed:
    for err in lemma_res.errors[:30]:
        print(" ", err)

if not validation_res.passed or not lemma_res.passed:
    sys.exit(1)

print("STATUS: OK — 200 sentences, de_a2_train_776–de_a2_train_975")