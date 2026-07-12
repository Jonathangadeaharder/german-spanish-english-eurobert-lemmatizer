"""Generate a2_new_001.conllu — de_a2_val_016 through de_a2_val_100."""

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 4 batches: 20 + 20 + 20 + 25 = 85
BATCHES = [
    # 016–035: Daily routine & home
    [
        "Ich räume jeden Samstag mein Zimmer auf.",
        "Mein Vater kocht gern am Sonntag für uns.",
        "Wir frühstücken immer zusammen in der Küche.",
        "Sie putzt das Bad, während ich koche.",
        "Er hat gestern die Wäsche gewaschen.",
        "Ich muss noch die Hausaufgaben machen.",
        "Die Kinder spielen gern im Garten draußen.",
        "Wir schlafen normalerweise um elf ein.",
        "Meine Mutter arbeitet von Montag bis Freitag.",
        "Er wacht jeden Morgen um sechs auf.",
        "Ich trinke morgens immer eine Tasse Kaffee.",
        "Sie hat das Geschirr schon abgewaschen.",
        "Wir essen um halb acht zu Abend.",
        "Er möchte heute früh ins Bett gehen.",
        "Ich föhne mir die Haare nach dem Duschen.",
        "Das Baby schläft gerade im Wohnzimmer.",
        "Wir haben das Sofa in die Ecke gestellt.",
        "Sie ist gestern mit dem Hund spazieren gegangen.",
        "Ich darf heute Abend fernsehen.",
        "Er hilft seiner Mutter beim Kochen.",
    ],
    # 036–055: Shopping & city
    [
        "Ich kaufe Brot und Milch im Supermarkt.",
        "Die Bluse kostet neununddreißig Euro.",
        "Wir haben gestern neue Schuhe gekauft.",
        "Sie sucht ein Geschenk für ihren Bruder.",
        "Er bezahlt alles mit seiner Kreditkarte.",
        "Das Geschäft öffnet um neun Uhr morgens.",
        "Ich möchte diese Jacke bitte anprobieren.",
        "Wir sind im Kaufhaus die Treppe hinaufgegangen.",
        "Sie findet den Pullover zu teuer.",
        "Er hat den Schlüssel im Laden verloren.",
        "Ich brauche eine Quittung für das Hemd.",
        "Das Angebot ist günstiger als gestern.",
        "Wir treffen uns vor dem Eingang des Marktes.",
        "Sie hat ihre Tasche an der Kasse vergessen.",
        "Ich gehe jeden Dienstag auf den Wochenmarkt.",
        "Er kauft frisches Gemüse vom Stand.",
        "Wir haben viel Geld für Geschenke ausgegeben.",
        "Sie möchte den Rock in Größe M.",
        "Der Verkäufer erklärt die Größen sehr gut.",
        "Ich nehme zwei Äpfel und eine Banane.",
    ],
    # 056–075: Health, appointments & school
    [
        "Ich habe morgen einen Termin beim Zahnarzt.",
        "Sie muss drei Tage zu Hause bleiben.",
        "Er fühlt sich heute viel besser als gestern.",
        "Wir gehen zur Apotheke um Medizin zu kaufen.",
        "Der Arzt sagt, dass ich mehr trinken soll.",
        "Ich habe mir gestern den Finger verletzt.",
        "Sie nimmt eine Tablette gegen die Kopfschmerzen.",
        "Er hustet seit zwei Tagen sehr stark.",
        "Wir werden nächste Woche zum Arzt gehen.",
        "Ich kann heute wegen der Grippe nicht kommen.",
        "Sie hat Fieber und bleibt im Bett.",
        "Er hat den Termin leider verschoben.",
        "Wir warten schon zwanzig Minuten beim Arzt.",
        "Ich lerne Deutsch, weil ich in Berlin wohne.",
        "Sie versteht die Aufgabe nicht ganz.",
        "Der Lehrer erklärt die Regel noch einmal.",
        "Wir schreiben morgen eine kleine Prüfung.",
        "Er hat die Hausaufgaben schon fertig gemacht.",
        "Ich frage meinen Freund nach der Lösung.",
        "Sie möchte nächstes Jahr studieren.",
    ],
    # 076–100: Travel, weather & mixed grammar
    [
        "Wir fahren im Sommer mit dem Auto ans Meer.",
        "Der Zug kommt heute leider zu spät.",
        "Ich habe meinen Koffer am Bahnhof vergessen.",
        "Sie bucht ein Zimmer im kleinen Hotel.",
        "Er fragt nach dem Weg zum Flughafen.",
        "Wir sind gestern Abend in München angekommen.",
        "Ich brauche eine Fahrkarte für die S-Bahn.",
        "Das Flugzeug startet um halb zehn.",
        "Sie wartet auf ihren Flug nach Wien.",
        "Heute regnet es, deshalb nehmen wir Schirme.",
        "Im Winter ist es hier oft sehr kalt.",
        "Morgen wird das Wetter wieder schöner.",
        "Weil es windig ist, bleiben wir drinnen.",
        "Ich ziehe mir die warme Jacke an.",
        "Er ist schneller als seine kleine Schwester.",
        "Sie singt so gut wie ihre große Schwester.",
        "Das ist das interessanteste Museum der Stadt.",
        "Wenn du Zeit hast, besuchen wir den Park.",
        "Ich rufe dich heute Abend an.",
        "Er gibt das Paket an der Post ab.",
        "Wir treffen uns um acht vor dem Café.",
        "Sie freut sich auf das Fest am Samstag.",
        "Er hat vergessen, seine Eltern anzurufen.",
        "Ich möchte lernen, wie man Kuchen backt.",
        "Obwohl es spät ist, gehen wir noch spazieren.",
    ],
]

sentences = [s for batch in BATCHES for s in batch]
assert len(sentences) == 85, f"Expected 85 sentences, got {len(sentences)}"

import stanza

print("Loading Stanza...")
nlp = stanza.Pipeline(lang="de", processors="tokenize,mwt,pos,lemma", use_gpu=False)

output_lines: list[str] = []
start_id = 16

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
    ("schlafen", "ein"): "einschlafen",
    ("wacht", "auf"): "aufwachen",
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
    "kalt": ("kalt", "ADJ"),
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
    "deshalb": ("deshalb", "ADV"),
    "normalerweise": ("normalerweise", "ADV"),
    "drinnen": ("drinnen", "ADV"),
    "draußen": ("draußen", "ADV"),
    "man": ("man", "PRON"),
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
    "räume": "aufräumen",
    "frühstücken": "frühstücken",
    "putzt": "putzen",
    "koche": "kochen",
    "gewaschen": "waschen",
    "schlafen": "schlafen",
    "wacht": "aufwachen",
    "föhne": "föhnen",
    "schläft": "schlafen",
    "gestellt": "stellen",
    "spazieren": "spazieren",
    "gegangen": "gehen",
    "fernsehen": "fernsehen",
    "hilft": "helfen",
    "kostet": "kosten",
    "sucht": "suchen",
    "bezahlt": "bezahlen",
    "öffnet": "öffnen",
    "anprobieren": "anprobieren",
    "hinaufgegangen": "hinaufgehen",
    "verloren": "verlieren",
    "günstiger": "günstig",
    "vergessen": "vergessen",
    "ausgegeben": "ausgeben",
    "nehme": "nehmen",
    "fühlt": "fühlen",
    "bleibt": "bleiben",
    "kaufen": "kaufen",
    "verletzt": "verletzen",
    "hustet": "husten",
    "kommen": "kommen",
    "verschoben": "verschieben",
    "lerne": "lernen",
    "versteht": "verstehen",
    "schreiben": "schreiben",
    "fertig": "fertig",
    "gemacht": "machen",
    "frage": "fragen",
    "studiieren": "studieren",
    "kommt": "kommen",
    "vergessen": "vergessen",
    "bucht": "buchen",
    "angekommen": "ankommen",
    "startet": "starten",
    "regnet": "regnen",
    "nehmen": "nehmen",
    "wird": "werden",
    "schöner": "schön",
    "schneller": "schnell",
    "singt": "singen",
    "interessanteste": "interessant",
    "besuchen": "besuchen",
    "backt": "backen",
    "anzurufen": "anrufen",
    "wohne": "wohnen",
    "soll": "sollen",
    "abgewaschen": "abwaschen",
    "abgegeben": "abgeben",
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
    "Samstag": "Samstag",
    "Sonntag": "Sonntag",
    "Montag": "Montag",
    "Freitag": "Freitag",
    "Dienstag": "Dienstag",
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
    "Küche": "Küche",
    "Wäsche": "Wäsche",
    "Wohnzimmer": "Wohnzimmer",
    "Sofa": "Sofa",
    "Ecke": "Ecke",
    "Bluse": "Bluse",
    "Kreditkarte": "Kreditkarte",
    "Geschäft": "Geschäft",
    "Kaufhaus": "Kaufhaus",
    "Treppe": "Treppe",
    "Pullover": "Pullover",
    "Schlüssel": "Schlüssel",
    "Laden": "Laden",
    "Hemd": "Hemd",
    "Angebot": "Angebot",
    "Eingang": "Eingang",
    "Tasche": "Tasche",
    "Kasse": "Kasse",
    "Wochenmarkt": "Wochenmarkt",
    "Gemüse": "Gemüse",
    "Stand": "Stand",
    "Rock": "Rock",
    "Verkäufer": "Verkäufer",
    "Größen": "Größe",
    "Banane": "Banane",
    "Zahnarzt": "Zahnarzt",
    "Apotheke": "Apotheke",
    "Medizin": "Medizin",
    "Finger": "Finger",
    "Tablette": "Tablette",
    "Kopfschmerzen": "Kopfschmerz",
    "Grippe": "Grippe",
    "Fieber": "Fieber",
    "Aufgabe": "Aufgabe",
    "Lehrer": "Lehrer",
    "Regel": "Regel",
    "Prüfung": "Prüfung",
    "Lösung": "Lösung",
    "Meer": "Meer",
    "Koffer": "Koffer",
    "Flugzeug": "Flugzeug",
    "Flug": "Flug",
    "Schirme": "Schirm",
    "Museum": "Museum",
    "Stadt": "Stadt",
    "Kuchen": "Kuchen",
    "Wind": "Wind",
}


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    if form.lower() in CONTRACTIONS:
        lemma, upos = CONTRACTIONS[form.lower()]
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
        if form.lower() in MODALS or lemma.lower() in MODALS:
            upos = "AUX"
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

        if lower == "wie" and i + 1 < len(aligned):
            next_form = aligned[i + 1][0].lower()
            if next_form == "man":
                upos = "ADV"
                lemma = "wie"
            elif i > 0:
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
    key = form.lower()
    if key not in CONTRACTION_EXPANSIONS:
        return None
    expansion = CONTRACTION_EXPANSIONS[key]
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
            if wi < len(words) and words[wi].text == form:
                wi += 1
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
    sent_id = f"de_a2_val_{start_id + idx:03d}"
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
target_path = project_root / "data/handcraft/de/val/a2_new_001.conllu"
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

print("STATUS: OK — 85 sentences, de_a2_val_016–de_a2_val_100")