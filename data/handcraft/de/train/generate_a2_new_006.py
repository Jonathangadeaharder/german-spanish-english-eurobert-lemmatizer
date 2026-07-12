"""Generate a2_new_006.conllu — de_a2_train_576 through de_a2_train_775."""

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 8 batches × 25 sentences = 200
BATCHES = [
    # 576–600: Restaurant
    [
        "Ich möchte bitte die Speisekarte sehen.",
        "Wir haben gestern im Restaurant Abendessen gegessen.",
        "Kann ich heute einen Tisch für zwei reservieren?",
        "Das Essen hier schmeckt sehr gut.",
        "Er bestellt ein Glas Wasser für sich.",
        "Sie trinkt Kaffee nach dem warmen Essen.",
        "Wir bezahlen die Rechnung mit Bargeld.",
        "Der Kellner bringt uns frisches Brot.",
        "Ich esse gern Pasta mit Tomatensauce.",
        "Möchten Sie etwas zum Nachtisch bestellen?",
        "Wir haben Pizza und Salat bestellt.",
        "Das Restaurant ist heute leider geschlossen.",
        "Ich nehme Suppe und ein Stück Kuchen.",
        "Sie wartet auf ihre heiße Suppe.",
        "Er findet das Gericht etwas zu scharf.",
        "Wir sitzen am Fenster im Restaurant.",
        "Ich habe Hunger und suche ein Café.",
        "Sie trinkt Tee statt Kaffee heute.",
        "Der Koch bereitet frischen Fisch zu.",
        "Wir haben das Menü schon gelesen.",
        "Kannst du mir das Salz geben?",
        "Er isst sein Brötchen mit Butter.",
        "Sie möchte ein vegetarisches Gericht essen.",
        "Wir werden morgen im Café frühstücken.",
        "Ich habe das Essen schon bezahlt.",
    ],
    # 601–625: Doctor / Health
    [
        "Ich habe einen Termin beim Arzt um zehn Uhr.",
        "Er fühlt sich heute nicht besonders gut.",
        "Sie hat seit gestern starke Kopfschmerzen.",
        "Wir müssen zur Apotheke gehen.",
        "Der Arzt untersucht meinen Arm.",
        "Ich habe Fieber und muss im Bett bleiben.",
        "Sie nimmt die Tabletten vor dem Essen.",
        "Er hat sich beim Sport den Fuß verletzt.",
        "Wir haben einen Termin beim Zahnarzt.",
        "Ich huste seit drei Tagen sehr stark.",
        "Sie trinkt heißen Tee gegen den Husten.",
        "Der Arzt hat mir Medizin verschrieben.",
        "Er ist morgen wieder gesund und fit.",
        "Wir müssen länger auf den Arzt warten.",
        "Ich habe Schmerzen im rechten Bein.",
        "Sie geht zum Arzt wegen ihrer Erkältung.",
        "Er hat den Termin leider vergessen.",
        "Wir werden nächste Woche zum Arzt gehen.",
        "Ich fühle mich nach dem Schlaf besser.",
        "Sie kann heute nicht zur Arbeit gehen.",
        "Der Arzt sagt, ich soll mich ausruhen.",
        "Ich habe mir die Hand beim Kochen verbrannt.",
        "Sie braucht ein neues Medikament vom Arzt.",
        "Er hat gestern im Krankenhaus gearbeitet.",
        "Wir können den Termin auf Freitag verschieben.",
    ],
    # 626–650: Directions / Travel
    [
        "Gehen Sie geradeaus und dann links.",
        "Die Bushaltestelle ist gleich um die Ecke.",
        "Wir fahren mit dem Zug nach Hamburg.",
        "Ich habe meinen Reisepass im Hotel vergessen.",
        "Sie sucht den Bahnhof in der Stadt.",
        "Er fragt nach dem Weg zum Flughafen.",
        "Wir sind gestern spät am Bahnhof angekommen.",
        "Ich brauche eine Fahrkarte für die U-Bahn.",
        "Sie nimmt das Taxi zum Flughafen.",
        "Der Zug fährt in zehn Minuten ab.",
        "Wir haben uns in der Stadt verlaufen.",
        "Ich kaufe Tickets am Schalter im Bahnhof.",
        "Sie wartet auf ihren Flug nach Berlin.",
        "Er zeigt mir den Weg zur nächsten Straße.",
        "Wir werden morgen früh mit dem Bus fahren.",
        "Ich habe mein Gepäck am Flughafen abgegeben.",
        "Sie findet das Hotel nicht ohne Karte.",
        "Er fährt mit dem Fahrrad zum Markt.",
        "Wir müssen an der nächsten Ecke rechts gehen.",
        "Ich habe eine Karte für die Stadtführung.",
        "Sie reist gern mit dem Zug durch Deutschland.",
        "Er hat den Bus verpasst und wartet.",
        "Wir sind mit dem Auto nach Wien gefahren.",
        "Ich frage den Mann nach der Adresse.",
        "Sie wird nächsten Monat nach Spanien fliegen.",
    ],
    # 651–675: Appointments
    [
        "Ich habe morgen um drei Uhr einen Termin.",
        "Wir treffen uns um acht Uhr vor dem Café.",
        "Sie hat ihren Termin beim Friseur verschoben.",
        "Er wartet seit zwanzig Minuten im Wartezimmer.",
        "Wir müssen den Termin leider absagen.",
        "Ich rufe an, um einen Termin zu machen.",
        "Sie kommt pünktlich zu unserem Treffen.",
        "Er hat den Termin im Kalender notiert.",
        "Wir haben uns für Montag verabredet.",
        "Ich kann den Termin am Donnerstag nicht.",
        "Sie bestätigt den Termin per E-Mail.",
        "Er ist zu spät zum wichtigen Termin gekommen.",
        "Wir werden uns am Samstag im Park treffen.",
        "Ich habe einen Termin bei der Bank.",
        "Sie sucht einen freien Termin beim Arzt.",
        "Er hat den Termin gestern abgesagt.",
        "Wir müssen früher zum Termin gehen.",
        "Ich schreibe den Termin in mein Handy.",
        "Sie erinnert mich an unseren Termin morgen.",
        "Er hat einen Termin beim Optiker.",
        "Wir treffen uns vor dem großen Kino.",
        "Ich kann den Termin auf nächste Woche legen.",
        "Sie hat den Termin online gebucht.",
        "Er vergisst oft seine Termine am Morgen.",
        "Wir haben heute keinen freien Termin mehr.",
    ],
    # 676–700: Describing people
    [
        "Mein Bruder ist groß und sehr sportlich.",
        "Sie hat lange braune Haare und blaue Augen.",
        "Er ist freundlich und hilft gern anderen.",
        "Wir haben einen netten Nachbarn im Haus.",
        "Ich finde meine Lehrerin sehr intelligent.",
        "Sie trägt heute ein rotes Kleid.",
        "Er sieht müde aus nach der langen Reise.",
        "Wir kennen einen lustigen Mann aus Italien.",
        "Ich beschreibe meinen Freund als sehr ruhig.",
        "Sie ist jung und lernt schnell Deutsch.",
        "Er hat einen dicken Hund und zwei Katzen.",
        "Wir finden unsere neue Kollegin sehr nett.",
        "Ich habe einen kleinen Bruder und eine Schwester.",
        "Sie sieht ihrer Mutter sehr ähnlich aus.",
        "Er ist immer pünktlich und sehr fleißig.",
        "Wir treffen eine alte Frau im Park.",
        "Ich finde den Mann im blauen Hemd nett.",
        "Sie hat ein freundliches Lächeln und lacht gern.",
        "Er ist nicht groß, aber sehr stark.",
        "Wir beschreiben den Mann der Polizei.",
        "Ich habe einen neuen Freund aus der Schule.",
        "Sie trägt eine Brille und einen Hut.",
        "Er wirkt heute sehr nervös und unruhig.",
        "Wir kennen eine Frau mit kurzen Haaren.",
        "Ich finde meine Tante sehr elegant und schön.",
    ],
    # 701–725: Describing places
    [
        "Das Hotel liegt direkt am schönen Strand.",
        "Wir wohnen in einer ruhigen Straße.",
        "Der Park ist groß und sehr grün.",
        "Ich finde die Stadt modern und interessant.",
        "Sie zeigt mir ihr kleines aber gemütliches Zimmer.",
        "Er beschreibt das Restaurant als sehr teuer.",
        "Wir haben ein helles Wohnzimmer mit Balkon.",
        "Das Museum ist alt aber sehr schön.",
        "Ich suche ein günstiges Hotel in der Stadt.",
        "Sie wohnt in einem hohen Haus im Zentrum.",
        "Er findet die Gegend hier sehr laut.",
        "Wir machen Urlaub in einem kleinen Dorf.",
        "Das Büro ist neu und sehr hell.",
        "Ich mag den Markt mit vielen Läden.",
        "Sie findet den Bahnhof groß und unübersichtlich.",
        "Er wohnt nah am Fluss und am Wald.",
        "Wir haben eine Wohnung mit zwei Zimmern.",
        "Das Café ist klein aber sehr beliebt.",
        "Ich finde die Altstadt wunderschön und ruhig.",
        "Sie beschreibt den Garten als groß und grün.",
        "Er sucht eine Wohnung in einer sicheren Gegend.",
        "Wir sitzen auf einer Terrasse mit Meerblick.",
        "Das Schwimmbad ist neu und sehr sauber.",
        "Ich finde das Zimmer zu klein und dunkel.",
        "Sie wohnt gern in der Nähe vom Supermarkt.",
    ],
    # 726–750: Work basics / Hobbies
    [
        "Ich arbeite als Verkäufer in einem Laden.",
        "Er hat gestern bis spät im Büro gearbeitet.",
        "Sie möchte im Sommer als Kellnerin arbeiten.",
        "Wir haben heute eine wichtige Besprechung im Büro.",
        "Ich muss morgen früh zur Arbeit gehen.",
        "Er liest in seiner Freizeit gern Bücher.",
        "Sie spielt jeden Samstag Fußball im Park.",
        "Wir gehen am Abend gern ins Kino.",
        "Ich habe angefangen, Gitarre zu spielen.",
        "Er fotografiert gern Landschaften auf Reisen.",
        "Sie tanzt seit zwei Jahren im Verein.",
        "Wir machen am Wochenende gern lange Spaziergänge.",
        "Ich interessiere mich sehr für Geschichte.",
        "Er kocht am Sonntag für seine Familie.",
        "Sie hat einen neuen Tanzkurs begonnen.",
        "Wir werden am Samstag zusammen Karten spielen.",
        "Ich arbeite nur halbtags in der Bäckerei.",
        "Er malt in seiner Freizeit kleine Bilder.",
        "Sie geht dreimal pro Woche ins Fitnessstudio.",
        "Wir haben gestern ein spannendes Brettspiel gespielt.",
        "Ich möchte später als Lehrer arbeiten.",
        "Er hört beim Joggen gern leise Musik.",
        "Sie sammelt alte Briefmarken als Hobby.",
        "Wir planen eine Fahrradtour am nächsten Sonntag.",
        "Ich habe mir ein neues Buch gekauft.",
    ],
    # 751–775: Perfekt / werden / modals
    [
        "Ich habe gestern viel im Garten gearbeitet.",
        "Wir sind am Wochenende nach Köln gefahren.",
        "Sie hat das Buch in zwei Tagen gelesen.",
        "Er hat seine Hausaufgaben schon erledigt.",
        "Wir haben gestern Abend Pizza gemacht.",
        "Ich werde nächstes Jahr Deutsch lernen.",
        "Sie wird morgen früh zum Arzt gehen.",
        "Er wird im Sommer nach Italien fahren.",
        "Wir werden das Wochenende zu Hause verbringen.",
        "Ich kann heute leider nicht zum Treffen kommen.",
        "Du musst die Medizin dreimal täglich nehmen.",
        "Wir wollen am Samstag zusammen kochen.",
        "Sie soll mehr Wasser trinken und schlafen.",
        "Er darf heute nicht im Garten spielen.",
        "Ich möchte ein Ticket für den Zug kaufen.",
        "Wir haben den Film schon zweimal gesehen.",
        "Sie ist gestern zu spät zur Arbeit gekommen.",
        "Er hat mir beim Umzug sehr geholfen.",
        "Wir werden uns morgen früh am Bahnhof treffen.",
        "Ich kann dir den Weg zum Arzt zeigen.",
        "Du sollst dich nach dem Essen ausruhen.",
        "Wir haben unser Zimmer schon aufgeräumt.",
        "Sie wird nächsten Monat eine neue Arbeit suchen.",
        "Er hat den ganzen Tag auf dich gewartet.",
        "Ich möchte morgen mit dir ins Restaurant gehen.",
    ],
]

sentences = [s for batch in BATCHES for s in batch]
assert len(sentences) == 200, f"Expected 200 sentences, got {len(sentences)}"

import stanza

print("Loading Stanza...")
nlp = stanza.Pipeline(lang="de", processors="tokenize,mwt,pos,lemma", use_gpu=False)

output_lines: list[str] = []
start_id = 576

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
}

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
    "netten": ("nett", "ADJ"),
    "nette": ("nett", "ADJ"),
    "gemütlichen": ("gemütlich", "ADJ"),
    "gemütliches": ("gemütlich", "ADJ"),
    "frischen": ("frisch", "ADJ"),
    "frisches": ("frisch", "ADJ"),
    "frische": ("frisch", "ADJ"),
    "warmen": ("warm", "ADJ"),
    "heiße": ("heiß", "ADJ"),
    "heißen": ("heiß", "ADJ"),
    "scharf": ("scharf", "ADJ"),
    "vegetarisches": ("vegetarisch", "ADJ"),
    "starke": ("stark", "ADJ"),
    "rechten": ("rechts", "ADJ"),
    "bitte": ("bitte", "ADV"),
    "braune": ("braun", "ADJ"),
    "blaue": ("blau", "ADJ"),
    "rotes": ("rot", "ADJ"),
    "langen": ("lang", "ADJ"),
    "lange": ("lang", "ADJ"),
    "dicken": ("dick", "ADJ"),
    "kurzen": ("kurz", "ADJ"),
    "alte": ("alt", "ADJ"),
    "freundliches": ("freundlich", "ADJ"),
    "ruhigen": ("ruhig", "ADJ"),
    "ruhige": ("ruhig", "ADJ"),
    "schönen": ("schön", "ADJ"),
    "schön": ("schön", "ADJ"),
    "wunderschön": ("wunderschön", "ADJ"),
    "modern": ("modern", "ADJ"),
    "günstiges": ("günstig", "ADJ"),
    "hohen": ("hoch", "ADJ"),
    "hellen": ("hell", "ADJ"),
    "helles": ("hell", "ADJ"),
    "sicheren": ("sicher", "ADJ"),
    "sauber": ("sauber", "ADJ"),
    "dunkel": ("dunkel", "ADJ"),
    "spannendes": ("spannend", "ADJ"),
    "leise": ("leise", "ADJ"),
    "wichtigen": ("wichtig", "ADJ"),
    "wichtige": ("wichtig", "ADJ"),
    "freien": ("frei", "ADJ"),
    "nächsten": ("nah", "ADJ"),
    "nächste": ("nah", "ADJ"),
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
    "statt": ("statt", "ADP"),
    "dreimal": ("dreimal", "ADV"),
    "täglich": ("täglich", "ADV"),
    "halbtags": ("halbtags", "ADV"),
    "online": ("online", "ADV"),
    "nah": ("nah", "ADV"),
}

# form/lemma -> infinitive for participles and irregular forms
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
    "frühstücken": "frühstücken",
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
    "fliegen": "fliegen",
    "wird": "werden",
    "werden": "werden",
    "wirst": "werden",
    "ist": "sein",
    "sind": "sein",
    "bin": "sein",
    "bist": "sein",
    "war": "sein",
    "habe": "haben",
    "hat": "haben",
    "haben": "haben",
    "hast": "haben",
    "huste": "husten",
    "nimmt": "nehmen",
    "bringt": "bringen",
    "bestellt": "bestellen",
    "sitzt": "sitzen",
    "bereitet": "bereiten",
    "isst": "essen",
    "trägt": "tragen",
    "sieht": "sehen",
    "kennen": "kennen",
    "kennt": "kennen",
    "beschreibt": "beschreiben",
    "beschreiben": "beschreiben",
    "findet": "finden",
    "liegt": "liegen",
    "wohnt": "wohnen",
    "mag": "mögen",
    "interessiere": "interessieren",
    "sammelt": "sammeln",
    "hört": "hören",
    "malt": "malen",
    "fotografiert": "fotografieren",
    "tanzt": "tanzen",
    "kocht": "kochen",
    "backt": "backen",
    "rufe": "anrufen",
    "an": "anrufen",
    "schreibe": "schreiben",
    "erinnert": "erinnern",
    "bestätigt": "bestätigen",
    "untersucht": "untersuchen",
    "fühlt": "fühlen",
    "frage": "fragen",
    "zeigt": "zeigen",
    "sucht": "suchen",
    "wartet": "warten",
    "nimmt": "nehmen",
    "kaufe": "kaufen",
    "reist": "reisen",
    "spielt": "spielen",
    "arbeite": "arbeiten",
    "arbeitet": "arbeiten",
    "lernt": "lernen",
    "planen": "planen",
    "treffen": "treffen",
    "trifft": "treffen",
    "kommt": "kommen",
    "kommen": "kommen",
    "gibt": "geben",
    "geben": "geben",
    "legt": "legen",
    "vergisst": "vergessen",
    "wirkt": "wirken",
    "lacht": "lachen",
    "hilft": "helfen",
    "sagt": "sagen",
    "braucht": "brauchen",
    "bleiben": "bleiben",
    "bleibt": "bleiben",
    "ausruhen": "ausruhen",
    "verbringen": "verbringen",
    "reservieren": "reservieren",
    "schmeckt": "schmecken",
    "esse": "essen",
    "trinkt": "trinken",
    "bestellen": "bestellen",
    "nehme": "nehmen",
    "warte": "warten",
    "sitzen": "sitzen",
    "suche": "suchen",
    "zubereitet": "zubereiten",
}

NOUN_LEMMA_MAP = {
    "Kinder": "Kind",
    "Hände": "Hand",
    "Augen": "Auge",
    "Haare": "Haar",
    "Katzen": "Katze",
    "Zimmern": "Zimmer",
    "Läden": "Laden",
    "Bücher": "Buch",
    "Bergen": "Berg",
    "Kopfschmerzen": "Kopfschmerz",
    "Großeltern": "Großeltern",
    "Minuten": "Minute",
    "Tabletten": "Tablette",
    "Kekse": "Keks",
    "Spaziergänge": "Spaziergang",
    "Briefmarken": "Briefmarke",
    "Hausaufgaben": "Hausaufgabe",
    "Medikament": "Medikament",
    "Medizin": "Medizin",
    "Fitnessstudio": "Fitnessstudio",
    "Abendessen": "Abendessen",
    "Tomatensauce": "Tomatensauce",
    "Nachtisch": "Nachtisch",
    "Brötchen": "Brötchen",
    "Bargeld": "Bargeld",
    "Rechnung": "Rechnung",
    "Speisekarte": "Speisekarte",
    "Wartezimmer": "Wartezimmer",
    "Stadtführung": "Stadtführung",
    "Bushaltestelle": "Bushaltestelle",
    "Fahrkarte": "Fahrkarte",
    "Gepäck": "Gepäck",
    "Reisepass": "Reisepass",
    "Handy": "Handy",
    "Brettspiel": "Brettspiel",
    "Tanzkurs": "Tanzkurs",
    "Bäckerei": "Bäckerei",
    "Kellnerin": "Kellnerin",
    "Verkäufer": "Verkäufer",
    "Kollegin": "Kollegin",
    "Lehrerin": "Lehrerin",
    "Nachbarn": "Nachbar",
    "Meerblick": "Meerblick",
    "Schwimmbad": "Schwimmbad",
    "Altstadt": "Altstadt",
    "Hemd": "Hemd",
    "Lächeln": "Lächeln",
    "Hut": "Hut",
    "Brille": "Brille",
    "Husten": "Husten",
    "Erkältung": "Erkältung",
    "Apotheke": "Apotheke",
    "Zahnarzt": "Zahnarzt",
    "Optiker": "Optiker",
    "Friseur": "Friseur",
    "Kalender": "Kalender",
    "Besprechung": "Besprechung",
    "Terrasse": "Terrasse",
    "Balkon": "Balkon",
    "Wohnzimmer": "Wohnzimmer",
    "Supermarkt": "Supermarkt",
    "Umzug": "Umzug",
    "Treffen": "Treffen",
    "Termin": "Termin",
    "Termine": "Termin",
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
        "weil", "dass", "obwohl", "wenn", "ob", "als", "wie", "während", "seit", "bis",
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

    return aligned


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
target_path = project_root / "data/handcraft/de/train/a2_new_006.conllu"
target_path.write_text(conllu_text, encoding="utf-8")
print(f"Wrote {target_path}")

validation_res = validate_text(conllu_text)
print("Validate:", validation_res.passed, "sentences:", validation_res.sentence_count)
if not validation_res.passed:
    for err in validation_res.errors[:20]:
        print(" ", err)

lemma_res = check_text(conllu_text, lang="de")
print("Lemma check:", lemma_res.passed)
if not lemma_res.passed:
    for err in lemma_res.errors[:20]:
        print(" ", err)

if not validation_res.passed or not lemma_res.passed:
    sys.exit(1)