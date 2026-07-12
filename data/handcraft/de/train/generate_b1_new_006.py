import os
import re
import sys
from pathlib import Path

# Add project root to path so we can import validator and checker
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# Define exactly 200 B1 German sentences
sentences = [
    # 305 - 324: Work and Career
    "Ich möchte mich für die Stelle als Verkäufer bewerben.",
    "Während meiner Ausbildung habe ich viel gelernt.",
    "Der Chef war mit meiner Arbeit sehr zufrieden.",
    "Könnten Sie mir bitte die Adresse der Firma schicken?",
    "Es ist wichtig, im Beruf immer pünktlich zu sein.",
    "Ich suche eine Arbeit, die mir Spaß macht.",
    "Mein Kollege hat mir bei dem Projekt geholfen.",
    "Die Besprechung fängt morgen um neun Uhr an.",
    "Ich habe beschlossen, eine Fortbildung zu machen.",
    "Wir müssen die Aufgaben gerecht im Team verteilen.",
    "Er hat seine Stelle gekündigt, weil er umzieht.",
    "Können wir den Termin auf nächsten Montag verschieben?",
    "Die Arbeitsbedingungen in diesem Betrieb sind sehr gut.",
    "Ich habe eine Zusage für das Praktikum bekommen.",
    "Sie arbeitet als Krankenschwester im Krankenhaus.",
    "Mein Ziel ist es, mich beruflich zu verbessern.",
    "Wir haben gestern über das neue Projekt gesprochen.",
    "Können Sie mir das Dokument bitte ausdrucken?",
    "Ich habe morgen ein wichtiges Vorstellungsgespräch.",
    "Er hat viel Erfahrung im Bereich Marketing gesammelt.",

    # 325 - 344: Travel and Transport
    "Wir haben unseren Urlaub am Meer verbracht.",
    "Wegen des schlechten Wetters hatte der Zug Verspätung.",
    "Ich würde gerne ein Auto für das Wochenende mieten.",
    "Vergiss nicht, deinen Reisepass mitzunehmen.",
    "Die Flugtickets waren überraschend günstig.",
    "Wir haben uns während der Reise verfahren.",
    "Gibt es in der Nähe einen Bahnhof?",
    "Ich freue mich darauf, eine neue Stadt zu entdecken.",
    "Wir mussten den Flug wegen eines Streiks stornieren.",
    "Die Fahrt mit dem Zug war sehr entspannend.",
    "Ich habe ein Zimmer im Hotel reserviert.",
    "Wo kann ich Fahrkarten für die U-Bahn kaufen?",
    "Der Ausflug in die Berge hat uns viel Spaß gemacht.",
    "Wir sind gestern spät am Flughafen angekommen.",
    "Es gibt eine direkte Verbindung nach Berlin.",
    "Die Landschaft in dieser Region ist wunderschön.",
    "Ich muss mein Fahrrad reparieren lassen.",
    "Wir haben eine Stadtführung durch die Altstadt gemacht.",
    "Bitte schnallen Sie sich während der Fahrt an.",
    "Die Autobahn war wegen eines Unfalls gesperrt.",

    # 345 - 364: Housing and Living
    "Wir suchen eine gemütliche Wohnung im Stadtzentrum.",
    "Die Miete für das Zimmer ist leider sehr hoch.",
    "Wir müssen am Wochenende die Möbel aufbauen.",
    "Unsere Nachbarn sind sehr nett und hilfsbereit.",
    "Die Heizung funktioniert seit gestern nicht mehr.",
    "Ich möchte mein Zimmer neu streichen.",
    "Wir haben uns eine neue Waschmaschine gekauft.",
    "Der Mietvertrag muss noch unterschrieben werden.",
    "Unsere Wohnung hat einen großen Balkon zum Garten.",
    "Ich fühle mich in meinem neuen Zuhause sehr wohl.",
    "Wir müssen den Müll trennen und rausbringen.",
    "Das Wohnzimmer ist sehr hell und freundlich.",
    "Können Sie mir beim Umzug am Samstag helfen?",
    "Die Nebenkosten sind in der Miete bereits enthalten.",
    "Ich habe den Schlüssel in der Wohnung vergessen.",
    "Wir wohnen in einer ruhigen Gegend am Stadtrand.",
    "Der Vermieter hat die Miete erhöht.",
    "Ich räume heute meine Küche auf.",
    "Der Aufzug im Haus ist leider kaputt.",
    "Wir haben gestern den Keller aufgeräumt.",

    # 365 - 384: Education and Learning
    "Ich lerne Deutsch, weil ich in Deutschland arbeiten möchte.",
    "Der Sprachkurs findet zweimal pro Woche statt.",
    "Ich muss mich auf die Prüfung vorbereiten.",
    "Unsere Lehrerin erklärt die Grammatik sehr gut.",
    "Ich habe Schwierigkeiten mit den deutschen Artikeln.",
    "Er hat die Prüfung mit einer guten Note bestanden.",
    "Wir haben in der Schule ein interessantes Thema besprochen.",
    "Es ist wichtig, jeden Tag Vokabeln zu wiederholen.",
    "Ich möchte mich an der Universität für Medizin einschreiben.",
    "Meine Tochter geht jetzt in die Grundschule.",
    "Wir müssen ein Referat über Umweltschutz halten.",
    "Ich habe sich für einen Online-Kurs angemeldet.",
    "Das Wörterbuch hilft mir beim Übersetzen.",
    "Er hat sein Studium nach fünf Jahren abgeschlossen.",
    "Wir machen morgen einen Ausflug mit der Klasse.",
    "Ich leihe mir oft Bücher aus der Bibliothek aus.",
    "Die Hausaufgaben waren heute besonders schwierig.",
    "Ich möchte eine neue Sprache lernen.",
    "Der Unterricht beginnt pünktlich um acht Uhr.",
    "Sie hat ein Stipendium für ein Auslandsjahr erhalten.",

    # 385 - 404: Health and Body
    "Ich fühle mich heute nicht besonders gut.",
    "Er muss wegen einer Erkältung im Bett bleiben.",
    "Ich habe einen Termin beim Arzt vereinbart.",
    "Gesunde Ernährung ist wichtig für den Körper.",
    "Ich treibe Sport, um fit zu bleiben.",
    "Der Arzt hat mir ein Medikament verschrieben.",
    "Ich habe seit drei Tagen starke Kopfschmerzen.",
    "Sie müssen diese Tabletten vor dem Essen einnehmen.",
    "Ich hoffe, dass du schnell wieder gesund wirst.",
    "Nach dem Unfall musste er ins Krankenhaus gebracht werden.",
    "Ich trinke viel Wasser, um gesund zu bleiben.",
    "Der Zahnarzt hat meine Zähne kontrolliert.",
    "Ich habe mir beim Fußballspielen den Fuß verletzt.",
    "Gegen den Husten trinke ich heißen Tee mit Honig.",
    "Er treibt jeden Morgen Sport im Park.",
    "Ich muss weniger Stress haben und mich entspannen.",
    "Die Untersuchung hat nicht lange gedauert.",
    "Ich habe einen Termin für eine Massage ausgemacht.",
    "Man sollte sich ausreichend an der frischen Luft bewegen.",
    "Der Patient muss sich nach der Operation ausruhen.",

    # 405 - 424: Media and Technology
    "Ich benutze mein Smartphone für fast alles.",
    "Hast du die neuesten Nachrichten im Radio gehört?",
    "Ich schaue mir gerne Dokumentationen im Fernsehen an.",
    "Wir haben eine E-Mail an den Kundenservice geschickt.",
    "Das Internet ist heute sehr langsam.",
    "Ich habe eine nützliche App auf meinem Handy installiert.",
    "Er verbringt zu viel Zeit in den sozialen Medien.",
    "Ich muss meinen Laptop neu starten.",
    "Wir haben den Film gestern im Kino gesehen.",
    "Kannst du mir den Link zu der Website schicken?",
    "Ich lese morgens immer die Zeitung auf meinem Tablet.",
    "Der Drucker ist nicht mit dem Computer verbunden.",
    "Ich habe mein Passwort vergessen und muss es ändern.",
    "Diese Software erleichtert die tägliche Arbeit sehr.",
    "Wir haben die Fotos auf eine Festplatte kopiert.",
    "Ich folge einem interessanten Blog über Kochen.",
    "Er hat ein neues Video auf YouTube hochgeladen.",
    "Ich höre beim Joggen gerne Podcasts.",
    "Die Batterie meines Laptops ist fast leer.",
    "Der Empfang in diesem Gebäude ist sehr schlecht.",

    # 425 - 444: Environment and Nature
    "Wir müssen mehr für den Umweltschutz tun.",
    "Im Frühling blühen die Blumen im Garten.",
    "Der Klimawandel ist eine große Herausforderung.",
    "Wir sollten Plastikmüll so gut wie möglich vermeiden.",
    "Die Luft in den Bergen ist sehr sauber.",
    "Wir haben einen langen Spaziergang im Wald gemacht.",
    "Viele Tiere sind durch die Zerstörung der Natur bedroht.",
    "Wir nutzen Solarzellen, um Strom zu erzeugen.",
    "Das Wetter war perfekt für einen Ausflug zum See.",
    "Der Fluss fließt durch das ganze Tal.",
    "Man sollte Energie sparen, indem man das Licht ausschaltet.",
    "Der Herbst bringt bunte Blätter an den Bäumen.",
    "Wir müssen das Wasser vor Verschmutzung schützen.",
    "Es hat den ganzen Tag geschneit.",
    "Wir haben im Garten einen Baum gepflanzt.",
    "Die Vögel singen morgens vor meinem Fenster.",
    "Es gibt viele Wanderwege in dieser Region.",
    "Wir sollten öffentliche Verkehrsmittel statt Autos nutzen.",
    "Der Bioladen verkauft Produkte aus der Region.",
    "Im Sommer kann man im Fluss schwimmen.",

    # 445 - 464: Shopping and Food
    "Ich gehe samstags immer auf den Wochenmarkt.",
    "Das Essen in diesem Restaurant schmeckt hervorragend.",
    "Ich muss noch Milch und Brot einkaufen.",
    "Können wir bitte die Speisekarte sehen?",
    "Ich habe einen Kuchen für den Geburtstag gebacken.",
    "Dieses Gericht ist mir etwas zu scharf.",
    "Wir haben einen Tisch für vier Personen reserviert.",
    "Ich bezahle lieber mit Karte als mit Bargeld.",
    "Das Rezept für die Suppe ist sehr einfach.",
    "Die Verkäuferin hat mich sehr freundlich beraten.",
    "Ich suche ein Geschenk für meine Mutter.",
    "Dieses Geschäft bietet viele reduzierte Produkte an.",
    "Das Gemüse ist frisch aus dem Garten.",
    "Ich möchte ein Kilo Äpfel kaufen.",
    "Wir haben zum Abendessen Fisch zubereitet.",
    "Gibt es hier ein vegetarisches Gericht?",
    "Ich trinke meinen Kaffee am liebsten schwarz.",
    "Die Schlange an der Kasse war sehr lang.",
    "Ich habe die Quittung für die Schuhe aufgehoben.",
    "Das Brot vom Bäcker nebenan schmeckt am besten.",

    # 465 - 484: Leisure and Hobbies
    "In meiner Freizeit lese ich leidenschaftlich gerne Romane.",
    "Wir spielen jeden Freitagabend zusammen Brettspiele.",
    "Ich habe mich für einen Tanzkurs angemeldet.",
    "Er fotografiert gerne Landschaften auf seinen Reisen.",
    "Wir haben das Wochenende in einem Ferienhaus verbracht.",
    "Ich gehe regelmäßig ins Fitnessstudio.",
    "Sie spielt seit ihrer Kindheit Klavier.",
    "Wir haben ein Ticket für das Konzert am Samstag gekauft.",
    "Ich treffe mich heute Abend mit meinen Freunden.",
    "Das Museum zeigt eine Ausstellung moderner Kunst.",
    "Wir gehen am liebsten im Park spazieren.",
    "Ich habe angefangen, Spanisch zu lernen.",
    "Wir schauen sonntags oft zusammen Filme.",
    "Er kocht gerne für seine ganze Familie.",
    "Ich gehe im Winter gerne in den Bergen skifahren.",
    "Wir verbringen den Abend gemütlich auf dem Sofa.",
    "Sie malt wunderschöne Bilder mit Ölfarben.",
    "Ich interessiere mich sehr für Geschichte.",
    "Wir planen eine Fahrradtour entlang des Flusses.",
    "Das Spiel gestern war unglaublich spannend.",

    # 485 - 504: Feelings, Relationships, and Future
    "Ich freue mich sehr über deinen Besuch.",
    "Wir haben uns auf der Party schnell angefreundet.",
    "Ich hoffe, dass wir in Kontakt bleiben.",
    "Er ist traurig, weil sein Hund weggelaufen ist.",
    "Ich bin stolz auf deine großartige Leistung.",
    "Wir müssen eine Lösung für das Problem finden.",
    "Ich bin mir sicher, dass alles gut gehen wird.",
    "Sie hat sich über die Nachricht sehr gewundert.",
    "Ich wünsche dir viel Erfolg bei der Prüfung.",
    "Wir haben eine wichtige Entscheidung getroffen.",
    "Er hat mir bei meiner schwierigen Aufgabe geholfen.",
    "Ich bin dankbar für deine Unterstützung.",
    "Wir haben uns vor zwei Jahren kennengelernt.",
    "Ich habe Angst vor großen Hunden.",
    "Sie hat sich sofort in die Katze verliebt.",
    "Ich bin gespannt auf das Ergebnis.",
    "Wir haben über unsere Zukunftspläne gesprochen.",
    "Er hat sich bei mir für den Fehler entschuldigt.",
    "Ich wünsche mir mehr Freizeit für meine Hobbys.",
    "Wir freuen uns auf ein baldiges Wiedersehen."
]

assert len(sentences) == 200, f"Expected 200 sentences, got {len(sentences)}"

# Stanza integration to parse sentences and build initial tokens
import stanza

print("Loading Stanza...")
nlp = stanza.Pipeline(lang='de', processors='tokenize,mwt,pos,lemma', use_gpu=False)

output_lines = []
start_id = 305

# Helper lists for modal verbs, aux verbs, etc.
MODALS = {"müssen", "können", "wollen", "sollen", "dürfen", "mögen", "möchte"}
AUX_LEMMAS = {"sein", "haben", "werden"}

# Known manual corrections/overrides to match handcrafted dataset's specific choices
SPECIAL_LEMMAS = {
    # word -> (lemma, upos)
    "mich": ("ich", "PRON"),
    "dich": ("du", "PRON"),
    "sich": ("sich", "PRON"),
    "uns": ("wir", "PRON"),
    "euch": ("ihr", "PRON"),
    "mir": ("ich", "PRON"),
    "dir": ("du", "PRON"),
    "ihm": ("er", "PRON"),
    "ihr": ("ihr", "DET"),  # default, checked in context below
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
    "euer": ("euer", "DET"),
    "euere": ("euer", "DET"),
    "eueren": ("euer", "DET"),
    "euerem": ("euer", "DET"),
    "euerer": ("euer", "DET"),
    "eueres": ("euer", "DET"),
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
    "alles": ("alle", "DET"),
    "alle": ("alle", "DET"),
    "viele": ("viel", "DET"),
    "vielen": ("viel", "DET"),
    "viel": ("viel", "DET"),
    "mehr": ("mehr", "DET"),
    "neue": ("neu", "ADJ"),
    "neuen": ("neu", "ADJ"),
    "neues": ("neu", "ADJ"),
    "neuer": ("neu", "ADJ"),
    "neuem": ("neu", "ADJ"),
    "schlechten": ("schlecht", "ADJ"),
    "schlechtes": ("schlecht", "ADJ"),
    "wichtiges": ("wichtig", "ADJ"),
    "wichtigen": ("wichtig", "ADJ"),
    "wichtige": ("wichtig", "ADJ"),
    "nächsten": ("nah", "ADJ"),
    "großen": ("groß", "ADJ"),
    "große": ("groß", "ADJ"),
    "großes": ("groß", "ADJ"),
    "gemütliche": ("gemütlich", "ADJ"),
    "gemütlichen": ("gemütlich", "ADJ"),
    "gemütliches": ("gemütlich", "ADJ"),
    "netten": ("nett", "ADJ"),
    "nette": ("nett", "ADJ"),
    "nettes": ("nett", "ADJ"),
    "runden": ("rund", "ADJ"),
    "gute": ("gut", "ADJ"),
    "guten": ("gut", "ADJ"),
    "gutes": ("gut", "ADJ"),
    "gutem": ("gut", "ADJ"),
    "bessere": ("gut", "ADJ"),
    "besseren": ("gut", "ADJ"),
    "besten": ("gut", "ADJ"),
    "deutschen": ("deutsch", "ADJ"),
    "deutsche": ("deutsch", "ADJ"),
    "Online-Kurs": ("Online-Kurs", "NOUN"),
    "Smartphone": ("Smartphone", "NOUN"),
    "YouTube": ("YouTube", "PROPN"),
    "Bioladen": ("Bioladen", "NOUN"),
    "vegetarisches": ("vegetarisch", "ADJ"),
    "Freitagabend": ("Freitagabend", "NOUN"),
    "heute": ("heute", "ADV"),
    "gestern": ("gestern", "ADV"),
    "morgen": ("morgen", "ADV"),
    "damals": ("damals", "ADV"),
    "bald": ("bald", "ADV"),
    "immer": ("immer", "ADV"),
    "nie": ("nie", "ADV"),
    "oft": ("oft", "ADV"),
    "schon": ("schon", "ADV"),
    "noch": ("noch", "ADV"),
    "nur": ("nur", "ADV"),
    "sehr": ("sehr", "ADV"),
    "etwas": ("etwas", "ADV"),
    "fast": ("fast", "ADV"),
    "vielleicht": ("vielleicht", "ADV"),
    "leider": ("leider", "ADV"),
    "gern": ("gern", "ADV"),
    "gerne": ("gern", "ADV"),
    "hier": ("hier", "ADV"),
    "dort": ("dort", "ADV"),
    "da": ("da", "ADV"),
}

for idx, sent in enumerate(sentences):
    sent_id = f"de_b1_train_{start_id + idx}"
    doc = nlp(sent)
    
    output_lines.append(f"# sent_id = {sent_id}")
    output_lines.append(f"# text = {sent}")
    
    token_counter = 1
    for stanza_sent in doc.sentences:
        for word in stanza_sent.words:
            # Handle multi-word tokens if any (Stanza lists them, we only want the word level ones)
            if not isinstance(word.id, int):
                continue
                
            form = word.text
            upos = word.upos
            lemma = word.lemma if word.lemma else form
            
            # Normalize according to exact handcraft guidelines:
            # 1. SPECIAL_LEMMAS override
            if form in SPECIAL_LEMMAS:
                lemma, upos = SPECIAL_LEMMAS[form]
            
            # Special manual case for "das" as relative pronoun vs determiner:
            if form.lower() == "das" and upos == "PRON":
                lemma = "der"
            if form.lower() == "die" and upos == "PRON":
                lemma = "der"
            if form.lower() == "der" and upos == "PRON":
                lemma = "der"
                
            # 2. NOUN: nominative singular capitalized
            if upos == "NOUN":
                # Ensure capitalized
                lemma = lemma[0].upper() + lemma[1:]
                
            # 3. VERB: infinitive lowercase, MODALS -> VERB
            if upos == "VERB" or lemma in MODALS or form.lower() in MODALS:
                upos = "VERB"
                lemma = lemma.lower()
                # Stanza sometimes gives inflected form as lemma or capitalizes, ensure infinitive.
                # Common verbs cleanup:
                if lemma == "bewerbe" or lemma == "beworben": lemma = "bewerben"
                if lemma == "gelernt" or lemma == "lerne": lemma = "lernen"
                if lemma == "schicken" or lemma == "schicke": lemma = "schicken"
                if lemma == "suche": lemma = "suchen"
                if lemma == "macht" or lemma == "gemacht": lemma = "machen"
                if lemma == "geholfen" or lemma == "hilft": lemma = "helfen"
                if lemma == "fängt": lemma = "fangen"
                if lemma == "beschlossen": lemma = "beschließen"
                if lemma == "verteilen": lemma = "verteilen"
                if lemma == "gekündigt": lemma = "kündigen"
                if lemma == "umzieht": lemma = "umziehen"
                if lemma == "verschieben": lemma = "verschieben"
                if lemma == "bekommen" or lemma == "bekommt": lemma = "bekommen"
                if lemma == "arbeitet": lemma = "arbeiten"
                if lemma == "verbessern": lemma = "verbessern"
                if lemma == "gesprochen" or lemma == "spricht": lemma = "sprechen"
                if lemma == "ausdrucken": lemma = "ausdrucken"
                if lemma == "gesammelt": lemma = "sammeln"
                if lemma == "verbracht": lemma = "verbringen"
                if lemma == "mieten": lemma = "mieten"
                if lemma == "mitzunehmen": lemma = "mitnehmen"
                if lemma == "verfahren": lemma = "verfahren"
                # etc, but let's automate it: if form is infinitive-like or participle
                # we map common irregular verbs:
                irr = {
                    "gelungen": "gelingen",
                    "gegangen": "gehen",
                    "gefahren": "fahren",
                    "geblieben": "bleiben",
                    "stornieren": "stornieren",
                    "storniert": "stornieren",
                    "reserviert": "reservieren",
                    "kaufen": "kaufen",
                    "angekommen": "ankommen",
                    "reparieren": "reparieren",
                    "gesperrt": "sperren",
                    "aufbauen": "aufbauen",
                    "streichen": "streichen",
                    "gekauft": "kaufen",
                    "unterschrieben": "unterschreiben",
                    "fühle": "fühlen",
                    "trennen": "trennen",
                    "rausbringen": "rausbringen",
                    "helfen": "helfen",
                    "enthalten": "enthalten",
                    "vergessen": "vergessen",
                    "wohnen": "wohnen",
                    "erhöht": "erhöhen",
                    "räume": "räumen",
                    "aufgeräumt": "aufräumen",
                    "vorbereiten": "vorbereiten",
                    "erklärt": "erklären",
                    "bestanden": "bestehen",
                    "besprochen": "besprechen",
                    "wiederholen": "wiederholen",
                    "einschreiben": "einschreiben",
                    "geht": "gehen",
                    "halten": "halten",
                    "angemeldet": "anmelden",
                    "abgeschlossen": "abschließen",
                    "leihe": "leihen",
                    "erhalten": "erhalten",
                    "bleiben": "bleiben",
                    "vereinbart": "vereinbaren",
                    "treibe": "treiben",
                    "verschrieben": "verschreiben",
                    "einnehmen": "einnehmen",
                    "wirst": "werden", # aux check will catch
                    "gebracht": "bringen",
                    "trinke": "trinken",
                    "kontrolliert": "kontrollieren",
                    "verletzt": "verletzen",
                    "entspannen": "entspannen",
                    "gedauert": "dauern",
                    "ausgemacht": "ausmachen",
                    "bewegen": "bewegen",
                    "ausruhen": "ausruhen",
                    "benutze": "benutzen",
                    "gehört": "hören",
                    "schaue": "schauen",
                    "geschickt": "schicken",
                    "installiert": "installieren",
                    "verbringt": "verbringen",
                    "starten": "starten",
                    "gesehen": "sehen",
                    "lese": "lesen",
                    "verbunden": "verbinden",
                    "ändern": "ändern",
                    "erleichtert": "erleichtern",
                    "kopiert": "kopieren",
                    "folge": "folgen",
                    "hochgeladen": "hochladen",
                    "höre": "hören",
                    "tun": "tun",
                    "blühen": "blühen",
                    "vermeiden": "vermeiden",
                    "erzeugen": "erzeugen",
                    "fließt": "fließen",
                    "sparen": "sparen",
                    "ausschaltet": "ausschalten",
                    "bringt": "bringen",
                    "schützen": "schützen",
                    "geschneit": "schneien",
                    "gepflanzt": "pflanzen",
                    "singen": "singen",
                    "nutzen": "nutzen",
                    "schwimmen": "schwimmen",
                    "einkaufen": "einkaufen",
                    "sehen": "sehen",
                    "gebacken": "backen",
                    "schmeckt": "schmecken",
                    "bezahle": "bezahlen",
                    "beraten": "beraten",
                    "bietet": "bieten",
                    "zubereitet": "zubereiten",
                    "aufgehoben": "aufheben",
                    "spielen": "spielen",
                    "fotografiert": "fotografieren",
                    "gehe": "gehen",
                    "treffe": "treffen",
                    "zeigt": "zeigen",
                    "angefangen": "anfangen",
                    "schauen": "schauen",
                    "kocht": "kochen",
                    "planen": "planen",
                    "freue": "freuen",
                    "angefreundet": "anfreunden",
                    "hoffe": "hoffen",
                    "gewundert": "wundern",
                    "wünsche": "wünschen",
                    "getroffen": "treffen",
                    "kennengelernt": "kennenlernen",
                    "verliebt": "verlieben",
                    "entschuldigt": "entschuldigen",
                    "freuen": "freuen",
                    "wiederzusehen": "wiedersehen",
                    "mitzunehmen": "mitnehmen",
                }
                if form in irr:
                    lemma = irr[form]
                elif lemma in irr:
                    lemma = irr[lemma]
                elif lemma.endswith("t") and not lemma.endswith("et"):
                    # likely a past participle or inflected form
                    pass
            
            # 4. AUX: sein/haben/werden (modals are VERB)
            if form.lower() in {"ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen"}:
                upos = "AUX"
                lemma = "sein"
            elif form.lower() in {"habe", "hat", "hatte", "hätte", "hätten", "haben", "habt", "gehabt"}:
                upos = "AUX"
                lemma = "haben"
            elif form.lower() in {"wird", "wurde", "werden", "würde", "würden", "geworden"}:
                upos = "AUX"
                lemma = "werden"
            elif upos == "AUX" and lemma not in AUX_LEMMAS:
                upos = "VERB" # modal or other verb misclassified by stanza
                
            # 5. ADJ: base lowercase
            if upos == "ADJ":
                lemma = lemma.lower()
                
            # 6. PROPN: capitalized
            if upos == "PROPN":
                lemma = lemma[0].upper() + lemma[1:]
                
            # 7. PUNCT: mark itself
            if upos == "PUNCT":
                lemma = form

            # Specific rule adjustments for pronouns/determiners/conjunctions
            if form.lower() in {"weil", "dass", "obwohl", "wenn", "ob", "da", "als", "wie", "während", "seit", "bis"}:
                if form.lower() in {"während", "seit", "bis"} and upos == "ADP":
                    pass # Keep ADP
                else:
                    upos = "SCONJ"
                    lemma = form.lower()
            elif form.lower() in {"und", "oder", "aber", "sondern"}:
                upos = "CCONJ"
                lemma = form.lower()
            elif form.lower() in {"um", "zu", "für", "mit", "von", "bei", "nach", "aus", "über", "vor", "unter", "durch", "ohne", "gegen", "wegen", "in", "an", "auf"}:
                if form.lower() == "zu" and upos == "PART":
                    pass # Keep particle for zu + infinitive
                else:
                    upos = "ADP"
                    lemma = form.lower()
            elif form.lower() in {"nicht", "ja", "nein"}:
                upos = "PART"
                lemma = form.lower()
            elif form.lower() in {"ich", "du", "er", "sie", "es", "wir", "ihr", "sie", "Sie"}:
                upos = "PRON"
                lemma = "Sie" if form == "Sie" or form == "Ihnen" else form.lower()

            cols = [
                str(token_counter),  # ID
                form,                # FORM
                lemma,               # LEMMA
                upos,                # UPOS
                "_",                 # XPOS
                "_",                 # FEATS
                "_",                 # HEAD
                "_",                 # DEPREL
                "_",                 # DEPS
                "_"                  # MISC
            ]
            output_lines.append("\t".join(cols))
            token_counter += 1
            
    output_lines.append("") # Blank line between sentences

# Final blank line
output_lines.append("")

conllu_text = "\n".join(output_lines)

# Write to file
target_path = project_root / "data/handcraft/de/train/b1_new_006.conllu"
target_path.parent.mkdir(parents=True, exist_ok=True)
target_path.write_text(conllu_text, encoding="utf-8")
print(f"Wrote to {target_path}")

# Run project's internal validation on the text
validation_res = validate_text(conllu_text)
print("Validate result passed:", validation_res.passed)
if not validation_res.passed:
    print("Errors:")
    for err in validation_res.errors:
        print(err)
    sys.exit(1)

lemma_res = check_text(conllu_text, lang="de")
print("Lemma check passed:", lemma_res.passed)
if not lemma_res.passed:
    print("Lemma errors:")
    for err in lemma_res.errors:
        print(err)
    sys.exit(1)
