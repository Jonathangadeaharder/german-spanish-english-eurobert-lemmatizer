"""Generate a2_new_005.conllu — en_a2_train_801 through en_a2_train_900."""

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 4 batches × 25 sentences = 100 (A2: hotel/travel, health, parties, services)
BATCHES = [
    # 801–825: Hotel & travel abroad
    [
        "We check in at the hotel reception desk.",
        "I pack my suitcase the night before.",
        "She shows her passport at the airport gate.",
        "He books a room with a sea view.",
        "We leave the hotel early in the morning.",
        "I ask the hotel for an early call.",
        "She puts the towels in the bathroom.",
        "We take the lift to the fifth floor.",
        "He fills in the hotel registration form.",
        "I call room service for extra pillows.",
        "She pays the hotel bill before we leave.",
        "We swim in the hotel pool after lunch.",
        "He finds his seat on the plane quickly.",
        "I put my bag in the overhead locker.",
        "She buys perfume at the airport shop.",
        "We wait for our luggage at the carousel.",
        "He rents a car at the arrival terminal.",
        "I take photos of the old city walls.",
        "She orders coffee on the hotel balcony.",
        "We explore the narrow streets near the harbour.",
        "He changes money at the travel agency.",
        "I wear sunscreen at the sandy beach.",
        "She collects shells along the quiet shore.",
        "We watch the sunset from the hilltop.",
        "He sends a postcard to his parents.",
    ],
    # 826–850: Health & pharmacy
    [
        "I make an appointment with the doctor.",
        "She feels dizzy after the long bus ride.",
        "He has a sore throat and cough.",
        "We buy cough syrup at the pharmacy.",
        "I take one tablet after each meal.",
        "She rests in bed with a high fever.",
        "He bandages his cut finger carefully.",
        "We call an ambulance for the injured man.",
        "I avoid dairy when my stomach hurts.",
        "She brushes her teeth after every snack.",
        "He visits the dentist twice a year.",
        "We wear masks in the crowded clinic.",
        "I drink warm tea for my sore throat.",
        "She checks her temperature with a thermometer.",
        "He follows the advice from the nurse.",
        "We pick up medicine at the local pharmacy.",
        "I stretch my legs after the long flight.",
        "She wears glasses when she reads books.",
        "He puts ice on his swollen ankle.",
        "We eat soup when we feel sick.",
        "I book a blood test for next week.",
        "She complains about pain in her back.",
        "He keeps the wound clean and dry.",
        "We thank the doctor after the quick visit.",
        "I feel better after two days of rest.",
    ],
    # 851–875: Parties & social life
    [
        "We invite friends to a birthday barbecue.",
        "She wraps the gift in colourful paper.",
        "He blows out the candles on the cake.",
        "I bring flowers to the dinner party.",
        "We dance to pop music at the club.",
        "She writes thank-you notes after the wedding.",
        "He accepts the invitation with a smile.",
        "I apologise for arriving late to dinner.",
        "We toast the happy couple with champagne.",
        "She serves cake to the hungry children.",
        "He decorates the hall with balloons and ribbons.",
        "I help clear the table after dessert.",
        "We shout happy birthday to our niece.",
        "She chooses a dress for the formal party.",
        "He hands out snacks to the arriving guests.",
        "I introduce my partner to my colleagues.",
        "We plan a surprise party for Dad.",
        "She opens her presents one by one.",
        "He makes a short speech at the celebration.",
        "I share sweets with my classmates today.",
        "We stay until midnight at the New Year party.",
        "She offers tea to the visiting neighbours.",
        "He refuses another slice of chocolate cake.",
        "I enjoy chatting with people at the event.",
        "We leave quietly when the baby falls asleep.",
    ],
    # 876–900: Services & city life
    [
        "I post a parcel at the post office.",
        "She gets a haircut at the local salon.",
        "He fixes the flat tyre on his bike.",
        "We watch a comedy at the cinema tonight.",
        "I borrow three novels from the public library.",
        "She queues for tickets at the box office.",
        "He repairs the broken zip on his jacket.",
        "We renew our membership at the sports club.",
        "I report the lost wallet to the police.",
        "She deposits cash at the ATM outside.",
        "He collects dry cleaning from the shop.",
        "We register our dog at the vet clinic.",
        "I reserve a table for four on Saturday.",
        "She scans her ticket at the museum entrance.",
        "He unblocks the sink in the kitchen.",
        "We complain about noise to the landlord.",
        "I water the plants on my small balcony.",
        "She empties the mailbox every afternoon.",
        "He paints the fence before the rain comes.",
        "We sign a contract for the new flat.",
        "I pick up bread from the bakery nearby.",
        "She puts coins into the parking meter.",
        "He checks the bus timetable on his phone.",
        "We join a guided tour of the castle.",
        "I feel safe in this friendly neighbourhood.",
    ],
]

sentences = [s for batch in BATCHES for s in batch]
assert len(sentences) == 100

start_id = 801

import stanza

nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma", use_gpu=False)

# --- English handcraft lemma conventions (see en_test.conllu) ---

AUX_BE = frozenset({"am", "is", "are", "was", "were", "be", "been", "'m", "'re", "'s"})
AUX_HAVE = frozenset({"have", "has", "had", "'ve"})
AUX_DO = frozenset({"do", "does", "did"})
AUX_WILL = frozenset({"will", "would", "'ll", "'d"})
AUX_CAN = frozenset({"can", "could"})
AUX_OTHER = frozenset({"must", "should", "may", "might", "shall"})

PRON_LEMMA_MAP = {
    "me": "I",
    "us": "we",
    "him": "he",
    "her": "she",
    "them": "they",
}

REFLEXIVE_PRONOUNS = frozenset({
    "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "yourselves", "themselves",
})

NOUN_LEMMA_MAP = {
    "children": "child",
    "mice": "mouse",
    "feet": "foot",
    "teeth": "tooth",
    "men": "man",
    "women": "woman",
    "people": "person",
    "leaves": "leaf",
    "knives": "knife",
    "wives": "wife",
    "lives": "life",
    "geese": "goose",
    "oxen": "ox",
    "bacteria": "bacterium",
    "criteria": "criterion",
    "phenomena": "phenomenon",
    "data": "datum",
    "media": "medium",
    "alumni": "alumnus",
    "cacti": "cactus",
    "fungi": "fungus",
    "indices": "index",
    "matrices": "matrix",
    "vertices": "vertex",
    "analyses": "analysis",
    "theses": "thesis",
    "crises": "crisis",
    "diagnoses": "diagnosis",
    "oases": "oasis",
    "radii": "radius",
    "stimuli": "stimulus",
    "appendices": "appendix",
    "biscuits": "biscuit",
    "potatoes": "potato",
    "tomatoes": "tomato",
    "heroes": "hero",
    "photos": "photo",
    "pianos": "piano",
    "radios": "radio",
    "studios": "studio",
    "zoos": "zoo",
    "keys": "key",
    "days": "day",
    "ways": "way",
    "boys": "boy",
    "toys": "toy",
    "cars": "car",
    "bars": "bar",
    "stars": "star",
    "bags": "bag",
    "dogs": "dog",
    "cats": "cat",
    "eggs": "egg",
    "legs": "leg",
    "arms": "arm",
    "hands": "hand",
    "friends": "friend",
    "parents": "parent",
    "neighbours": "neighbour",
    "neighbors": "neighbor",
    "words": "word",
    "books": "book",
    "shops": "shop",
    "dishes": "dish",
    "questions": "question",
    "tickets": "ticket",
    "flowers": "flower",
    "hours": "hour",
    "minutes": "minute",
    "weeks": "week",
    "months": "month",
    "years": "year",
    "rooms": "room",
    "doors": "door",
    "windows": "window",
    "lights": "light",
    "plates": "plate",
    "glasses": "glass",
    "shoes": "shoe",
    "clothes": "clothes",
    "vegetables": "vegetable",
    "apples": "apple",
    "oranges": "orange",
    "cookies": "cookie",
    "sandwiches": "sandwich",
    "medicines": "medicine",
    "centimetres": "centimetre",
    "centimeters": "centimeter",
    "kilometres": "kilometre",
    "kilometers": "kilometer",
    "trainers": "trainer",
    "fireworks": "firework",
    "candles": "candle",
    "euros": "euro",
    "coins": "coin",
    "banks": "bank",
    "runners": "runner",
    "classmates": "classmate",
    "newspapers": "newspaper",
    "masks": "mask",
    "presents": "present",
    "gifts": "gift",
    "sweets": "sweet",
    "drums": "drum",
    "lanterns": "lantern",
    "dancers": "dancer",
    "sports": "sport",
    "coats": "coat",
    "hills": "hill",
    "sunglasses": "sunglass",
    "clothes": "clothes",
    "stamps": "stamp",
    "comics": "comic",
    "scarves": "scarf",
    "cookies": "cookie",
    "herbs": "herb",
    "videos": "video",
    "podcasts": "podcast",
    "puzzles": "puzzle",
    "games": "game",
    "files": "file",
    "apps": "app",
    "notifications": "notification",
    "bottles": "bottle",
    "trees": "tree",
    "signs": "sign",
    "streets": "street",
    "scooters": "scooter",
    "pictures": "picture",
    "stories": "story",
    "galleries": "gallery",
    "shows": "show",
    "songs": "song",
    "planes": "plane",
    "birds": "bird",
    "buttons": "button",
    "maps": "map",
    "marks": "mark",
    "answers": "answer",
    "notes": "note",
    "colours": "colour",
    "colors": "color",
    "members": "member",
    "countries": "country",
    "evenings": "evening",
    "days": "day",
    "updates": "update",
    "messages": "message",
    "films": "film",
    "calls": "call",
    "neighbours": "neighbour",
    "neighbors": "neighbor",
    "tools": "tool",
    "flowers": "flower",
    "leaves": "leaf",
    "trousers": "trouser",
    "jeans": "jean",
    "gloves": "glove",
    "sweaters": "sweater",
    "shirts": "shirt",
    "badges": "badge",
    "colleagues": "colleague",
    "clients": "client",
    "documents": "document",
    "emails": "email",
    "reports": "report",
    "skills": "skill",
    "results": "result",
    "butterflies": "butterfly",
    "hamsters": "hamster",
    "crickets": "cricket",
    "squirrels": "squirrel",
    "rabbits": "rabbit",
    "ducks": "duck",
    "owls": "owl",
    "bees": "bee",
    "whales": "whale",
    "frogs": "frog",
    "gutters": "gutter",
    "screws": "screw",
    "tiles": "tile",
    "fingerprints": "fingerprint",
    "nephews": "nephew",
    "kayaks": "kayak",
    "students": "student",
    "guests": "guest",
    "bins": "bin",
    "bicycles": "bicycle",
    "shelves": "shelf",
    "holes": "hole",
    "leaves": "leaf",
}

VERB_LEMMA_MAP = {
    "went": "go",
    "gone": "go",
    "got": "get",
    "gotten": "get",
    "bought": "buy",
    "brought": "bring",
    "thought": "think",
    "taught": "teach",
    "caught": "catch",
    "fought": "fight",
    "sought": "seek",
    "wrote": "write",
    "written": "write",
    "rode": "ride",
    "ridden": "ride",
    "drove": "drive",
    "driven": "drive",
    "ate": "eat",
    "eaten": "eat",
    "drank": "drink",
    "drunk": "drink",
    "sang": "sing",
    "sung": "sing",
    "ran": "run",
    "swam": "swim",
    "swum": "swim",
    "gave": "give",
    "given": "give",
    "took": "take",
    "taken": "take",
    "made": "make",
    "came": "come",
    "became": "become",
    "knew": "know",
    "known": "know",
    "saw": "see",
    "seen": "see",
    "spoke": "speak",
    "spoken": "speak",
    "broke": "break",
    "broken": "break",
    "chose": "choose",
    "chosen": "choose",
    "froze": "freeze",
    "frozen": "freeze",
    "stole": "steal",
    "stolen": "steal",
    "woke": "wake",
    "woken": "wake",
    "fell": "fall",
    "fallen": "fall",
    "forgot": "forget",
    "forgotten": "forget",
    "left": "leave",
    "lost": "lose",
    "found": "find",
    "kept": "keep",
    "slept": "sleep",
    "sent": "send",
    "spent": "spend",
    "built": "build",
    "felt": "feel",
    "heard": "hear",
    "held": "hold",
    "stood": "stand",
    "understood": "understand",
    "won": "win",
    "met": "meet",
    "read": "read",
    "led": "lead",
    "fed": "feed",
    "paid": "pay",
    "said": "say",
    "told": "tell",
    "sold": "sell",
    "began": "begin",
    "begun": "begin",
    "rang": "ring",
    "rung": "ring",
    "sank": "sink",
    "sunk": "sink",
    "shook": "shake",
    "shaken": "shake",
    "shut": "shut",
    "hit": "hit",
    "hurt": "hurt",
    "cut": "cut",
    "put": "put",
    "let": "let",
    "set": "set",
    "lit": "light",
    "had": "have",
    "did": "do",
    "was": "be",
    "were": "be",
    "rains": "rain",
    "shines": "shine",
    "covers": "cover",
    "blooms": "bloom",
    "appears": "appear",
    "drops": "drop",
    "turns": "turn",
    "revise": "revise",
    "practises": "practise",
    "practices": "practice",
    "knits": "knit",
    "sews": "sew",
    "downloads": "download",
    "uploads": "upload",
    "streams": "stream",
    "scans": "scan",
    "mutes": "mute",
    "donates": "donate",
    "recycles": "recycle",
    "volunteers": "volunteer",
    "applies": "apply",
    "irons": "iron",
    "queues": "queue",
    "offers": "offer",
    "resets": "reset",
    "deletes": "delete",
    "installs": "install",
    "blocks": "block",
    "validates": "validate",
    "adopts": "adopt",
    "complains": "complain",
    "recommends": "recommend",
    "reserves": "reserve",
    "collects": "collect",
    "solves": "solve",
    "connects": "connect",
    "logs": "log",
    "replies": "reply",
    "charges": "charge",
    "copies": "copy",
    "records": "record",
    "wraps": "wrap",
    "waters": "water",

    "plants": "plant",
    "hangs": "hang",
    "folds": "fold",
    "knocks": "knock",
    "locks": "lock",
    "votes": "vote",
    "signs": "sign",
    "rents": "rent",
    "attends": "attend",
    "fixes": "fix",
    "compares": "compare",
    "exchanges": "exchange",
    "matches": "match",
    "prefers": "prefer",
    "organises": "organise",
    "organizes": "organize",
    "discusses": "discuss",
    "arrives": "arrive",
    "feels": "feel",
    "seems": "seem",
    "protects": "protect",
    "photographs": "photograph",
    "spots": "spot",
    "respects": "respect",
    "tightens": "tighten",
    "unplugs": "unplug",
    "sands": "sand",
    "measures": "measure",
    "replaces": "replace",
    "oils": "oil",
    "scrubs": "scrub",
    "assembles": "assemble",
    "seals": "seal",
    "clears": "clear",
    "wipes": "wipe",
    "tests": "test",
    "adjusts": "adjust",
    "mops": "mop",
    "rehangs": "rehang",
    "stores": "store",
    "delivers": "deliver",
    "tutors": "tutor",
    "sorts": "sort",
    "leads": "lead",
    "greets": "greet",
    "rehearses": "rehearse",
    "boards": "board",
    "thanks": "thank",
    "babysits": "babysit",
    "hosts": "host",
    "practices": "practice",
    "practises": "practise",
    "coaches": "coach",
    "packs": "pack",
    "avoids": "avoid",
    "draws": "draw",
    "climbs": "climb",
    "drills": "drill",
    "moves": "move",
    "rang": "ring",
    "sounded": "sound",
    "started": "start",
    "arrived": "arrive",
    "entered": "enter",
    "broke": "break",
    "played": "play",
    "went": "go",
    "taken": "take",
    "jumped": "jump",
    "called": "call",
    "died": "die",
    "lived": "live",
    "finished": "finish",
    "visited": "visit",
    "tried": "try",
    "lost": "lose",
    "worked": "work",
    "known": "know",
    "eaten": "eat",
    "bought": "buy",
    "seen": "see",
    "broken": "break",
    "written": "write",
    "cleaned": "clean",
    "waited": "wait",
    "forgotten": "forget",
    "read": "read",
    "passed": "pass",
    "saved": "save",
    "cut": "cut",
    "heard": "hear",
    "moved": "move",
    "grown": "grow",
    "hurt": "hurt",
    "met": "meet",
    "pets": "pet",
    "carries": "carry",
    "feeds": "feed",
    "keeps": "keep",
}

PROPN_LEMMA_MAP = {
    "london": "London",
    "rome": "Rome",
    "spain": "Spain",
    "greece": "Greece",
    "french": "French",
    "english": "English",
    "italian": "Italian",
    "christmas": "Christmas",
    "easter": "Easter",
    "halloween": "Halloween",
    "eve": "Eve",
    "tv": "TV",
    "atm": "ATM",
    "pin": "PIN",
    "id": "ID",
    "may": "May",
    "january": "January",
    "october": "October",
    "june": "June",
    "friday": "Friday",
    "monday": "Monday",
    "sunday": "Sunday",
    "saturday": "Saturday",
    "rainbow": "Rainbow",
    "wi-fi": "Wi-Fi",
    "wifi": "Wi-Fi",
    "qr": "QR",
    "salsa": "Salsa",
    "paris": "Paris",
    "may": "May",
    "hr": "HR",
    "tv": "TV",
}

SPECIAL_UPOS = {
    "not": ("not", "PART"),
    "please": ("please", "INTJ"),
    "yes": ("yes", "INTJ"),
    "no": ("no", "INTJ"),
    "to": ("to", "PART"),
    "than": ("than", "SCONJ"),
    "as": ("as", "SCONJ"),
    "because": ("because", "SCONJ"),
    "when": ("when", "SCONJ"),
    "if": ("if", "SCONJ"),
    "while": ("while", "SCONJ"),
    "after": ("after", "ADP"),
    "before": ("before", "ADP"),
    "during": ("during", "ADP"),
    "without": ("without", "ADP"),
    "with": ("with", "ADP"),
    "for": ("for", "ADP"),
    "from": ("from", "ADP"),
    "in": ("in", "ADP"),
    "on": ("on", "ADP"),
    "at": ("at", "ADP"),
    "by": ("by", "ADP"),
    "of": ("of", "ADP"),
    "to": ("to", "ADP"),
    "into": ("into", "ADP"),
    "near": ("near", "ADP"),
    "under": ("under", "ADP"),
    "over": ("over", "ADP"),
    "about": ("about", "ADP"),
    "through": ("through", "ADP"),
    "across": ("across", "ADP"),
    "between": ("between", "ADP"),
    "against": ("against", "ADP"),
    "around": ("around", "ADP"),
    "behind": ("behind", "ADP"),
    "beside": ("beside", "ADP"),
    "inside": ("inside", "ADP"),
    "outside": ("outside", "ADP"),
    "up": ("up", "ADP"),
    "down": ("down", "ADP"),
    "off": ("off", "ADP"),
    "out": ("out", "ADP"),
    "away": ("away", "ADP"),
    "back": ("back", "ADP"),
    "along": ("along", "ADP"),
    "and": ("and", "CCONJ"),
    "or": ("or", "CCONJ"),
    "but": ("but", "CCONJ"),
    "nor": ("nor", "CCONJ"),
    "so": ("so", "CCONJ"),
    "yet": ("yet", "ADV"),
    "very": ("very", "ADV"),
    "too": ("too", "ADV"),
    "also": ("also", "ADV"),
    "always": ("always", "ADV"),
    "never": ("never", "ADV"),
    "often": ("often", "ADV"),
    "sometimes": ("sometimes", "ADV"),
    "usually": ("usually", "ADV"),
    "today": ("today", "ADV"),
    "tomorrow": ("tomorrow", "ADV"),
    "yesterday": ("yesterday", "ADV"),
    "tonight": ("tonight", "ADV"),
    "now": ("now", "ADV"),
    "here": ("here", "ADV"),
    "there": ("there", "ADV"),
    "well": ("well", "ADV"),
    "quickly": ("quickly", "ADV"),
    "slowly": ("slowly", "ADV"),
    "carefully": ("carefully", "ADV"),
    "clearly": ("clearly", "ADV"),
    "patiently": ("patiently", "ADV"),
    "gracefully": ("gracefully", "ADV"),
    "together": ("together", "ADV"),
    "again": ("again", "ADV"),
    "already": ("already", "ADV"),
    "still": ("still", "ADV"),
    "just": ("just", "ADV"),
    "only": ("only", "ADV"),
    "really": ("really", "ADV"),
    "quite": ("quite", "ADV"),
    "more": ("more", "ADV"),
    "less": ("less", "ADV"),
    "most": ("most", "ADV"),
    "late": ("late", "ADV"),
    "hard": ("hard", "ADV"),
    "monthly": ("monthly", "ADV"),
    "weekly": ("weekly", "ADV"),
    "the": ("the", "DET"),
    "a": ("a", "DET"),
    "an": ("an", "DET"),
    "this": ("this", "DET"),
    "that": ("that", "DET"),
    "these": ("this", "DET"),
    "those": ("that", "DET"),
    "my": ("my", "PRON"),
    "your": ("your", "PRON"),
    "his": ("his", "PRON"),
    "her": ("her", "PRON"),
    "its": ("its", "PRON"),
    "our": ("our", "PRON"),
    "their": ("their", "PRON"),
    "mine": ("mine", "PRON"),
    "yours": ("your", "PRON"),
    "ours": ("our", "PRON"),
    "theirs": ("their", "PRON"),
    "i": ("I", "PRON"),
    "you": ("you", "PRON"),
    "he": ("he", "PRON"),
    "she": ("she", "PRON"),
    "it": ("it", "PRON"),
    "we": ("we", "PRON"),
    "they": ("they", "PRON"),
    "who": ("who", "PRON"),
    "what": ("what", "PRON"),
    "which": ("which", "PRON"),
    "where": ("where", "ADV"),
    "how": ("how", "ADV"),
    "why": ("why", "ADV"),
    "cannot": ("can", "AUX"),
    "can't": ("can", "AUX"),
    "won't": ("will", "AUX"),
    "don't": ("do", "AUX"),
    "doesn't": ("do", "AUX"),
    "didn't": ("do", "AUX"),
    "isn't": ("be", "AUX"),
    "aren't": ("be", "AUX"),
    "wasn't": ("be", "AUX"),
    "weren't": ("be", "AUX"),
    "haven't": ("have", "AUX"),
    "hasn't": ("have", "AUX"),
    "hadn't": ("have", "AUX"),
    "wouldn't": ("will", "AUX"),
    "couldn't": ("can", "AUX"),
    "shouldn't": ("should", "AUX"),
    "mustn't": ("must", "AUX"),
}

COMPARATIVE_ADJ = frozenset({
    "smaller", "bigger", "larger", "taller", "shorter", "faster", "slower",
    "cheaper", "dearer", "nicer", "better", "worse", "older", "younger",
    "colder", "warmer", "hotter", "busier", "heavier", "lighter", "longer",
    "sweeter", "harder", "easier", "quicker", "stronger", "weaker",
    "newer", "narrower", "louder", "happier", "closer", "fitter",
    "more", "less",
})

COMPARATIVE_BASE = {
    "smaller": "small",
    "bigger": "big",
    "larger": "large",
    "taller": "tall",
    "shorter": "short",
    "faster": "fast",
    "slower": "slow",
    "cheaper": "cheap",
    "dearer": "dear",
    "nicer": "nice",
    "better": "good",
    "worse": "bad",
    "older": "old",
    "younger": "young",
    "colder": "cold",
    "warmer": "warm",
    "hotter": "hot",
    "busier": "busy",
    "heavier": "heavy",
    "lighter": "light",
    "longer": "long",
    "sweeter": "sweet",
    "harder": "hard",
    "easier": "easy",
    "quicker": "quick",
    "stronger": "strong",
    "weaker": "weak",
    "newer": "new",
    "narrower": "narrow",
    "louder": "loud",
    "happier": "happy",
    "closer": "close",
    "fitter": "fit",
    "more": "much",
    "less": "little",
    "exciting": "exciting",
    "crowded": "crowded",
}

PHRASAL_PARTICLES = frozenset({"up", "down", "on", "off", "out", "away", "in", "back"})


def _strip_verb_suffix(lemma: str) -> str:
    if lemma in VERB_LEMMA_MAP:
        return VERB_LEMMA_MAP[lemma]
    if lemma.endswith("ies") and len(lemma) > 3:
        return lemma[:-3] + "y"
    if lemma.endswith("ing") and len(lemma) > 4:
        base = lemma[:-3]
        if base.endswith(base[-1] * 2):
            base = base[:-1]
        if base.endswith("e") and not lemma.endswith("eing"):
            base = base[:-1]
        return base
    if lemma.endswith("ed") and len(lemma) > 3:
        base = lemma[:-2]
        if base.endswith("i"):
            return base[:-1] + "y"
        if base.endswith(base[-1] * 2):
            base = base[:-1]
        return base
    if lemma.endswith("es") and len(lemma) > 3:
        return lemma[:-2]
    if lemma.endswith("s") and len(lemma) > 2 and not lemma.endswith("ss"):
        return lemma[:-1]
    return lemma


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    lower = form.lower()

    if form in ".,;:!?":
        return form, "PUNCT"

    if lower in REFLEXIVE_PRONOUNS:
        return lower, "PRON"

    if lower in SPECIAL_UPOS:
        return SPECIAL_UPOS[lower]

    if lower in PRON_LEMMA_MAP and upos == "PRON":
        return PRON_LEMMA_MAP[lower], "PRON"

    if upos == "PROPN" or form[0].isupper() and upos in {"PROPN", "NOUN", "ADJ"}:
        upos = "PROPN"
        if lower in PROPN_LEMMA_MAP:
            return PROPN_LEMMA_MAP[lower], "PROPN"
        return form if form[0].isupper() else form.capitalize(), "PROPN"

    if lower in AUX_BE or lemma.lower() in AUX_BE:
        return "be", "AUX"
    if lower in AUX_HAVE or lemma.lower() in AUX_HAVE:
        return "have", "AUX"
    if lower in AUX_DO or lemma.lower() in AUX_DO:
        return "do", "AUX"
    if lower in AUX_WILL or lemma.lower() in AUX_WILL:
        return "will", "AUX"
    if lower in AUX_CAN or lemma.lower() in AUX_CAN:
        return "can", "AUX"
    if lower in AUX_OTHER:
        return lower, "AUX"

    if upos == "NOUN" or (upos == "X" and lower not in AUX_BE):
        if form in NOUN_LEMMA_MAP:
            return NOUN_LEMMA_MAP[form], "NOUN"
        if lemma in NOUN_LEMMA_MAP:
            return NOUN_LEMMA_MAP[lemma], "NOUN"
        noun_lemma = lemma.lower() if lemma else lower
        if noun_lemma.endswith("ies"):
            noun_lemma = noun_lemma[:-3] + "y"
        elif noun_lemma.endswith("ves"):
            noun_lemma = noun_lemma[:-3] + "f"
        elif noun_lemma.endswith("ses") and noun_lemma not in {"glasses", "clothes"}:
            noun_lemma = noun_lemma[:-2]
        elif noun_lemma.endswith("s") and not noun_lemma.endswith("ss") and noun_lemma not in {
            "news", "physics", "mathematics", "glasses", "clothes", "o'clock",
        }:
            noun_lemma = noun_lemma[:-1]
        return noun_lemma, "NOUN"

    if upos in {"VERB", "AUX"}:
        if lower in VERB_LEMMA_MAP:
            return VERB_LEMMA_MAP[lower], "VERB"
        if lemma in VERB_LEMMA_MAP:
            return VERB_LEMMA_MAP[lemma], "VERB"
        verb_lemma = _strip_verb_suffix(lemma.lower() if lemma else lower)
        return verb_lemma, "VERB"

    if upos == "ADJ" or lower in COMPARATIVE_ADJ:
        if lower in COMPARATIVE_BASE:
            return COMPARATIVE_BASE[lower], "ADJ"
        if lemma and lemma.lower() in COMPARATIVE_BASE:
            return COMPARATIVE_BASE[lemma.lower()], "ADJ"
        adj_lemma = lemma.lower() if lemma else lower
        if adj_lemma.endswith("ier"):
            return adj_lemma[:-3] + "y", "ADJ"
        if adj_lemma.endswith("est"):
            base = adj_lemma[:-3]
            if base.endswith("i"):
                return base[:-1] + "y", "ADJ"
            return base, "ADJ"
        if adj_lemma.endswith("er"):
            base = adj_lemma[:-2]
            if base.endswith("i"):
                return base[:-1] + "y", "ADJ"
            return base, "ADJ"
        return adj_lemma, "ADJ"

    if upos == "ADV":
        return (lemma.lower() if lemma else lower), "ADV"

    if upos == "DET":
        return (lemma.lower() if lemma else lower), "DET"

    if upos == "NUM":
        return (lemma.lower() if lemma else lower), "NUM"

    if upos == "PUNCT":
        return form, "PUNCT"

    return (lemma.lower() if lemma else lower), upos


def post_process_tokens(aligned: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    for i, (form, lemma, upos) in enumerate(aligned):
        lower = form.lower()

        if lower == "to" and i + 1 < len(aligned) and aligned[i + 1][2] == "VERB":
            lemma, upos = "to", "PART"

        if lower == "than" and i > 0 and aligned[i - 1][2] in {"ADJ", "ADV"}:
            lemma, upos = "than", "SCONJ"

        if lower == "as":
            if i > 0 and aligned[i - 1][2] in {"ADJ", "ADV", "DET"}:
                lemma, upos = "as", "SCONJ"
            elif i + 1 < len(aligned) and aligned[i + 1][0].lower() in {
                "tall", "well", "fast", "strong", "tired", "good", "fit", "far",
                "quiet", "gracefully", "crowded",
            }:
                lemma, upos = "as", "SCONJ"

        if lower in PHRASAL_PARTICLES and i > 0 and aligned[i - 1][2] == "VERB":
            lemma, upos = lower, "ADP"

        if form == "'s" or form == "’s":
            lemma, upos = "'s", "PART"

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

        head = words[wi]
        lemma, upos = normalize_token(form, head.upos or "X", head.lemma or form)
        aligned.append((form, lemma, upos))
        wi += 1

    return post_process_tokens(aligned)


def count_tokens(sentence: str) -> int:
    return len(tokenize_text(sentence))


for i, sent in enumerate(sentences, start=1):
    n = count_tokens(sent)
    if n < 5 or n > 12:
        raise ValueError(f"Sentence {i} has {n} tokens (need 5–12): {sent}")

output_lines: list[str] = []

for idx, sent in enumerate(sentences):
    sent_id = f"en_a2_train_{start_id + idx:03d}"
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
target_path = project_root / "data/handcraft/en/train/a2_new_005.conllu"
target_path.write_text(conllu_text, encoding="utf-8")
print(f"Wrote {target_path}")

validation_res = validate_text(conllu_text)
print("Validate:", validation_res.passed, "sentences:", validation_res.sentence_count)
if not validation_res.passed:
    for err in validation_res.errors[:50]:
        print(" ", err)

lemma_res = check_text(conllu_text, lang="en")
print("Lemma check:", lemma_res.passed)
if not lemma_res.passed:
    for err in lemma_res.errors[:50]:
        print(" ", err)

if not validation_res.passed or not lemma_res.passed:
    sys.exit(1)

print("STATUS: OK — 100 sentences, en_a2_train_801–en_a2_train_900")