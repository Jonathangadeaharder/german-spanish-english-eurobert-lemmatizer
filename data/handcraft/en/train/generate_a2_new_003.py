"""Generate a2_new_003.conllu — en_a2_train_401 through en_a2_train_600."""

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 8 batches × 25 sentences = 200 (A2: weather, school, restaurant, hobbies, future, directions, tech, mixed)
BATCHES = [
    # 401–425: Weather & seasons
    [
        "It rains a lot in autumn here.",
        "The sky is clear and blue today.",
        "We wear warm coats in the winter.",
        "She takes an umbrella when it rains.",
        "The sun shines brightly in the summer.",
        "I like walking in the spring rain.",
        "It is windy on the coast today.",
        "They play outside on sunny days.",
        "Snow covers the hills every January.",
        "The weather forecast says rain tomorrow.",
        "He feels cold without his thick scarf.",
        "We enjoy hot days at the beach.",
        "The leaves turn brown in October.",
        "I stay indoors when the storm comes.",
        "She opens the window on warm evenings.",
        "Fog makes driving difficult in the morning.",
        "We pick fruit from the garden in summer.",
        "The temperature drops at night quickly.",
        "He checks the weather app every morning.",
        "Ice appears on the lake in winter.",
        "I wear sunglasses on bright summer days.",
        "They build a snowman after the snowfall.",
        "Rainbow colours appear after the heavy rain.",
        "We dry our clothes in the hot sun.",
        "Spring flowers bloom in the park soon.",
    ],
    # 426–450: School & learning
    [
        "She studies maths at school every day.",
        "We have a test on Friday morning.",
        "He writes notes in his exercise book.",
        "I ask the teacher about the homework.",
        "They sit in the front row today.",
        "She learns new words in English class.",
        "We read a story in the library.",
        "He forgets his pencil case at home.",
        "I finish the science project this week.",
        "She completes the exam with good marks.",
        "We draw maps in the geography lesson.",
        "He listens carefully during the history talk.",
        "I practise spelling with my classmate.",
        "She joins the school choir after lessons.",
        "We copy the answers from the board.",
        "He packs his bag before the bell sounds.",
        "I revise for the test every evening.",
        "She explains the rule to her friend.",
        "We visit the museum with our class.",
        "He finds his classroom on the second floor.",
        "I borrow a ruler from the teacher.",
        "She writes her name on the paper.",
        "We clap when the play ends.",
        "He misses the bus after school today.",
        "I learn about animals in biology class.",
    ],
    # 451–475: Restaurant & café
    [
        "We order soup and salad at the café.",
        "She asks for the bill after dinner.",
        "I reserve a table for four people.",
        "He tries the special dish of the day.",
        "They serve fresh bread with the meal.",
        "I drink water with lemon at lunch.",
        "She tips the waiter after the meal.",
        "We wait for a table near the window.",
        "He complains about the cold coffee.",
        "I choose chicken instead of fish tonight.",
        "She eats dessert after the main course.",
        "We share a pizza at the restaurant.",
        "He reads the menu before he orders.",
        "I recommend this pasta to my friend.",
        "She adds pepper to her tomato soup.",
        "We leave a good review for the chef.",
        "He pays the bill with his credit card.",
        "I taste the sauce before I serve it.",
        "She orders tea without sugar this time.",
        "We sit outside on the sunny terrace.",
        "He returns the wrong dish to the waiter.",
        "I book lunch at the new Italian place.",
        "She wipes her hands with a clean napkin.",
        "We celebrate her birthday at the restaurant.",
        "He enjoys the view from the rooftop café.",
    ],
    # 476–500: Hobbies & free time
    [
        "I paint pictures in my free time.",
        "She collects stamps from different countries.",
        "We play board games on rainy evenings.",
        "He builds model planes in the garage.",
        "I read comics before I go to bed.",
        "She knits scarves for her family members.",
        "We go fishing at the lake on Sundays.",
        "He takes photos of birds in the forest.",
        "I learn guitar songs from online videos.",
        "She draws cartoons in her sketchbook.",
        "We bake cookies with the children today.",
        "He watches comedy shows on TV tonight.",
        "I join a chess club at the community centre.",
        "She grows herbs on her kitchen window sill.",
        "We cycle along the river on Saturday.",
        "He repairs old bikes as a hobby.",
        "I write short stories in a notebook.",
        "She dances salsa with her friends weekly.",
        "We camp in the woods during the holiday.",
        "He makes videos for his small online channel.",
        "I visit art galleries in the city centre.",
        "She sews buttons on her favourite jacket.",
        "We listen to podcasts on long train rides.",
        "He solves crossword puzzles every morning.",
        "I play video games with my brother.",
    ],
    # 501–525: Future (going to)
    [
        "I am going to visit my aunt tomorrow.",
        "She is going to study abroad next year.",
        "We are going to move house in June.",
        "He is going to fix the broken shelf.",
        "They are going to open a small café.",
        "I am going to call you this evening.",
        "She is going to learn French at school.",
        "We are going to cook pasta for dinner.",
        "He is going to buy a new laptop soon.",
        "I am going to start jogging next Monday.",
        "She is going to wear a blue dress tonight.",
        "We are going to take the early train.",
        "He is going to paint the bedroom walls.",
        "I am going to save money for the trip.",
        "She is going to meet her friend at noon.",
        "We are going to plant trees in the garden.",
        "He is going to clean the car this weekend.",
        "I am going to send the package by post.",
        "She is going to practise piano every evening.",
        "We are going to watch the match on TV.",
        "He is going to ask his boss for help.",
        "I am going to arrive before eight o'clock.",
        "She is going to change her phone number soon.",
        "We are going to invite them for Sunday lunch.",
        "He is going to finish the report by Friday.",
    ],
    # 526–550: Directions & transport
    [
        "Turn left at the traffic lights ahead.",
        "The bus stop is opposite the bank.",
        "We walk across the bridge to the station.",
        "She asks where the post office is.",
        "Go straight on until you see the church.",
        "The pharmacy is next to the supermarket.",
        "I take the underground to the city centre.",
        "He misses his stop and gets off late.",
        "We follow the signs to the airport exit.",
        "She shows him the way on the map.",
        "The museum is behind the old town hall.",
        "I buy a ticket at the machine outside.",
        "He parks the car near the market square.",
        "We walk down this street to the harbour.",
        "She catches the tram at the main square.",
        "The hotel entrance is on your right side.",
        "I get a taxi from the train station.",
        "He reads the timetable on the platform wall.",
        "We ride scooters through the narrow streets.",
        "She turns right after the second corner.",
        "The cinema is at the end of the road.",
        "I validate my ticket before I board the train.",
        "He waits at the corner for the green light.",
        "We walk past the library on the way home.",
        "She finds the correct gate at the airport.",
    ],
    # 551–575: Technology & social media
    [
        "I download music onto my phone tonight.",
        "She posts photos on her social media page.",
        "He connects his laptop to the hotel Wi-Fi.",
        "We watch videos online during the break.",
        "I update my apps before the trip starts.",
        "She deletes old files from her computer.",
        "He shares a link with his work team.",
        "We chat online with friends from abroad.",
        "I back up my photos to the cloud.",
        "She logs into her email every morning.",
        "He forgets his password and resets it.",
        "We search for cheap flights on the website.",
        "I reply to messages before I go out.",
        "She streams films on her tablet at night.",
        "He installs a new game on his console.",
        "We follow news updates on our phones daily.",
        "I print the tickets from my home printer.",
        "She blocks spam calls on her mobile phone.",
        "He scans the QR code at the museum.",
        "We record a short video for the blog.",
        "I charge my tablet before the long flight.",
        "She copies the address into the navigation app.",
        "He mutes the microphone during the online call.",
        "We upload holiday pictures to the family album.",
        "I turn off notifications when I study.",
    ],
    # 576–600: Mixed A2 topics
    [
        "I visit the dentist twice a year.",
        "She adopts a small dog from the shelter.",
        "We rent a flat near the university campus.",
        "He fixes the leaky tap in the bathroom.",
        "I vote in the local election this spring.",
        "She donates clothes to the charity shop.",
        "We recycle plastic bottles every week.",
        "He plants flowers along the garden path.",
        "I volunteer at the animal centre on Saturdays.",
        "She applies for a summer job at the hotel.",
        "We attend a wedding in the village church.",
        "He rents tools from the hardware store.",
        "I sign the contract at the estate office.",
        "She knocks on the door and waits outside.",
        "We lock the bikes outside the sports hall.",
        "He brings seeds for the birds in the park.",
        "I iron my shirt before the job interview.",
        "She folds the map and puts it away.",
        "We hang pictures on the living room wall.",
        "He collects tickets from the cinema counter.",
        "I wrap the gift in colourful paper.",
        "She waters the garden after the hot day.",
        "We queue quietly at the post office desk.",
        "He offers his seat to an older passenger.",
        "I feel grateful for my helpful neighbours.",
    ],
]

sentences = [s for batch in BATCHES for s in batch]
assert len(sentences) == 200

start_id = 401

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
target_path = project_root / "data/handcraft/en/train/a2_new_003.conllu"
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

print("STATUS: OK — 200 sentences, en_a2_train_401–en_a2_train_600")