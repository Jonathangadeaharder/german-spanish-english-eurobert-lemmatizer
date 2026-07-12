"""Generate a2_new_004.conllu — en_a2_train_601 through en_a2_train_800."""

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 8 batches × 25 sentences = 200 (A2: shopping, jobs, feelings, past continuous, present perfect, animals, home, mixed)
BATCHES = [
    # 601–625: Shopping & clothes
    [
        "I try on the blue jacket in the shop.",
        "She buys a warm scarf at the market.",
        "We look for shoes on the second floor.",
        "He pays with cash at the checkout.",
        "I return the shirt because it is small.",
        "She wears a hat on sunny days.",
        "We compare prices at two different shops.",
        "He picks a tie for his job interview.",
        "I need a new belt for these trousers.",
        "She folds the clothes after she washes them.",
        "We wait in line at the fitting room.",
        "He chooses jeans that fit him well.",
        "I pack my coat in the shopping bag.",
        "She exchanges the dress for a larger size.",
        "We spend too much money on clothes.",
        "He irons his shirt before the meeting.",
        "I hang my jacket on the bedroom door.",
        "She matches her shoes with her handbag.",
        "We buy gloves before the cold winter.",
        "He checks the size on the clothing label.",
        "I prefer cotton shirts to wool sweaters.",
        "She saves money for a new winter coat.",
        "We find a sale at the department store.",
        "He washes his trainers after the long run.",
        "I feel happy with my new red dress.",
    ],
    # 626–650: Jobs & workplace
    [
        "She starts work at nine every morning.",
        "We have a meeting in the small office.",
        "He writes reports for his manager each week.",
        "I send emails to clients before lunch.",
        "She takes a break at eleven thirty.",
        "We finish the project by next Friday.",
        "He works from home on Mondays.",
        "I ask my boss for a day off.",
        "She trains new staff at the company.",
        "We share ideas during the team lunch.",
        "He checks the schedule on the wall.",
        "I type notes on my work laptop.",
        "She answers calls at the front desk.",
        "We print the documents for the meeting.",
        "He leaves the office at six o'clock.",
        "I bring lunch from home to save money.",
        "She organises files in the shared folder.",
        "We talk about the plan with our colleagues.",
        "He signs papers at the reception desk.",
        "I learn new skills at the training course.",
        "She arrives early for the job interview.",
        "We wear badges inside the building.",
        "He fixes the printer in the copy room.",
        "I fill in a form for the HR department.",
        "She feels nervous before her first day.",
    ],
    # 651–675: Feelings & emotions
    [
        "I feel nervous before the big exam.",
        "She looks worried about her sick cat.",
        "We are excited about the summer trip.",
        "He seems angry after the long delay.",
        "I am bored during the slow meeting.",
        "She feels proud of her exam results.",
        "We are surprised by the sudden news.",
        "He looks tired after the night shift.",
        "I am happy to see my old friends.",
        "She feels lonely in the new city.",
        "We are scared during the loud storm.",
        "He seems relaxed on his day off.",
        "I feel grateful for my family's help.",
        "She looks sad when she hears the news.",
        "We are curious about the new neighbours.",
        "He feels confident before the job talk.",
        "I am jealous of her new bicycle.",
        "She feels calm after her yoga class.",
        "We are disappointed by the match result.",
        "He looks upset after his small mistake.",
        "I feel hopeful about the new plan.",
        "She seems pleased with her test score.",
        "We are tense before the long journey.",
        "He feels relieved when the test ends.",
        "I am shy when I meet new people.",
    ],
    # 676–700: Past continuous / when-while
    [
        "I was cooking when he called me.",
        "She was reading while he watched TV.",
        "We were walking when it started raining.",
        "He was sleeping when the alarm sounded.",
        "I was working while my sister studied.",
        "She was driving when she saw the deer.",
        "We were eating when the guests arrived.",
        "He was playing when he hurt his knee.",
        "I was listening to music while I cleaned.",
        "She was talking when the teacher entered.",
        "We were waiting when the bus finally came.",
        "He was writing when his pen broke.",
        "I was shopping when I met my friend.",
        "She was dancing while the band played.",
        "We were studying when the lights went out.",
        "He was running when he lost his key.",
        "I was painting when someone knocked.",
        "She was cooking while her son did homework.",
        "We were laughing when the photo was taken.",
        "He was fixing the bike when it rained.",
        "I was texting when my battery died.",
        "She was sitting when the dog jumped up.",
        "We were planning when she called us back.",
        "He was packing when his taxi arrived.",
        "I was thinking while I walked home.",
    ],
    # 701–725: Present perfect (basic)
    [
        "I have lived here for two years.",
        "She has finished her homework already.",
        "We have visited Paris three times.",
        "He has never tried sushi before.",
        "I have lost my keys again today.",
        "She has worked at the bank since May.",
        "We have known each other since school.",
        "He has just arrived at the station.",
        "I have eaten too much cake tonight.",
        "She has bought a new phone recently.",
        "We have seen that film twice already.",
        "He has broken his glasses this morning.",
        "I have written three emails so far.",
        "She has cleaned the flat this week.",
        "We have waited for an hour already.",
        "He has forgotten his wallet at home.",
        "I have read half of this long book.",
        "She has taken her driving test finally.",
        "We have saved enough money for the trip.",
        "He has cut his finger while cooking.",
        "I have heard that song many times.",
        "She has moved to a new flat recently.",
        "We have grown vegetables in our garden.",
        "He has hurt his back at the gym.",
        "I have met her parents only once.",
    ],
    # 726–750: Animals & nature
    [
        "We feed the birds in the park daily.",
        "She walks her dog along the river.",
        "He watches ducks on the small lake.",
        "I love cats but I am allergic.",
        "We see rabbits in the field at dawn.",
        "She pets the horse at the farm.",
        "He photographs butterflies in the meadow.",
        "I hear owls in the woods at night.",
        "We protect bees in our wild garden.",
        "She adopts a rabbit from the shelter.",
        "He cleans the fish tank every Sunday.",
        "I follow a fox trail in the snow.",
        "We plant trees for the local birds.",
        "She carries food for the hungry stray.",
        "He builds a birdhouse in the garden.",
        "I watch whales from the boat deck.",
        "We spot deer near the forest path.",
        "She learns names of common wild flowers.",
        "He finds a frog near the garden pond.",
        "I listen to crickets on warm evenings.",
        "We avoid litter in the nature reserve.",
        "She draws animals in her sketch pad.",
        "He keeps hamsters as quiet classroom pets.",
        "I see squirrels climb the tall oak tree.",
        "We respect rules in the national park.",
    ],
    # 751–775: House & home maintenance
    [
        "He paints the kitchen walls this weekend.",
        "I change the light bulb in the hall.",
        "She fixes the loose handle on the door.",
        "We clean the gutters after the storm.",
        "He drills holes for the new shelves.",
        "I tighten the screws on the chair.",
        "She unplugs the fridge before she cleans it.",
        "We move the sofa to the other room.",
        "He repairs the fence behind the house.",
        "I sand the old wooden table carefully.",
        "She measures the wall for the new shelf.",
        "We replace the broken window pane today.",
        "He oils the squeaky hinge on the gate.",
        "I vacuum under the bed every week.",
        "She scrubs the bathroom tiles on Saturday.",
        "We assemble the new flat pack furniture.",
        "He seals the gap under the front door.",
        "I clear leaves from the garden drain.",
        "She wipes fingerprints off the glass door.",
        "We test the smoke alarm every month.",
        "He adjusts the thermostat in the bedroom.",
        "I mop the floor after the muddy walk.",
        "She rehangs the picture above the sofa.",
        "We store tools in the garden shed.",
        "He checks the roof after heavy rain.",
    ],
    # 776–800: Mixed A2 topics
    [
        "I learn to drive with a patient instructor.",
        "She babysits her nephew on Friday night.",
        "We host a barbecue in the back garden.",
        "He delivers newspapers early every morning.",
        "I join a book club at the library.",
        "She practices driving in the empty car park.",
        "We rent a kayak for the lake trip.",
        "He tutors maths for younger school students.",
        "I book tickets for the theatre online.",
        "She knits a blanket for the baby shower.",
        "We donate blood at the local hospital.",
        "He coaches children at the youth centre.",
        "I sort recycling into separate coloured bins.",
        "She leads a short tour at the museum.",
        "We babysit the neighbours' dog this weekend.",
        "He learns first aid at the community class.",
        "I rent a costume for the fancy dress party.",
        "She packs lunches for the school trip.",
        "We greet guests at the hotel entrance.",
        "He collects litter along the beach path.",
        "I rehearse my speech before the presentation.",
        "She gives the fish food while we are away.",
        "We board the ferry with our bicycles.",
        "He signs up for the local running race.",
        "I thank the nurse after the quick check-up.",
    ],
]

sentences = [s for batch in BATCHES for s in batch]
assert len(sentences) == 200

start_id = 601

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
target_path = project_root / "data/handcraft/en/train/a2_new_004.conllu"
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

print("STATUS: OK — 200 sentences, en_a2_train_601–en_a2_train_800")