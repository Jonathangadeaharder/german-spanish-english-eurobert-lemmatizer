"""Generate a2_new_001.conllu — en_a2_train_001 through en_a2_train_200."""

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 8 batches × 25 sentences = 200 (A2: 5–12 tokens, daily life / travel / health / past / modals / comparisons)
BATCHES = [
    # 001–025: Daily life — morning & home
    [
        "I wake up at seven every morning.",
        "She brushes her teeth before breakfast.",
        "We eat toast and jam for breakfast.",
        "He takes a shower after his workout.",
        "My mother makes coffee in the kitchen.",
        "The children put on their school clothes quickly.",
        "I always check my email in the morning.",
        "She waters the plants on the balcony.",
        "We clean the house every Saturday morning.",
        "He feeds the cat before he leaves home.",
        "I put my keys in my bag every day.",
        "She hangs her coat in the hallway closet.",
        "We watch the news after dinner together.",
        "He turns off the lights before bed.",
        "I set the alarm for six tomorrow morning.",
        "She folds the clean laundry on the bed.",
        "We open the windows when it is warm.",
        "He vacuums the living room on Sundays.",
        "I wash the dishes after every meal.",
        "She makes the bed before she goes out.",
        "We take out the rubbish every evening.",
        "He repairs the broken chair in the kitchen.",
        "I lock the door when I leave home.",
        "She listens to music while she cooks dinner.",
        "We sit on the sofa and read books.",
    ],
    # 026–050: Daily life — shopping & food
    [
        "I go to the supermarket on Fridays.",
        "She buys fresh fruit at the market.",
        "We need milk, bread, and eggs today.",
        "He pays for the groceries with his card.",
        "I cook pasta with tomato sauce for dinner.",
        "She chops the vegetables for the salad.",
        "We drink tea with honey in the evening.",
        "He bakes a cake for his sister's birthday.",
        "I order a pizza for dinner tonight.",
        "She tastes the soup and adds more salt.",
        "We share a large pizza at the restaurant.",
        "He cuts the bread into small pieces.",
        "I boil the eggs for ten minutes.",
        "She serves the food on blue plates.",
        "We finish our meal and pay the bill.",
        "He puts the milk in the fridge quickly.",
        "I heat the soup in the microwave.",
        "She adds sugar to her cup of coffee.",
        "We try a new recipe from the cookbook.",
        "He eats an apple after his lunch break.",
        "I pack a sandwich for work tomorrow.",
        "She washes the fruit before she eats it.",
        "We buy cheese and butter at the shop.",
        "He grills chicken for the family dinner.",
        "I stir the rice while it is cooking.",
    ],
    # 051–075: Travel
    [
        "We arrive at the airport at noon.",
        "I pack my suitcase for the trip tomorrow.",
        "She shows her passport at the border.",
        "He buys a ticket for the morning train.",
        "We wait at the bus stop near the hotel.",
        "I check in at the hotel reception desk.",
        "She asks for directions to the museum.",
        "He takes a taxi to the city centre.",
        "We visit the old castle on the hill.",
        "I book a room with a sea view.",
        "She takes photos of the famous bridge.",
        "He misses the bus and walks to work.",
        "We wait at the traffic lights before we walk.",
        "I look at the map on my phone.",
        "She catches the train to London today.",
        "He rents a car for the weekend trip.",
        "We explore the market in the old town.",
        "I leave my bag at the hotel desk.",
        "She changes money at the bank nearby.",
        "He finds a cheap hostel near the station.",
        "We ride the ferry to the small island.",
        "I send a postcard to my parents.",
        "She boards the plane at gate twelve.",
        "He returns the rental car on Monday.",
        "We enjoy the view from the hotel balcony.",
    ],
    # 076–100: Health
    [
        "I feel sick and stay at home today.",
        "She has a headache after a long day.",
        "We call the doctor for an appointment.",
        "He takes medicine for his sore throat.",
        "I rest in bed when I have a fever.",
        "She drinks warm tea for her cold.",
        "We go to the pharmacy after lunch.",
        "He hurts his knee while playing football.",
        "I need to see the dentist this week.",
        "She wears a bandage on her cut finger.",
        "We eat healthy food and drink water.",
        "He sleeps eight hours every night.",
        "I exercise three times a week at home.",
        "She feels better after a good rest.",
        "We wash our hands before every meal.",
        "He coughs a lot during the night.",
        "I avoid sugar when I feel unwell.",
        "She checks her temperature with a thermometer.",
        "We walk in the park for fresh air.",
        "He visits the nurse at the school clinic.",
        "I stretch my back after sitting too long.",
        "She puts cream on her dry skin.",
        "We drink soup when we are ill.",
        "He recovers quickly after the small operation.",
        "I feel tired but I go to work.",
    ],
    # 101–125: Past tense
    [
        "I visited my grandparents last weekend.",
        "She cooked dinner for the whole family.",
        "We walked to the park after school.",
        "He played football with his friends yesterday.",
        "I watched a film at the cinema.",
        "She cleaned her room on Saturday morning.",
        "We stayed at a small hotel in Rome.",
        "He called his mother on Sunday evening.",
        "I finished my homework before dinner.",
        "She bought a new dress at the shop.",
        "We met our friends at the coffee bar.",
        "He learned ten new words in class.",
        "I lost my wallet on the bus yesterday.",
        "She wrote a letter to her aunt.",
        "We danced at the party all night.",
        "He helped his father fix the bike.",
        "I forgot my umbrella at the restaurant.",
        "She opened the window because it was hot.",
        "We enjoyed the concert in the park.",
        "He closed the shop at eight o'clock.",
        "I received a package from my sister.",
        "She painted the wall in her bedroom.",
        "We travelled by train across the country.",
        "He answered all the questions on the test.",
        "I remembered her name after a moment.",
    ],
    # 126–150: Modals
    [
        "I can swim but I cannot dive well.",
        "She must finish her work before five.",
        "We should leave now or we will be late.",
        "He will visit his uncle next month.",
        "I would like a glass of water, please.",
        "You can sit here if you want.",
        "She must take this medicine twice a day.",
        "We can meet at the cafe at three.",
        "He should call the doctor about his cough.",
        "I will help you carry those heavy bags.",
        "They cannot come to the party tonight.",
        "She can speak English and a little French.",
        "We must show our tickets at the gate.",
        "He will not be at home this evening.",
        "I can cook dinner if you buy the food.",
        "You should wear a coat because it is cold.",
        "She will start her new job on Monday.",
        "We can take the bus to the airport.",
        "He must return the book to the library.",
        "I would like to visit Spain next summer.",
        "She can help you find the right street.",
        "We should book the tickets in advance.",
        "He will bring dessert to the dinner party.",
        "I can meet you after work on Thursday.",
        "You must turn off your phone in class.",
    ],
    # 151–175: Comparisons
    [
        "My room is smaller than your room.",
        "She runs faster than her older brother.",
        "This hotel is cheaper than the other one.",
        "He is taller than me by two centimetres.",
        "The train is quicker than the bus today.",
        "I am as tall as my younger sister.",
        "This soup is better than the salad.",
        "She dances as well as her music teacher.",
        "The blue shirt is nicer than the green one.",
        "He works harder than his colleague at the office.",
        "Winter is colder than autumn in this city.",
        "I eat less sugar than my brother does.",
        "The market is busier than the shop nearby.",
        "She is younger than I thought at first.",
        "This street is longer than the next one.",
        "He drives more carefully than his friend.",
        "The cake tastes sweeter than the biscuits.",
        "We are as tired as you are tonight.",
        "Her bag is heavier than mine by far.",
        "The new park is bigger than the old one.",
        "I speak English better than my cousin.",
        "Today is warmer than yesterday was.",
        "He is as strong as his father now.",
        "The film was more boring than the book.",
        "She walks as fast as her best friend.",
    ],
    # 176–200: Mixed A2 topics
    [
        "I work in an office near the station.",
        "She studies English at the language school.",
        "We live in a flat above the bakery.",
        "He rides his bike to school every day.",
        "I like reading books about travel and food.",
        "She works part time at the local cafe.",
        "We plan a picnic in the park on Sunday.",
        "He listens to the radio in the morning.",
        "I send a message to my friend every day.",
        "She wears glasses when she reads small text.",
        "We celebrate her birthday with a small party.",
        "He saves money for his holiday in Greece.",
        "I borrow books from the public library.",
        "She teaches maths at the secondary school.",
        "We invite our neighbours for dinner on Friday.",
        "He fixes his phone with help from the shop.",
        "I prefer tea but my wife prefers coffee.",
        "She grows flowers in pots on the terrace.",
        "We wait for the rain to stop outside.",
        "He plays the guitar in a small band.",
        "I need a new jacket for the cold weather.",
        "She finds her glasses under the newspaper.",
        "We choose a table near the window.",
        "He returns home late after the football match.",
        "I hope to see you again very soon.",
    ],
]

sentences = [s for batch in BATCHES for s in batch]
assert len(sentences) == 200

start_id = 1

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
    "biscuits": "biscuit",
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
    "lit": "light",
}

PROPN_LEMMA_MAP = {
    "london": "London",
    "rome": "Rome",
    "spain": "Spain",
    "greece": "Greece",
    "french": "French",
    "english": "English",
}

SPECIAL_UPOS = {
    "not": ("not", "PART"),
    "please": ("please", "INTJ"),
    "yes": ("yes", "INTJ"),
    "no": ("no", "INTJ"),
    "to": ("to", "PART"),  # infinitive marker; overridden when ADP
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
    "more": "much",
    "less": "little",
    "boring": "boring",
}


def _strip_verb_suffix(lemma: str) -> str:
    """Reduce -ing/-ed/-s verb lemmas to base when Stanza misses."""
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

        # "to" as infinitive marker before verb
        if lower == "to" and i + 1 < len(aligned) and aligned[i + 1][2] == "VERB":
            lemma, upos = "to", "PART"

        # "than" after adjective/adverb
        if lower == "than" and i > 0 and aligned[i - 1][2] in {"ADJ", "ADV"}:
            lemma, upos = "than", "SCONJ"

        # "as ... as" comparison
        if lower == "as":
            if i > 0 and aligned[i - 1][2] in {"ADJ", "ADV", "DET"}:
                lemma, upos = "as", "SCONJ"
            elif i + 1 < len(aligned) and aligned[i + 1][0].lower() in {"tall", "well", "fast", "strong", "tired", "good"}:
                lemma, upos = "as", "SCONJ"

        # phrasal particle
        if lower == "up" and i > 0 and aligned[i - 1][2] == "VERB":
            lemma, upos = "up", "ADP"

        # possessive 's
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


# Validate token counts before generation
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
target_path = project_root / "data/handcraft/en/train/a2_new_001.conllu"
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

print("STATUS: OK — 200 sentences, en_a2_train_001–en_a2_train_200")