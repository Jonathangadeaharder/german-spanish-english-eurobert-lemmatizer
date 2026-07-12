"""Generate a2_new_002.conllu — en_a2_train_201 through en_a2_train_400."""

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# 8 batches × 25 sentences = 200 (A2: sports, festivals, phone, bank, comparisons, phrasal, reflexive)
BATCHES = [
    # 201–225: Sports
    [
        "He plays tennis every Saturday morning.",
        "She runs five kilometres in the park.",
        "We watch the football match on TV.",
        "I swim in the pool three times weekly.",
        "They train hard before the big game.",
        "He scores a goal in the second half.",
        "She joins a yoga class after work.",
        "We cheer for our team at the stadium.",
        "I ride my bike along the river path.",
        "He wins the race by two seconds.",
        "She stretches her legs before the run.",
        "We give the ball to our teammate.",
        "I lose the match but I had fun.",
        "He kicks the ball into the net.",
        "She wears new trainers for the race.",
        "We practice basketball in the school gym.",
        "I catch the ball with both hands.",
        "He hurt his ankle during the match.",
        "She throws the ball to her friend.",
        "We warm up before we start running.",
        "I skip rope in the garden sometimes.",
        "He lifts weights at the sports centre.",
        "She moves faster than the other runners.",
        "We rest on the bench after the game.",
        "I join the swimming club this autumn.",
    ],
    # 226–250: Festivals
    [
        "We celebrate Christmas with our whole family.",
        "She wears a colourful dress at the festival.",
        "They dance in the street during the parade.",
        "I eat traditional food at the summer fair.",
        "He lights candles on the birthday cake.",
        "We watch fireworks by the river tonight.",
        "She learns songs with her school choir.",
        "They decorate the house with bright lights.",
        "I give presents to my friends at Easter.",
        "We drink hot chocolate at the winter market.",
        "He plays music at the village festival.",
        "She buys gifts at the Christmas market.",
        "They wear masks at the carnival party.",
        "I take photos of the street parade.",
        "We clap when the band finishes playing.",
        "He brings flowers to the spring festival.",
        "She makes lanterns for the autumn event.",
        "They share cake at the wedding party.",
        "I watch dancers in the town square.",
        "We meet friends at the music festival.",
        "He tells jokes at the family gathering.",
        "She paints her face for the parade.",
        "They eat sweets at the Halloween party.",
        "I hear drums from the festival stage.",
        "We stay up late on New Year's Eve.",
    ],
    # 251–275: Phone calls
    [
        "I call my mother every Sunday evening.",
        "She answers the phone at work quickly.",
        "He hangs up after a short conversation.",
        "We speak to the manager on the phone.",
        "I leave a message on her voicemail.",
        "She dials the number from her contact list.",
        "He talks to the doctor about his cough.",
        "We call the hotel to book a room.",
        "I wait on hold for five long minutes.",
        "She calls back after she reads the text.",
        "He asks for help on the support line.",
        "We hear a busy tone and try again.",
        "I pick up the phone in the kitchen.",
        "She speaks louder because the line is bad.",
        "He forgets to call his friend on time.",
        "We talk about the plan over the phone tonight.",
        "I save his new number in my phone.",
        "She says goodbye and ends the call.",
        "He misses an important call at lunch.",
        "We chat with grandma for an hour.",
        "I text him when he does not answer.",
        "She turns down the call during dinner.",
        "He explains the problem on the phone clearly.",
        "We arrange a meeting time by phone.",
        "I charge my phone before the long trip.",
    ],
    # 276–300: Bank
    [
        "I open a bank account at the local branch.",
        "She pays her bills at the bank today.",
        "He withdraws cash from the ATM outside.",
        "We transfer money to our savings account.",
        "I check my balance on the banking app.",
        "She deposits her salary every month.",
        "He signs the form at the bank desk.",
        "We need a PIN code for the card.",
        "I lose my bank card on the bus.",
        "She asks about the loan at the counter.",
        "He saves fifty euros each week.",
        "We pay the rent by bank transfer.",
        "I show my ID at the bank office.",
        "She changes her address at the bank.",
        "He receives a receipt after the payment.",
        "We compare rates at two different banks.",
        "I borrow money for my new laptop.",
        "She closes her old account this week.",
        "He counts the coins in his wallet.",
        "We wait in line at the busy bank.",
        "I type my password on the screen.",
        "She keeps her card in a safe place.",
        "He pays interest on his small loan.",
        "We receive a statement by email monthly.",
        "I report a lost card to the bank.",
    ],
    # 301–325: Comparisons
    [
        "This gym is bigger than the old one.",
        "She plays tennis better than her sister.",
        "The festival was louder than the concert.",
        "He is as fit as his coach now.",
        "Today feels colder than it did yesterday.",
        "My phone is newer than your phone.",
        "She speaks more clearly than he does.",
        "The bank is closer than the post office.",
        "He runs as far as the bridge every day.",
        "This team is stronger than the other team.",
        "I sleep less than my roommate does.",
        "The park is as quiet as the library.",
        "She earns more money than her colleague.",
        "Winter sports are harder than summer sports.",
        "He is younger than his older brother.",
        "The cake is sweeter than the bread.",
        "We arrive earlier than they do tonight.",
        "This street is narrower than the main road.",
        "She dances as gracefully as her teacher.",
        "The match was more exciting than the film.",
        "I feel happier than I did last week.",
        "His car is faster than my old bike.",
        "The market is as crowded as the mall.",
        "She reads faster than most of her classmates.",
        "This jacket is warmer than that thin shirt.",
    ],
    # 326–350: Separable / phrasal verbs
    [
        "I turn on the radio in the morning.",
        "She takes off her shoes at the door.",
        "He picks up the kids after school.",
        "We sit down and talk about the plan.",
        "I look up the word in my dictionary.",
        "She fills out the form at the bank.",
        "He gives up smoking after the check-up.",
        "We clean up the kitchen after dinner.",
        "I write down the address on the paper.",
        "She wakes up late on Sunday morning.",
        "He puts away his clothes in the closet.",
        "We set up the tent at the campsite.",
        "I throw away the old newspapers today.",
        "She looks after her little brother tonight.",
        "He runs out of milk before breakfast.",
        "We grow up in the same small town.",
        "I switch off the TV before bed.",
        "She hands in her homework on Monday.",
        "He carries on working after the break.",
        "We cheer up our sad friend after school.",
        "I try on the blue jacket in the shop.",
        "She calms down after the phone call ends.",
        "He checks in at the hotel reception.",
        "We work out at the gym every Tuesday.",
        "I call off the picnic because of rain.",
    ],
    # 351–375: Reflexive patterns
    [
        "I hurt myself while I was playing football.",
        "She checks herself in the mirror before the show.",
        "He blames himself for the lost match.",
        "We enjoy ourselves at the music festival.",
        "I teach myself new words every week.",
        "She buys herself a new pair of shoes.",
        "They help themselves to food at the party.",
        "He reminds himself to call the bank.",
        "I wash myself quickly before the meeting.",
        "She prepares herself for the big race.",
        "We introduce ourselves to the new neighbours.",
        "He pushes himself harder during every workout.",
        "I ask myself why the plan failed.",
        "She looks at herself in the mirror.",
        "They seat themselves near the festival stage.",
        "He finds himself alone at the bus stop.",
        "I tell myself to stay calm on the phone.",
        "She treats herself to coffee after the run.",
        "We prepare ourselves for the dance show tonight.",
        "He cuts himself while he chops vegetables.",
        "I pride myself on my good handwriting.",
        "She distances herself from the loud crowd.",
        "They organize themselves into two small teams.",
        "He challenges himself with harder exercises.",
        "I remind myself to save money each month.",
    ],
    # 376–400: Mixed A2 topics
    [
        "I join my friends for a picnic in May.",
        "She sends an email to the bank manager.",
        "We rent bikes and ride along the coast.",
        "He repairs his racket before the tennis match.",
        "I pack snacks for the long festival day.",
        "She shares her seat on the crowded bus.",
        "We plan a short trip for the spring break.",
        "He writes a short note for his teacher.",
        "I borrow a book about sports from the library.",
        "She finds a quiet spot near the river bank.",
        "We laugh together at the funny carnival masks.",
        "He orders tea and sits by the window.",
        "I return the sports equipment after the lesson.",
        "She learns a new dance for the village fair.",
        "We thank the staff for their friendly help.",
        "He keeps his festival tickets in his wallet.",
        "I meet my coach at the sports ground.",
        "She listens carefully to the phone instructions.",
        "We choose a gift at the market stall.",
        "He smiles when he sees his team win.",
        "I read the festival programme before we go.",
        "She waits patiently in the bank queue.",
        "We walk home slowly after the late concert.",
        "He turns up the volume on the radio.",
        "I feel proud after my first swimming medal.",
    ],
]

sentences = [s for batch in BATCHES for s in batch]
assert len(sentences) == 200

start_id = 201

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
}

PROPN_LEMMA_MAP = {
    "london": "London",
    "rome": "Rome",
    "spain": "Spain",
    "greece": "Greece",
    "french": "French",
    "english": "English",
    "christmas": "Christmas",
    "easter": "Easter",
    "halloween": "Halloween",
    "eve": "Eve",
    "tv": "TV",
    "atm": "ATM",
    "pin": "PIN",
    "id": "ID",
    "may": "May",
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
target_path = project_root / "data/handcraft/en/train/a2_new_002.conllu"
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

print("STATUS: OK — 200 sentences, en_a2_train_201–en_a2_train_400")