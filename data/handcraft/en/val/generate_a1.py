"""Generate a1.conllu — en_a1_val_001 through en_a1_val_100."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

START_ID = 1
BATCH_SIZE = 25

# 4 batches × 25 = 100 A1 validation sentences (3–8 tokens, present tense)
# Topics: farm, post office, zoo, camping — distinct from train data
SENTENCE_BATCHES: list[list[str]] = [
    # 001–025: Farm & countryside
    [
        "I see the big barn.",
        "The cow eats grass.",
        "We give hay to horses.",
        "He milks the cow.",
        "The farm is quiet.",
        "She picks fresh eggs.",
        "The hen is brown.",
        "I like farm life.",
        "We ride the tractor.",
        "The field is green.",
        "He chops the wood.",
        "The pig is fat.",
        "We grow corn here.",
        "She carries a bucket.",
        "The barn door is open.",
        "I hear the rooster.",
        "The sheep eat hay.",
        "He fixes the fence.",
        "We watch the ducks.",
        "The goat is small.",
        "I water the crops.",
        "She sells fresh milk.",
        "The path is muddy.",
        "Do you like farms?",
        "He drives the truck.",
    ],
    # 026–050: Post office & mail
    [
        "I mail this letter.",
        "She buys a stamp.",
        "The box is heavy.",
        "We send a package.",
        "He writes the address.",
        "The mail is here.",
        "I wait in line.",
        "She seals the envelope.",
        "The stamp is blue.",
        "We pick up mail.",
        "He signs the form.",
        "The clerk is busy.",
        "I want more stamps.",
        "She posts the card.",
        "The line is long.",
        "We weigh the box.",
        "He tapes the package.",
        "Is the mail late?",
        "I get a parcel.",
        "She reads the notice.",
        "The office is open.",
        "We pay the fee.",
        "He asks for help.",
        "The letter is thin.",
        "Can you mail this?",
    ],
    # 051–075: Zoo & wildlife park
    [
        "I see a big lion.",
        "The tiger is wild.",
        "We watch the bears.",
        "She gives fish to penguins.",
        "The zoo is fun.",
        "He likes the monkeys.",
        "The cage is strong.",
        "We hear loud roars.",
        "I take zoo photos.",
        "The giraffe is tall.",
        "She points at seals.",
        "The elephant is huge.",
        "We buy zoo tickets.",
        "He waves at apes.",
        "The pond has ducks.",
        "I read the sign.",
        "The zebra has stripes.",
        "She loves the pandas.",
        "The path goes left.",
        "We rest on bench.",
        "He spots a hippo.",
        "The birds are free.",
        "I buy zoo snacks.",
        "She maps the zoo.",
        "Where is the bear?",
    ],
    # 076–100: Camping & hiking
    [
        "We set the tent.",
        "The fire is warm.",
        "I pack my bag.",
        "She sleeps in tent.",
        "The trail is long.",
        "He builds a fire.",
        "We cook on fire.",
        "The creek is clear.",
        "I find dry wood.",
        "She ties the rope.",
        "The moon is bright.",
        "We tell camp stories.",
        "He wears warm boots.",
        "The night is cold.",
        "I boil creek water.",
        "She checks the map.",
        "The path is steep.",
        "We walk the trail.",
        "He lights the lamp.",
        "The bag is light.",
        "I hear wild owls.",
        "She shares warm soup.",
        "The tent is dry.",
        "Are we near camp?",
        "He rolls his mat.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 100, f"Expected 100 sentences, got {len(SENTENCES)}"

AUX_BE = {"am", "is", "are", "was", "were", "be", "been", "being"}
AUX_HAVE = {"have", "has", "had", "having"}
AUX_DO = {"do", "does", "did", "doing"}
AUX_MODALS = {"will", "would", "can", "could", "may", "might", "must", "should", "shall"}
AUX_FORMS = AUX_BE | AUX_HAVE | AUX_DO | AUX_MODALS

PROPN_NAMES = {
    "Mom", "Dad", "Tom", "Anna", "Ben", "London", "Paris", "English",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "June", "July", "January", "TV",
}

SPECIAL_LEMMAS: dict[str, tuple[str, str]] = {
    "I": ("I", "PRON"),
    "i": ("I", "PRON"),
    "me": ("I", "PRON"),
    "my": ("my", "PRON"),
    "My": ("my", "PRON"),
    "mine": ("my", "PRON"),
    "you": ("you", "PRON"),
    "You": ("you", "PRON"),
    "your": ("your", "PRON"),
    "Your": ("your", "PRON"),
    "yours": ("your", "PRON"),
    "he": ("he", "PRON"),
    "He": ("he", "PRON"),
    "him": ("he", "PRON"),
    "his": ("his", "PRON"),
    "His": ("his", "PRON"),
    "she": ("she", "PRON"),
    "She": ("she", "PRON"),
    "her": ("her", "PRON"),
    "Her": ("her", "PRON"),
    "hers": ("her", "PRON"),
    "it": ("it", "PRON"),
    "It": ("it", "PRON"),
    "its": ("its", "PRON"),
    "we": ("we", "PRON"),
    "We": ("we", "PRON"),
    "us": ("we", "PRON"),
    "our": ("our", "PRON"),
    "Our": ("our", "PRON"),
    "ours": ("our", "PRON"),
    "they": ("they", "PRON"),
    "They": ("they", "PRON"),
    "them": ("they", "PRON"),
    "their": ("their", "PRON"),
    "Their": ("their", "PRON"),
    "theirs": ("their", "PRON"),
    "the": ("the", "DET"),
    "The": ("the", "DET"),
    "a": ("a", "DET"),
    "A": ("a", "DET"),
    "an": ("an", "DET"),
    "An": ("an", "DET"),
    "this": ("this", "DET"),
    "This": ("this", "DET"),
    "that": ("that", "DET"),
    "That": ("that", "DET"),
    "these": ("this", "DET"),
    "These": ("this", "DET"),
    "those": ("that", "DET"),
    "Those": ("that", "DET"),
    "some": ("some", "DET"),
    "Some": ("some", "DET"),
    "many": ("many", "DET"),
    "Many": ("many", "DET"),
    "more": ("more", "ADJ"),
    "More": ("more", "ADJ"),
    "one": ("one", "NUM"),
    "One": ("one", "NUM"),
    "what": ("what", "PRON"),
    "What": ("what", "PRON"),
    "who": ("who", "PRON"),
    "Who": ("who", "PRON"),
    "where": ("where", "ADV"),
    "Where": ("where", "ADV"),
    "when": ("when", "ADV"),
    "When": ("when", "ADV"),
    "why": ("why", "ADV"),
    "Why": ("why", "ADV"),
    "how": ("how", "ADV"),
    "How": ("how", "ADV"),
    "not": ("not", "PART"),
    "please": ("please", "INTJ"),
    "Please": ("please", "INTJ"),
    "thanks": ("thanks", "NOUN"),
    "Thanks": ("thanks", "NOUN"),
    "hello": ("hello", "INTJ"),
    "Hello": ("hello", "INTJ"),
    "hi": ("hi", "INTJ"),
    "Hi": ("hi", "INTJ"),
    "goodbye": ("goodbye", "INTJ"),
    "Goodbye": ("goodbye", "INTJ"),
    "good": ("good", "ADJ"),
    "Good": ("good", "ADJ"),
    "nice": ("nice", "ADJ"),
    "Nice": ("nice", "ADJ"),
    "very": ("very", "ADV"),
    "Very": ("very", "ADV"),
    "today": ("today", "ADV"),
    "Today": ("today", "ADV"),
    "now": ("now", "ADV"),
    "Now": ("now", "ADV"),
    "here": ("here", "ADV"),
    "Here": ("here", "ADV"),
    "there": ("there", "ADV"),
    "There": ("there", "ADV"),
    "tonight": ("tonight", "ADV"),
    "Tonight": ("tonight", "ADV"),
    "early": ("early", "ADV"),
    "Early": ("early", "ADV"),
    "late": ("late", "ADV"),
    "Late": ("late", "ADV"),
    "always": ("always", "ADV"),
    "Always": ("always", "ADV"),
    "never": ("never", "ADV"),
    "Never": ("never", "ADV"),
    "every": ("every", "DET"),
    "Every": ("every", "DET"),
    "each": ("each", "DET"),
    "Each": ("each", "DET"),
    "to": ("to", "ADP"),
    "To": ("to", "ADP"),
    "of": ("of", "ADP"),
    "Of": ("of", "ADP"),
    "in": ("in", "ADP"),
    "In": ("in", "ADP"),
    "on": ("on", "ADP"),
    "On": ("on", "ADP"),
    "at": ("at", "ADP"),
    "At": ("at", "ADP"),
    "for": ("for", "ADP"),
    "For": ("for", "ADP"),
    "with": ("with", "ADP"),
    "With": ("with", "ADP"),
    "by": ("by", "ADP"),
    "By": ("by", "ADP"),
    "from": ("from", "ADP"),
    "From": ("from", "ADP"),
    "up": ("up", "ADP"),
    "Up": ("up", "ADP"),
    "after": ("after", "ADP"),
    "After": ("after", "ADP"),
    "and": ("and", "CCONJ"),
    "And": ("and", "CCONJ"),
    "or": ("or", "CCONJ"),
    "Or": ("or", "CCONJ"),
    "but": ("but", "CCONJ"),
    "But": ("but", "CCONJ"),
    "do": ("do", "AUX"),
    "Do": ("do", "AUX"),
    "does": ("do", "AUX"),
    "Does": ("do", "AUX"),
    "is": ("be", "AUX"),
    "Is": ("be", "AUX"),
    "are": ("be", "AUX"),
    "Are": ("be", "AUX"),
    "am": ("be", "AUX"),
    "Am": ("be", "AUX"),
    "was": ("be", "AUX"),
    "were": ("be", "AUX"),
    "be": ("be", "AUX"),
    "Be": ("be", "AUX"),
    "been": ("be", "AUX"),
    "being": ("be", "AUX"),
    "have": ("have", "AUX"),
    "Have": ("have", "AUX"),
    "has": ("have", "AUX"),
    "Has": ("have", "AUX"),
    "had": ("have", "AUX"),
    "can": ("can", "AUX"),
    "Can": ("can", "AUX"),
    "will": ("will", "AUX"),
    "Will": ("will", "AUX"),
    "may": ("may", "AUX"),
    "May": ("may", "AUX"),
    "must": ("must", "AUX"),
    "Must": ("must", "AUX"),
    "should": ("should", "AUX"),
    "Should": ("should", "AUX"),
    "so": ("so", "ADV"),
    "So": ("so", "ADV"),
    "sometimes": ("sometimes", "ADV"),
    "Sometimes": ("sometimes", "ADV"),
    "without": ("without", "ADP"),
    "Without": ("without", "ADP"),
    "down": ("down", "ADV"),
    "Down": ("down", "ADV"),
    "all": ("all", "DET"),
    "All": ("all", "DET"),
    "bit": ("bit", "NOUN"),
    "Bit": ("bit", "NOUN"),
    "like": ("like", "ADP"),
    "Like": ("like", "ADP"),
    "long": ("long", "ADJ"),
    "Long": ("long", "ADJ"),
    "near": ("near", "ADP"),
    "Near": ("near", "ADP"),
    "wild": ("wild", "ADJ"),
    "Wild": ("wild", "ADJ"),
    "dry": ("dry", "ADJ"),
    "Dry": ("dry", "ADJ"),
    "warm": ("warm", "ADJ"),
    "Warm": ("warm", "ADJ"),
    "cold": ("cold", "ADJ"),
    "Cold": ("cold", "ADJ"),
    "loud": ("loud", "ADJ"),
    "Loud": ("loud", "ADJ"),
    "light": ("light", "ADJ"),
    "Light": ("light", "ADJ"),
    "thin": ("thin", "ADJ"),
    "Thin": ("thin", "ADJ"),
    "steep": ("steep", "ADJ"),
    "Steep": ("steep", "ADJ"),
    "muddy": ("muddy", "ADJ"),
    "Muddy": ("muddy", "ADJ"),
    "clear": ("clear", "ADJ"),
    "Clear": ("clear", "ADJ"),
    "bright": ("bright", "ADJ"),
    "Bright": ("bright", "ADJ"),
    "strong": ("strong", "ADJ"),
    "Strong": ("strong", "ADJ"),
    "huge": ("huge", "ADJ"),
    "Huge": ("huge", "ADJ"),
    "fat": ("fat", "ADJ"),
    "Fat": ("fat", "ADJ"),
    "busy": ("busy", "ADJ"),
    "Busy": ("busy", "ADJ"),
    "free": ("free", "ADJ"),
    "Free": ("free", "ADJ"),
    "left": ("left", "ADJ"),
    "Left": ("left", "ADJ"),
}

NOUN_LEMMA_MAP = {
    "children": "child",
    "mice": "mouse",
    "feet": "foot",
    "teeth": "tooth",
    "geese": "goose",
    "men": "man",
    "women": "woman",
    "people": "person",
    "leaves": "leaf",
    "sheep": "sheep",
    "fish": "fish",
    "horses": "horse",
    "hens": "hen",
    "pigs": "pig",
    "ducks": "duck",
    "goats": "goat",
    "buckets": "bucket",
    "fences": "fence",
    "roosters": "rooster",
    "crops": "crop",
    "stamps": "stamp",
    "parcels": "parcel",
    "notices": "notice",
    "envelopes": "envelope",
    "penguins": "penguin",
    "monkeys": "monkey",
    "giraffes": "giraffe",
    "elephants": "elephant",
    "apes": "ape",
    "hippos": "hippo",
    "zebras": "zebra",
    "pandas": "panda",
    "bears": "bear",
    "seals": "seal",
    "birds": "bird",
    "snacks": "snack",
    "tents": "tent",
    "trails": "trail",
    "owls": "owl",
    "mats": "mat",
    "stories": "story",
    "boots": "boot",
    "photos": "photo",
    "tickets": "ticket",
    "roars": "roar",
    "books": "book",
    "words": "word",
    "friends": "friend",
    "trees": "tree",
    "flowers": "flower",
    "animals": "animal",
    "stripes": "stripe",
}

VERB_LEMMA_MAP = {
    "says": "say",
    "goes": "go",
    "does": "do",
    "has": "have",
    "is": "be",
    "are": "be",
    "am": "be",
    "was": "be",
    "were": "be",
    "gets": "get",
    "makes": "make",
    "takes": "take",
    "wakes": "wake",
    "brushes": "brush",
    "washes": "wash",
    "likes": "like",
    "lives": "live",
    "loves": "love",
    "hugs": "hug",
    "eats": "eat",
    "drinks": "drink",
    "cooks": "cook",
    "buys": "buy",
    "wants": "want",
    "needs": "need",
    "reads": "read",
    "writes": "write",
    "watches": "watch",
    "listens": "listen",
    "cleans": "clean",
    "opens": "open",
    "closes": "close",
    "sits": "sit",
    "stands": "stand",
    "runs": "run",
    "plays": "play",
    "sleeps": "sleep",
    "works": "work",
    "studies": "study",
    "waits": "wait",
    "meets": "meet",
    "talks": "talk",
    "learns": "learn",
    "speaks": "speak",
    "starts": "start",
    "feels": "feel",
    "paints": "paint",
    "cuts": "cut",
    "tastes": "taste",
    "shares": "share",
    "finds": "find",
    "counts": "count",
    "calls": "call",
    "helps": "help",
    "visits": "visit",
    "misses": "miss",
    "sees": "see",
    "wears": "wear",
    "puts": "put",
    "rains": "rain",
    "gives": "give",
    "lays": "lay",
    "pets": "pet",
    "hurts": "hurt",
    "touches": "touch",
    "moves": "move",
    "covers": "cover",
    "beats": "beat",
    "bites": "bite",
    "folds": "fold",
    "ties": "tie",
    "grows": "grow",
    "flies": "fly",
    "sells": "sell",
    "drives": "drive",
    "looks": "look",
    "seems": "seem",
    "holds": "hold",
    "smiles": "smile",
    "smells": "smell",
    "kicks": "kick",
    "throws": "throw",
    "wins": "win",
    "loses": "lose",
    "jumps": "jump",
    "rides": "ride",
    "catches": "catch",
    "scores": "score",
    "climbs": "climb",
    "dances": "dance",
    "passes": "pass",
    "walks": "walk",
    "shops": "shop",
    "crosses": "cross",
    "maps": "map",
    "comes": "come",
    "plans": "plan",
    "leaves": "leave",
    "picks": "pick",
    "sings": "sing",
    "hikes": "hike",
    "digs": "dig",
    "plants": "plant",
    "boils": "boil",
    "mixes": "mix",
    "stirs": "stir",
    "adds": "add",
    "bakes": "bake",
    "sets": "set",
    "pours": "pour",
    "chops": "chop",
    "peels": "peel",
    "serves": "serve",
    "grills": "grill",
    "heats": "heat",
    "rings": "ring",
    "draws": "draw",
    "uses": "use",
    "asks": "ask",
    "raises": "raise",
    "copies": "copy",
    "checks": "check",
    "forgets": "forget",
    "wipes": "wipe",
    "fixes": "fix",
    "locks": "lock",
    "turns": "turn",
    "joins": "join",
    "boards": "board",
    "parks": "park",
    "packs": "pack",
    "tries": "try",
    "pays": "pay",
    "saves": "save",
    "invites": "invite",
    "laughs": "laugh",
    "waves": "wave",
    "trusts": "trust",
    "brings": "bring",
    "signs": "sign",
    "types": "type",
    "finishes": "finish",
    "builds": "build",
    "orders": "order",
    "tips": "tip",
    "wraps": "wrap",
    "decorates": "decorate",
    "claps": "clap",
    "snaps": "snap",
    "blows": "blow",
    "skips": "skip",
    "lights": "light",
    "feeds": "feed",
    "drops": "drop",
    "milks": "milk",
    "carries": "carry",
    "hears": "hear",
    "waters": "water",
    "mails": "mail",
    "sends": "send",
    "seals": "seal",
    "posts": "post",
    "weighs": "weigh",
    "tapes": "tape",
    "points": "point",
    "spots": "spot",
    "rolls": "roll",
    "rests": "rest",
    "goes": "go",
}


def _strip_trailing_punct(text: str) -> str:
    return re.sub(r"[.,;:!?]+$", "", text)


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    if lemma and _strip_trailing_punct(lemma) != lemma and _strip_trailing_punct(form) == form:
        lemma = _strip_trailing_punct(lemma)

    if form in SPECIAL_LEMMAS:
        lemma, upos = SPECIAL_LEMMAS[form]

    if form in PROPN_NAMES:
        upos = "PROPN"
        lemma = form

    lower = form.lower()

    if lower in AUX_FORMS:
        upos = "AUX"
        if lower in AUX_BE:
            lemma = "be"
        elif lower in AUX_HAVE:
            lemma = "have"
        elif lower in AUX_DO:
            lemma = "do"
        elif lower in {"will", "would"}:
            lemma = "will"
        elif lower in {"can", "could"}:
            lemma = "can"
        elif lower in {"may", "might"}:
            lemma = "may"
        elif lower == "must":
            lemma = "must"
        elif lower in {"should", "shall"}:
            lemma = "should"
    elif upos in {"VERB", "AUX"}:
        if lower in AUX_FORMS:
            pass
        else:
            upos = "VERB"
            lemma = lemma.lower() if lemma else lower
            if form in VERB_LEMMA_MAP:
                lemma = VERB_LEMMA_MAP[form]
            elif lemma in VERB_LEMMA_MAP:
                lemma = VERB_LEMMA_MAP[lemma]

    if upos == "NOUN":
        if form in NOUN_LEMMA_MAP:
            lemma = NOUN_LEMMA_MAP[form]
        elif lemma:
            lemma = lemma.lower()

    if upos == "ADJ":
        lemma = lemma.lower() if lemma else lower
        if form in SPECIAL_LEMMAS:
            lemma, _ = SPECIAL_LEMMAS[form]

    if upos == "PROPN" and lemma:
        lemma = lemma[0].upper() + lemma[1:] if lemma else form

    if upos == "NUM":
        lemma = form.lower() if form.isdigit() else form

    if upos == "PUNCT":
        lemma = form

    if lower in {"my", "your", "his", "her", "its", "our", "their"}:
        upos = "PRON"
        lemma = lower

    if lower in {"what", "who", "whom", "which"} and upos in {"DET", "PRON"}:
        lemma = lower
        upos = "PRON"

    if lower in {"where", "when", "why", "how"}:
        upos = "ADV"
        lemma = lower

    if lower in {"and", "or", "but"}:
        upos = "CCONJ"
        lemma = lower

    if lower in {
        "in", "on", "at", "to", "for", "with", "by", "from", "of", "up",
        "into", "through", "over", "under", "about", "after", "before",
        "without", "like", "as", "near",
    }:
        upos = "ADP"
        lemma = lower

    if lower in {
        "the", "a", "an", "this", "that", "these", "those", "some", "every",
        "many", "all", "each",
    }:
        if lower in {"my", "your", "his", "her", "its", "our", "their"}:
            upos = "PRON"
            lemma = lower
        else:
            upos = "DET"
            lemma = lower

    if lower == "not":
        upos = "PART"
        lemma = "not"

    if lower in {
        "so", "very", "sometimes", "down", "here", "now", "today", "always",
        "never", "early", "late",
    }:
        upos = "ADV"
        lemma = lower

    if form == "I":
        upos = "PRON"
        lemma = "I"

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
        stanza_form = _strip_trailing_punct(head.text)
        stanza_lemma = head.lemma or stanza_form
        if stanza_form.lower() == form.lower() or form in PROPN_NAMES:
            lemma, upos = normalize_token(form, head.upos or "X", stanza_lemma)
        else:
            lemma, upos = normalize_token(form, "X", form)
        aligned.append((form, lemma, upos))
        wi += 1

    return aligned


def count_tokens(sentence: str) -> int:
    return len(tokenize_text(sentence))


for i, sent in enumerate(SENTENCES, start=START_ID):
    n = count_tokens(sent)
    if n < 3 or n > 8:
        raise ValueError(f"Sentence {i} has {n} tokens (need 3–8): {sent}")


def main() -> None:
    import stanza

    print("Loading Stanza...")
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,lemma",
        use_gpu=False,
        verbose=False,
    )

    target_path = project_root / "data/handcraft/en/val/a1.conllu"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    output_lines: list[str] = []
    global_idx = 0

    for batch_num, batch in enumerate(SENTENCE_BATCHES):
        assert len(batch) == BATCH_SIZE, f"Batch {batch_num} has {len(batch)} sentences"
        print(
            f"Processing batch {batch_num + 1}/4 "
            f"(sentences {START_ID + global_idx}–{START_ID + global_idx + BATCH_SIZE - 1})"
        )
        for sent in batch:
            sent_id = f"en_a1_val_{START_ID + global_idx:03d}"
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
            global_idx += 1

    conllu_text = "\n".join(output_lines) + "\n"
    target_path.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {global_idx} sentences to {target_path}")

    validation_res = validate_text(conllu_text)
    print(f"Validate: count={validation_res.sentence_count}, passed={validation_res.passed}")
    if not validation_res.passed:
        for err in validation_res.errors:
            print(f"  {err}")
        sys.exit(1)

    lemma_res = check_text(conllu_text, lang="en")
    print(f"Lemma check: passed={lemma_res.passed}")
    if not lemma_res.passed:
        for err in lemma_res.errors:
            print(f"  {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()