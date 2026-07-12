"""Generate a1_new_002.conllu — en_a1_train_201 through en_a1_train_400."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

START_ID = 201
BATCH_SIZE = 25

# 8 batches × 25 = 200 A1 sentences (3–8 tokens, present tense)
# Topics: animals, body, clothing, seasons, professions, emotions (+ mixed)
SENTENCE_BATCHES: list[list[str]] = [
    # 201–225: Animals
    [
        "I see a big cat.",
        "The dog is brown.",
        "She has a small bird.",
        "We like our pet fish.",
        "He gives the rabbit food.",
        "The horse is fast.",
        "I hear a loud cow.",
        "The pig is pink.",
        "She sees white sheep.",
        "The lion is strong.",
        "A tiger has stripes.",
        "The bear is big.",
        "I love little mice.",
        "The frog can jump.",
        "We see a gray duck.",
        "The chicken lays eggs.",
        "He pets the soft cat.",
        "My dog runs fast.",
        "The bird chirps now.",
        "She watches the fish.",
        "Do you like dogs?",
        "The zoo has elephants.",
        "I want a pet bird.",
        "The cat sleeps here.",
        "We visit the farm.",
    ],
    # 226–250: Body
    [
        "I have two eyes.",
        "She has blue eyes.",
        "He has brown hair.",
        "My nose is small.",
        "I open my mouth.",
        "She shows her teeth.",
        "I hurt my left hand.",
        "He uses his right arm.",
        "My leg is long.",
        "She hurts her foot.",
        "I touch my face.",
        "He moves his head.",
        "We wash our hands.",
        "She cuts her finger.",
        "My back hurts now.",
        "I feel pain today.",
        "He has strong arms.",
        "She has long legs.",
        "I close my eyes.",
        "He covers his ears.",
        "My heart beats fast.",
        "She has soft skin.",
        "I bite my lip.",
        "Where is your nose?",
        "He shows his hands.",
    ],
    # 251–275: Clothing
    [
        "I wear a red shirt.",
        "She likes her blue dress.",
        "He puts on pants.",
        "My coat is warm.",
        "She buys new shoes.",
        "I want a warm hat.",
        "He wears black socks.",
        "Her skirt is long.",
        "I find my green scarf.",
        "We wash dirty clothes.",
        "She folds the shirt.",
        "He ties his shoes.",
        "My jacket is new.",
        "I like this sweater.",
        "She wears a white blouse.",
        "He takes off coat.",
        "The gloves are small.",
        "I wear my belt.",
        "She puts on boots.",
        "His tie is blue.",
        "Where is my shirt?",
        "I want new pants.",
        "She has a pink hat.",
        "We buy winter coats.",
        "These shoes are nice.",
    ],
    # 276–300: Seasons
    [
        "Spring is very warm.",
        "Summer is hot here.",
        "Fall has cool days.",
        "Winter is very cold.",
        "I like warm spring.",
        "She loves hot summer.",
        "We play in autumn.",
        "He skis in winter.",
        "It snows in winter.",
        "The leaves fall down.",
        "Flowers grow in spring.",
        "We swim in summer.",
        "I wear coats in fall.",
        "The days are short.",
        "Spring rain is nice.",
        "Summer sun is bright.",
        "Winter ice is cold.",
        "I see autumn colors.",
        "She likes all seasons.",
        "When does spring start?",
        "Winter days are cold.",
        "Summer nights are warm.",
        "Fall wind is strong.",
        "Spring air smells good.",
        "What season is it?",
    ],
    # 301–325: Professions
    [
        "She is a good teacher.",
        "He is a young doctor.",
        "I am a new student.",
        "My mom is a nurse.",
        "He works as a cook.",
        "She is a kind nurse.",
        "The farmer has cows.",
        "A pilot flies planes.",
        "The baker makes bread.",
        "I want to be teacher.",
        "He is a police officer.",
        "She is a fast runner.",
        "The chef cooks food.",
        "My dad is a driver.",
        "A vet helps animals.",
        "He is a tall waiter.",
        "She runs like runner.",
        "The artist paints pictures.",
        "I meet the dentist.",
        "Who is your teacher?",
        "He drives a big bus.",
        "She sells fresh fruit.",
        "The guard is here.",
        "I help the doctor.",
        "What is your job?",
    ],
    # 326–350: Emotions
    [
        "I am very happy.",
        "She feels very sad.",
        "He is angry now.",
        "We are so excited.",
        "I feel tired today.",
        "She looks very scared.",
        "He is not afraid.",
        "I am a bit nervous.",
        "She feels calm now.",
        "We are very proud.",
        "He seems very bored.",
        "I feel lonely today.",
        "She is surprised now.",
        "They are very worried.",
        "I am so angry.",
        "He feels much better.",
        "She is shy today.",
        "We feel great now.",
        "Why are you sad?",
        "I am not scared.",
        "He looks very tired.",
        "She is very kind.",
        "I feel love today.",
        "Are you happy now?",
        "He is very calm.",
    ],
    # 351–375: Mixed animals + body
    [
        "The cat has green eyes.",
        "My dog hurt its leg.",
        "The bird has small feet.",
        "I wash my dog today.",
        "She pets the soft fur.",
        "The horse has long legs.",
        "Fish have no legs.",
        "The bear has big paws.",
        "I see a long tail.",
        "Cats have sharp teeth.",
        "The rabbit has long ears.",
        "A snake has no feet.",
        "My cat licks the paw.",
        "Birds use their wings.",
        "The lion has strong teeth.",
        "I touch the soft fur.",
        "Dogs wag their tails.",
        "She holds the baby duck.",
        "The frog has big eyes.",
        "He gives the hungry dog food.",
        "My hamster has small paws.",
        "The cow has a wet nose.",
        "We see many farm animals.",
        "The parrot says hello.",
        "Does your cat bite?",
    ],
    # 376–400: Mixed clothing + seasons + professions + emotions
    [
        "I wear boots in winter.",
        "She wears a hat in summer.",
        "He puts on coat in fall.",
        "We want rain coats in spring.",
        "The teacher is happy today.",
        "The doctor feels tired now.",
        "My warm scarf helps a lot.",
        "She is sad in winter.",
        "Summer clothes are light.",
        "The nurse wears white shoes.",
        "I feel cold without coat.",
        "He buys gloves for winter.",
        "The farmer wears old boots.",
        "Spring makes me happy.",
        "She is excited for summer.",
        "Winter hats are very warm.",
        "The cook wears a white hat.",
        "I wear shorts in summer.",
        "Fall makes her sad sometimes.",
        "The student is very nervous.",
        "He wears a tie to work.",
        "My red dress is for spring.",
        "The baker is proud today.",
        "Are you cold in winter?",
        "She smiles in warm spring.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

AUX_BE = {"am", "is", "are", "was", "were", "be", "been", "being"}
AUX_HAVE = {"have", "has", "had", "having"}
AUX_DO = {"do", "does", "did", "doing"}
AUX_MODALS = {"will", "would", "can", "could", "may", "might", "must", "should", "shall"}
AUX_FORMS = AUX_BE | AUX_HAVE | AUX_DO | AUX_MODALS

PROPN_NAMES = {
    "Mom", "Dad", "Tom", "Anna", "Ben", "London", "Paris", "English",
    "Spring", "Summer", "Fall", "Winter",
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
    "an": ("a", "DET"),
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
    "every": ("every", "DET"),
    "Every": ("every", "DET"),
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
    "pants": "pants",
    "clothes": "clothes",
    "shorts": "shorts",
    "boots": "boot",
    "gloves": "glove",
    "socks": "sock",
    "shoes": "shoe",
    "coats": "coat",
    "colors": "color",
    "pictures": "picture",
    "animals": "animal",
    "wings": "wing",
    "tails": "tail",
    "paws": "paw",
    "eggs": "egg",
    "stripes": "stripe",
    "elephants": "elephant",
    "eyes": "eye",
    "arms": "arm",
    "legs": "leg",
    "hands": "hand",
    "ears": "ear",
    "days": "day",
    "nights": "night",
    "hats": "hat",
    "dogs": "dog",
    "cats": "cat",
    "cows": "cow",
    "planes": "plane",
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
    "chirps": "chirp",
    "hurts": "hurt",
    "touches": "touch",
    "moves": "move",
    "covers": "cover",
    "beats": "beat",
    "bites": "bite",
    "folds": "fold",
    "ties": "tie",
    "skis": "ski",
    "snows": "snow",
    "grows": "grow",
    "swims": "swim",
    "flies": "fly",
    "sells": "sell",
    "drives": "drive",
    "looks": "look",
    "seems": "seem",
    "licks": "lick",
    "holds": "hold",
    "wags": "wag",
    "smiles": "smile",
    "smells": "smell",
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
        "without", "like", "as",
    }:
        upos = "ADP"
        lemma = lower

    if lower in {"the", "a", "an", "this", "that", "these", "those", "some", "every", "many", "all"}:
        if lower in {"my", "your", "his", "her", "its", "our", "their"}:
            upos = "PRON"
            lemma = lower
        else:
            upos = "DET"
            lemma = lower

    if lower == "not":
        upos = "PART"
        lemma = "not"

    if lower in {"so", "very", "sometimes", "down", "here", "now", "today"}:
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

    target_path = project_root / "data/handcraft/en/train/a1_new_002.conllu"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    output_lines: list[str] = []
    global_idx = 0

    for batch_num, batch in enumerate(SENTENCE_BATCHES):
        assert len(batch) == BATCH_SIZE, f"Batch {batch_num} has {len(batch)} sentences"
        print(
            f"Processing batch {batch_num + 1}/8 "
            f"(sentences {START_ID + global_idx}–{START_ID + global_idx + BATCH_SIZE - 1})"
        )
        for sent in batch:
            sent_id = f"en_a1_train_{START_ID + global_idx:03d}"
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