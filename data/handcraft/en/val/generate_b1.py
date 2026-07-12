"""Generate b1.conllu (en_b1_val_001–100) for English B1 validation."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import EN_DET_LEMMAS, EN_IRREGULAR_PLURALS

# 100 B1 validation sentences (8–15 tokens, mixed B1 structures)
SENTENCES: list[str] = [
    # 001–020: daily life and housing
    "I would wake up earlier if I did not stay up so late every evening.",
    "The apartment that we viewed yesterday is much brighter than our current flat.",
    "Without a reservation, you may have to wait a long time for a table.",
    "He decided to move into a smaller house after his children left for university.",
    "We have to pay the rent before the first day of each month.",
    "The bedroom was painted by our neighbors while we were away on holiday.",
    "I am looking forward to meeting my new roommate next weekend.",
    "If you require help with the move, please call me anytime.",
    "Despite the rain, we went for a walk as we had planned.",
    "She forgot to take out the rubbish before the collection truck arrived.",
    "The key that you lent me is still lying on the kitchen table.",
    "We complained to the landlord about the rising cost of heating bills.",
    "I could not attend the party because I had a dentist appointment.",
    "The dinner was prepared by my aunt with fresh vegetables from her garden.",
    "Although the kitchen is small, we enjoy cooking together at weekends.",
    "He wants us to postpone the housewarming until next month.",
    "Because of the construction work, we had to find another route home.",
    "I am very interested in sustainable living in modern apartment buildings.",
    "The lamp that I bought last week stopped working during the storm.",
    "If we leave earlier, we will catch the last train home.",
    # 021–040: food and restaurants
    "The menu was changed because several dishes were too expensive for students.",
    "If customers order online, they can often collect their meals more quickly.",
    "Many diners believe that restaurants should label allergens more clearly.",
    "The recipe was shared by a chef who specializes in Mediterranean cuisine.",
    "Although the soup was spicy, most guests said they enjoyed the flavor.",
    "New tasting menus were introduced to attract younger visitors to the bistro.",
    "I think cafés should offer reusable cups for regular morning customers.",
    "The waiter explained that desserts would be served after the main courses.",
    "If ingredients are not fresh, the quality of the meal may disappoint guests.",
    "The bakery was renovated while staff continued serving customers outside.",
    "Many people feel that food waste should be reduced in school canteens.",
    "The reservation was confirmed after the manager checked the booking system.",
    "She said she would try the vegetarian option if it contained enough protein.",
    "Cooking classes were organized by volunteers who wanted to teach basic skills.",
    "We talked about whether home cooked meals are healthier than fast food.",
    "The oven was repaired when the technician discovered a faulty heating element.",
    "If spices are stored incorrectly, they may lose their aroma within months.",
    "I believe local markets should support farmers who grow seasonal produce.",
    "The complaint was investigated after customers reported poor hygiene standards.",
    "Although the restaurant is busy, the staff always greet regular guests warmly.",
    # 041–060: sports and leisure
    "The match was postponed because the pitch was flooded after heavy rain.",
    "If athletes warm up properly, they are less likely to suffer serious injuries.",
    "Many fans believe that ticket prices should be lower for young supporters.",
    "The trophy was won by a team that had trained together for several years.",
    "Although the hike was tiring, we reached the summit before sunset.",
    "New cycling paths were built to encourage residents to leave their cars at home.",
    "I think swimming pools should offer affordable lessons for adult beginners.",
    "The coach explained that practice sessions would begin after the school term ends.",
    "If equipment is not maintained, accidents may happen during team sports.",
    "The gym was expanded while members continued to work out in temporary studios.",
    "Many players feel that referees should explain controversial decisions more clearly.",
    "The membership was renewed after the club offered a discount for families.",
    "She said she would join the running group if her knee injury healed completely.",
    "Tennis lessons were arranged by parents who wanted children to stay active.",
    "We talked about whether competitive sports put too much pressure on teenagers.",
    "The stadium was opened when the mayor cut a ribbon at the entrance gate.",
    "If weather forecasts are wrong, outdoor events may be canceled at short notice.",
    "I believe community clubs should welcome members of all ages and abilities.",
    "The tournament was streamed online after organizers installed new broadcast equipment.",
    "Although the course is challenging, most participants complete it within six weeks.",
    # 061–080: travel and transport
    "The flight was delayed because fog reduced visibility at the regional airport.",
    "If travelers book tickets early, they can often save money on long journeys.",
    "Many passengers believe that luggage allowances should be explained more clearly.",
    "The hotel was recommended by a guidebook that deals with budget accommodation.",
    "Although the journey was long, we arrived safely before midnight.",
    "New bus routes were introduced to connect suburbs with the city center.",
    "I think railway stations should provide better information for disabled travelers.",
    "The driver explained that the service would resume after the track repair.",
    "If passports expire soon, applicants may have to request urgent replacements.",
    "The bridge was closed while engineers inspected damage caused by flooding.",
    "Many tourists feel that travel insurance should cover medical emergencies abroad.",
    "The booking was changed after the airline offered an alternative departure time.",
    "She said she would visit the museum if her guided tour was confirmed.",
    "Maps were distributed by volunteers who wanted to help visitors find landmarks.",
    "We talked about whether car sharing can reduce traffic in crowded cities.",
    "The ferry was cancelled when strong winds made the crossing too dangerous.",
    "If luggage is lost, travelers may wait days before their bags are returned.",
    "I believe public transport should be reliable during evening and weekend hours.",
    "The complaint was filed after several commuters failed to catch important connections.",
    "Although the hostel is basic, backpackers praised its friendly atmosphere.",
    # 081–100: health and wellness
    "The clinic was expanded because the waiting list had grown during the winter.",
    "If patients rest properly, they often recover more quickly from minor illnesses.",
    "Many residents believe that mental health services should be easier to obtain.",
    "The prescription was issued by a doctor who specializes in chronic conditions.",
    "Although the treatment was uncomfortable, most patients reported positive results.",
    "New wellness programs were launched to help employees manage workplace stress.",
    "I think pharmacies should offer clear advice about common over the counter medicines.",
    "The nurse explained that test results would be available within three working days.",
    "If symptoms persist, patients should contact their doctor without delay.",
    "The dental surgery was renovated while emergencies were handled in a nearby unit.",
    "Many families feel that vaccination reminders should be sent before school starts.",
    "The appointment was rescheduled after the specialist was called to an emergency.",
    "She said she would try yoga classes if the schedule fit her working hours.",
    "Support groups were formed by patients who wanted to share practical coping strategies.",
    "We talked about whether screen time affects sleep quality among young adults.",
    "The physiotherapy session was postponed when the therapist caught a seasonal illness.",
    "If exercise is neglected, people may experience stiffness and low energy levels.",
    "I believe hospitals should communicate clearly with relatives during long treatments.",
    "The health survey was published after researchers removed all personal identifiers.",
    "Although recovery takes time, most volunteers said the rehabilitation program was worthwhile.",
]

assert len(SENTENCES) == 100, f"Expected 100 sentences, got {len(SENTENCES)}"

START_ID = 1
SENT_ID_PREFIX = "en_b1_val"

AUX_LEMMAS = {"be", "have", "do", "will", "can", "may", "might", "shall", "should", "must"}
BE_FORMS = {
    "am", "is", "are", "was", "were", "be", "been", "being",
}
HAVE_FORMS = {"have", "has", "had", "having"}
DO_FORMS = {"do", "does", "did", "doing"}
WILL_FORMS = {"will", "would", "'ll", "wo"}
CAN_FORMS = {"can", "could", "'d"}
MAY_FORMS = {"may", "might"}
SHALL_FORMS = {"shall", "should"}
MUST_FORMS = {"must"}

SPECIAL_LEMMAS: dict[str, tuple[str, str]] = {
    "I": ("I", "PRON"),
    "me": ("I", "PRON"),
    "my": ("my", "PRON"),
    "mine": ("my", "PRON"),
    "you": ("you", "PRON"),
    "your": ("your", "PRON"),
    "yours": ("your", "PRON"),
    "he": ("he", "PRON"),
    "him": ("he", "PRON"),
    "his": ("his", "PRON"),
    "she": ("she", "PRON"),
    "her": ("her", "PRON"),
    "hers": ("her", "PRON"),
    "it": ("it", "PRON"),
    "its": ("its", "PRON"),
    "we": ("we", "PRON"),
    "us": ("we", "PRON"),
    "our": ("our", "PRON"),
    "ours": ("our", "PRON"),
    "they": ("they", "PRON"),
    "them": ("they", "PRON"),
    "their": ("their", "PRON"),
    "theirs": ("their", "PRON"),
    "the": ("the", "DET"),
    "a": ("a", "DET"),
    "an": ("a", "DET"),
    "this": ("this", "DET"),
    "that": ("that", "DET"),
    "these": ("this", "DET"),
    "those": ("that", "DET"),
    "what": ("what", "DET"),
    "which": ("which", "DET"),
    "whose": ("whose", "DET"),
    "please": ("please", "INTJ"),
    "not": ("not", "PART"),
    "n't": ("not", "PART"),
    "to": ("to", "PART"),
    "o'clock": ("o'clock", "NOUN"),
}

SCONJS = {
    "because", "that", "if", "when", "while", "although", "though", "unless",
    "since", "before", "after", "until", "whether", "as", "than", "so",
}
CCONJS = {"and", "or", "but", "nor", "yet", "so"}
ADPS = {
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about", "into",
    "over", "under", "between", "through", "during", "without", "against",
    "among", "within", "across", "off", "up", "down", "out", "around", "via",
    "per", "near", "behind", "beyond", "despite", "toward", "towards", "upon",
}
PARTS = {"not", "to"}


def _aux_lemma(form: str) -> str:
    lower = form.lower()
    if lower in BE_FORMS:
        return "be"
    if lower in HAVE_FORMS:
        return "have"
    if lower in DO_FORMS:
        return "do"
    if lower in WILL_FORMS:
        return "will"
    if lower in CAN_FORMS:
        return "can"
    if lower in MAY_FORMS:
        return "may"
    if lower in SHALL_FORMS:
        return "should" if lower == "should" else "shall"
    if lower in MUST_FORMS:
        return "must"
    return lower


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    """Apply handcraft lemma/UPOS conventions per en_test.conllu."""
    if form in SPECIAL_LEMMAS:
        lemma, upos = SPECIAL_LEMMAS[form]

    lower = form.lower()

    if lower in BE_FORMS | HAVE_FORMS | DO_FORMS | WILL_FORMS | CAN_FORMS | MAY_FORMS | SHALL_FORMS | MUST_FORMS:
        if lower == "been" and upos == "VERB":
            lemma = "be"
        else:
            upos = "AUX"
            lemma = _aux_lemma(form)
    elif upos == "AUX" and lemma not in AUX_LEMMAS:
        lemma = lemma.lower()

    if upos == "VERB":
        lemma = lemma.lower()
        if lemma.endswith("ing") and len(lemma) > 5:
            base = lemma[:-3]
            if base.endswith(("t", "d")):
                base = base + "e"
            lemma = base
        elif lemma.endswith("ed") and len(lemma) > 4 and form.lower().endswith("ed"):
            base = lemma[:-2]
            if base.endswith(("t", "d")):
                base = base + "e"
            lemma = base

    if upos == "NOUN" and lower in EN_IRREGULAR_PLURALS:
        lemma = EN_IRREGULAR_PLURALS[lower]

    if upos == "NOUN" and lemma:
        lemma = lemma.lower()
        if lemma.endswith("s") and lower.endswith("s") and lower not in EN_IRREGULAR_PLURALS:
            if not lemma.endswith("ss") and len(lemma) > 3:
                lemma = lemma[:-1]
                if lemma.endswith("ie"):
                    lemma = lemma[:-2] + "y"
                elif lemma.endswith("ve"):
                    lemma = lemma[:-2] + "f"

    if upos == "ADJ":
        lemma = lemma.lower()
        if lemma.endswith("er") and len(lemma) > 4:
            base = lemma[:-2]
            if lower.endswith("ier"):
                lemma = base + "y"
            elif lower.endswith(lower[:-2] + "er"):
                lemma = base
        if lemma.endswith("est") and len(lemma) > 5:
            lemma = lemma[:-3]
            if lemma.endswith("i"):
                lemma = lemma[:-1] + "y"

    if upos == "PROPN" and lemma:
        lemma = lemma[0].upper() + lemma[1:]

    if upos == "PUNCT":
        lemma = form

    if upos == "DET" and lower in EN_DET_LEMMAS:
        lemma = EN_DET_LEMMAS[lower]

    if lower in SCONJS:
        if lower not in {"before", "after", "since", "until"} or upos != "ADP":
            upos = "SCONJ"
            lemma = lower
    elif lower in CCONJS:
        upos = "CCONJ"
        lemma = lower
    elif lower in ADPS:
        if not (lower == "to" and upos == "PART"):
            upos = "ADP"
            lemma = lower
    elif lower in PARTS:
        if lower == "to" and upos == "PART":
            pass
        else:
            upos = "PART"
            lemma = lower

    if lower == "there" and upos in {"ADV", "PRON"}:
        upos = "PRON"
        lemma = "there"

    if lower == "more" and upos == "ADJ":
        upos = "ADV"
        lemma = "more"

    if lower == "most" and upos == "ADV":
        lemma = "most"

    return lemma, upos


def sentence_to_conllu(sent_id: str, sent: str, nlp) -> list[str]:
    lines: list[str] = [f"# sent_id = {sent_id}", f"# text = {sent}"]
    doc = nlp(sent)
    token_counter = 1
    for stanza_sent in doc.sentences:
        for word in stanza_sent.words:
            if not isinstance(word.id, int):
                continue
            form = word.text
            upos = word.upos or "X"
            lemma = word.lemma if word.lemma else form
            lemma, upos = normalize_token(form, upos, lemma)
            cols = [
                str(token_counter),
                form,
                lemma,
                upos,
                "_", "_", "_", "_", "_", "_",
            ]
            lines.append("\t".join(cols))
            token_counter += 1
    lines.append("")
    return lines


def main() -> None:
    import stanza

    print("Loading Stanza...")
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma",
        use_gpu=False,
        verbose=False,
    )

    target_path = project_root / "data/handcraft/en/val/b1.conllu"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    all_lines: list[str] = []
    for idx, sent in enumerate(SENTENCES):
        sent_id = f"{SENT_ID_PREFIX}_{START_ID + idx:03d}"
        all_lines.extend(sentence_to_conllu(sent_id, sent, nlp))
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/100 sentences")

    conllu_text = "\n".join(all_lines) + "\n"
    target_path.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {len(SENTENCES)} sentences to {target_path}")

    validation_res = validate_text(conllu_text)
    print(
        f"Validate: count={validation_res.sentence_count}, "
        f"tokens={validation_res.token_count}, passed={validation_res.passed}"
    )
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

    print("All checks passed.")


if __name__ == "__main__":
    main()