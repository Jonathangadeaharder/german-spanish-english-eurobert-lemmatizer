"""Generate b1_new_005.conllu (en_b1_train_801–900) in sub-batches of 25."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import EN_DET_LEMMAS, EN_IRREGULAR_PLURALS

# 100 B1 sentences: technology, food, wellbeing, arts and mixed
SENTENCE_BATCHES: list[list[str]] = [
    # 801–825: technology, apps, and online services
    [
        "The software update was delayed because developers discovered a security flaw in the login system.",
        "If users enable two factor authentication, their accounts are usually better protected against hackers.",
        "Many customers believe that tech firms should explain clearly how personal data is collected.",
        "The website was redesigned while engineers tested new features for mobile payment options.",
        "Although the app is popular, some reviewers complained that customer support responded too slowly.",
        "New privacy settings were introduced to help users control which contacts can view their profiles.",
        "I think online platforms should remove misleading health claims before they reach large audiences.",
        "The technician explained that repairs would begin after spare parts arrived from the supplier.",
        "If servers crash during peak hours, shoppers may lose items saved in their online baskets.",
        "The cloud service was expanded because remote teams required secure storage for shared documents.",
        "Many employees feel that video meetings should be shorter when the same points are repeated.",
        "The password reset link was sent after the user confirmed ownership of the registered email address.",
        "She said she would switch providers if the company did not improve broadband speed in her area.",
        "Digital literacy courses were offered because libraries wanted to help seniors use smartphones confidently.",
        "We talked about whether artificial intelligence chatbots can replace human support for simple requests.",
        "The router was replaced while technicians checked whether cables in the building were outdated.",
        "If phishing emails look convincing, even careful users may click harmful links by mistake.",
        "I believe public wireless networks should display warnings about risks before users connect automatically.",
        "The data breach was reported after experts found that encrypted files had been stolen illegally.",
        "Many parents worry that children spend too much time on games instead of outdoor activities.",
        "The startup was funded by investors who wanted to develop affordable tools for small retailers.",
        "If backup systems fail, hospitals may struggle to retrieve patient records during power outages.",
        "He explained that the subscription had been canceled through the official account settings page.",
        "Although the tablet is old, students said it was reliable enough for reading and note taking.",
        "The provider promised that faster upload speeds would be available in rural areas next year.",
    ],
    # 826–850: food, restaurants, and healthy eating
    [
        "The restaurant was booked out because a local food festival attracted visitors from nearby towns.",
        "If menus list allergens clearly, diners with food sensitivities can choose meals more safely.",
        "Many nutritionists believe that school lunches should include more vegetables and whole grains.",
        "The kitchen was renovated while chefs tested recipes that used seasonal produce from local farms.",
        "Although the soup was simple, guests praised the balance of herbs and fresh ingredients.",
        "New cooking classes were introduced to help families prepare affordable meals at home.",
        "I think food labels should show sugar content in a way that children can understand easily.",
        "The manager explained that reservations would be confirmed after the deposit was paid online.",
        "If supply shortages continue, cafés may have to change their menus several times each week.",
        "The farmers market was expanded because residents wanted direct access to organic fruit and vegetables.",
        "Many diners feel that tipping practices should be explained clearly to tourists visiting the city.",
        "The complaint was investigated after several customers reported undercooked chicken at a chain restaurant.",
        "She said she would open a bakery if she could secure a loan for commercial kitchen equipment.",
        "Meal delivery kits were promoted because the company wanted to reduce food waste in households.",
        "We talked about whether plant based diets can provide enough protein for active teenagers.",
        "The dining room was rearranged while staff prepared tables for a large wedding reception.",
        "If refrigeration fails overnight, restaurants may be required to discard large quantities of food.",
        "I believe community gardens should receive support to teach residents how to grow herbs locally.",
        "The hygiene inspection was approved after the owner replaced damaged tiles near the preparation area.",
        "Many shoppers worry that packaged snacks are marketed too aggressively to young children.",
        "The cookbook was published by chefs who specialize in traditional dishes from coastal regions.",
        "If prices rise sharply, students may cook together to share costs for weekly groceries.",
        "He explained that the catering order had been changed after the client reduced the guest list.",
        "Although the café is tiny, regular visitors said the staff remember their favorite coffee orders.",
        "The council promised that new street food permits would be issued before the summer festival.",
    ],
    # 851–875: mental health, wellbeing, and community support
    [
        "The counseling service was expanded because waiting times had increased during the winter months.",
        "If people talk openly about stress, colleagues may feel more comfortable seeking help early.",
        "Many therapists believe that workplace mental health programs should be available to all employees.",
        "The support group was formed while volunteers trained to facilitate discussions in several languages.",
        "Although recovery takes time, patients said regular walks helped them manage anxiety more effectively.",
        "New wellbeing workshops were introduced after schools reported rising pressure among examination students.",
        "I think employers should offer flexible schedules when staff require time for medical appointments.",
        "The nurse explained that referrals would be handled after the initial assessment was completed.",
        "If sleep problems persist, doctors may recommend changes to evening routines and screen use.",
        "The helpline was staffed by trained listeners who wanted to assist callers during night shifts.",
        "Many residents feel that parks and quiet spaces should be protected in densely built neighborhoods.",
        "The appointment was rescheduled after the clinic received an urgent request from another patient.",
        "She said she would join the yoga class if the schedule fit around her evening work shifts.",
        "Mindfulness sessions were offered because the community center wanted to support carers under pressure.",
        "We talked about whether social media use affects sleep quality more than television before bedtime.",
        "The therapy room was refurbished while staff moved sessions temporarily to a nearby health center.",
        "If stigma remains strong, some people may avoid treatment despite experiencing serious symptoms.",
        "I believe schools should teach students practical strategies for managing exam stress and conflict.",
        "The wellness program was praised after participants reported improved mood and better concentration.",
        "Many managers worry that burnout will increase if overtime becomes normal in busy departments.",
        "The retreat was organized by coaches who specialize in resilience training for healthcare workers.",
        "If crisis teams are understaffed, response times may lengthen during weekends and public holidays.",
        "He explained that the support plan had been updated after monthly progress reviews with his doctor.",
        "Although the group is small, members said they valued honest conversations about daily challenges.",
        "The charity promised that free counseling slots would be added before the start of the new term.",
    ],
    # 876–900: arts, culture, entertainment, and mixed B1 structures
    [
        "The theater production was canceled because the lead actor became ill two days before opening night.",
        "If museums offer discounted tickets, more families may visit exhibitions during school holidays.",
        "Many artists believe that public funding should support creative projects in smaller towns.",
        "The gallery was renovated while curators prepared a collection of photographs from local photographers.",
        "Although the film was long, audiences praised the acting and the detailed historical setting.",
        "New music lessons were introduced to help children learn instruments regardless of family income.",
        "I think libraries should host author talks that encourage teenagers to read more widely.",
        "The director explained that rehearsals would resume after the stage lighting system was repaired.",
        "If ticket sales are low, independent cinemas may struggle to keep evening screenings profitable.",
        "The festival was organized by volunteers who wanted to celebrate traditional dance and crafts.",
        "Many listeners feel that podcasts should include transcripts for people with hearing difficulties.",
        "The exhibition was extended after critics praised the interactive displays about urban architecture.",
        "She said she would audition for the choir if rehearsals did not conflict with her university exams.",
        "Street performances were permitted because the council hoped to attract visitors to the historic quarter.",
        "We talked about whether streaming services reduce attendance at live concerts and theater shows.",
        "The studio was soundproofed while engineers installed equipment for recording educational podcasts.",
        "If copyright rules are ignored, small publishers may lose income from digital reproductions.",
        "I believe community theaters should receive grants to keep ticket prices affordable for local residents.",
        "The documentary was nominated after filmmakers spent two years interviewing workers in coastal industries.",
        "Many critics worry that remakes are replacing original stories in mainstream cinema schedules.",
        "The concert hall was reopened when inspectors confirmed that safety exits met current regulations.",
        "If rehearsal spaces are limited, youth orchestras may have to share rooms with adult ensembles.",
        "He explained that the painting had been restored after humidity damaged the frame and varnish.",
        "Although the venue is modest, performers said the audience responded warmly to new compositions.",
        "I am convinced that creative communities, informed audiences, and open debate can enrich public life.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 100, f"Expected 100 sentences, got {len(SENTENCES)}"

START_ID = 801
BATCH_SIZE = 25

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

    target_path = project_root / "data/handcraft/en/train/b1_new_005.conllu"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    all_lines: list[str] = []
    global_idx = 0

    for batch_num, batch in enumerate(SENTENCE_BATCHES):
        assert len(batch) == BATCH_SIZE, f"Batch {batch_num} has {len(batch)} sentences"
        print(
            f"Processing batch {batch_num + 1}/4 "
            f"(sentences {START_ID + global_idx}–{START_ID + global_idx + BATCH_SIZE - 1})"
        )
        for sent in batch:
            sent_id = f"en_b1_train_{START_ID + global_idx:03d}"
            all_lines.extend(sentence_to_conllu(sent_id, sent, nlp))
            global_idx += 1

    conllu_text = "\n".join(all_lines) + "\n"
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