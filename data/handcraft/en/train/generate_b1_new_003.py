"""Generate b1_new_003.conllu (en_b1_train_401–600) in sub-batches of 25."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import EN_DET_LEMMAS, EN_IRREGULAR_PLURALS

# 200 B1 sentences: housing, travel, workplace skills, sports, culture, finance, transport, mixed
SENTENCE_BATCHES: list[list[str]] = [
    # 401–425: housing and renting
    [
        "The landlord agreed to repair the heating system after tenants reported several cold nights.",
        "If rent increases continue, many young families may have to move to cheaper suburbs.",
        "Many renters believe that deposit rules should be clearer for first time tenants.",
        "The apartment was inspected because the building manager received complaints about damp walls.",
        "Although the flat is small, the location makes it convenient for commuting to work.",
        "New regulations were introduced to limit sudden rent rises in popular city districts.",
        "I think tenants should receive written notice before major renovation work begins.",
        "The agent explained that the lease must be signed before the keys can be collected.",
        "If the contract is broken early, tenants may lose part of their security deposit.",
        "The housing office was expanded while staff handled a growing number of applications.",
        "Many residents feel that affordable housing should be built near public transport links.",
        "The report was published after researchers compared rent levels in several major cities.",
        "She said she would negotiate the rent if the kitchen appliances were not replaced.",
        "Shared flats were advertised because students wanted cheaper options near the campus.",
        "We talked about whether short term rentals reduce the supply of long term housing.",
        "The council promised that new social housing units would be completed by next autumn.",
        "If maintenance requests are ignored, landlords may face fines under updated tenancy law.",
        "I believe housing advice centers should be available in every district with high demand.",
        "The eviction notice was challenged because the tenant had paid rent on time.",
        "Many neighbors worry that empty office buildings will not be converted into homes quickly.",
        "The viewing was arranged after the tenant submitted proof of stable monthly income.",
        "If energy bills rise again, low income households may struggle to heat their homes.",
        "He explained that the deposit had been returned after the final inspection was completed.",
        "Although the area is quiet, some buyers hesitate because parking spaces are limited.",
        "The housing association announced that balcony repairs would start in early March.",
    ],
    # 426–450: travel and tourism
    [
        "The flight was delayed because strong winds prevented planes from landing safely.",
        "If travelers book early, they can often find cheaper tickets for summer holidays.",
        "Many tourists believe that local guides should receive fair pay for their work.",
        "The hotel was renovated while guests stayed in temporary rooms on another floor.",
        "Although the resort is popular, some visitors complain about crowded beaches in August.",
        "New safety rules were introduced after several hiking accidents were reported last year.",
        "I think travel insurance should cover medical costs when people visit foreign countries.",
        "The receptionist explained that check in would begin after two o'clock in the afternoon.",
        "If passports expire soon, passengers may be refused boarding at the departure gate.",
        "The tour was canceled because the guide became ill the night before departure.",
        "Many travelers feel that baggage fees should be included in the advertised ticket price.",
        "The brochure was updated after customers asked for clearer information about local transport.",
        "She said she would extend her stay if the weather remained sunny during the week.",
        "Train tickets were sold out because a festival attracted thousands of visitors to the town.",
        "We talked about whether tourism creates more benefits than problems to small communities.",
        "The museum pass was recommended by staff who wanted visitors to save money on entry fees.",
        "If luggage is lost, airlines are expected to help passengers track their bags quickly.",
        "I believe tourist areas should protect historic sites from damage caused by large crowds.",
        "The cruise was postponed after health officials inspected the ship for safety violations.",
        "Many hosts worry that strict rental rules will reduce income from holiday apartments.",
        "The route was changed because roadworks blocked the main highway to the coast.",
        "If exchange rates fall, imported goods may become more expensive for foreign visitors.",
        "He explained that the reservation had been confirmed through the online booking system.",
        "Although the city is beautiful, peak season prices are too high for many families.",
        "The travel agency promised that refunds would be issued within ten business days.",
    ],
    # 451–475: workplace skills and human resources
    [
        "The training session was organized to help staff improve their presentation skills.",
        "If employees receive regular feedback, they are more likely to develop professionally.",
        "Many workers believe that flexible schedules should be offered to parents with young children.",
        "The contract was reviewed because the company wanted to clarify remote work conditions.",
        "Although the role is demanding, most applicants said the salary was competitive.",
        "New guidelines were introduced to prevent discrimination during recruitment interviews.",
        "I think teamwork skills should be evaluated alongside technical knowledge in job applications.",
        "The manager explained that annual leave requests must be submitted at least two weeks early.",
        "If overtime increases, staff may become exhausted and less productive during busy periods.",
        "The onboarding program was expanded while HR recruited specialists for several departments.",
        "Many employees feel that mental health support should be part of standard workplace benefits.",
        "The handbook was revised after staff requested clearer rules about expense claims.",
        "She said she would apply for the internal vacancy if her current project ended in June.",
        "Workshops were offered because the firm wanted to improve communication between teams.",
        "We talked about whether performance targets should be adjusted during economic downturns.",
        "The mentor program was launched by managers who wanted to support junior colleagues.",
        "If conflicts are ignored, they can damage cooperation within a department over time.",
        "I believe employers should explain promotion criteria more clearly to all staff members.",
        "The appraisal was postponed because the supervisor was attending an urgent client meeting.",
        "Many graduates worry that unpaid internships limit access to valuable work experience.",
        "The policy was updated so that parental leave would be available to all eligible employees.",
        "If budgets are cut, training courses may be reduced despite their long term benefits.",
        "He explained that his reference had been sent directly to the recruitment team.",
        "Although the office is modern, some staff prefer working from home two days a week.",
        "The company promised that pay reviews would take place before the end of the fiscal year.",
    ],
    # 476–500: sports and leisure
    [
        "The match was postponed because heavy rain made the pitch unsafe for players.",
        "If athletes train consistently, they are more likely to avoid injuries during competitions.",
        "Many fans believe that ticket prices for major events should be more affordable.",
        "The swimming pool was renovated while members used facilities at a nearby sports center.",
        "Although the team lost, supporters praised the players for their effort until the end.",
        "New safety measures were introduced after several accidents occurred on the cycling route.",
        "I think local clubs should receive more funding to maintain public sports facilities.",
        "The coach explained that practice sessions would start earlier during the summer camp.",
        "If membership fees rise, some families may cancel their subscriptions to the gym.",
        "The tournament was organized by volunteers who wanted to promote youth football locally.",
        "Many residents feel that parks should include areas for outdoor exercise and recreation.",
        "The schedule was changed because the referee became unavailable for the weekend fixture.",
        "She said she would join the running club if training times suited her work schedule.",
        "Equipment donations were collected because the school could not afford new basketball hoops.",
        "We talked about whether professional sport receives too much public attention and funding.",
        "The trail was closed after storm damage made several sections dangerous for hikers.",
        "If rules are unclear, disputes between players may interrupt games more frequently.",
        "I believe leisure centers should offer discounted access for students and older people.",
        "The championship was broadcast live because millions of viewers followed the national team.",
        "Many parents worry that children spend too little time on active play after school.",
        "The fitness class was canceled when the instructor reported illness on the morning of the session.",
        "If facilities are poorly maintained, participation in community sport may decline steadily.",
        "He explained that his place in the marathon had been secured through an early registration.",
        "Although the club is successful, some members want more focus on amateur development.",
        "The council promised that the new stadium would include accessible seating for all visitors.",
    ],
    # 501–525: culture and arts
    [
        "The exhibition was extended because visitor numbers exceeded expectations during the first month.",
        "If museums offer evening openings, they can attract audiences who work during the day.",
        "Many artists believe that public funding should support creative projects in smaller towns.",
        "The theater was restored while performances continued in a temporary venue nearby.",
        "Although the play is challenging, most critics praised the acting and stage design.",
        "New grants were introduced to help young musicians record their first professional albums.",
        "I think cultural events should be promoted in schools to encourage early interest in the arts.",
        "The curator explained that guided tours would be available in three different languages.",
        "If tickets sell out quickly, additional shows may be scheduled for the following weekend.",
        "The festival was founded by volunteers who wanted to celebrate local traditions and crafts.",
        "Many residents feel that libraries should host more concerts and community art workshops.",
        "The manuscript was displayed after experts confirmed its historical importance to the region.",
        "She said she would attend the premiere if she could arrange childcare for the evening.",
        "Workshops were organized because the gallery wanted to involve families in creative activities.",
        "We talked about whether free admission to museums improves access for disadvantaged groups.",
        "The concert hall was upgraded while engineers improved acoustics and seating comfort.",
        "If funding falls, several independent theaters may have to reduce their annual programs.",
        "I believe artists should be paid fairly when their work is used in commercial projects.",
        "The film screening was canceled because the projector failed minutes before the start.",
        "Many students worry that arts education receives less attention than science subjects.",
        "The collection was loaned to another city while the museum prepared a major renovation.",
        "If copyright rules are ignored, creators may lose income from their original work.",
        "He explained that the sculpture had been commissioned for the new public square.",
        "Although the novel is famous, some readers find the language difficult at first.",
        "The ministry promised that heritage sites would receive additional protection against damage.",
    ],
    # 526–550: personal finance and banking
    [
        "The bank account was frozen after suspicious transactions were reported by the customer.",
        "If interest rates fall, savers may receive lower returns on fixed term deposits.",
        "Many customers believe that banking fees should be explained more clearly in monthly statements.",
        "The loan application was approved because the applicant had a stable employment history.",
        "Although the offer looked attractive, the contract included hidden charges for early repayment.",
        "New security features were introduced after several online accounts were opened illegally.",
        "I think financial advice should be available to people who are managing debt for the first time.",
        "The adviser explained that the mortgage documents must be signed before the funds are released.",
        "If payments are not made, borrowers may damage their credit rating for several years.",
        "The branch was renovated while customers used online services for routine transactions.",
        "Many households feel that budget planning tools should be offered free by major banks.",
        "The report was published after analysts studied household savings rates across the country.",
        "She said she would switch banks if customer service remained slow during urgent inquiries.",
        "Savings accounts were promoted because the institution wanted to attract younger customers.",
        "We talked about whether contactless payments make it easier to overspend without noticing.",
        "The fraud alert was issued when the system detected unusual purchases abroad.",
        "If inflation rises, the real value of cash savings may decline over time.",
        "I believe banks should warn clients clearly before changing terms on existing accounts.",
        "The overdraft limit was reduced because the customer had exceeded it several times.",
        "Many citizens worry that pension reforms will affect their income after retirement.",
        "The payment was delayed while the bank verified the recipient's account details.",
        "If budgets are tight, families may postpone large purchases until their situation improves.",
        "He explained that the transfer had been completed through the mobile banking app.",
        "Although the app is convenient, some users prefer speaking to staff for complex matters.",
        "The regulator promised that misleading financial products would be investigated more thoroughly.",
    ],
    # 551–575: public transport and mobility
    [
        "The bus route was changed because roadworks blocked access to the main station.",
        "If ticket prices rise again, commuters may consider cycling or walking to work.",
        "Many passengers believe that timetables should be updated more quickly after service disruptions.",
        "The tram line was extended while engineers tested signals on the new section.",
        "Although the service is frequent, some riders complain about overcrowding during rush hour.",
        "New bike rental stations were introduced to encourage alternatives to short car journeys.",
        "I think public transport should be affordable for people who do not own private cars.",
        "The driver explained that delays were caused by an accident on the motorway bridge.",
        "If strikes continue, travelers may have to plan longer journeys using replacement buses.",
        "The station was renovated while temporary platforms were built for continuing services.",
        "Many residents feel that night buses should operate on weekends in entertainment districts.",
        "The survey was conducted after officials reviewed complaints about unreliable suburban trains.",
        "She said she would buy a monthly pass if the discount covered all local zones.",
        "Accessibility improvements were funded because the operator wanted to meet legal requirements.",
        "We talked about whether free public transport would reduce traffic in city centers.",
        "The ferry service was suspended because strong winds made the crossing unsafe.",
        "If maintenance is delayed, older vehicles may break down more often during winter.",
        "I believe transport planners should consult communities before closing rural bus routes.",
        "The smart card was introduced so that passengers could pay fares through one system.",
        "Many commuters worry that fare increases are not matched by better service quality.",
        "The timetable was revised when the operator added extra trains during peak periods.",
        "If park and ride facilities expand, fewer cars may enter the downtown area daily.",
        "He explained that his season ticket had been renewed automatically through direct debit.",
        "Although the network is extensive, some villages still lack reliable connections to hospitals.",
        "The authority promised that real time arrival information would be available on all major lines.",
    ],
    # 576–600: mixed B1 structures across topics
    [
        "The rental contract was reviewed because the tenant requested permission to keep a pet.",
        "If museums cooperate with schools, more pupils may visit exhibitions during term time.",
        "Many bank customers complained that fraud warnings were sent too late to prevent losses.",
        "The cycling event was supported by sponsors who wanted to promote healthy lifestyles locally.",
        "Although the hotel was fully booked, staff found rooms for guests whose train was canceled.",
        "Workplace mentors were recruited when the company launched a program for new graduates.",
        "I think public transport and housing policy should be planned together in growing cities.",
        "The gallery curator explained that loans from private collectors would arrive next week.",
        "If gym memberships are canceled early, members may still owe fees under the contract terms.",
        "The travel refund was approved after the airline admitted that the delay exceeded six hours.",
        "We talked about whether cultural funding should prioritize community projects over major institutions.",
        "The branch manager said overtime payments would be calculated according to the revised policy.",
        "She said she would appeal the rent increase if the flat had not been maintained properly.",
        "Sports facilities were upgraded while the council prepared a plan for youth participation.",
        "If bus services are reduced, residents in outer districts may become more dependent on cars.",
        "The savings plan was recommended by advisers who specialize in long term financial planning.",
        "Many tourists feel that heritage sites should limit visitor numbers during fragile restoration work.",
        "The team coach explained that injured players would be examined before the next league match.",
        "Although the concert was sold out, organizers arranged a live stream for remote audiences.",
        "Tenant advice sessions were offered because several households faced sudden eviction notices.",
        "I believe reliable transport links are essential if rural areas are to attract new residents.",
        "The loan officer announced that applications would be reviewed within five business days.",
        "If ticket machines fail, passengers should still be able to buy fares from staff on board.",
        "The festival program was designed by artists who wanted to reflect the city's diverse communities.",
        "I am convinced that balanced investment in housing, culture, and mobility can improve urban life.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

START_ID = 401
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

    target_path = project_root / "data/handcraft/en/train/b1_new_003.conllu"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    all_lines: list[str] = []
    global_idx = 0

    for batch_num, batch in enumerate(SENTENCE_BATCHES):
        assert len(batch) == BATCH_SIZE, f"Batch {batch_num} has {len(batch)} sentences"
        print(
            f"Processing batch {batch_num + 1}/8 "
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