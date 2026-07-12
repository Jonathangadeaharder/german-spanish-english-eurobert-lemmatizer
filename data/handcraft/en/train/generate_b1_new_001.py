"""Generate b1_new_001.conllu (en_b1_train_001–200) in sub-batches of 25."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import EN_DET_LEMMAS, EN_IRREGULAR_PLURALS

# 200 B1 sentences: 8–15 tokens, subordinate clauses, conditionals, passive, opinions, work/society
SENTENCE_BATCHES: list[list[str]] = [
    # 001–025: work and careers
    [
        "I applied for the marketing position because the company offers flexible working hours.",
        "If you have any questions about the role, please contact the recruitment team.",
        "She was promoted to team leader after she completed her management training course.",
        "The interview will be conducted online unless the candidate prefers meeting in person.",
        "I believe that remote work can improve productivity when employees have proper support.",
        "He has been searching for a new job since his contract expired last month.",
        "The salary for this position is higher than the average in the local industry.",
        "We must finish the quarterly report before the board meeting on Friday morning.",
        "Although the workload is heavy, most colleagues say that they enjoy the challenge.",
        "The new policy was introduced to help employees balance work and personal life.",
        "If I had more experience, I would apply for the senior analyst role immediately.",
        "Many workers feel that their opinions are not always heard by senior management.",
        "The training program was designed by experts from several international companies.",
        "She explained that she would like to work part time after her maternity leave.",
        "I think the company should invest more in professional development for all staff.",
        "The contract must be signed by both parties before the project can officially start.",
        "When the economy slows down, many businesses reduce their advertising budgets significantly.",
        "He was asked to lead the project because he has strong communication skills.",
        "I would appreciate it if you could send me the job description by email.",
        "The office was renovated during the summer while most employees were working remotely.",
        "We talked about whether the team requires additional support to meet the deadline.",
        "The manager said that performance reviews will take place in the coming weeks.",
        "If the client accepts the proposal, we will begin implementation in early April.",
        "I am not convinced that reducing staff is the best solution to the problem.",
        "Several applications were received after the vacancy was posted on the company website.",
    ],
    # 026–050: opinions and public debate
    [
        "In my opinion, public transport should be cheaper for students and poorer families.",
        "I think that social media has changed the way people talk about political issues.",
        "Many citizens believe that local governments should listen more to community concerns.",
        "She argued that education funding must increase if we want better schools.",
        "If people voted more often, politicians might pay closer attention to public needs.",
        "He feels that the new law will not solve the underlying social problems.",
        "We talked about whether the museum should remain free for all visitors on Sundays.",
        "I do not agree with the view that economic growth always improves living standards.",
        "The article claims that inequality has risen in many European countries recently.",
        "Although opinions differ, most participants supported the proposed community center.",
        "I would say that honesty is essential in any serious public discussion.",
        "They suggested that the debate should center on practical solutions rather than blame.",
        "If the council approved the plan, construction could start within the next year.",
        "Many residents think that noise levels have increased since the road was expanded.",
        "The speaker explained why she supports stronger regulations on brief rental contracts.",
        "I believe young people should have more opportunities to influence local policy.",
        "It seems that public trust in institutions has declined over the past decade.",
        "We were told that the survey results would be published at the end of May.",
        "In my view, cooperation between neighbors can improve safety in urban areas.",
        "He claimed that the report had been misunderstood by several news outlets.",
        "If we ignore these warnings, future generations may face serious social challenges.",
        "I find it unfair that some groups still do not have basic public services.",
        "The panel agreed that more research is required before major reforms are introduced.",
        "She said she would support the initiative if it received sufficient public funding.",
        "Most voters feel that candidates should explain their policies more clearly.",
    ],
    # 051–075: society, law, and civic life
    [
        "The new regulation was approved to protect consumers from misleading online advertising.",
        "If citizens report problems early, authorities can often prevent larger social conflicts.",
        "Many people worry that rising rents will force families to leave the city center.",
        "The court ruled that the evidence was not strong enough to support the charge.",
        "We learned that volunteer programs play an important role in local communities.",
        "Although crime rates have fallen, residents still want better street lighting at night.",
        "The charity was founded by activists who wanted to support homeless young adults.",
        "I think legal advice should be available to everyone, not only the wealthy.",
        "If the law is changed, tenants will receive more protection against sudden evictions.",
        "The report shows that immigration has contributed to economic growth in several regions.",
        "She explained that the organization helps refugees find housing and language courses.",
        "Public meetings were held so residents could comment on the redevelopment plan.",
        "Many experts argue that early intervention can reduce lasting social inequality.",
        "The policy was criticized because it did not cover childcare costs for working parents.",
        "If community centers close, teenagers may have fewer safe places to meet after school.",
        "I believe civic education should be strengthened in schools and adult training programs.",
        "The protest was peaceful, although tensions remained high throughout the afternoon.",
        "Volunteers were recruited when the food bank announced that supplies were running low.",
        "He said that trust between citizens and police must be rebuilt through dialogue.",
        "We talked about how public spaces can be designed to encourage social interaction.",
        "The initiative was supported by groups representing older people and disabled residents.",
        "If funding is cut, several local libraries may be forced to reduce their opening hours.",
        "Many parents feel that after school programs are essential for children's development.",
        "The mayor promised that the housing project would include affordable units for families.",
        "I am convinced that transparent decision making improves confidence in public institutions.",
    ],
    # 076–100: economy and business
    [
        "The factory was closed because demand for the product had fallen sharply last year.",
        "If interest rates rise again, many small businesses may struggle to repay their loans.",
        "Investors believe that the technology sector will continue to grow over the next decade.",
        "The company announced that it would open a new branch in the northern region.",
        "Although profits increased, wages for poorly paid employees remained almost unchanged.",
        "Several shops were affected when the main road was blocked by construction work.",
        "I think competition between firms can benefit consumers through lower prices and innovation.",
        "The export figures were published after the government revised its annual forecast.",
        "If suppliers delay deliveries, production lines may have to stop for several days.",
        "Many economists argue that inflation hurts households with fixed incomes the most.",
        "The merger was approved on the condition that two regional offices would be retained.",
        "We were informed that the budget would be reviewed during the next committee meeting.",
        "The startup was funded by investors who wanted to support sustainable packaging solutions.",
        "She said the firm would hire more staff if sales continued to improve this quarter.",
        "Rising energy costs have forced some manufacturers to raise prices for basic goods.",
        "The market report suggests that consumer confidence has weakened since early spring.",
        "If the strike continues, deliveries to supermarkets could be disrupted next week.",
        "I do not think short term profits should be prioritized over long term business stability.",
        "The contract was negotiated by lawyers representing both the buyer and the seller.",
        "Many customers complained that service quality declined after the company changed owners.",
        "The tax reform was introduced to encourage small firms to invest in new equipment.",
        "If demand recovers quickly, the warehouse may require additional workers during summer.",
        "He explained that the business plan had been revised to reflect higher material costs.",
        "We talked about whether the price increase would reduce sales in the domestic market.",
        "The annual results were presented to shareholders at a meeting in the city hall.",
    ],
    # 101–125: education and training
    [
        "The university announced that tuition fees would remain unchanged for the next academic year.",
        "If students receive better guidance, they are more likely to choose suitable career paths.",
        "Many teachers believe that class sizes should be reduced to improve learning outcomes.",
        "The exam was postponed because several schools were closed due to severe weather.",
        "Although the course is demanding, most participants said they found it very useful.",
        "Scholarships were offered to applicants who could demonstrate strong academic performance.",
        "I think practical experience is just as important as theoretical knowledge for graduates.",
        "The lecturer explained that the assignment must be submitted by the end of the month.",
        "If funding increases, more schools could provide modern science laboratories for students.",
        "The library was renovated while classes continued in temporary rooms across the campus.",
        "Many parents feel that homework loads should be balanced with extracurricular activities.",
        "The curriculum was updated to include digital skills and media literacy for teenagers.",
        "She said she would enroll in the evening course if her work schedule allowed it.",
        "Students were encouraged to join study groups because collaboration improves understanding.",
        "I believe universities should offer more support to international students during their first year.",
        "The workshop was led by trainers who specialize in communication and presentation skills.",
        "If attendance falls, the college may cancel some optional modules next semester.",
        "We talked about whether online learning can replace traditional classroom teaching completely.",
        "The research project was funded by a grant from the national education ministry.",
        "Many graduates struggle to find jobs because employers expect relevant work experience.",
        "The seminar was attended by teachers who wanted to learn about inclusive education methods.",
        "If the exam rules change, students will require clearer instructions before the test day.",
        "I am convinced that lifelong learning is necessary in a rapidly changing labor market.",
        "The textbook was written by professors who have taught the subject for many years.",
        "Although results improved, officials said more investment in teacher training was required.",
    ],
    # 126–150: health and public services
    [
        "The hospital was expanded because the number of patients had increased every year.",
        "If symptoms persist for more than a week, you should contact your doctor immediately.",
        "Many patients believe that waiting times could be reduced with better appointment systems.",
        "The clinic announced that it would offer evening consultations for working adults.",
        "Although the treatment was effective, recovery took longer than the patient expected.",
        "New guidelines were introduced to help staff prevent infections in care homes.",
        "I think mental health services should receive the same priority as physical healthcare.",
        "The nurse explained that the test results would be available within two business days.",
        "If the pharmacy closes early, residents may have difficulty collecting urgent prescriptions.",
        "The vaccination campaign was supported by local organizations and community volunteers.",
        "Many families worry that healthcare costs will continue to rise during the next decade.",
        "The report was published after researchers analyzed data from several public hospitals.",
        "She said she would change doctors if appointments remained difficult to book online.",
        "Emergency services were praised because they responded quickly to the motorway accident.",
        "We talked about whether more doctors should be trained in rural areas with aging populations.",
        "The health center was renovated while outpatient services moved to a nearby building.",
        "If budgets are reduced, some preventive programs may be cut despite their lasting benefits.",
        "I believe patients should be given clearer information about possible side effects.",
        "The policy was changed so that carers could receive additional respite support each month.",
        "Many citizens feel that public transport links to hospitals require urgent improvement.",
        "The survey showed that satisfaction with local health services had declined slightly.",
        "If the new system works, patients will be able to manage appointments through one portal.",
        "He explained that the referral had been sent to a specialist clinic in the capital.",
        "Although staff worked hard, the department still lacked enough trained nurses.",
        "The government promised that waiting lists would be reduced through targeted recruitment.",
    ],
    # 151–175: environment and community
    [
        "The river was polluted after waste from a nearby factory entered the water system.",
        "If communities plant more trees, urban temperatures may fall during hot summer months.",
        "Many residents believe that recycling rules should be simpler and easier to follow.",
        "The park was restored by volunteers who wanted to create a safer space for families.",
        "Although air quality has improved, experts say further action is still necessary.",
        "New bike lanes were built because the council wanted to reduce car traffic downtown.",
        "I think companies should be encouraged to cut emissions through clearer environmental standards.",
        "The campaign explained that small changes at home can reduce energy use significantly.",
        "If the drought continues, farmers may lose a large part of their harvest this year.",
        "The nature reserve was protected after local groups campaigned against commercial development.",
        "Many people feel that plastic packaging should be reduced in supermarkets and restaurants.",
        "The report was prepared by scientists who study climate effects on coastal communities.",
        "She said she would join the cleanup project if the event was held on Saturday morning.",
        "Recycling centers were expanded when the city introduced stricter waste separation rules.",
        "We talked about whether public funding should support renewable energy projects more actively.",
        "The flood barriers were inspected after warnings were issued for several flood prone districts.",
        "If waste collection is delayed, residents are asked to store rubbish safely until pickup.",
        "I believe urban planning should prioritize green spaces and safe walking routes for children.",
        "The festival was organized to raise awareness about water conservation in dry regions.",
        "Although progress is slow, many towns have invested in solar panels for public buildings.",
        "The forest path was closed because storm damage made the trail dangerous for hikers.",
        "If emissions targets are not met, governments may face stronger pressure from young activists.",
        "Many neighbors support the plan because it would convert an empty lot into a community garden.",
        "The environmental audit was carried out before the company renewed its operating license.",
        "He explained that the wetland had been damaged by illegal dumping over several years.",
    ],
    # 176–200: technology, media, and mixed B1 structures
    [
        "The website was updated because users complained that the payment process was confusing.",
        "If passwords are reused, accounts may be exposed to serious security risks online.",
        "Many journalists believe that checking facts should be strengthened on major news platforms.",
        "The app was developed by a team that wanted to help people manage daily expenses.",
        "Although the device is convenient, some customers worry about how their data is stored.",
        "New privacy rules were introduced after several companies reported large data breaches.",
        "I think children should learn how to evaluate information they find on the internet.",
        "The editor explained that the article would be revised before publication next Monday.",
        "If the server fails, employees will not be able to open shared files from home.",
        "The documentary was filmed in communities affected by rapid changes in local industry.",
        "Many users feel that social networks should give people more control over notifications.",
        "The software update was released while technicians monitored systems for unexpected errors.",
        "She said she would delete the account if the platform did not improve its privacy settings.",
        "Online courses were promoted because the college wanted to reach students in rural areas.",
        "We talked about whether artificial intelligence will create more jobs than it replaces.",
        "The podcast was recorded by reporters who cover technology and labor market trends.",
        "If the network is overloaded, video calls may become unstable during peak working hours.",
        "I am not sure that automation alone can solve productivity problems in every sector.",
        "The press release was issued after the firm confirmed plans to invest in cybersecurity.",
        "Although the tool is popular, experts warn that it should not replace professional advice.",
        "The platform was criticized because misleading content spread quickly during the election.",
        "If users report abuse, moderators are expected to review the posts within twenty four hours.",
        "Many readers think that local newspapers remain essential for informed civic participation.",
        "The interview was published after the minister answered questions about future digital policy.",
        "I believe reliable information is necessary if societies are to make responsible collective decisions.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

START_ID = 1
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

    target_path = project_root / "data/handcraft/en/train/b1_new_001.conllu"
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