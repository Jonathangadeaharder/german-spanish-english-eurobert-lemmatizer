"""Generate b1_new_002.conllu (en_b1_train_201–400) in sub-batches of 25."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import EN_DET_LEMMAS, EN_IRREGULAR_PLURALS

# 200 B1 sentences: immigration, education, consumer rights, health insurance, social media, volunteering
SENTENCE_BATCHES: list[list[str]] = [
    # 201–225: immigration
    [
        "The visa application was rejected because several required documents were not included in the file.",
        "If immigrants learn the local language, they often find it easier to use public services.",
        "Many families moved abroad after they were offered better job opportunities in another country.",
        "The residence permit must be renewed before it expires at the end of this month.",
        "Although the process is slow, most applicants receive a decision within twelve weeks.",
        "She explained that she would apply for citizenship after living here for five years.",
        "The border office was expanded because the number of travelers had increased every summer.",
        "If refugees receive housing support, they can concentrate more on language courses and training.",
        "I believe integration programs should include practical help with housing and employment.",
        "The policy was introduced to help skilled workers obtain work permits more quickly.",
        "When the law changed, foreign students were allowed to stay longer after graduation.",
        "He was asked to provide proof of address before his appointment could be confirmed.",
        "Many residents feel that immigration rules should be clearer for families with children.",
        "The embassy announced that appointment slots would be released online every Monday morning.",
        "If the appeal is accepted, the applicant will receive a new interview date.",
        "We talked about whether newcomers should receive more support during their first year.",
        "The report shows that migration has helped fill shortages in the healthcare sector.",
        "Although opinions differ, most participants supported better access to legal advice.",
        "The charity was founded by volunteers who wanted to assist asylum seekers with paperwork.",
        "I think employers should not discriminate against candidates because of their nationality.",
        "The family was reunited after immigration authorities approved their application for reunion.",
        "If documents are translated incorrectly, applications may be delayed for several months.",
        "She said she would contact a lawyer if her work permit was not renewed.",
        "Public meetings were held so residents could talk about plans for a new welcome center.",
        "The government promised that processing times would improve through additional staff.",
    ],
    # 226–250: education
    [
        "The college announced that evening classes would be offered for adults who work full time.",
        "If students attend regularly, they are more likely to succeed in the final examination.",
        "Many parents believe that schools should provide more career guidance for teenagers.",
        "The exam was rescheduled because the heating system failed during the winter storm.",
        "Although the module is difficult, most learners said the teacher explained concepts clearly.",
        "Scholarships were awarded to students who demonstrated strong results in science subjects.",
        "I think group projects help learners develop teamwork skills for future employment.",
        "The tutor explained that the essay must be submitted through the online portal.",
        "If class sizes fall, the school may cancel some optional courses next term.",
        "The laboratory was upgraded while lessons continued in temporary rooms on another floor.",
        "Many teachers feel that digital tools should support learning rather than replace teachers.",
        "The syllabus was revised to include practical skills and workplace communication training.",
        "She said she would enroll in the certificate course if her employer covered the fees.",
        "Students were encouraged to use the library because it offers quiet study spaces.",
        "I believe adult education should be affordable for people who want to change careers.",
        "The workshop was led by trainers who specialize in inclusive teaching methods.",
        "If attendance drops, the university may merge two small seminars into one group.",
        "We talked about whether homework loads should be reduced for younger pupils.",
        "The training program was funded by a grant from the regional education authority.",
        "Many graduates struggle because employers expect both qualifications and work experience.",
        "The conference was attended by teachers who wanted to share ideas about assessment.",
        "If exam regulations change, students will require updated instructions before the test.",
        "I am convinced that lifelong learning is essential in a changing labor market.",
        "The textbook was chosen by lecturers who have taught the subject for many years.",
        "Although results improved, officials said more investment in teacher training was required.",
    ],
    # 251–275: consumer rights
    [
        "The refund was approved after the customer proved that the product arrived damaged.",
        "If a shop refuses a valid return, you can contact the consumer protection agency.",
        "Many buyers believe that online sellers should provide clearer information about delivery times.",
        "The contract was canceled because the company failed to deliver the goods on time.",
        "Although the price was low, the warranty did not cover repairs for electrical faults.",
        "New rules were introduced to stop businesses from using misleading advertising online.",
        "I think customers should receive written confirmation before recurring payments are charged.",
        "The adviser explained that the complaint must be filed within thirty days of purchase.",
        "If the item is faulty, the retailer is expected to offer a repair or replacement.",
        "The product recall was announced while stores removed affected batches from their shelves.",
        "Many consumers feel that cancellation policies for subscriptions should be easier to understand.",
        "The report was published after investigators reviewed complaints from several major retailers.",
        "She said she would dispute the charge if the service was not provided as promised.",
        "Online reviews were checked because the platform wanted to reduce fake customer ratings.",
        "We talked about whether loyalty programs should give consumers more control over personal data.",
        "The store was fined after inspectors found that prices did not match shelf labels.",
        "If a delivery is late, buyers may be entitled to compensation under consumer law.",
        "I believe companies should respond to complaints within a reasonable period of time.",
        "The warranty claim was rejected because the device had been repaired by an unauthorized technician.",
        "Many shoppers worry that hidden fees are added during the final step of checkout.",
        "The helpline was expanded when the agency received a large number of billing disputes.",
        "If you sign a contract online, you should save a copy of the terms immediately.",
        "He explained that the payment had been taken twice due to a technical error.",
        "Although the offer looked attractive, the contract included expensive cancellation penalties.",
        "The regulator promised that misleading sales practices would be investigated more thoroughly.",
    ],
    # 276–300: health insurance
    [
        "The insurance policy was updated because the provider changed its coverage for dental treatment.",
        "If you change employers, you should check whether your health plan remains valid.",
        "Many patients believe that insurance forms should be simpler and easier to complete online.",
        "The claim was rejected because the treatment was not listed in the approved benefits.",
        "Although the premium is high, the plan covers emergency care in most European countries.",
        "New guidelines were introduced to help members compare different insurance packages more easily.",
        "I think preventive checkups should be included in basic coverage for working adults.",
        "The agent explained that the application must be submitted before the enrollment deadline.",
        "If the pharmacy is not in network, patients may have to pay higher out of pocket costs.",
        "The coverage extension was approved while the member recovered from surgery at home.",
        "Many families worry that rising premiums will make insurance unaffordable next year.",
        "The report was published after analysts studied claims data from several regional insurers.",
        "She said she would switch providers if customer service remained difficult to reach.",
        "Emergency treatment was covered because the patient could not wait for prior authorization.",
        "We talked about whether mental health counseling should receive the same reimbursement rates.",
        "The benefits guide was revised while members received notices about upcoming policy changes.",
        "If deductibles increase, some households may delay necessary medical appointments.",
        "I believe patients should be told clearly which services require additional approval.",
        "The appeal was accepted so that the insurer would review the denied prescription claim.",
        "Many citizens feel that public information about insurance options requires urgent improvement.",
        "The survey showed that satisfaction with claim processing had declined slightly this year.",
        "If the new portal works, members will be able to upload documents through one system.",
        "He explained that the referral had been sent to a specialist covered by his plan.",
        "Although staff were helpful, the company still lacked enough bilingual support advisers.",
        "The ministry promised that waiting times for benefit decisions would be reduced.",
    ],
    # 301–325: social media
    [
        "The account was suspended because the user repeatedly posted content that violated community rules.",
        "If passwords are weak, social media profiles may be exposed to serious security risks.",
        "Many teenagers believe that platforms should give users more control over personal data.",
        "The post was removed after several members reported that it contained misleading information.",
        "Although the app is popular, some parents worry about how much time children spend online.",
        "New privacy settings were introduced after users complained about unwanted targeted advertising.",
        "I think schools should teach students how to evaluate information they find online.",
        "The moderator explained that appeals would be reviewed within forty eight business hours.",
        "If the network fails, users will not be able to upload photos from their phones.",
        "The campaign was launched by activists who wanted to promote respectful online discussion.",
        "Many users feel that notification settings should be easier to customize on mobile devices.",
        "The platform update was released while engineers monitored systems for unexpected login errors.",
        "She said she would delete the profile if the company did not improve privacy controls.",
        "Online events were promoted because the group wanted to reach members in rural areas.",
        "We talked about whether social media influences political opinions more than traditional news.",
        "The podcast was recorded by journalists who cover technology and digital communication trends.",
        "If users report harassment, moderators are expected to respond within one business day.",
        "I am not sure that blocking alone can solve every problem with online abuse.",
        "The statement was published after the firm confirmed plans to strengthen content moderation.",
        "Although the tool is convenient, experts warn that it should not replace face to face contact.",
        "The page was criticized because false rumors spread quickly during the local election.",
        "If teenagers share location data, their accounts may become visible to strangers online.",
        "Many readers think that community guidelines should be explained more clearly to new users.",
        "The interview was posted after the director answered questions about future platform policy.",
        "I believe reliable information is necessary if users are to make responsible sharing decisions.",
    ],
    # 326–350: volunteering
    [
        "The food bank was supported by volunteers who collected donations every Saturday morning.",
        "If more helpers join the project, the charity can serve additional families each week.",
        "Many residents believe that local organizations should recruit volunteers through schools and libraries.",
        "The cleanup event was postponed because heavy rain made the riverside path unsafe.",
        "Although the work is tiring, most participants said they enjoyed meeting new neighbors.",
        "Training sessions were offered to volunteers who wanted to support elderly residents at home.",
        "I think community service should be promoted as a way to strengthen social connections.",
        "The coordinator explained that shifts must be booked through the online volunteer calendar.",
        "If funding falls, several outreach programs may have to reduce their weekly activities.",
        "The shelter was renovated while volunteers helped residents move into temporary accommodation nearby.",
        "Many students feel that volunteering experience can improve their chances of finding employment.",
        "The initiative was launched by groups who wanted to mentor young people after school.",
        "She said she would lead the reading club if enough parents offered to assist.",
        "Donation drives were organized because supplies at the community kitchen were running low.",
        "We talked about whether companies should give employees paid time off for volunteering.",
        "The garden project was praised because it created a safe meeting place for families.",
        "If volunteer numbers drop, the helpline may not be able to operate every evening.",
        "I believe charities should provide clear guidance for people who offer help for the first time.",
        "The appeal was answered when dozens of residents signed up to deliver meals locally.",
        "Many neighbors worry that youth centers will close without enough adult volunteers.",
        "The survey showed that participation in local volunteering had increased since last spring.",
        "If the new system works, helpers will be able to track hours through one mobile app.",
        "He explained that the mentoring program had been expanded to three additional schools.",
        "Although demand is high, the organization still requires more trained volunteers on weekends.",
        "The council promised that community projects would receive more support for volunteer coordination.",
    ],
    # 351–375: mixed B1 structures across topics
    [
        "The immigration office was criticized because appointment waiting times had increased sharply.",
        "If students skip too many classes, they may lose access to financial support.",
        "Many consumers complained that refund requests were ignored by the online retailer.",
        "The insurer announced that preventive screenings would be covered under the revised plan.",
        "Although the post was popular, moderators removed it for spreading unverified health claims.",
        "Volunteers were recruited when the refugee center said it required language tutors urgently.",
        "I think schools and charities should cooperate to support families who recently arrived.",
        "The lecturer explained that plagiarism rules apply to assignments submitted through social platforms.",
        "If a warranty expires, buyers may have to pay the full cost of repairs.",
        "The health app was developed by a team that wanted to simplify insurance claims.",
        "We talked about whether immigration policy should prioritize family reunion cases.",
        "The training course was funded by donors who support adult literacy programs locally.",
        "She said she would report the seller if the advertised discount was not applied.",
        "Online support groups were moderated by volunteers with experience in mental health care.",
        "If premium subsidies are cut, low income families may lose basic health coverage.",
        "The platform was updated because users asked for stronger tools against online harassment.",
        "Many applicants feel that visa instructions should be available in more languages.",
        "The college library was staffed by volunteers who helped newcomers use public computers.",
        "Although the contract looked fair, the cancellation clause contained unexpected charges.",
        "Emergency volunteers were praised because they responded quickly during the regional floods.",
        "I believe consumer education should include lessons about safe behavior on social media.",
        "The seminar was attended by teachers who support migrant pupils in mainstream classes.",
        "If claim forms are incomplete, insurance payments may be delayed for several weeks.",
        "The community page was created to share verified information about local support services.",
        "He explained that his volunteer visa would allow him to work for the nonprofit organization.",
    ],
    # 376–400: mixed B1 structures across topics
    [
        "The residence card was issued after the applicant completed a mandatory integration course.",
        "If teachers receive better resources, they can offer more support to struggling learners.",
        "Many shoppers believe that product labels should list allergens in plain language.",
        "The dental benefit was added because members requested broader coverage for routine treatment.",
        "Although the video went viral, fact checkers warned that several claims were inaccurate.",
        "Weekend volunteers prepared meals while staff coordinated deliveries to isolated residents.",
        "I think immigration advisers should explain rights and responsibilities during the first appointment.",
        "The university said it would expand online modules if student demand continued to grow.",
        "If a company misleads customers, regulators can impose fines under consumer protection law.",
        "The insurance helpline was expanded when open enrollment generated thousands of new questions.",
        "We talked about whether teenagers should limit social media use during exam periods.",
        "The mentorship scheme was designed by volunteers who wanted to reduce school dropout rates.",
        "She said she would appeal the visa decision if new evidence became available.",
        "Refund policies were reviewed after customers reported confusion about return shipping costs.",
        "If mental health services are underfunded, patients may wait months for counseling appointments.",
        "The account settings were changed so that location sharing would be disabled by default.",
        "Many families feel that schools should inform parents about available community volunteers.",
        "The language class was taught by volunteers who had themselves immigrated ten years earlier.",
        "Although the insurance brochure was detailed, key exclusions were difficult to find quickly.",
        "Donation centers were opened because relief groups expected a rise in requests for aid.",
        "I am convinced that transparent policies improve trust in public institutions and online platforms.",
        "The exam board announced that results would be published on the student portal Friday.",
        "If sellers hide fees, buyers can file complaints with the national trading standards office.",
        "The wellness program was supported by insurers who wanted to encourage healthier lifestyles.",
        "I believe cooperation between schools, charities, and local government can strengthen community resilience.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

START_ID = 201
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

    target_path = project_root / "data/handcraft/en/train/b1_new_002.conllu"
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