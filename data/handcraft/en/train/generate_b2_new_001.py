"""Generate b2_new_001.conllu (en_b2_train_001–200) in sub-batches of 25."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import EN_DET_LEMMAS, EN_IRREGULAR_PLURALS

# 8 batches × 25 = 200 B2 sentences (10–18 tokens, complex syntax, abstract topics)
SENTENCE_BATCHES: list[list[str]] = [
    # Batch 1 (001–025): Globalization
    [
        "Although global trade has expanded rapidly, not every region benefits equally from open markets.",
        "It is feared that rising protectionism could weaken international cooperation over the long term.",
        "The economist explained that dependence on foreign raw materials is increasing in many industries.",
        "Despite falling transport costs, environmental regulations continue to burden global supply chains significantly.",
        "If trade barriers were reduced further, the exchange of goods and services might increase substantially.",
        "Multinational corporations, whose influence shapes global standards, are debated intensely in public forums.",
        "It was reported that regional economic blocs will gain strategic importance in the coming decades.",
        "While tariffs have been lowered, technical trade barriers still prevent many exporters from entering new markets.",
        "The minister emphasized that open markets must not conflict with social responsibility and fair labor practices.",
        "Had fair trading conditions been demanded earlier, the gap between nations would be smaller today.",
        "The integration of world markets requires shared standards and reliable international trade agreements.",
        "It is expected that the next round of talks will establish new rules for digital services.",
        "Although supply chains have become more efficient, they are also more vulnerable to global disruptions.",
        "The relocation of production abroad has sparked debates about employment levels and wage stagnation.",
        "Developing countries should be able to participate fairly in the global economic system, experts argued.",
        "Global brands have reshaped consumer behavior in many societies, as several recent studies have demonstrated.",
        "It is assumed that coordinated tax policies could reduce harmful competition among large economies.",
        "While corporations operate worldwide, national tax regulations often remain fragmented and poorly coordinated.",
        "The expansion of international finance was monitored closely by regulators after the last financial crisis.",
        "Fair supply chains are being promoted to improve working conditions in manufacturing countries abroad.",
        "National economic policy must adapt to global trends, which remains a central challenge for governments.",
        "The analyst noted that migration of skilled workers affects local labor markets in complex ways.",
        "Strengthening multilateral institutions is considered essential for maintaining a stable world economy.",
        "It is not a lack of political will but of enforceable international mechanisms that hinders progress.",
        "The impact of globalization on local cultures is being studied intensively by researchers across disciplines.",
    ],
    # Batch 2 (026–050): Human Rights
    [
        "Human rights are widely regarded as the universal foundation of every democratic legal order.",
        "Although many states have signed the convention, fundamental rights are still violated around the world.",
        "It is expected that the new legislation will significantly strengthen protection for minority communities.",
        "Guaranteeing freedom of expression is an indispensable component of human dignity in modern societies.",
        "Despite alarming reports, effective sanctions against persistent violators are often still lacking.",
        "The commissioner explained that conditions in several prisons had deteriorated over the past year.",
        "Combating forced labor requires international cooperation and consistent prosecution of offenders.",
        "Refugees must be treated in a lawful and humane manner, the rapporteur said during the hearing.",
        "The enforcement of the ban on torture was demanded forcefully by leading human rights organizations.",
        "If victims were better protected, the threshold for committing serious abuses might rise considerably.",
        "Although the resolution was adopted, its practical implementation remains questionable in many countries.",
        "It is feared that restricting assembly rights could weaken democratic processes over the long term.",
        "The rights of people with disabilities are legally enshrined in the constitutions of numerous states.",
        "Had the international community intervened earlier, the suffering of many civilians would have been reduced.",
        "The report indicates that child labor is still widespread in certain industrial and agricultural sectors.",
        "Independent monitoring of the human rights situation is considered indispensable by legal scholars.",
        "Although progress has been made, gender equality continues to be an urgent concern in many regions.",
        "The problem is not a shortage of international norms but a lack of consistent enforcement mechanisms.",
        "Safe corridors for humanitarian aid were described as an urgent measure during the emergency session.",
        "The judge emphasized that nobody should be detained arbitrarily or convicted without a fair trial.",
        "The expansion of asylum rights is debated emotionally and controversially in several European parliaments.",
        "It is recommended that the rights of indigenous peoples be integrated more fully into national law.",
        "Although press freedom is guaranteed, independent journalists face growing pressure in various countries.",
        "Condemning discriminatory practices should send a clear signal to all institutions and public officials.",
        "Upholding human dignity remains a non negotiable obligation of the state even during severe crises.",
    ],
    # Batch 3 (051–075): Scientific Ethics
    [
        "The ethical evaluation of new research methods requires a careful weighing of all foreseeable risks.",
        "Although the study appears promising, participants must not be enrolled without informed consent.",
        "Scientific integrity is emphasized as the foundation of every credible publication in medical research.",
        "Animal experiments must comply with strict ethical and legal requirements before approval is granted.",
        "Despite the value of the data, using genetic information raises substantial ethical questions for society.",
        "The ethics board explained that limits on research involving human embryos must be clearly defined.",
        "Preventing data falsification requires transparent procedures and independent oversight mechanisms.",
        "Researchers should disclose potential conflicts of interest openly, the committee stated in its report.",
        "The application of artificial intelligence in medicine raises questions about responsibility and transparency.",
        "If long term consequences were ignored, unintended harm to patients could arise from rushed deployment.",
        "Although funding was secured, the project was suspended temporarily because of unresolved ethical concerns.",
        "It is expected that new clinical trial guidelines will be published by the end of the year.",
        "Compliance with data protection standards in research projects was reviewed strictly by the commission.",
        "Had participants been better informed, the study would have continued in a more ethically sound manner.",
        "The report recommends conducting the public debate on gene editing in an evidence based way.",
        "The responsibility of science toward society is debated intensely during periods of rapid innovation.",
        "Although the technology is available, its application to humans remains highly controversial ethically.",
        "The issue is not a lack of rules but their consistent implementation across all research institutions.",
        "Independent peer review is regarded as a central quality criterion in academic publishing worldwide.",
        "The professor emphasized that scientific progress must not come at the expense of fundamental ethical principles.",
        "Vaccine development must be both effective and equitably accessible across the entire global population.",
        "It is assumed that ethics advisory bodies will play an even greater role in future funding decisions.",
        "Although the results are impressive, experts urge greater caution when interpreting preliminary findings.",
        "Promoting open science should make manipulation harder and improve the reproducibility of published studies.",
        "Respecting the dignity of research subjects remains the highest priority even during urgent medical breakthroughs.",
    ],
    # Batch 4 (076–100): Urban Development
    [
        "Urbanization is transforming the appearance of many metropolitan areas at an unprecedented pace.",
        "Although cities are growing, affordable housing for low income households remains severely insufficient.",
        "It is expected that most people will live in urban areas within the next few decades.",
        "Planning sustainable transport systems is crucial for quality of life in densely populated regions.",
        "Despite new districts being built, social segregation in many urban centers continues to increase steadily.",
        "The planner explained that green spaces must be incorporated mandatorily into every major construction project.",
        "Renovating derelict industrial sites offers opportunities for mixed residential and commercial development.",
        "Public transport should be expanded before further suburban areas are developed, the mayor argued.",
        "Densifying inner city land is intended to reduce sprawl and shorten long daily commuting distances.",
        "If more investment were directed toward social housing, the housing shortage could be eased noticeably.",
        "Although infrastructure was modernized, the rollout of digital networks in peripheral areas remains slow.",
        "It is feared that rising rents will displace many long term residents from their established neighborhoods.",
        "Developing walkable neighborhoods helps reduce car traffic in city centers, according to recent surveys.",
        "Had greater emphasis been placed on sustainable mobility earlier, air pollution levels would be lower today.",
        "The report documents how heat islands intensify in densely built areas during the summer months.",
        "Resident participation in urban planning decisions is viewed as an element of sound municipal policy.",
        "Although population numbers are rising, preserving historic city centers remains a major planning challenge.",
        "The difficulty lies not in available concepts but in sufficient funding for their consistent implementation.",
        "Promoting micro mobility is intended to make short everyday trips more efficient and environmentally friendly.",
        "The mayor emphasized that livable cities must be residential spaces, not merely centers of economic activity.",
        "Adapting drainage systems to frequent heavy rainfall events requires long term infrastructure investment planning.",
        "It is recommended that vacant commercial properties be converted systematically into additional housing units.",
        "Although satellite towns are being planned, their connection to regional core networks remains inadequate.",
        "Creating public meeting spaces should strengthen coexistence in heterogeneous urban societies, planners believe.",
        "Managing growth without destroying valuable landscape areas remains a core task for regional authorities.",
    ],
    # Batch 5 (101–125): Demographic Change
    [
        "Demographic change is forcing fundamental adjustments in social security systems and labor markets.",
        "Although life expectancy is rising, the number of people requiring care is growing faster than expected.",
        "It is assumed that the proportion of older adults will increase significantly over the coming decades.",
        "Securing the supply of skilled workers requires targeted measures in education, migration, and training.",
        "Despite low birth rates, population has increased in some regions through sustained immigration flows.",
        "The demographer explained that rural areas will be affected more strongly by aging and outward migration.",
        "Balancing family and career is regarded as key to stable long term demographic development.",
        "Older employees should be able to contribute their experience more flexibly in working life, experts said.",
        "Adapting pension systems to rising life expectancy was described as unavoidable by several economists.",
        "If more were invested in childcare, work life balance for young parents could improve substantially.",
        "Although the city is expanding, many communities in rural areas have been shrinking for years.",
        "It is feared that a shortage of caregivers will affect quality of life for many affected individuals.",
        "Promoting intergenerational housing is intended to counter loneliness among older people more effectively.",
        "Had preventive health programs been expanded earlier, care costs might have remained lower overall.",
        "The report shows how age structure influences demand for medical services across different regions.",
        "Strengthening the silver economy is seen as an opportunity for innovation and new employment fields.",
        "Although immigration is widely debated, social acceptance depends on many economic and cultural factors.",
        "The challenge is not political debate but long term viable solutions for the labor market.",
        "Improving working conditions in care professions should alleviate the severe shortage of qualified staff.",
        "The minister emphasized that demographic change must not become a burden borne by a single generation.",
        "Developing age appropriate housing requires cooperation among municipalities, builders, and care providers.",
        "It is expected that labor force participation among older workers will continue to rise in coming years.",
        "Although mortality rates remained stable, the pandemic altered public perceptions of risk and prevention.",
        "Supporting families with several children is frequently cited as a contribution to long term demographic stability.",
        "Building an age secure society remains a central task for policymakers and the private sector alike.",
    ],
    # Batch 6 (126–150): Energy Transition
    [
        "The energy transition requires a comprehensive transformation of generation, networks, and consumption patterns.",
        "Although renewable capacity is expanding, supply security still depends partly on fossil fuel sources.",
        "It is expected that electricity prices will remain volatile until decarbonization is largely completed.",
        "Reducing greenhouse gas emissions is the central objective of national and European climate policy.",
        "Despite numerous proposals, grid expansion is advancing slowly in several regions of the country.",
        "The energy minister explained that the coal exit must be planned in a fair way.",
        "Storing renewable energy presents technical and economic challenges that researchers are actively tackling.",
        "Household energy consumption should be reduced through renovation and efficiency measures, officials stated.",
        "Funding for hydrogen projects was increased in the budget to accelerate pilot initiatives nationwide.",
        "If approval procedures were streamlined, the expansion of wind power could advance more rapidly.",
        "Although solar installations are increasing, cross regional transmission capacity remains insufficient for demand.",
        "It is feared that delays in grid expansion could jeopardize the country's medium term climate targets.",
        "Converting industry to climate neutral processes requires high investment and reliable long term planning security.",
        "Had more been invested in grid infrastructure earlier, bottlenecks in electricity distribution would be smaller.",
        "The report documents how the coal exit could affect economically weaker regions over the next decade.",
        "Strengthening community energy cooperatives is viewed as a way to increase local acceptance of projects.",
        "Although the technology exists, financing large offshore installations remains complex for many investors.",
        "The obstacle is not political commitment but the speed of practical implementation on the ground.",
        "Introducing dynamic electricity tariffs should align consumption more flexibly with fluctuating renewable supply.",
        "The expert emphasized that the energy transition must not compromise supply security under any circumstances.",
        "Replacing old heating systems is being facilitated through subsidies and advisory services for households.",
        "It is assumed that electric mobility and renewables will become increasingly intertwined in future systems.",
        "Although emissions are falling, the transport sector remains one of the largest climate policy challenges.",
        "Expanding geothermal and biomass capacity should increase diversity in regional energy mixes across the country.",
        "Achieving climate neutrality by mid century requires decisive action from all economic and political actors.",
    ],
    # Batch 7 (151–175): Digital Privacy
    [
        "Protecting digital privacy is gaining importance amid the widespread collection of personal online data.",
        "Although new laws have been adopted, controlling large technology platforms remains a major challenge.",
        "It is expected that stricter requirements for handling personal data will take effect next year.",
        "Transparency in algorithmic decision making is a central concern for many users in everyday digital life.",
        "Despite available encryption, many services still store sensitive information in inadequately protected form.",
        "The data protection officer explained that data minimization must be integrated consistently into business processes.",
        "User consent must be understandable, voluntary, and revocable at any time under current regulations.",
        "Citizens should know which profiles are created about them online, the regulator said in a statement.",
        "Monitoring online behavior through advertising networks raises significant legal and ethical questions for society.",
        "If tracking were restricted by default, trust in digital services could increase noticeably among consumers.",
        "Although the application is popular, its privacy policy was criticized as unclear by consumer advocates.",
        "It is feared that state surveillance tools could restrict fundamental rights on the internet permanently.",
        "Anonymizing research data should protect sensitive information in scientific studies more effectively.",
        "Had security vulnerabilities been closed earlier, data loss affecting millions of users could have been avoided.",
        "The report recommends incorporating privacy by design from the outset in new product development cycles.",
        "Enforcing the data protection regulation requires sufficient staff and effective financial penalties for violations.",
        "Although users want more control, technical settings are often too complex for non expert audiences.",
        "The problem is not insufficient legal standards but weak oversight capacity in several member states.",
        "Restricting third party cookies should reduce invisible profiling practices across major commercial websites.",
        "The commissioner emphasized that privacy must not be traded for convenience in public digital services.",
        "Biometric identification systems are being debated because of their impact on anonymity in public spaces.",
        "It is recommended that children receive stronger protection against manipulative design in online platforms.",
        "Although companies promise greater transparency, internal documentation on data flows is rarely made public.",
        "Promoting digital literacy should enable citizens to evaluate privacy risks when using online services.",
        "Balancing innovation and privacy protection remains a persistent challenge for legislators and technology firms.",
    ],
    # Batch 8 (176–200): International Relations
    [
        "International relations are being reshaped by shifting alliances, economic rivalry, and shared global risks.",
        "Although diplomatic talks continue, trust between several major powers has deteriorated markedly in recent years.",
        "It is expected that multilateral forums will play a decisive role in managing future security crises.",
        "Coordinated sanctions were imposed to pressure the government without resorting immediately to military action.",
        "Despite humanitarian appeals, delivering aid across closed borders remains extremely difficult in practice.",
        "The ambassador explained that regional stability depends on credible commitments to conflict prevention measures.",
        "Arms control agreements must be updated to reflect new technologies and changing strategic environments.",
        "Neighboring states should resolve territorial disputes peacefully, the secretary general said at the summit.",
        "The deployment of peacekeeping forces was authorized after lengthy negotiations in the security council.",
        "If dialogue channels were maintained, escalation between rival factions might be prevented more effectively.",
        "Although a ceasefire was announced, violations were reported within hours by independent observers on the ground.",
        "It is feared that prolonged instability could trigger large scale displacement across several border regions.",
        "Strengthening international law is regarded as essential for limiting the use of force in global politics.",
        "Had mediation efforts begun earlier, the humanitarian crisis might have remained less severe overall.",
        "The report notes that economic interdependence can both constrain and intensify political conflict simultaneously.",
        "Confidence building measures are viewed as a prerequisite for substantive negotiations on long standing disputes.",
        "Although alliances were renewed, burden sharing arrangements continue to provoke disagreement among member states.",
        "The difficulty lies not in declared foreign policy goals but in coordinating action among diverse partners.",
        "Promoting cultural exchange should reduce prejudice and support cooperation between societies with different traditions.",
        "The foreign minister emphasized that national interests must not override collective security obligations indefinitely.",
        "Nuclear proliferation remains a priority despite setbacks in several high profile diplomatic negotiations.",
        "It is assumed that emerging powers will demand greater influence in institutions governing global governance.",
        "Although trade relations were normalized, strategic competition in technology sectors persists unabated.",
        "Establishing verification mechanisms should make future disarmament agreements more credible and enforceable.",
        "Managing global challenges peacefully remains the defining test of international cooperation in the twenty first century.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

START_ID = 1
BATCH_SIZE = 25
MIN_TOKENS = 10
MAX_TOKENS = 18

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
        for suffix in ("ing", "ed", "es"):
            if lemma.endswith(suffix) and len(lemma) > len(suffix) + 1:
                candidate = lemma[: -len(suffix)]
                if suffix == "ing" and candidate.endswith(("t", "d")):
                    candidate = candidate + "e"
                if len(candidate) >= 2:
                    lemma = candidate
                break
        else:
            fl = form.lower()
            if (
                lemma == fl
                and lemma.endswith("s")
                and len(lemma) > 2
                and not lemma.endswith("ss")
                and not fl.endswith(("ss", "us", "is", "ness", "ing", "ed"))
            ):
                lemma = lemma[:-1]

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




def count_tokens(sentence: str) -> int:
    import re
    return len(re.findall(r"[\w']+|[^\w\s]", sentence))


def main() -> None:
    import stanza


    for i, sent in enumerate(SENTENCES, START_ID):
        tc = count_tokens(sent)
        if tc < MIN_TOKENS or tc > MAX_TOKENS:
            print(f"PRE-CHECK en_b2_train_{i:03d}: {tc} tokens — {sent}")
            sys.exit(1)

    print("Loading Stanza...")
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma",
        use_gpu=False,
        verbose=False,
    )

    target_path = project_root / "data/handcraft/en/train/b2_new_001.conllu"
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
            sent_id = f"en_b2_train_{START_ID + global_idx:03d}"
            all_lines.extend(sentence_to_conllu(sent_id, sent, nlp))
            global_idx += 1

    conllu_text = "\n".join(all_lines) + "\n"
    target_path.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {global_idx} sentences to {target_path}")

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


if __name__ == "__main__":
    main()