"""Generate b2_new_004.conllu (en_b2_train_601–800) in sub-batches of 25."""

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
    # Batch: Climate Adaptation (601–625)
    [
        "Coastal cities are investing in flood barriers, yet rising seas still threaten low lying neighborhoods disproportionately.",
        "Although droughts intensified, farmers lacked insurance products tailored to climate related crop losses.",
        "It is expected that heat action plans will become mandatory for municipalities in regions facing extreme temperatures.",
        "Protecting wetlands should reduce flood damage, yet urban expansion continues draining ecosystems that absorb stormwater naturally.",
        "Despite early warning systems, vulnerable households often receive evacuation notices too late during rapidly developing storms.",
        "The planner explained that retrofitting buildings should prioritize homes occupied by elderly residents with limited mobility.",
        "Promoting drought resistant crops requires extension services that reach smallholders farming marginal lands in arid zones.",
        "If mangrove restoration expanded, coastal communities might withstand storm surges more effectively during cyclone seasons annually.",
        "Although emissions fell modestly, adaptation funding still lags behind humanitarian needs in disaster prone developing nations.",
        "It is feared that insurers may withdraw coverage from flood prone zones after repeated costly claims.",
        "Community preparedness drills are regarded as cost effective measures for reducing casualties during sudden flash flood events.",
        "Had cooling centers been mapped earlier, authorities would have sheltered more residents during severe summer heatwaves.",
        "The report shows how informal settlements suffer disproportionately when drainage infrastructure fails during monsoon rainfall intensification.",
        "Ensuring climate resilient infrastructure requires coordinating engineers, planners, and social workers serving affected neighborhoods consistently.",
        "Although forests were protected, wildfires still spread faster across landscapes altered by decades of suppression.",
        "The challenge is not uncertainty but reluctance to restrict development in coastal floodplains and erosion zones.",
        "Expanding green roofs should lower urban temperatures while managing stormwater runoff in densely built commercial districts.",
        "The minister emphasized that adaptation must tackle inequalities, not only protect high value coastal properties.",
        "Developing early season forecasts is being piloted to help pastoral communities relocate livestock before drought conditions worsen.",
        "It is recommended that building codes incorporate future rainfall projections rather than historical averages from previous decades.",
        "Although desalination expanded, energy costs limit its viability as a sole strategy for water security.",
        "Strengthening river basin governance should prevent upstream projects from worsening floods downstream in neighboring municipalities.",
        "Promoting nature based solutions should restore ecosystems that buffer communities against storms while supporting local livelihoods.",
        "Balancing adaptation costs with fiscal constraints remains contentious when voters prioritize immediate services over long term resilience.",
        "Building climate resilient societies will require redesigning insurance, housing, and emergency response together.",
    ],
    # Batch: Artificial Intelligence & Ethics (626–650)
    [
        "Artificial intelligence is transforming hiring, yet opaque algorithms may reproduce biases in historical employment data.",
        "Although regulators drafted guidelines, developers still deploy risky models without independent audits of fairness.",
        "It is expected that liability rules will clarify who bears responsibility when autonomous systems cause public harm.",
        "Protecting patient privacy requires strict limits on how medical algorithms retrieve sensitive records across interconnected hospital networks.",
        "Despite transparency pledges, proprietary training data remain inaccessible to researchers evaluating discriminatory outcomes in automated decisions.",
        "The ethicist explained that explainability should be mandatory for systems influencing credit, housing, and sentencing.",
        "Promoting human oversight in critical domains should prevent automated decisions from overriding professional judgment without appeal.",
        "If consent mechanisms were strengthened, users might understand how personal data fuels recommendation engines on platforms.",
        "Although productivity gains were celebrated, workplace surveillance tools intensified monitoring of employees without negotiated safeguards.",
        "It is feared that synthetic media could undermine trust in journalism unless verification tools reach citizens widely.",
        "Algorithmic impact assessments are regarded as necessary steps before governments deploy systems affecting welfare eligibility and benefits.",
        "Had dataset documentation been required earlier, researchers would have identified demographic gaps skewing facial recognition performance.",
        "The inquiry shows how predictive policing can reinforce over policing in neighborhoods already subject to intensive surveillance.",
        "Ensuring equitable access to artificial intelligence benefits requires public investment, not only private corporate innovation.",
        "Although chatbots improved service delivery, vulnerable users struggled when complex cases required empathetic human caseworker intervention.",
        "The obstacle is not capacity but incentives that reward speed over careful evaluation of social consequences.",
        "Expanding participatory design should involve communities in shaping systems that classify, rank, and allocate resources.",
        "The commissioner emphasized that security applications must respect civil liberties and avoid indiscriminate bulk data collection.",
        "Developing model card standards is being promoted to communicate limitations and known failure modes transparently.",
        "It is recommended that procurement rules require vendors to disclose environmental costs of training large language models.",
        "Although automation reduced routine tasks, displaced workers received insufficient support for transitioning into new skilled occupations.",
        "Strengthening cross border cooperation should harmonize rules governing autonomous vehicles operating near pedestrians in dense urban areas.",
        "Promoting open benchmarks for safety should enable independent researchers to test systems before regulators approve public deployment.",
        "Balancing innovation and precaution remains contentious when industry leaders argue regulation could slow domestic technological competitiveness.",
        "Governing artificial intelligence responsibly will require institutions capable of updating rules as capabilities evolve faster than legislation.",
    ],
    # Batch: Food Security & Agriculture (651–675)
    [
        "Global food prices surged after disruptions, yet small farmers received little benefit from higher retail margins.",
        "Although harvests improved locally, export restrictions abroad still threatened access to staple grains for dependent nations.",
        "It is expected that precision tools will spread as farmers seek to reduce fertilizer and water consumption.",
        "Protecting soil health should sustain yields, yet monoculture continues depleting nutrients across vast commercial farming regions.",
        "Despite food aid deliveries, conflict zones experienced severe malnutrition among children cut off from distribution networks.",
        "The agronomist explained that seed diversity should preserve crops resilient to pests and changing rainfall patterns.",
        "Promoting fair contracts requires tackling buyer power that squeezes margins for growers in global commodity chains.",
        "If storage facilities were upgraded, post harvest losses might decline where spoilage wastes substantial annual production.",
        "Although organic demand grew, certification costs barred many smallholders from entering premium markets in wealthy cities.",
        "It is feared that land grabbing could accelerate as investors acquire farmland for speculative purposes abroad.",
        "Agroecological practices are viewed as promising alternatives to input intensive models dependent on volatile global fertilizer markets.",
        "Had irrigation been maintained earlier, rural communities would have avoided crop failures during prolonged dry seasons.",
        "The assessment shows how trade liberalization exposed producers to import surges that undercut local market prices.",
        "Ensuring nutritious school meals requires coordinating agriculture, education, and health ministries serving impoverished rural children.",
        "Although aquaculture expanded, antibiotic overuse raised concerns about long term sustainability of intensive fish farming.",
        "The challenge is not production alone but distribution that leaves nutritious food unaffordable in low income neighborhoods.",
        "Expanding farmer cooperatives should improve bargaining power and access to credit for members cultivating heterogeneous small plots.",
        "The minister emphasized that agricultural subsidies must support environmental stewardship rather than encourage wasteful overproduction alone.",
        "Developing climate smart extension services is being piloted to advise pastoralists on grazing during increasingly erratic seasons.",
        "It is recommended that public procurement favor locally sourced produce to stabilize demand for regional farming communities.",
        "Although biotechnology advanced, public skepticism slowed adoption of improved varieties targeting vitamin deficiencies in staple crops.",
        "Strengthening food safety inspection should prevent contaminated products from reaching markets served by informal street vendors.",
        "Promoting urban gardening should supplement household diets where fresh produce remains scarce and expensive in inner cities.",
        "Balancing export earnings with domestic food security remains difficult when governments prioritize foreign exchange from cash crops.",
        "Building resilient food systems will require integrating trade policy, rural development, and nutrition programs nationwide.",
    ],
    # Batch: Housing & Urban Planning (676–700)
    [
        "Affordable housing shortages intensified as investors purchased units primarily for short term rental income rather than occupancy.",
        "Although zoning reforms were announced, restrictive rules still blocked multifamily housing near transit corridors.",
        "It is expected that inclusionary policies will require developers to reserve units for moderate income households.",
        "Protecting tenants from displacement should accompany regeneration, yet rising rents pushed longtime residents from revitalized districts.",
        "Despite vacancy taxes, empty investment properties remained common where local workers could not afford market rents.",
        "The architect explained that mixed use developments should integrate shops, offices, and homes reducing commutes.",
        "Promoting cooperative ownership requires financing mechanisms accessible to households excluded from conventional mortgage lending markets.",
        "If building permits were streamlined responsibly, municipalities might accelerate construction without sacrificing safety or environmental standards.",
        "Although public housing expanded, maintenance backlogs left residents with mold, heating failures, and leaks.",
        "It is feared that speculative bubbles could burst, leaving buyers with negative equity in overheated markets.",
        "Transit oriented planning is regarded as essential for curbing car dependency in rapidly growing metropolitan regions.",
        "Had anti speculation measures been enforced, foreign investment in residential property would have been less disruptive.",
        "The study documents how redlining legacies still shape racial segregation patterns across many American cities today.",
        "Ensuring accessible housing for disabled residents requires universal design standards enforced during new construction and renovation.",
        "Although tiny homes were promoted, zoning barriers limited their potential as emergency shelter solutions.",
        "The obstacle is not innovation but opposition from homeowners benefiting from scarcity driven property value increases.",
        "Expanding community land trusts should keep housing affordable by separating land ownership from structures occupied by residents.",
        "The mayor emphasized that demolition without replacement should not occur when affordable units leave tight markets.",
        "Developing modular construction capacity is being explored to reduce costs for building social housing at scale.",
        "It is recommended that energy retrofits be subsidized for low income tenants in poorly insulated rental buildings.",
        "Although rent controls were debated, economists disagreed about whether caps reduce investment without improving affordability.",
        "Strengthening tenant organizing should empower renters to challenge unlawful evictions and negotiate repairs with negligent owners.",
        "Promoting walkable neighborhoods should improve public health while lowering transportation expenses for households without private vehicles.",
        "Balancing density with neighborhood character remains contentious when residents oppose taller buildings near historic streetscapes.",
        "Creating inclusive cities will require confronting markets that treat homes primarily as assets rather than necessities.",
    ],
    # Batch: Mental Health & Wellbeing (701–725)
    [
        "Mental health services remain underfunded, yet counseling demand surged after crises disrupted routines and connections.",
        "Although awareness campaigns expanded, stigma still prevents many men from seeking help for depression and anxiety.",
        "It is expected that schools will employ psychologists to support students coping with bullying and academic pressure.",
        "Protecting staff wellbeing should be prioritized, yet high caseloads leave clinicians vulnerable to burnout and fatigue.",
        "Despite teletherapy options, rural residents lacked broadband access for confidential sessions with qualified mental health professionals.",
        "The psychiatrist explained that early intervention should target adolescents before mild symptoms develop into severe conditions.",
        "Promoting peer support requires training volunteers who recognize warning signs and guide friends toward professional assistance.",
        "If crisis hotlines were staffed adequately, fewer individuals might reach emergency rooms without timely counseling first.",
        "Although wellness initiatives multiplied, they often failed to tackle insecure contracts and excessive overtime demands.",
        "It is feared that social media algorithms could worsen body image among teenagers exposed to filtered content.",
        "Trauma informed care is regarded as essential, yet many providers lack specialized training and supervision.",
        "Had community centers remained open, isolated elderly residents would have maintained ties reducing loneliness related decline.",
        "The survey shows how financial stress correlates with sleep disturbances among households facing unstable employment.",
        "Ensuring culturally competent therapy requires clinicians who understand how migration and discrimination shape clients lived experiences.",
        "Although antidepressant use rose, psychotherapy remained limited by long waiting lists in publicly funded health systems.",
        "The barrier is not indifference but fragmented services forcing patients to navigate referrals without coordinated management.",
        "Expanding school based mental health teams should identify struggling pupils before academic failure becomes entrenched.",
        "The director emphasized that involuntary treatment must follow safeguards protecting autonomy and dignity during acute crisis.",
        "Developing digital therapeutics is being evaluated as supplements when qualified therapists are unavailable locally.",
        "It is recommended that employers provide paid mental health days without requiring disclosure of sensitive diagnoses.",
        "Although meditation apps proliferated, evidence based treatments should not be replaced by products lacking clinical oversight.",
        "Strengthening suicide prevention should combine means restriction, crisis training, and follow up after hospital discharge.",
        "Promoting sleep hygiene education should tackle how shift work and screen exposure disrupt circadian rhythms among adults.",
        "Balancing confidentiality with safety remains difficult when therapists must breach privacy to prevent imminent self harm.",
        "Building mentally healthy societies will require integrating psychological support into education, labor policy, and healthcare.",
    ],
    # Batch: International Development (726–750)
    [
        "Development agencies pledged increased aid, yet disbursement delays left partner governments unable to fund urgent programs.",
        "Although poverty rates declined globally, fragile states experienced conflict that reversed gains in health and education.",
        "It is expected that climate finance will channel resources toward adaptation in low income countries facing disasters.",
        "Protecting aid effectiveness requires transparent procurement, yet corruption scandals undermined trust in financed infrastructure schemes.",
        "Despite debt relief initiatives, servicing obligations consumed budgets for teachers, vaccines, and rural roads.",
        "The economist explained that local procurement should strengthen domestic industries rather than importing donor country goods.",
        "Promoting women's economic participation requires legal reforms and childcare support enabling mothers to enter labor markets.",
        "If evaluation standards were harmonized, donors might compare program outcomes fairly across competing intervention models.",
        "Although microfinance expanded, over indebted borrowers faced harassment from lenders charging unsustainable interest rates.",
        "It is feared that geopolitical rivalry could politicize assistance, diverting funds toward allies rather than need.",
        "Community driven development is viewed as promising when residents control priorities for small scale investments in villages.",
        "Had sanitation projects been maintained, childhood mortality from waterborne diseases would have fallen in rural districts.",
        "The review documents how loan conditionality forced governments to cut social spending during economic downturns.",
        "Ensuring humanitarian access requires negotiating with armed groups while protecting aid workers delivering food and supplies.",
        "Although digital identity systems spread, undocumented groups remained excluded from social protection and banking services.",
        "The challenge is not generosity alone but coordination failures duplicating efforts while leaving gaps in border regions.",
        "Expanding south south cooperation should enable countries with recent experience to share lessons beyond traditional donors.",
        "The ambassador emphasized that sovereignty concerns must be respected when designing programs through national institutions.",
        "Developing impact bonds for education is being tested to align returns with measurable improvements in learning outcomes.",
        "It is recommended that aid agencies publish disaggregated data on benefits reaching women and remote communities.",
        "Although remittances supported families, volatile exchange rates eroded purchasing power in overseas dependent economies.",
        "Strengthening tax administration should broaden domestic revenue bases reducing reliance on unpredictable external grant financing.",
        "Promoting fair trade certification should help cooperatives obtain stable prices for coffee, cocoa, and exports.",
        "Balancing growth and environmental protection remains contentious when governments prioritize extractive projects promising rapid revenues.",
        "Achieving sustainable development will require aligning trade, migration, and climate policies operating in separate silos.",
    ],
    # Batch: Sports & Society (751–775)
    [
        "Professional leagues generate enormous revenue, yet grassroots clubs still lack safe facilities and qualified volunteer coaches.",
        "Although anti doping rules tightened, poor nations struggled to obtain testing resources of wealthy federations.",
        "It is expected that women's competitions will receive more coverage as sponsors respond to growing audience demand.",
        "Protecting young athletes should include limits on training hours, yet academies sometimes prioritize results over welfare.",
        "Despite inclusion policies, disabled participants still encountered inaccessible venues and inadequate adaptive equipment at events.",
        "The coach explained that fair play education should begin early to counter abuse in competitive youth leagues.",
        "Promoting physical activity in schools requires reversing cuts to recess and physical education in overcrowded campuses.",
        "If concussion protocols were enforced consistently, fewer players might return to contact sports before recovering fully.",
        "Although mega events boosted tourism, host cities incurred debt maintaining stadiums underused after tournaments concluded.",
        "It is feared that gambling sponsorship could normalize betting among adolescents watching matches on popular online platforms.",
        "Community sports programs are regarded as valuable for integration when they bring together residents from diverse backgrounds.",
        "Had safeguarding policies been implemented earlier, abusive coaches would have been removed before harming vulnerable minors.",
        "The investigation shows how transfer markets treat players as commodities, ignoring mental health impacts of relocations.",
        "Ensuring gender equitable funding requires auditing federations allocating disproportionate resources to men's teams and partnerships.",
        "Although esports gained recognition, debates continue over whether screen based competition belongs alongside traditional athletic disciplines.",
        "The obstacle is not talent but governance shielding officials from accountability after corruption and match fixing scandals.",
        "Expanding adaptive sports leagues should provide competitive pathways for participants with disabilities seeking meaningful achievement.",
        "The president emphasized that human rights must guide decisions about hosting events in countries restricting expression.",
        "Developing low cost equipment initiatives is being promoted to enable children in informal settlements to play sports.",
        "It is recommended that brain injury research receive funding proportional to revenues generated by contact sports industries.",
        "Although nationalism intensified rivalries, international competitions still fostered moments of solidarity transcending political tensions temporarily.",
        "Strengthening whistleblower protections should encourage insiders to report bribery affecting selection decisions in sports federations.",
        "Promoting lifelong recreational sport should reduce chronic disease burdens linked to sedentary lifestyles in aging populations.",
        "Balancing commercialization with integrity remains contentious when owners prioritize relocation profits over loyal local fan communities.",
        "Using sport for social good will require reforming institutions concentrating power among officials distant from athletes.",
    ],
    # Batch: Science Communication (776–800)
    [
        "Scientists struggle to convey uncertainty, yet policymakers demand definitive answers during fast moving public health crises.",
        "Although open access expanded, paywalls still block clinicians in poor countries from reading cutting edge research.",
        "It is expected that preprint servers will share findings before peer review during future epidemic responses.",
        "Protecting credibility requires correcting errors publicly, yet retractions remain rare when prominent labs resist acknowledging flaws.",
        "Despite outreach programs, communities affected by pollution still distrust researchers perceived as aligned with industry funders.",
        "The communicator explained that metaphors should simplify concepts without distorting vaccine, climate, or genetic evidence.",
        "Promoting science literacy in media requires training journalists to evaluate statistics and distinguish correlation from causation.",
        "If citizen science platforms expanded, volunteers might collect environmental data complementing professional monitoring in underserved regions.",
        "Although podcasts popularized research, sensational episode titles sometimes misrepresented tentative findings as settled scientific consensus.",
        "It is feared that harassment campaigns could silence early career researchers sharing evidence on controversial topics online.",
        "Interactive museum exhibits are effective ways to engage families skeptical of abstract arguments in policy briefs.",
        "Had conflict disclosures been clearer, public debate over dietary guidelines would have examined funding sources transparently.",
        "The panel shows how algorithmic amplification rewards outrage, making nuanced commentary less visible than provocative misinformation.",
        "Ensuring equitable participation in research requires translating consent forms into languages spoken by study volunteers.",
        "Although press offices grew, overstated announcements sometimes preceded independent replication of headline grabbing discoveries.",
        "The difficulty lies not in curiosity but in educational systems providing insufficient statistical reasoning before adulthood.",
        "Expanding classroom dialogue with scientists should humanize research and counter stereotypes portraying experts as detached elites.",
        "The editor emphasized that peer review must evolve to detect fraud faster without delaying legitimate urgent findings.",
        "Developing plain language summaries is being mandated so participants understand how studies using their data influence practice.",
        "It is recommended that platforms label content referencing health claims with links to verified institutional information sources.",
        "Although visualization tools improved, misleading charts circulated when axes were truncated to exaggerate trend changes.",
        "Strengthening media partnerships should help local outlets cover environmental science beyond brief wire reports on disasters.",
        "Promoting constructive skepticism should teach students to question sources while respecting evidence from replicated experimental work.",
        "Balancing speed and accuracy remains contentious when scientists feel pressured to comment before data are analyzed thoroughly.",
        "Strengthening public trust in science will require transparency and institutions funding communication as seriously as discovery.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

START_ID = 601
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

    target_path = project_root / "data/handcraft/en/train/b2_new_004.conllu"
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