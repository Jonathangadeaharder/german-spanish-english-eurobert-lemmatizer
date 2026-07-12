"""Generate b2_new_002.conllu (en_b2_train_201–400) in sub-batches of 25."""

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
    # Batch 1 (201–225): Globalization
    [
        "Cross border capital flows have grown sharply, yet smaller economies still struggle to attract sustainable investment.",
        "It is argued that fragmented regulatory standards increase compliance costs for firms operating in multiple jurisdictions.",
        "Although outsourcing lowered production costs, critics claim it weakened domestic manufacturing capacity over time.",
        "The committee noted that export oriented industries remain highly exposed to sudden shifts in global demand.",
        "If emerging markets were integrated more fairly, global growth patterns might become less uneven across regions.",
        "Trade agreements often reflect power asymmetries, which smaller partners find difficult to negotiate effectively.",
        "It is expected that digital platforms will reshape cross border commerce far beyond traditional retail channels.",
        "Despite lower shipping fees, geopolitical tensions continue to disrupt established maritime trade corridors worldwide.",
        "Multilateral negotiations stalled because member states disagreed sharply over tariff exemptions for sensitive sectors.",
        "The analyst warned that capital flight from unstable regions could trigger broader financial contagion across markets.",
        "While consumers benefit from cheaper imports, domestic producers face intense pressure from foreign competitors.",
        "It is feared that rising inequality within countries may undermine public support for further trade liberalization.",
        "Offshore financial centers, whose opacity attracts scrutiny, are central to debates on tax fairness globally.",
        "Had supply chain mapping been mandatory earlier, companies would have identified critical dependencies more quickly.",
        "Global labor mobility creates opportunities, but it also intensifies competition for skilled positions in host economies.",
        "The minister argued that national resilience should not be sacrificed for short term gains from globalization.",
        "Although foreign direct investment rose, local value added in manufacturing sectors remained disappointingly low.",
        "It is assumed that intellectual property rules will dominate the next phase of international economic negotiations.",
        "Promoting responsible sourcing standards should reduce environmental harm linked to global commodity supply chains.",
        "The report showed that currency pegs, once considered stable, can collapse under sustained external pressure.",
        "Economic interdependence limits conflict, yet it also makes societies vulnerable to disruptions beyond their borders.",
        "It is recommended that local producers receive support when exposed to unfair dumping practices from abroad.",
        "Although globalization expanded choice, many rural communities lost traditional livelihoods connected to local markets.",
        "The obstacle is not openness itself but the absence of institutions that distribute gains more equitably.",
        "Managing global risks cooperatively remains essential as climate shocks and pandemics ignore national boundaries completely.",
    ],
    # Batch 2 (226–250): Human Rights
    [
        "Civil society groups argued that emergency powers should never be used to suspend basic judicial safeguards.",
        "Although the treaty was ratified, enforcement mechanisms for protecting displaced persons remain seriously underfunded.",
        "It is expected that the court will clarify state obligations toward persons held in immigration detention.",
        "Freedom of association enables workers to organize collectively without fear of reprisal from employers or officials.",
        "Despite international condemnation, independent observers were denied access to detention facilities in the conflict zone.",
        "The rapporteur explained that online harassment had become a persistent threat to human rights defenders.",
        "Protecting privacy in public health programs requires balancing data collection with respect for individual autonomy.",
        "Victims of state violence should receive compensation and meaningful access to truth finding processes, advocates said.",
        "If witness protection were strengthened, more survivors might testify against powerful perpetrators of serious crimes.",
        "Although legislation was amended, discriminatory profiling continues to affect minority communities in daily policing.",
        "It is feared that shrinking civic space could undermine accountability for abuses committed by security forces.",
        "The right to education must be guaranteed even during protracted conflicts, the agency emphasized repeatedly.",
        "Had early warning systems functioned properly, mass atrocities in the region might have been prevented.",
        "LGBT communities face legal and social barriers that limit equal participation in public life across several states.",
        "Independent fact finding missions are regarded as vital tools for documenting violations in closed societies.",
        "Although shelters were expanded, funding for legal aid for survivors of domestic violence remains inadequate.",
        "The problem is not insufficient rhetoric but concrete measures to protect journalists facing credible threats.",
        "Banning indefinite detention without charge should be a minimum standard in every democratic legal system.",
        "The commissioner emphasized that humanitarian corridors must remain open to civilians trapped by active fighting.",
        "Expanding access to justice for rural populations requires mobile courts and stronger local legal assistance.",
        "It is recommended that reparations programs remedy historical injustices suffered by marginalized indigenous communities.",
        "Although progress was celebrated, trafficking networks continue exploiting vulnerable migrants along major transit routes.",
        "Condemning hate speech online should not be used to justify broad censorship of legitimate political dissent.",
        "Upholding the presumption of innocence remains fundamental even when public opinion demands rapid punitive action.",
        "Ensuring equal political representation for women requires sustained reform of electoral rules and party structures.",
    ],
    # Batch 3 (251–275): Scientific Ethics
    [
        "The review board insisted that risky experiments must not proceed without transparent justification and independent oversight.",
        "Although the trial produced promising results, placebo controls were criticized as ethically insufficient by several experts.",
        "It is expected that stricter authorship rules will reduce disputes over credit in collaborative research projects.",
        "Dual use research in biotechnology raises concerns about misuse, even when conducted with legitimate scientific aims.",
        "Despite informed consent forms being signed, participants often misunderstood the scope of data sharing arrangements.",
        "The committee explained that commercial pressure should never override patient safety in clinical development programs.",
        "Preventing selective reporting of outcomes requires registering study protocols before recruitment begins in medical trials.",
        "Researchers must avoid exploiting vulnerable populations, the ethics council stated during its annual public hearing.",
        "The deployment of predictive algorithms raises questions about bias, accountability, and informed patient choice.",
        "If retractions were handled more transparently, public trust in published science might recover more quickly.",
        "Although funding was approved, the gene therapy study was paused pending review of unforeseen safety signals.",
        "It is feared that competitive publication culture could encourage corner cutting in laboratories under intense performance pressure.",
        "Animal welfare standards should be harmonized across institutions to prevent regulatory arbitrage in experimental research.",
        "Had conflicts of interest been disclosed earlier, the advisory panel would have reached a different recommendation.",
        "The report recommends strengthening community engagement before launching large scale genomic screening initiatives nationwide.",
        "Scientific responsibility extends beyond laboratories, especially when findings influence policy during public health emergencies.",
        "Although open access is expanding, paywalls still limit equitable participation in global scientific discourse significantly.",
        "The issue is not whether innovation matters but how societies govern technologies with irreversible societal consequences.",
        "Independent replication studies are viewed as essential safeguards against overstated claims in high impact journals.",
        "The director emphasized that expedited approval must not weaken ethical review for experimental treatments on children.",
        "Sharing negative results should improve resource allocation and reduce unnecessary repetition of failed research approaches.",
        "It is assumed that ethics training will become mandatory for all principal investigators receiving public research grants.",
        "Although biosafety protocols exist, enforcement gaps were identified at multiple facilities handling dangerous pathogens.",
        "Promoting responsible communication of preliminary findings should limit harmful misinterpretation in media coverage worldwide.",
        "Respecting cultural values in field research remains obligatory even when local customs complicate standard methodological procedures.",
    ],
    # Batch 4 (276–300): Urbanization
    [
        "Rapid urbanization is reshaping land use patterns and placing unprecedented strain on municipal infrastructure systems.",
        "Although megacities attract investment, informal settlements on their peripheries often lack basic sanitation and services.",
        "It is expected that climate migration will accelerate population growth in already congested coastal metropolitan regions.",
        "Integrating public transit with cycling networks is regarded as essential for reducing congestion in expanding urban cores.",
        "Despite new housing projects, gentrification continues displacing lower income families from centrally located neighborhoods.",
        "The architect explained that mixed use development could revive underused commercial districts without encouraging sprawl.",
        "Upgrading informal settlements requires secure tenure rights and coordinated investment in water and electricity networks.",
        "City authorities should prioritize flood resilient design before approving construction in low lying urban expansion zones.",
        "If parking minimums were reduced, developers might build more compact housing near existing transit corridors.",
        "Although smart city initiatives expanded, digital services often failed to reach residents in peripheral urban districts.",
        "It is feared that unchecked speculation will make home ownership unattainable for younger households in major cities.",
        "Creating shaded pedestrian routes helps mitigate extreme heat exposure in densely built urban environments during summer.",
        "Had zoning rules encouraged infill development earlier, suburban land consumption would have remained significantly lower.",
        "The study documents how informal economies sustain millions of urban residents excluded from regulated labor markets.",
        "Participatory budgeting is viewed as a practical way to align public spending with neighborhood priorities in cities.",
        "Although skyscrapers dominate skylines, ground level public spaces remain scarce in many rapidly urbanizing districts.",
        "The challenge is not visionary master plans but reliable funding to maintain infrastructure as cities keep expanding.",
        "Promoting transit oriented development should reduce car dependency among residents commuting into congested city centers.",
        "The mayor emphasized that urban growth must include affordable rental options, not only luxury residential developments.",
        "Retrofitting aging apartment blocks for energy efficiency requires coordinated subsidies and technical support for owners.",
        "It is recommended that brownfield sites near rail stations be converted into climate resilient mixed income housing.",
        "Although satellite cities were promoted, weak regional governance delayed integration of transport and utility networks.",
        "Establishing night time safety programs should improve mobility for women using public transport in large urban areas.",
        "Managing informal street vending requires balanced policies that protect livelihoods while maintaining public order downtown.",
        "Building inclusive cities remains a defining challenge as urban populations are projected to rise for decades ahead.",
    ],
    # Batch 5 (301–325): Energy Transition
    [
        "The energy transition demands coordinated investment in generation, storage, and flexible distribution across entire regions.",
        "Although wind capacity increased, local opposition delayed several onshore projects deemed critical for national targets.",
        "It is expected that industrial electrification will accelerate once grid reliability improves in manufacturing heartlands.",
        "Phasing out fossil subsidies is regarded as necessary to align market incentives with long term decarbonization goals.",
        "Despite falling battery costs, integrating variable renewables still requires substantial upgrades to transmission infrastructure.",
        "The regulator explained that capacity markets must reward flexibility, not only continuous output from legacy plants.",
        "Expanding rooftop solar should empower households while reducing peak demand on overstretched urban electricity networks.",
        "Municipalities should retrofit public buildings first, officials argued, to demonstrate leadership in energy efficiency policy.",
        "If methane leaks were measured more accurately, climate impact assessments of gas infrastructure would become stricter.",
        "Although coal generation declined, gas fired plants continue operating during periods of low renewable output nationwide.",
        "It is feared that delayed permitting could undermine investor confidence in large scale offshore wind developments.",
        "Developing district heating networks offers a pathway to replace oil boilers in dense northern urban neighborhoods.",
        "Had building codes been tightened earlier, heating demand in new construction would have fallen more sharply.",
        "The assessment shows how job losses in fossil sectors may concentrate in regions with limited economic diversification.",
        "Community benefit agreements are viewed as effective tools for securing local acceptance of nearby renewable installations.",
        "Although hydrogen is promising, experts caution that production must rely on renewables to avoid emissions shifting.",
        "The barrier is not technological readiness but administrative complexity that slows deployment of clean energy projects.",
        "Introducing time variable tariffs should encourage consumers to shift usage toward periods of abundant renewable supply.",
        "The minister emphasized that affordability safeguards must accompany ambitious targets for phasing out fossil heating systems.",
        "Modernizing distribution grids is being financed through long term bonds tied to measurable reliability improvements.",
        "It is assumed that vehicle to grid services will become viable as electric car adoption expands rapidly.",
        "Although emissions fell in power generation, aviation and shipping remain difficult sectors for deep decarbonization efforts.",
        "Expanding pumped hydro storage should stabilize grids facing rising shares of intermittent solar and wind generation.",
        "Achieving net zero targets requires binding sectoral roadmaps, not merely aspirational declarations at international climate summits.",
        "Balancing supply security with climate goals remains the central policy dilemma facing energy planners across Europe.",
    ],
    # Batch 6 (326–350): Digital Privacy
    [
        "Digital privacy has become a fundamental concern as connected devices collect detailed behavioral data every day.",
        "Although consent banners proliferate, many users still lack meaningful control over how personal information is handled.",
        "It is expected that regulators will impose heavier fines on firms that ignore data minimization obligations systematically.",
        "Explaining automated decisions in plain language is regarded as essential for fair treatment in credit and hiring.",
        "Despite encryption improvements, metadata from messaging apps can still reveal sensitive patterns about users' social networks.",
        "The inspector explained that cross border data transfers require stronger safeguards after recent court rulings abroad.",
        "Limiting employee monitoring tools should protect dignity in workplaces adopting pervasive digital surveillance technologies.",
        "Citizens ought to review and correct inaccurate profiles held by data brokers, the ombudsman said.",
        "If default settings favored privacy, fewer users would unknowingly share location histories with third party advertisers.",
        "Although the platform updated policies, researchers found persistent gaps in documentation of automated content moderation.",
        "It is feared that facial recognition in public transport could normalize continuous identification without democratic debate.",
        "Pseudonymizing datasets should reduce reidentification risks when researchers analyze sensitive health records at scale.",
        "Had breach notifications been issued promptly, affected customers could have changed compromised passwords more quickly.",
        "The guidance recommends embedding privacy impact assessments into procurement processes for all new government software.",
        "Enforcing children's rights online requires age appropriate design standards and limits on profiling for commercial gain.",
        "Although users demand transparency, corporate transparency reports rarely explain how automated systems rank or filter information.",
        "The difficulty lies not in privacy laws but inconsistent enforcement capacity across national supervisory authorities.",
        "Restricting data retention periods should prevent indefinite storage of communications metadata by telecommunications providers.",
        "The advocate emphasized that security measures must not become pretexts for mass surveillance of lawful civic activity.",
        "Biometric payment systems are being questioned because they may erode anonymity in routine commercial transactions downtown.",
        "It is recommended that public agencies publish clear inventories of databases containing identifiable personal information.",
        "Although firms pledged compliance, independent audits revealed recurring failures to honor deletion requests from users.",
        "Promoting privacy preserving analytics should enable innovation without exposing individuals to harmful profiling practices online.",
        "Balancing fraud prevention with data protection remains contentious in debates over digital identity systems nationwide.",
        "Protecting digital autonomy in an age of pervasive tracking is a defining test for democratic institutions today.",
    ],
    # Batch 7 (351–375): Cross-cutting B2 topics
    [
        "Global supply chains and urban labor markets intersect as manufacturing hubs attract migrants seeking better wages.",
        "Although human rights norms are universal, their enforcement often weakens when economic growth is prioritized uncritically.",
        "It is expected that ethical review boards will scrutinize artificial intelligence tools used in public welfare programs.",
        "Rising sea levels threaten coastal cities, forcing planners to reconsider long term settlement patterns near shorelines.",
        "Despite renewable investment, energy poverty in urban peripheries continues limiting access to reliable electricity services.",
        "The panel explained that data driven policing raises privacy risks while promising more efficient crime prevention strategies.",
        "Protecting whistleblowers in multinational firms should strengthen accountability for labor abuses in overseas factories.",
        "Researchers argued that informed consent standards must evolve as wearable devices collect continuous health related data.",
        "If cross border tax cooperation improved, governments could fund social housing in rapidly urbanizing metropolitan areas.",
        "Although climate targets were announced, fossil dependent regions fear job losses without credible transition support packages.",
        "It is feared that algorithmic hiring tools could reproduce discrimination unless audits examine outcomes across demographic groups.",
        "Strengthening local governance is viewed as crucial for managing infrastructure pressures caused by sustained urbanization trends.",
        "Had privacy by default been mandated earlier, mobile applications would collect far less intrusive user data.",
        "The report links globalization of food systems with nutritional inequality in densely populated low income urban districts.",
        "Ensuring asylum seekers' digital identities are protected requires strict limits on data sharing among state agencies.",
        "Although scientific collaboration expanded online, unequal access to research networks persists between wealthy and poorer institutions.",
        "The challenge is not declaring rights but building institutions capable of delivering services in overcrowded urban settlements.",
        "Promoting community owned solar projects should combine energy transition goals with local economic participation in cities.",
        "The commissioner emphasized that trade policy must not undermine labor standards in countries competing for foreign investment.",
        "Developing ethical frameworks for urban surveillance cameras requires public deliberation beyond narrow security justifications alone.",
        "It is assumed that interoperable privacy standards will facilitate safer international data flows for medical research networks.",
        "Although megaprojects promised renewal, residents criticized insufficient consultation on displacement and cultural heritage preservation.",
        "Condemning forced evictions without resettlement plans should remain a priority for organizations monitoring urban housing rights.",
        "Balancing open science with participant confidentiality remains difficult when genomic datasets are shared across global platforms.",
        "Managing interconnected global risks demands cooperation among states, cities, and scientists facing shared vulnerabilities.",
    ],
    # Batch 8 (376–400): Cross-cutting B2 topics
    [
        "International trade in critical minerals shapes both energy transition strategies and geopolitical competition among major economies.",
        "Although privacy regulations tightened, state agencies still request vast communications data during national security investigations.",
        "It is expected that urban climate adaptation plans will prioritize vulnerable neighborhoods exposed to recurring heat waves.",
        "Respecting research participants' withdrawal rights must remain enforceable even when studies rely on large shared biobanks.",
        "Despite humanitarian law, attacks on hospitals in conflict zones continue provoking condemnation from human rights bodies.",
        "The analyst explained that decentralized energy systems could make cities more resilient to grid failures during storms.",
        "Preventing exploitation of migrant workers in global logistics hubs requires stronger labor inspections and accessible complaint mechanisms.",
        "Scientists should communicate uncertainty clearly, the editor argued, when preliminary findings influence public health guidance.",
        "If building retrofits were subsidized generously, older urban housing stock could meet stricter efficiency standards sooner.",
        "Although digital services expanded access to education, unequal connectivity deepened learning gaps in marginalized urban communities.",
        "It is feared that weak oversight of biometric databases could facilitate surveillance of activists and minority groups.",
        "Strengthening regional power grids is regarded as essential for integrating offshore wind into national energy transition plans.",
        "Had due diligence rules been enforced earlier, supply chain abuses in manufacturing countries might have declined.",
        "The briefing notes that urban air pollution disproportionately harms children living near busy roads in metropolitan areas.",
        "Guaranteeing safe online spaces for adolescents requires combining privacy protections with effective reporting tools against harassment.",
        "Although open trade created jobs, wage stagnation in deindustrialized regions fueled political skepticism toward globalization itself.",
        "The obstacle is not public support for renewables but bureaucratic delays that stall locally supported energy projects.",
        "Promoting ethical artificial intelligence in public administration should prevent opaque systems from deciding access to essential services.",
        "The minister emphasized that climate migration policy must respect human dignity while supporting receiving cities under strain.",
        "Modernizing water infrastructure in fast growing cities requires long term financing beyond short electoral budget cycles.",
        "It is recommended that researchers share deidentified datasets only under contracts specifying strict reuse and security conditions.",
        "Although smart meters improve efficiency, residents fear consumption data may reveal intrusive details about their habits.",
        "Establishing independent monitors for detention centers should improve accountability for abuses reported by civil society organizations.",
        "Balancing economic openness with social protection remains contentious in debates over globalization's long term domestic consequences.",
        "Building sustainable and rights respecting societies will require sustained cooperation across borders and policy domains.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

START_ID = 201
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

    target_path = project_root / "data/handcraft/en/train/b2_new_002.conllu"
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