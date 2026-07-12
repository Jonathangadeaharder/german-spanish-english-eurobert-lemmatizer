"""Generate b2_new_003.conllu (en_b2_train_401–600) in sub-batches of 25."""

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
    # Batch 1 (401–425): Education Policy
    [
        "Although funding increased, schools in deprived districts still struggle to recruit qualified teachers.",
        "It is expected that digital literacy programs will become mandatory in national secondary curricula.",
        "Inclusive classrooms benefit all learners, yet many teachers lack training in differentiated instruction.",
        "The ministry explained that vocational pathways should be valued equally alongside university entrance routes.",
        "Despite smaller class sizes, achievement gaps between advantaged and disadvantaged students remain wide.",
        "Promoting critical thinking requires moving beyond rote memorization and narrow standardized test preparation.",
        "If counseling services were expanded, dropout rates in technical colleges might decline more rapidly.",
        "Although tuition was frozen, living costs still push low income students toward precarious part time work.",
        "It is feared that chronic underfunding could undermine faith in education as a pathway for mobility.",
        "Universities whose research attracts funding often neglect teaching quality in overcrowded undergraduate programs.",
        "Had literacy coaching begun earlier, struggling readers in primary schools would have improved faster.",
        "Lifelong learning policies should enable mid career workers to adapt to changing labor market demands.",
        "Although rankings dominate headlines, graduate employment outcomes vary widely across comparable institutions.",
        "The challenge is not curriculum ambition but reliable support for teachers implementing ambitious reforms.",
        "Expanding early childhood education should reduce later remediation costs and narrow initial learning gaps.",
        "The rector emphasized that academic freedom must not be sacrificed for short term reputational concerns.",
        "It is recommended that assessment methods reward analytical writing rather than repetitive factual recall.",
        "Although online platforms expanded access, digital divides limit participation for rural and poorer households.",
        "Strengthening school leadership is regarded as essential for sustaining improvement across disadvantaged communities.",
        "Banning mobile devices during lessons may improve focus, yet engaging instructional design still matters most.",
        "It is assumed that bilingual programs will expand as cities integrate growing numbers of migrant children.",
        "Although grants were announced, libraries in public schools still lack current textbooks and learning materials.",
        "Encouraging student participation in governance should foster responsibility and democratic habits from early ages.",
        "Managing classroom diversity effectively remains a defining professional skill for teachers in urban schools.",
        "Building equitable education systems requires sustained investment beyond symbolic reforms announced during election cycles.",
    ],
    # Batch 2 (426–450): Public Health
    [
        "Public health agencies argued that prevention programs deserve stable funding, not only emergency crisis allocations.",
        "Although vaccination rates rose, vaccine hesitancy persists in communities with limited access to trusted information.",
        "It is expected that mental health services will be integrated more fully into primary care networks nationwide.",
        "Reducing air pollution near schools should lower asthma rates among children in densely populated urban districts.",
        "Despite expanded screening, late diagnosis of chronic diseases continues imposing heavy costs on health systems.",
        "The epidemiologist explained that contact tracing systems must protect privacy while enabling rapid outbreak response.",
        "Promoting healthy diets requires tackling food deserts and aggressive marketing of heavily refined products.",
        "If harm reduction programs were scaled up, overdose deaths in affected neighborhoods might fall noticeably.",
        "Although hospitals upgraded equipment, staffing shortages limit capacity during seasonal surges of respiratory illness.",
        "It is feared that antimicrobial resistance could undermine routine treatments unless prescribing practices change quickly.",
        "Community health workers are viewed as vital links between marginalized populations and formal medical services.",
        "Had tobacco taxes been raised earlier, smoking prevalence among adolescents would have declined more sharply.",
        "The report shows how social determinants shape health outcomes more than clinical interventions alone can explain.",
        "Ensuring equitable vaccine distribution during pandemics requires planning that reaches informal workers and remote villages.",
        "Although telemedicine expanded, elderly patients without digital skills were excluded from convenient remote consultations.",
        "The obstacle is not medical knowledge but fragmented governance that slows coordinated responses to outbreaks.",
        "Expanding mental health first aid training should help workplaces identify colleagues at risk of serious crisis.",
        "The minister emphasized that health data sharing must respect consent and avoid punitive surveillance of patients.",
        "Developing resilient supply chains for essential medicines requires diversifying production beyond a few global suppliers.",
        "It is recommended that sugary beverage taxes fund programs promoting physical activity in underserved communities.",
        "Although life expectancy improved, health inequalities between regions widened after recent cuts to preventive services.",
        "Strengthening wastewater surveillance could provide early warnings of viral outbreaks in rapidly growing metropolitan areas.",
        "Promoting rest and recovery among healthcare staff should reduce burnout linked to chronic understaffing pressures.",
        "Balancing economic reopening with infection control remains contentious when case data are incomplete or delayed.",
        "Protecting population health in aging societies will require rethinking how care, housing, and transport interconnect.",
    ],
    # Batch 3 (451–475): Migration & Integration
    [
        "Migration policy debates often overlook the economic contributions migrants make to aging labor markets.",
        "Although asylum applications fell, processing backlogs still leave applicants in prolonged uncertainty without work rights.",
        "It is expected that integration courses will emphasize language skills alongside practical knowledge of civic institutions.",
        "Host communities benefit when migrants fill vacancies in sectors facing persistent shortages of qualified workers.",
        "Despite integration programs, discrimination in housing and hiring continues limiting opportunities for newly arrived families.",
        "The advisor explained that family reunification rules should balance security concerns with humanitarian obligations consistently.",
        "Promoting intercultural dialogue in schools should reduce tensions arising from rapid demographic change in towns.",
        "If recognition of foreign credentials improved, skilled migrants could contribute sooner to national innovation ecosystems.",
        "Although border controls tightened, irregular routes still expose vulnerable people to exploitation by criminal networks.",
        "It is feared that inflammatory rhetoric could erode public support for sensible, evidence based migration policies.",
        "Local authorities are key actors for delivering housing, language training, and employment support services.",
        "Had reception centers been funded adequately, overcrowding and poor conditions would have been avoided entirely.",
        "The study documents how remittances support households abroad while migrants rebuild lives in unfamiliar host societies.",
        "Ensuring safe legal pathways for labor migration should reduce reliance on dangerous undocumented cross border journeys.",
        "Although diversity is celebrated nationally, migrants in rural areas often experience isolation and limited service access.",
        "The challenge is not cultural difference itself but institutions that fail to translate diversity into equal opportunity.",
        "Expanding mentorship schemes should help young migrants navigate education systems and early career decisions successfully.",
        "The commissioner emphasized that deportation procedures must respect due process and protect unaccompanied minors consistently.",
        "Developing mobile language classes is being piloted to reach workers employed far from urban integration centers.",
        "It is recommended that employers receive incentives to offer apprenticeships tailored to migrants with relevant experience.",
        "Although refugees were welcomed initially, housing shortages gradually strained relations in several receiving municipalities.",
        "Strengthening anti discrimination enforcement should improve outcomes for second generation citizens facing subtle labor market barriers.",
        "Promoting accurate media coverage of migration should counter stereotypes that fuel resentment during economic downturns.",
        "Balancing humanitarian protection with capacity limits remains politically difficult in societies anxious about rapid population change.",
        "Building cohesive multicultural communities requires sustained investment in inclusion, not only border management measures alone.",
    ],
    # Batch 4 (476–500): Media & Journalism
    [
        "Independent journalism faces growing pressure from political attacks, corporate consolidation, and declining advertising revenues.",
        "Although fact checking expanded, misleading content still spreads faster than corrections on major social platforms.",
        "It is expected that public broadcasters will face renewed scrutiny over editorial independence and government funding models.",
        "Protecting sources enables reporters to investigate corruption, yet legal threats against whistleblowers have intensified recently.",
        "Despite subscriptions, local news outlets in smaller towns keep closing, leaving coverage gaps for communities.",
        "The editor explained that investigative teams require sustained funding beyond short term grants tied to trending topics.",
        "Promoting media literacy in schools should help adolescents evaluate sensational headlines and manipulated images critically.",
        "If platform algorithms favored verified reporting, misinformation might circulate less widely during election campaign seasons.",
        "Although press freedom is constitutionally guaranteed, journalists covering protests have faced harassment and arbitrary detention abroad.",
        "It is feared that self censorship could grow when outlets depend on powerful advertisers or state contracts.",
        "Collaborative cross border reporting is viewed as essential for exposing tax evasion and environmental crimes spanning jurisdictions.",
        "Had transparency laws been stronger, ownership structures behind influential media conglomerates would be clearer to citizens.",
        "The inquiry shows how sensational coverage can amplify public anxiety beyond levels justified by available evidence.",
        "Ensuring reporter safety in conflict zones requires training, insurance, and diplomatic support from governments.",
        "Although podcasts diversified commentary, they also enabled unchecked conspiracy narratives to reach large audiences uncritically.",
        "The problem is not audience demand for news but business models that reward speed over careful verification.",
        "Expanding community radio stations should give marginalized groups space to debate local issues on their own terms.",
        "The publisher emphasized that editorial standards must not be weakened to maximize clicks and engagement metrics online.",
        "Developing ethical guidelines for artificial intelligence generated content should prevent fabricated stories from flooding newsfeeds.",
        "It is recommended that archives of public interest reporting remain accessible after outlets merge or cease operations.",
        "Although regulations were proposed, enforcement against coordinated disinformation campaigns remains uneven across different national jurisdictions.",
        "Strengthening protections for freelance journalists should close gaps in legal support and safety resources they currently face.",
        "Promoting diverse newsrooms should improve coverage of stories affecting communities historically ignored by mainstream outlets nationwide.",
        "Balancing national security claims with press freedom remains contentious when governments seek to restrict reporting on intelligence.",
        "Sustaining quality journalism in digital markets will require new funding models that reward accuracy rather than outrage.",
    ],
    # Batch 5 (501–525): Criminal Justice Reform
    [
        "Criminal justice reform advocates argued that sentencing guidelines should prioritize rehabilitation over purely punitive measures.",
        "Although crime rates fell, prison overcrowding persists because pretrial detention is used excessively in many cases.",
        "It is expected that restorative justice programs will expand for nonviolent offenses involving juvenile defendants nationwide.",
        "Reducing racial disparities in policing requires collecting transparent data and holding departments accountable for biased practices.",
        "Despite body camera mandates, footage is rarely released promptly when misconduct allegations arise during street encounters.",
        "The judge explained that cash bail systems unfairly detain poor defendants who pose no serious flight risk.",
        "Promoting diversion programs for mental illness should keep vulnerable individuals out of overcrowded correctional facilities unnecessarily.",
        "If parole boards received better training, reintegration outcomes for former prisoners might improve across disadvantaged neighborhoods.",
        "Although drug courts expanded, treatment waiting lists still delay help for defendants struggling with severe addiction.",
        "It is feared that mandatory minimum sentences could return if politicians respond to isolated violent incidents emotionally.",
        "Community policing is regarded as promising, yet it requires sustained funding beyond short pilot project cycles.",
        "Had eyewitness procedures been reformed earlier, wrongful convictions stemming from unreliable testimony would have decreased significantly.",
        "The review documents how solitary confinement harms mental health without demonstrably improving safety inside prison institutions.",
        "Ensuring competent legal representation for indigent defendants remains a constitutional obligation that many jurisdictions underfund systematically.",
        "Although expungement laws were approved, complicated application processes deter many eligible citizens from clearing old records.",
        "The obstacle is not reform proposals but political resistance from groups benefiting from current punitive institutional arrangements.",
        "Expanding vocational training in prisons should improve employment prospects and reduce repeat offending after release substantially.",
        "The attorney emphasized that juvenile records must be sealed to prevent lifelong stigma from adolescent mistakes.",
        "Developing independent oversight of internal affairs investigations is being debated after repeated failures to discipline abusive officers.",
        "It is recommended that victim support services receive funding proportional to resources devoted to prosecution and incarceration.",
        "Although decriminalization reduced arrests, public health responses to substance use disorders still lack adequate treatment capacity.",
        "Strengthening reentry housing programs should prevent homelessness among former prisoners immediately after they leave institutional settings.",
        "Promoting transparent prosecution guidelines should reduce perception that justice depends on geography or prosecutorial discretion alone.",
        "Balancing public safety with civil liberties remains difficult when surveillance technologies expand faster than legal safeguards develop.",
        "Building fairer justice systems requires confronting structural inequities, not only introducing isolated procedural reforms in courts.",
    ],
    # Batch 6 (526–550): Cultural Heritage & Identity
    [
        "Cultural heritage preservation faces competing pressures from tourism development, urban renewal, and insufficient maintenance budgets.",
        "Although museums digitized collections, many rural communities still lack access to traveling exhibitions and cultural programs.",
        "It is expected that repatriation debates will intensify as source countries demand return of artifacts acquired colonially.",
        "Safeguarding intangible traditions requires supporting living practitioners, not only cataloging performances for archival documentation purposes.",
        "Despite UNESCO listings, historic districts continue losing authentic character as souvenir shops replace local artisan workshops.",
        "The curator explained that community led curation should give residents voice in how neighborhood histories are presented.",
        "Promoting multilingual cultural programming should reflect diverse identities in cities transformed by decades of migration.",
        "If restoration grants targeted neglected monuments, regional heritage sites might attract visitors beyond famous capital landmarks.",
        "Although festivals revived locally, climate change threatens outdoor celebrations dependent on seasonal weather patterns and water.",
        "It is feared that commercial streaming platforms could displace independent filmmakers chronicling marginalized cultural experiences authentically.",
        "Indigenous land rights are viewed as inseparable from protecting sacred sites threatened by mining and infrastructure projects.",
        "Had fire prevention been upgraded earlier, irreplaceable manuscripts in historic libraries would have survived recent blazes.",
        "The exhibition shows how diaspora artists reinterpret homeland memories through hybrid forms blending global and local influences.",
        "Ensuring equitable access to culture requires subsidized tickets and transport for schools in underfunded peripheral municipalities.",
        "Although monuments were removed, public discussions about commemoration often remain polarized and superficial in media coverage.",
        "The challenge is not celebrating diversity rhetorically but funding institutions that sustain minority languages and artistic traditions.",
        "Expanding oral history projects should document working class experiences often omitted from official national narrative accounts.",
        "The director emphasized that renovation must respect original materials rather than replacing historic fabric with generic facades.",
        "Developing heritage trails linking villages is being promoted to distribute tourism income beyond overcrowded urban center attractions.",
        "It is recommended that looted objects be returned when provenance research establishes evidence of unlawful acquisition abroad.",
        "Although creative industries grew, precarious contracts leave many cultural workers without pensions or stable income security.",
        "Strengthening craft apprenticeships should transmit specialized skills threatened by mass production and imported imitation goods flooding markets.",
        "Promoting respectful cultural exchange should avoid reducing living communities to exotic attractions for visiting international tourists alone.",
        "Balancing heritage conservation with housing needs remains contentious when protected buildings limit construction in crowded historic neighborhoods.",
        "Preserving cultural memory in changing societies will require institutions willing to confront uncomfortable chapters of national history.",
    ],
    # Batch 7 (551–575): Economic Inequality
    [
        "Economic inequality has widened as asset prices rose faster than wages for ordinary workers in many countries.",
        "Although minimum wages increased, inflation eroded purchasing power for households dependent on low paid service jobs.",
        "It is expected that wealth taxes will be debated again as governments seek revenue for social programs.",
        "Progressive taxation is regarded as necessary, yet political resistance from wealthy interests over funding remains fierce.",
        "Despite economic growth, child poverty persists in neighborhoods lacking affordable childcare and reliable public transportation links.",
        "The economist explained that intergenerational mobility declines when elite education and housing markets become increasingly segregated financially.",
        "Promoting worker ownership models should distribute corporate profits more fairly among employees who generate value daily.",
        "If antitrust enforcement strengthened, dominant platforms might face competition that lowers prices for consumers and businesses.",
        "Although stimulus measures helped, temporary relief did not tackle structural weaknesses in labor markets facing automation.",
        "It is feared that austerity could deepen hardship among households already struggling with debt and rising rents.",
        "Universal basic services are viewed as alternatives to fragmented welfare programs that leave citizens without adequate support.",
        "Had financial regulation tightened earlier, speculative bubbles inflating housing costs would have been less destructive nationwide.",
        "The analysis shows how gender pay gaps persist among graduates with comparable qualifications and similar work experience.",
        "Ensuring living wages for care workers should recognize essential labor traditionally undervalued because it employs mostly women.",
        "Although philanthropy increased, charitable giving does not substitute for tax financed safety nets covering basic needs.",
        "The barrier is not public awareness but concentrated political influence protecting preferential treatment for capital income.",
        "Expanding cooperative housing should reduce speculation that displaces renters from neighborhoods undergoing rapid gentrification and redevelopment.",
        "The minister emphasized that fiscal policy must not favor asset holders at the expense of young households.",
        "Developing regional investment funds is proposed to channel capital toward areas left behind by national growth patterns.",
        "It is recommended that executive pay reflect long term performance rather than short term stock market returns.",
        "Although unemployment fell, underemployment and insecure contracts leave many workers without savings or predictable monthly incomes.",
        "Strengthening collective bargaining should improve wages in sectors where individualized negotiation leaves employees with limited bargaining power.",
        "Promoting financial education should help families avoid predatory lending products marketed aggressively in economically struggling communities nationwide.",
        "Balancing growth and redistribution remains contentious when policymakers fear higher taxes might discourage domestic investment flows.",
        "Reducing entrenched inequality will require sustained commitment beyond symbolic gestures announced before elections and quickly abandoned.",
    ],
    # Batch 8 (576–600): Workplace & Labor
    [
        "Workplace automation is reshaping job requirements, forcing unions and employers to negotiate retraining programs collaboratively.",
        "Although remote work expanded flexibility, blurred boundaries increased stress for employees managing childcare and domestic duties.",
        "It is expected that gig economy regulations will clarify employment status for platform workers seeking labor protections.",
        "Ensuring safe warehouse conditions requires enforcing standards, not relying on corporate self reporting alone without inspections.",
        "Despite diversity pledges, leadership teams in major corporations still underrepresent women and minorities at senior levels.",
        "The negotiator explained that wage indexation should protect purchasing power when inflation spikes unexpectedly during economic disruptions.",
        "Promoting four day workweek pilots should test whether shorter hours maintain productivity while improving employee wellbeing measurably.",
        "If whistleblower protections were strengthened, workers might report safety violations without fear of retaliation from management.",
        "Although strikes were averted, unresolved disputes over automation threaten trust between labor unions and employers.",
        "It is feared that algorithmic scheduling could intensify precarity for retail workers whose hours fluctuate without notice.",
        "Sectoral bargaining agreements are regarded as effective tools for setting fair wages across industries with many firms.",
        "Had parental leave been equalized earlier, career penalties for mothers in professional roles would have diminished.",
        "The survey documents how burnout spreads when organizations reward overwork while understaffing teams during high demand.",
        "Guaranteeing the right to organize should remain fundamental when companies threaten relocation to weaker labor jurisdictions.",
        "Although training budgets grew, older workers in declining industries receive too little support for new jobs.",
        "The difficulty lies not in recognizing shortages but coordinating education providers with regional employers seeking apprentices.",
        "Expanding paid sick leave should reduce contagion and protect low wage employees unable to afford unpaid absence.",
        "The chair emphasized that layoffs should be last resort after exploring reduced hours and internal redeployment first.",
        "Developing portable pension schemes is debated as workers change employers more frequently across fragmented labor markets today.",
        "It is recommended that harassment reporting channels be independent from hierarchies that might discourage victims from reporting.",
        "Although productivity rose, wage shares declined as firms channeled profits toward shareholders rather than workforce compensation.",
        "Strengthening inspection capacity should expose subcontracting chains that hide labor abuses in global manufacturing and logistics networks.",
        "Promoting ergonomic design should prevent chronic injuries among employees performing repetitive tasks in factories and service settings.",
        "Balancing flexibility and security remains contentious as freelancers demand benefits traditionally tied to permanent employment contracts nationwide.",
        "Building fair labor markets will require updating institutions designed for industrial workplaces of the previous century.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

START_ID = 401
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

    target_path = project_root / "data/handcraft/en/train/b2_new_003.conllu"
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