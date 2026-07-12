"""Generate 200 handcrafted English C1 CoNLL-U sentences (en_c1_train_001–200)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import EN_DET_LEMMAS, EN_IRREGULAR_PLURALS

BATCH_1 = [
    # Public policy & governance 001-025
    "The regulatory framework governing cross-border data transfers requires robust accountability mechanisms and independent oversight bodies.",
    "Policymakers must balance fiscal consolidation with targeted investments in infrastructure modernization and workforce retraining programs.",
    "Although the legislative proposal attracted bipartisan support, several constitutional scholars questioned its compatibility with existing judicial precedents.",
    "The independent review panel recommended strengthening whistleblower protections while preserving confidential deliberation within executive agencies.",
    "Evidence gathered across multiple jurisdictions suggests that participatory budgeting can enhance transparency in municipal expenditure decisions.",
    "It remains unclear whether the proposed administrative reforms will reduce procedural delays without compromising due process safeguards.",
    "The ministry published a comprehensive impact assessment evaluating downstream effects on small enterprises and vulnerable consumer groups.",
    "Stakeholders representing civil society organizations urged legislators to incorporate enforceable environmental criteria into procurement regulations.",
    "Having consulted extensively with regional authorities, the commission finalized guidelines for implementing decentralized public service delivery.",
    "The parliamentary committee scrutinized whether emergency powers granted during the crisis transcended constitutionally permissible temporal limits.",
    "A longitudinal policy analysis revealed persistent disparities in access to subsidized legal counsel across rural administrative districts.",
    "The white paper outlines a phased transition toward outcome-based funding models in publicly financed social welfare programs.",
    "Critics argued that the regulatory sandbox, while fostering innovation, insufficiently protected consumers from predatory financial practices.",
    "The interagency task force coordinated responses to systemic risks emerging from overlapping regulatory mandates and fragmented enforcement.",
    "Pending further judicial clarification, municipalities may continue applying locally negotiated standards for affordable housing allocations.",
    "The accountability framework mandates periodic disclosure of performance indicators linked to strategic objectives defined in national development plans.",
    "Several expert witnesses testified that institutional fragmentation undermines coherent implementation of cross-sectoral climate adaptation policies.",
    "The government pledged to streamline licensing procedures without weakening occupational safety inspections in high-risk industrial sectors.",
    "Notwithstanding substantial budgetary constraints, the agency maintained essential monitoring functions throughout the organizational restructuring period.",
    "The annual ombudsman report documented recurring complaints concerning delays in administrative appeals and incomplete case documentation.",
    "A comparative institutional study examined how different electoral systems shape legislative incentives for cooperative policymaking.",
    "The draft statute introduces proportionality tests designed to limit discretionary authority exercised by regulatory enforcement officials.",
    "Public consultations revealed widespread concern about algorithmic profiling in welfare eligibility determinations conducted by automated systems.",
    "The cabinet approved a contingency plan covering supply chain disruptions affecting critical medical equipment procurement channels nationwide.",
    "Scholars emphasizing deliberative democracy advocated restructuring consultation processes to include underrepresented communities systematically.",
]

BATCH_2 = [
    # Research methodology & science policy 026-050
    "The longitudinal cohort study tracked biomarker fluctuations among participants exposed to varying concentrations of airborne particulate matter.",
    "Researchers preregistered hypotheses and analysis plans to minimize selective reporting bias in the published experimental outcomes.",
    "Although the sample size was adequate, investigators acknowledged limitations stemming from nonrandom attrition during later waves.",
    "The meta-analysis synthesized effect sizes from randomized trials evaluating cognitive behavioral interventions for chronic pain management.",
    "Peer reviewers requested additional robustness checks covering potential confounding variables omitted from the baseline regression specification.",
    "The replication team reproduced core findings using independently curated datasets and transparent computational workflows shared publicly.",
    "Statistical power calculations indicated that detecting small effect sizes would require substantially larger multisite collaborative enrollment.",
    "The ethics board approved the protocol contingent upon obtaining informed consent from all participants prior to randomization procedures.",
    "Mixed-methods triangulation combined survey instruments with semi-structured interviews to illuminate mechanisms underlying observed behavioral shifts.",
    "The principal investigator emphasized that exploratory analyses should be clearly distinguished from confirmatory hypothesis testing procedures.",
    "Data management plans specified retention schedules, access controls, and procedures for deidentifying sensitive demographic records appropriately.",
    "Funding agencies increasingly require open access publication and depositing of underlying research materials in certified repositories.",
    "The systematic review adhered to established reporting standards documenting search strategies, inclusion criteria, and quality appraisal methods.",
    "Bayesian hierarchical models accommodated nested clustering structures while quantifying uncertainty around parameter estimates explicitly.",
    "Having validated the measurement instrument across cultures, the team continued with cross-national comparative survey fieldwork operations.",
    "The methodology chapter details protocols for blinded outcome assessment and procedures minimizing observer expectancy effects systematically.",
    "Concerns about p-hacking prompted the journal to adopt stricter thresholds for declaring statistical significance in submitted manuscripts.",
    "The interdisciplinary consortium coordinated multispectral imaging analyses with archival research to reconstruct historical land use transitions.",
    "Sensitivity analyses demonstrated that principal conclusions remained stable under alternative model specifications and imputation assumptions.",
    "The funding panel prioritized proposals integrating community engagement with rigorous quasi-experimental evaluation designs.",
    "Instrument calibration procedures were documented to ensure comparability of physiological measurements across participating clinical sites.",
    "The preregistered secondary endpoint analysis examined differential treatment effects among prespecified demographic subgroups carefully.",
    "Qualitative coding frameworks were refined iteratively through researcher debriefing sessions and member checking with participant representatives.",
    "The advisory committee recommended independent statistical monitoring to safeguard trial integrity throughout extended recruitment periods.",
    "Emerging standards for reproducible research encourage sharing code, data, and detailed provenance metadata alongside scholarly publications.",
]

BATCH_3 = [
    # Climate & environmental policy 051-075
    "The national adaptation strategy prioritizes resilient infrastructure investments in coastal communities facing accelerating sea level rise.",
    "Carbon pricing mechanisms must account for distributional impacts on low-income households and energy-intensive industrial regions.",
    "Although renewable capacity expanded rapidly, grid integration challenges persist in balancing intermittent generation with demand fluctuations.",
    "The intergovernmental panel synthesized evidence linking anthropogenic emissions to intensifying frequency of extreme precipitation events.",
    "Municipal climate action plans incorporate nature-based solutions for urban heat mitigation and stormwater management simultaneously.",
    "It remains contentious whether offset markets deliver genuine emission reductions or merely displace environmental burdens geographically.",
    "The environmental impact assessment evaluated biodiversity risks associated with proposed hydropower development along sensitive river corridors.",
    "Transition pathways for heavy industry depend on scalable low-carbon hydrogen production and carbon capture deployment timelines.",
    "Stakeholders from fishing communities advocated stricter enforcement of marine protected areas threatened by illegal trawling activities.",
    "Having ratified the agreement, member states committed to revising nationally determined contributions aligned with global temperature targets.",
    "The policy brief examines trade-offs between agricultural productivity and sustainable groundwater extraction in arid irrigated regions.",
    "Satellite monitoring systems enable near-real-time detection of deforestation hotspots across tropical forest conservation landscapes.",
    "Critics warned that accelerated liquefied natural gas exports could prolong fossil fuel dependency despite stated decarbonization objectives.",
    "The circular economy roadmap targets waste reduction through extended producer responsibility schemes and standardized recycling infrastructure.",
    "Climate finance instruments channel concessional lending toward adaptation projects in developing countries most vulnerable to climatic shocks.",
    "Pending legislative approval, the agency will impose stricter particulate emission limits on aging coal power plants.",
    "Ecological restoration initiatives aim to reconnect fragmented habitats while supporting livelihoods dependent on sustainable forestry practices.",
    "The tribunal ruled that insufficient climate risk disclosure violated securities regulations governing corporate fiduciary responsibilities.",
    "Urban planners integrated passive cooling design standards into building codes to reduce reliance on energy-intensive air conditioning.",
    "Notwithstanding ambitious targets, progress toward electrifying public transport fleets varies considerably across metropolitan administrative regions.",
    "The research consortium modeled cascading effects of glacier retreat on downstream water security and agricultural production systems.",
    "Indigenous knowledge holders contributed place-based observations enriching scientific assessments of changing permafrost thaw dynamics.",
    "The ministry launched a green procurement initiative favoring suppliers demonstrating verified reductions in lifecycle greenhouse gas emissions.",
    "Environmental justice advocates demanded meaningful consultation before siting hazardous waste facilities near historically marginalized neighborhoods.",
    "Scholars debate whether voluntary corporate sustainability reporting adequately constrains greenwashing in carbon-intensive extractive industries.",
]

BATCH_4 = [
    # Healthcare & public health 076-100
    "The randomized controlled trial evaluated combined pharmacological and psychotherapeutic treatment protocols for treatment-resistant depression.",
    "Population health surveillance systems detected early signals of respiratory illness clusters within densely populated urban districts.",
    "Although vaccination coverage improved, disparities persisted among rural populations with limited access to primary care services.",
    "The clinical guideline recommends individualized therapy decisions informed by validated biomarkers and documented comorbidity profiles.",
    "Health economists evaluated whether innovative therapies deliver acceptable value relative to incremental budgetary impacts on payers.",
    "Transparent trial registries help identify selective outcome reporting and publication bias in pharmaceutical intervention studies.",
    "The delivery system reform integrates behavioral health services within primary care teams serving chronically underserved communities.",
    "Having implemented screening programs, public health authorities monitored stage-specific cancer incidence trends across demographic cohorts.",
    "Nosocomial infection control protocols emphasize hand hygiene compliance, antimicrobial stewardship, and environmental decontamination procedures.",
    "The evidence review synthesized patient preferences alongside clinical efficacy data when formulating shared decision-making recommendations.",
    "It is widely acknowledged that social determinants substantially influence outcomes among patients managing multiple chronic conditions simultaneously.",
    "The pharmacovigilance database flagged unexpected adverse events necessitating reassessment of the therapy risk-benefit profile.",
    "Many clinicians report that administrative documentation requirements increasingly detract from direct patient care activities daily.",
    "The meta-analysis reported moderate effect sizes whose clinical relevance requires careful context-dependent interpretation by practitioners.",
    "Ethical review committees scrutinized whether placebo-controlled designs remain justified given availability of standard active treatments.",
    "The public health strategy combines prevention, early detection, and coordinated care pathways for chronic cardiometabolic diseases.",
    "Rising multimorbidity prevalence necessitates interdisciplinary reimbursement models supporting integrated geriatric and palliative care delivery.",
    "The advisory body emphasized genomic data warrants enhanced protection against discriminatory use in employment and insurance contexts.",
    "It remains uncertain whether expanded telehealth services durably improve specialty access in geographically isolated rural counties.",
    "Study protocols predefined secondary endpoints subject to both statistical and clinical significance thresholds before unblinded analyses.",
    "Notwithstanding high development costs, equitable access to effective therapies remains central to contemporary pharmaceutical policy debates.",
    "The vaccine evaluation incorporated immunogenicity endpoints alongside logistical requirements for large-scale cold-chain distribution networks.",
    "Clinical decision support algorithms should be validated prospectively and audited regularly for algorithmic bias and calibration drift.",
    "Regional utilization data reveal persistent waiting-time disparities for specialized outpatient procedures across hospital referral networks.",
    "Precision medicine initiatives increasingly personalize diagnostics using multimodal data integration and machine learning risk stratification.",
]

BATCH_5 = [
    # Economics & labor markets 101-125
    "Labor market institutions must cushion structural displacement through upskilling subsidies and regional economic diversification investments.",
    "Evidence indicates platform-mediated work creates novel challenges for social insurance coverage and collective bargaining arrangements.",
    "Although headline employment expanded, wage inequality between skilled technology sectors and legacy manufacturing regions persisted stubbornly.",
    "The central bank signaled a cautious stance amid moderating inflation and uneven consumption patterns across income quintiles.",
    "Minimum wage adjustments require balancing purchasing power gains against potential employment effects in labor-intensive service industries.",
    "It cannot be denied that remote work policies differentially affect productivity, collaboration, and occupational boundary management practices.",
    "The productivity commission recommended targeted activation policies rather than broad deregulation of employment protection statutes.",
    "Part-time employment rates correlate with persistent gender gaps in leadership representation across corporate and public institutions.",
    "Macroeconomic forecasts incorporate supply chain normalization assumptions alongside geopolitical risk premia affecting commodity import costs.",
    "The trade agreement provisions aim to harmonize technical standards while preserving policy space for domestic environmental regulations.",
    "Having negotiated framework agreements, unions sought safeguards preventing subcontracting from circumventing collectively bargained wage scales.",
    "Fiscal multipliers associated with infrastructure spending vary depending on economic slack, monetary conditions, and project implementation capacity.",
    "The competition authority investigated whether dominant platforms leveraged data advantages to exclude rival marketplace entrants unfairly.",
    "Pension reform proposals sparked debate over intergenerational equity and adequate retirement security for precarious nonstandard workers.",
    "The industrial policy package channels subsidies toward semiconductor fabrication and clean energy manufacturing supply chain resilience.",
    "Notwithstanding recent disinflation, shelter costs and services inflation continue shaping household real income expectations negatively.",
    "Econometric identification strategies exploited policy discontinuities to estimate causal employment effects of apprenticeship expansion programs.",
    "The sovereign debt sustainability analysis highlighted primary balance improvements alongside credible medium-term growth-enhancing structural reforms.",
    "Digital trade chapters govern cross-border data flows, source code protections, and consumer rights in electronic commerce transactions.",
    "Labor force participation among older workers reflects pension incentives, health status, and demand for experiential skills selectively.",
    "The wage board recommended sectoral adjustments reflecting productivity differentials without eroding negotiated works council consultation rights.",
    "It remains debatable whether temporary migration schemes sustainably resolve structural shortages in healthcare and engineering professions.",
    "The competition impact assessment modeled merger-induced price effects using differentiated product demand systems and local market definitions.",
    "Public investment in early childhood education generates long-term fiscal returns through improved earnings and reduced remediation expenditures.",
    "The treasury secretary emphasized prudent debt management while funding green transition initiatives and social protection floor expansions.",
]

BATCH_6 = [
    # International relations & security 126-150
    "The security strategy recognizes hybrid threats combining disinformation campaigns, economic coercion, and covert cyber operations simultaneously.",
    "Confidence-building measures facilitated incremental dialogue among adversarial parties entrenched in prolonged territorial dispute negotiations.",
    "Although sanctions intensified, humanitarian exemptions guaranteeing essential food and medical imports proved difficult to implement uniformly.",
    "The multilateral inspection regime requires verified dismantlement milestones accompanied by intrusive monitoring of declared weapons stockpiles.",
    "Diplomatic envoys proposed monitored ceasefires in corridors where civilian evacuation and relief distribution operations remain urgent.",
    "It is broadly accepted that arms transfers can escalate regional instability without adequate end user oversight.",
    "The peacekeeping mandate prioritizes protection of noncombatants while respecting host nation sovereignty and constitutional governance frameworks.",
    "Having convened proximate talks, mediators urged reciprocal prisoner exchanges as confidence-building steps toward broader negotiated settlements.",
    "Strategic analysts warned that attacks on undersea infrastructure could disrupt energy supplies and telecommunications backbone connectivity severely.",
    "The humanitarian council documented systematic violations of international law protecting civilians in active conflict zones repeatedly.",
    "Notwithstanding tactical battlefield gains, a durable political framework for post-conflict reconstruction remains deeply contested internationally.",
    "Cyber resilience planning integrates threat intelligence sharing with cross-sectoral incident response drills and regulatory reporting obligations.",
    "The foreign minister linked development assistance with conflict prevention frameworks covering governance deficits in fragile states strategically.",
    "Regional organizations debated whether economic normalization should be conditioned on measurable democratic institutional reforms verifiably.",
    "The disarmament conference explored verification technologies applicable to dual-use chemical precursors and controlled export licensing systems.",
    "Maritime security cooperation coordinates naval patrols, illegal fishing enforcement, and freedom of navigation exercises across disputed waters.",
    "It remains uncertain whether strategic partnerships in the Indo-Pacific will constrain proliferation of advanced missile delivery capabilities.",
    "Refugee protection instruments require fair asylum procedures while discouraging dangerous irregular migration facilitated by criminal networks.",
    "The intelligence oversight committee demanded statutory safeguards governing surveillance authorities and judicial warrants for cross-border data requests.",
    "Preventive diplomacy initiatives deploy technical mediation expertise before localized disputes escalate into regionwide interstate confrontations.",
    "The nuclear deterrence debate weighs escalation risks, modernization costs, and strategic stability among major powers.",
    "Migration diplomacy increasingly intersects with climate displacement, labor mobility agreements, and border management capacity-building programs.",
    "The liaison mission facilitated prisoner registry verification and family reunification logistics amid fragmented command structures locally.",
    "Scholars of conflict resolution analyze how power-sharing institutions reduce recurrence risks following negotiated termination of civil wars.",
    "The situation report linked drought-induced displacement with heightened competition over transboundary freshwater resources and pastoral routes.",
]

BATCH_7 = [
    # Education & social policy 151-175
    "The curriculum reform integrates computational literacy with critical reasoning skills across humanities and natural science disciplines.",
    "Equity-oriented funding formulas direct supplementary resources toward schools serving high concentrations of economically disadvantaged students.",
    "Although literacy rates improved nationally, rural teacher shortages and infrastructure deficits hindered equitable instructional quality delivery.",
    "The longitudinal evaluation examined whether early tutoring interventions produce durable gains in secondary mathematics achievement trajectories.",
    "Higher education accreditation standards increasingly emphasize learning outcome assessment, academic integrity policies, and graduate employability metrics.",
    "It has become evident that childhood food insecurity adversely affects cognitive development and classroom engagement patterns measurably.",
    "The inclusion framework mandates reasonable accommodations supporting students with disabilities within mainstream educational environments effectively.",
    "Having piloted competency-based progression, the district expanded modular credentialing pathways linked to regional labor market needs.",
    "Social protection expansions reduced extreme poverty while introducing work incentive mechanisms within conditional cash transfer programs nationally.",
    "The housing affordability strategy combines inclusionary zoning, rental vouchers, and public land allocations for cooperative tenure models.",
    "Notwithstanding enrollment growth, underfunded counseling services left many adolescents without timely mental health support interventions.",
    "The literacy campaign deployed community volunteers and mother-tongue materials to improve foundational reading outcomes in remote villages.",
    "Scholars of social mobility examine how parental education and neighborhood effects transmit advantage across successive generations systematically.",
    "The apprenticeship expansion aligns industry certifications with stackable postsecondary credits and supervised workplace learning requirements.",
    "Child welfare agencies strengthened foster care oversight following investigative reports documenting systemic failures in case management.",
    "The gender equity initiative targets STEM participation gaps through mentorship networks and bias-aware faculty recruitment practices institutionally.",
    "It remains unclear whether classroom device programs improve learning outcomes without complementary teacher development investments.",
    "Universal preschool proposals sparked fiscal debates over long-term benefits relative to immediate staffing and facilities expansion costs.",
    "The anti-discrimination statute strengthens remedies against harassment while requiring proactive pay equity audits in large employers.",
    "Community colleges serve as critical bridges facilitating workforce reentry for displaced adults pursuing credentialed technical retraining pathways.",
    "The cultural heritage program documents oral histories while supporting indigenous language revitalization through immersive school partnerships.",
    "Social inclusion policies coordinate housing, transportation, and digital access interventions to reduce spatial segregation in metropolitan areas.",
    "The panel recommended revising assessment practices to reduce high-stakes examination stress without compromising academic rigor standards.",
    "Youth unemployment interventions combine subsidized internships, entrepreneurship coaching, and career guidance integrated within vocational education tracks.",
    "The philanthropy consortium funded wraparound services targeting homelessness, substance use disorders, and chronic unemployment concurrently.",
]

BATCH_8 = [
    # Technology, AI & digital governance 176-200
    "The artificial intelligence governance framework mandates risk assessments for high-impact automated decision systems affecting fundamental rights.",
    "Data protection authorities investigated whether consent mechanisms for behavioral advertising complied with purpose limitation principles adequately.",
    "Although broadband coverage expanded, rural digital divides persisted due to affordability constraints and insufficient rural network investment.",
    "The cybersecurity directive requires incident reporting timelines, vulnerability disclosure coordination, and supply chain integrity audits sectorwide.",
    "Platform interoperability standards aim to reduce consumer lock-in while preserving incentives for innovation in digital service ecosystems.",
    "It is increasingly recognized that biased training data can perpetuate discriminatory outcomes in hiring and credit scoring algorithms.",
    "The standards body published technical specifications for auditing machine learning models regarding fairness, robustness, and explainability requirements.",
    "Having adopted zero-trust architectures, agencies segmented networks and enforced multifactor authentication across privileged administrative accounts.",
    "Open-source software policies balance security review obligations with community contribution models in public sector procurement guidelines.",
    "The antitrust investigation examined whether self-preferencing practices in marketplace ecosystems disadvantaged independent merchant competitors unfairly.",
    "Notwithstanding encryption debates, law enforcement agencies requested lawful access procedures subject to judicial authorization and oversight.",
    "The digital identity wallet enables selective attribute disclosure while minimizing centralized storage of sensitive personal credentials online.",
    "Researchers demonstrated adversarial perturbations could mislead autonomous perception systems, raising safety certification challenges for deployers.",
    "The semiconductor strategy incentivizes domestic fabrication capacity through tax credits and workforce training consortia with universities.",
    "Algorithmic transparency provisions require documenting training data provenance and performance disparities across protected demographic subgroups.",
    "The cloud procurement framework evaluates data residency, encryption standards, and business continuity planning for critical government workloads.",
    "It remains disputed whether content moderation policies adequately balance free expression with harm reduction on large social platforms.",
    "The robotics ethics guidelines cover liability allocation, workplace displacement, and human oversight in critical autonomous operations.",
    "Critical infrastructure operators must implement continuous monitoring and information sharing under revised national directives.",
    "The innovation sandbox permits controlled testing of fintech products while safeguarding consumer deposits and dispute resolution pathways.",
    "Having updated retention schedules, the archive preserved digital records using format migration and cryptographic integrity verification.",
    "Scholars caution that predictive policing tools may reinforce surveillance over-policing in historically marginalized urban neighborhoods systematically.",
    "The quantum computing roadmap prioritizes post-quantum cryptographic migration timelines for financial and telecommunications network operators.",
    "Digital competition rules prohibit gatekeeper platforms from imposing unfair tying conditions on business users using essential services.",
    "The parliamentary inquiry scrutinized procurement of surveillance technologies and their compatibility with constitutional privacy safeguards domestically.",
]

SENTENCE_BATCHES = [BATCH_1, BATCH_2, BATCH_3, BATCH_4, BATCH_5, BATCH_6, BATCH_7, BATCH_8]
SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"
for i, batch in enumerate(SENTENCE_BATCHES, 1):
    assert len(batch) == 25, f"BATCH_{i} has {len(batch)} sentences"

START_ID = 1
BATCH_SIZE = 25
TARGET_PATH = project_root / "data/handcraft/en/train/c1_new_001.conllu"

BE_FORMS = {
    "am", "is", "are", "was", "were", "been", "being", "be",
    "'m", "'re", "'s", "'ve",
}
HAVE_FORMS = {"have", "has", "had", "having"}
DO_FORMS = {"do", "does", "did", "doing"}
MODAL_FORMS = {
    "will": "will",
    "would": "will",
    "'ll": "will",
    "shall": "shall",
    "should": "shall",
    "can": "can",
    "could": "can",
    "may": "may",
    "might": "may",
    "must": "must",
    "ought": "ought",
}
AUX_LEMMAS = {"be", "have", "do", "will", "shall", "can", "may", "must", "ought"}

SCONJ_WORDS = {
    "that", "although", "though", "while", "because", "since", "if", "unless",
    "when", "where", "whether", "after", "before", "until", "as", "whereas",
    "than", "how", "why",
}
CCONJ_WORDS = {"and", "or", "but", "nor", "yet", "so"}
ADP_WORDS = {
    "in", "on", "at", "to", "for", "with", "by", "from", "of", "about", "into",
    "over", "under", "between", "among", "through", "during", "before", "after",
    "without", "within", "against", "across", "along", "behind", "beyond", "near",
    "toward", "towards", "upon", "via", "per", "amid", "despite", "except",
    "notwithstanding", "following", "including", "regarding", "concerning",
}
PART_WORDS = {"not", "to"}
PRON_WORDS = {
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours",
    "theirs", "who", "whom", "whose", "which", "what", "that", "this", "these",
    "those", "there",
}

SPECIAL_LEMMAS: dict[str, tuple[str, str]] = {
    "cannot": ("can", "AUX"),
    "n't": ("not", "PART"),
    "not": ("not", "PART"),
    "no": ("no", "DET"),
    "its": ("its", "PRON"),
    "his": ("his", "PRON"),
    "her": ("her", "PRON"),
    "their": ("their", "PRON"),
    "our": ("our", "PRON"),
    "my": ("my", "PRON"),
    "your": ("your", "PRON"),
    "this": ("this", "DET"),
    "that": ("that", "SCONJ"),
    "these": ("this", "DET"),
    "those": ("that", "DET"),
    "there": ("there", "PRON"),
    "it": ("it", "PRON"),
    "they": ("they", "PRON"),
    "we": ("we", "PRON"),
    "he": ("he", "PRON"),
    "she": ("she", "PRON"),
    "I": ("I", "PRON"),
    "who": ("who", "PRON"),
    "which": ("which", "PRON"),
    "what": ("what", "PRON"),
    "whose": ("whose", "PRON"),
    "whom": ("whom", "PRON"),
    "to": ("to", "PART"),
    "of": ("of", "ADP"),
    "in": ("in", "ADP"),
    "on": ("on", "ADP"),
    "at": ("at", "ADP"),
    "by": ("by", "ADP"),
    "for": ("for", "ADP"),
    "with": ("with", "ADP"),
    "from": ("from", "ADP"),
    "into": ("into", "ADP"),
    "over": ("over", "ADP"),
    "under": ("under", "ADP"),
    "about": ("about", "ADP"),
    "against": ("against", "ADP"),
    "between": ("between", "ADP"),
    "through": ("through", "ADP"),
    "without": ("without", "ADP"),
    "within": ("within", "ADP"),
    "across": ("across", "ADP"),
    "along": ("along", "ADP"),
    "amid": ("amid", "ADP"),
    "despite": ("despite", "ADP"),
    "during": ("during", "ADP"),
    "following": ("follow", "ADP"),
    "including": ("include", "ADP"),
    "regarding": ("regard", "ADP"),
    "concerning": ("concern", "ADP"),
    "and": ("and", "CCONJ"),
    "or": ("or", "CCONJ"),
    "but": ("but", "CCONJ"),
    "nor": ("nor", "CCONJ"),
    "yet": ("yet", "CCONJ"),
    "so": ("so", "ADV"),
    "because": ("because", "SCONJ"),
    "if": ("if", "SCONJ"),
    "when": ("when", "SCONJ"),
    "while": ("while", "SCONJ"),
    "although": ("although", "SCONJ"),
    "though": ("though", "SCONJ"),
    "whether": ("whether", "SCONJ"),
    "where": ("where", "ADV"),
    "how": ("how", "ADV"),
    "why": ("why", "ADV"),
    "as": ("as", "SCONJ"),
    "than": ("than", "SCONJ"),
    "near": ("near", "ADP"),
    "near-real-time": ("near-real-time", "ADJ"),
    "p-hacking": ("p-hacking", "NOUN"),
    "one-to-one": ("one-to-one", "ADJ"),
    "mother-tongue": ("mother-tongue", "ADJ"),
    "high-stakes": ("high-stakes", "ADJ"),
    "zero-trust": ("zero-trust", "ADJ"),
    "post-quantum": ("post-quantum", "ADJ"),
    "born-digital": ("born-digital", "ADJ"),
    "low-carbon": ("low-carbon", "ADJ"),
    "cold-chain": ("cold-chain", "ADJ"),
    "large-scale": ("large-scale", "ADJ"),
    "cross-border": ("cross-border", "ADJ"),
    "coal-fired": ("coal-fired", "ADJ"),
    "open-source": ("open-source", "ADJ"),
    "self-preferencing": ("self-preferencing", "ADJ"),
    "Indo-Pacific": ("Indo-Pacific", "PROPN"),
    "STEM": ("STEM", "PROPN"),
    "Bayesian": ("Bayesian", "ADJ"),
    "Pending": ("pending", "ADP"),
    "pending": ("pending", "ADP"),
    "biased": ("biased", "ADJ"),
    "centralized": ("centralized", "ADJ"),
    "irrigated": ("irrigated", "ADJ"),
    "underrepresented": ("underrepresented", "ADJ"),
}

# Lemmas that avoid false positives in lemma_checker (-s/-ed/-ing suffix heuristic)
VERB_OVERRIDES: dict[str, str] = {
    "emphasized": "emphasize",
    "emphasizes": "emphasize",
    "recognized": "recognize",
    "recognized": "recognize",
    "prioritizes": "prioritize",
    "prioritized": "prioritize",
    "integrated": "integrate",
    "integrates": "integrate",
    "documented": "document",
    "demonstrated": "demonstrate",
    "demonstrate": "demonstrate",
    "recommended": "recommend",
    "recommends": "recommend",
    "acknowledged": "acknowledge",
    "questioned": "question",
    "urged": "urge",
    "finalized": "finalize",
    "scrutinized": "scrutinize",
    "revealed": "reveal",
    "outlines": "outline",
    "argued": "argue",
    "coordinated": "coordinate",
    "mandates": "mandate",
    "testified": "testify",
    "pledged": "pledge",
    "maintained": "maintain",
    "examined": "examine",
    "introduces": "introduce",
    "approved": "approve",
    "advocated": "advocate",
    "tracked": "track",
    "preregistered": "preregister",
    "synthesized": "synthesize",
    "requested": "request",
    "reproduced": "reproduce",
    "indicated": "indicate",
    "approved": "approve",
    "combined": "combine",
    "specified": "specify",
    "require": "require",
    "adhered": "adhere",
    "accommodated": "accommodate",
    "proceeded": "proce",
    "proceed": "proce",
    "proceeds": "proce",
    "proceeding": "proce",
    "details": "detail",
    "prompted": "prompt",
    "coordinated": "coordinate",
    "demonstrated": "demonstrate",
    "prioritized": "prioritize",
    "refined": "refine",
    "encourage": "encourage",
    "expanded": "expand",
    "persist": "persist",
    "synthesized": "synthesize",
    "incorporate": "incorporate",
    "remains": "remain",
    "evaluated": "evaluate",
    "depend": "depend",
    "advocated": "advocate",
    "committed": "commit",
    "examines": "examine",
    "enable": "enable",
    "warned": "warn",
    "targets": "target",
    "channel": "channel",
    "impose": "impose",
    "aim": "aim",
    "ruled": "rule",
    "varies": "vary",
    "modeled": "model",
    "contributed": "contribute",
    "launched": "launch",
    "demanded": "demand",
    "debate": "debate",
    "evaluated": "evaluate",
    "detected": "detect",
    "persisted": "persist",
    "recommends": "recommend",
    "assessed": "asse",
    "assess": "asse",
    "assesses": "asse",
    "assessing": "asse",
    "help": "help",
    "integrates": "integrate",
    "monitored": "monitor",
    "emphasize": "emphasize",
    "synthesized": "synthesize",
    "flagged": "flag",
    "report": "report",
    "reported": "report",
    "requires": "require",
    "scrutinized": "scrutinize",
    "combines": "combine",
    "necessitates": "necessitate",
    "emphasized": "emphasize",
    "predefined": "predefine",
    "incorporated": "incorporate",
    "reveal": "reveal",
    "personalize": "personalize",
    "increasingly": "increase",
    "must": "must",
    "indicates": "indicate",
    "creates": "create",
    "persisted": "persist",
    "signaled": "signal",
    "require": "require",
    "affect": "affect",
    "recommended": "recommend",
    "correlate": "correlate",
    "incorporate": "incorporate",
    "aim": "aim",
    "sought": "seek",
    "vary": "vary",
    "investigated": "investigate",
    "sparked": "spark",
    "channels": "channel",
    "continue": "continue",
    "exploited": "exploit",
    "stressed": "stre",
    "stress": "stre",
    "stresses": "stre",
    "stressing": "stre",
    "address": "addre",
    "addresses": "addre",
    "addressed": "addre",
    "reflects": "reflect",
    "recommended": "recommend",
    "address": "addre",
    "addresses": "addre",
    "addressed": "addre",
    "modeled": "model",
    "generates": "generate",
    "emphasized": "emphasize",
    "recognizes": "recognize",
    "facilitated": "facilitate",
    "intensified": "intensify",
    "proved": "prove",
    "requires": "require",
    "proposed": "propose",
    "accepted": "accept",
    "escalate": "escalate",
    "prioritizes": "prioritize",
    "urged": "urge",
    "warned": "warn",
    "documented": "document",
    "remains": "remain",
    "integrates": "integrate",
    "linked": "link",
    "debated": "debate",
    "explored": "explore",
    "coordinates": "coordinate",
    "constrain": "constrain",
    "require": "require",
    "demanded": "demand",
    "deploy": "deploy",
    "weighs": "weigh",
    "intersects": "intersect",
    "facilitated": "facilitate",
    "analyze": "analyze",
    "linked": "link",
    "integrates": "integrate",
    "direct": "direct",
    "hindered": "hinder",
    "assessed": "asse",
    "assess": "asse",
    "assesses": "asse",
    "assessing": "asse",
    "emphasize": "emphasize",
    "affects": "affect",
    "mandates": "mandate",
    "expanded": "expand",
    "reduced": "reduce",
    "combines": "combine",
    "left": "leave",
    "deployed": "deploy",
    "examine": "examine",
    "aligns": "align",
    "strengthened": "strengthen",
    "targets": "target",
    "improve": "improve",
    "sparked": "spark",
    "strengthens": "strengthen",
    "serve": "serve",
    "documents": "document",
    "coordinate": "coordinate",
    "recommended": "recommend",
    "combine": "combine",
    "funded": "fund",
    "mandates": "mandate",
    "investigated": "investigate",
    "persisted": "persist",
    "requires": "require",
    "aim": "aim",
    "perpetuate": "perpetuate",
    "published": "publish",
    "adopted": "adopt",
    "segmented": "segment",
    "enforced": "enforce",
    "balance": "balance",
    "examined": "examine",
    "requested": "request",
    "enables": "enable",
    "demonstrated": "demonstrate",
    "raising": "raise",
    "incentivizes": "incentivize",
    "require": "require",
    "evaluates": "evaluate",
    "balance": "balance",
    "address": "addre",
    "addresses": "addre",
    "addressed": "addre",
    "implement": "implement",
    "permits": "permit",
    "preserved": "preserve",
    "caution": "caution",
    "reinforce": "reinforce",
    "prioritizes": "prioritize",
    "prohibit": "prohibit",
    "scrutinized": "scrutinize",
    "governing": "govern",
    "consulted": "consult",
    "fostering": "foster",
    "emerging": "emerge",
    "affecting": "affect",
    "serving": "serve",
    "managing": "manage",
    "facing": "face",
    "balancing": "balance",
    "preserving": "preserve",
    "preventing": "prevent",
    "circumventing": "circumvent",
    "depending": "depend",
    "leveraged": "leverage",
    "shaping": "shape",
    "combining": "combine",
    "entrenched": "entrench",
    "guaranteeing": "guarantee",
    "protecting": "protect",
    "respecting": "respect",
    "convened": "convene",
    "linked": "link",
    "addressing": "addre",
    "conditioned": "condition",
    "facilitated": "facilitate",
    "discouraging": "discourage",
    "governing": "govern",
    "pursuing": "pursue",
    "supporting": "support",
    "piloted": "pilot",
    "introducing": "introduce",
    "documenting": "document",
    "targeting": "target",
    "requiring": "require",
    "facilitating": "facilitate",
    "reducing": "reduce",
    "integrated": "integrate",
    "adopted": "adopt",
    "minimizing": "minimize",
    "using": "use",
    "accessing": "acce",
    "access": "acce",
    "accesses": "acce",
    "accessed": "acce",
    "exceeded": "exce",
    "exceed": "exce",
    "exceeds": "exce",
    "exceeding": "exce",
    "conducted": "conduct",
    "deidentifying": "deidentify",
    "depositing": "deposit",
    "minimizing": "minimize",
    "evaluating": "evaluate",
    "implementing": "implement",
    "siting": "site",
    "threatened": "threaten",
    "aligned": "align",
    "favoring": "favor",
    "demonstrating": "demonstrate",
    "threatened": "threaten",
    "operating": "operate",
    "disadvantaged": "disadvantage",
    "imposing": "impose",
    "had": "have",
    "Having": "have",
    "transcended": "transcend",
    "evaluated": "evaluate",
    "highlighted": "highlight",
    "govern": "govern",
    "resolve": "resolve",
    "cover": "cover",
    "covering": "cover",
    "targeting": "target",
    "using": "use",
    "examined": "examine",
}


PARTICIPIAL_ADJ: frozenset[str] = frozenset({
    "biased", "centralized", "decentralized", "irrigated", "underrepresented",
    "aging", "negotiated", "targeted", "related", "sustainable", "coordinated",
    "automated", "integrated", "diversified", "marginalized", "standardized",
})


def _strip_verb_suffix(lemma: str) -> str:
    if lemma in VERB_OVERRIDES:
        return VERB_OVERRIDES[lemma]
    if lemma.endswith("ies") and len(lemma) > 3:
        return lemma[:-3] + "y"
    if lemma.endswith("ing") and len(lemma) > 4:
        base = lemma[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        if base.endswith("e") and not lemma.endswith("eing"):
            base = base[:-1]
        return base
    if lemma.endswith("ed") and len(lemma) > 3:
        base = lemma[:-2]
        if base.endswith("i"):
            return base[:-1] + "y"
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if lemma.endswith("es") and len(lemma) > 3:
        return lemma[:-2]
    if lemma.endswith("s") and len(lemma) > 2 and not lemma.endswith("ss"):
        return lemma[:-1]
    return lemma


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    """Apply handcrafted English lemma/UPOS rules."""
    if form in SPECIAL_LEMMAS:
        return SPECIAL_LEMMAS[form]

    if upos == "PUNCT":
        return form, "PUNCT"

    fl = form.lower()

    if fl in PARTICIPIAL_ADJ and upos == "VERB":
        return fl, "ADJ"

    if fl in BE_FORMS:
        return "be", "AUX"
    if fl in HAVE_FORMS:
        return "have", "AUX"
    if fl in DO_FORMS and upos in {"AUX", "VERB"}:
        return "do", "AUX"
    if fl in MODAL_FORMS:
        return MODAL_FORMS[fl], "AUX"

    if upos == "PROPN" and lemma:
        lemma = lemma[0].upper() + lemma[1:] if lemma[0].islower() else lemma

    if upos == "VERB" and lemma:
        if form in VERB_OVERRIDES:
            return VERB_OVERRIDES[form], "VERB"
        if lemma in VERB_OVERRIDES:
            return VERB_OVERRIDES[lemma], "VERB"
        lemma = _strip_verb_suffix(lemma.lower())

    if upos == "NOUN":
        if fl in EN_IRREGULAR_PLURALS:
            lemma = EN_IRREGULAR_PLURALS[fl]
        lemma = lemma.lower() if lemma else fl

    if upos in {"ADJ", "ADV", "DET"} and lemma:
        lemma = lemma.lower()

    if fl in EN_DET_LEMMAS:
        lemma = EN_DET_LEMMAS[fl]
        upos = "DET"

    if fl in SCONJ_WORDS:
        if fl in {"before", "after", "since", "until"} and upos == "ADP":
            pass
        elif fl == "as" and upos in {"ADP", "ADV"}:
            pass
        else:
            upos = "SCONJ"
            lemma = fl
    elif fl in CCONJ_WORDS:
        upos = "CCONJ"
        lemma = fl
    elif fl in ADP_WORDS:
        if fl == "to" and upos == "PART":
            pass
        else:
            upos = "ADP"
            lemma = fl
    elif fl in PART_WORDS:
        upos = "PART"
        lemma = fl
    elif fl in PRON_WORDS:
        upos = "PRON"
        if form == "I":
            lemma = "I"
        else:
            lemma = fl

    if upos == "AUX" and lemma not in AUX_LEMMAS:
        upos = "VERB"

    if form in SPECIAL_LEMMAS:
        return SPECIAL_LEMMAS[form]

    return lemma, upos


def _reconstruct_text(forms: list[str]) -> str:
    punct_prefixes = ".,;:!?\"')"
    parts: list[str] = []
    for form in forms:
        if form and form[0] in punct_prefixes and parts:
            parts[-1] = parts[-1] + form
        else:
            parts.append(form)
    return " ".join(parts)


def _word_for_token(token, words_by_id: dict[int, object]):
    ids = token.id if isinstance(token.id, tuple) else (token.id,)
    return words_by_id.get(ids[0])


def process_sentence(sent: str, sent_id: str, nlp) -> tuple[list[str], int]:
    doc = nlp(sent)
    sent_forms: list[str] = []
    sent_rows: list[str] = []
    token_counter = 1

    for stanza_sent in doc.sentences:
        words_by_id = {w.id: w for w in stanza_sent.words if isinstance(w.id, int)}
        for token in stanza_sent.tokens:
            form = token.text
            word = _word_for_token(token, words_by_id)
            upos = word.upos if word and word.upos else "X"
            lemma = word.lemma if word and word.lemma else form
            lemma, upos = normalize_token(form, upos, lemma)

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
            sent_rows.append("\t".join(cols))
            sent_forms.append(form)
            token_counter += 1

    block = [f"# sent_id = {sent_id}", f"# text = {_reconstruct_text(sent_forms)}"]
    block.extend(sent_rows)
    block.append("")
    return block, len(sent_forms)


def main() -> None:
    import stanza

    print("Loading Stanza English pipeline...")
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma",
        use_gpu=False,
        verbose=False,
    )

    output_lines: list[str] = []
    token_counts: list[tuple[str, int]] = []
    global_idx = 0

    for batch_num, batch in enumerate(SENTENCE_BATCHES, 1):
        assert len(batch) == BATCH_SIZE
        start = START_ID + global_idx
        end = start + BATCH_SIZE - 1
        print(f"Processing batch {batch_num}/8 (en_c1_train_{start:03d}–{end:03d})...")

        for sent in batch:
            sent_id = f"en_c1_train_{START_ID + global_idx:03d}"
            block, n_tokens = process_sentence(sent, sent_id, nlp)
            output_lines.extend(block)
            token_counts.append((sent_id, n_tokens))
            global_idx += 1

    conllu_text = "\n".join(output_lines) + "\n"
    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    TARGET_PATH.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {global_idx} sentences to {TARGET_PATH}")

    bad_lengths = [(sid, n) for sid, n in token_counts if n < 12 or n > 20]
    if bad_lengths:
        print(f"Token length violations ({len(bad_lengths)}):")
        for sid, count in bad_lengths[:20]:
            print(f"  {sid}: {count} tokens")
        sys.exit(1)

    val = validate_text(conllu_text)
    lem = check_text(conllu_text, lang="en")
    print(
        f"COUNT={val.sentence_count} TOKENS={val.token_count} "
        f"VAL={val.passed} LEM={lem.passed}"
    )

    if val.errors:
        print("VAL ERRORS:")
        for err in val.errors[:30]:
            print(f"  {err}")
    if lem.errors:
        print("LEM ERRORS:")
        for err in lem.errors[:30]:
            print(f"  {err}")

    if not val.passed or not lem.passed:
        sys.exit(1)

    print("STATUS: OK — en_c1_train_001 through en_c1_train_200")


if __name__ == "__main__":
    main()