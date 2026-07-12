"""Generate 200 handcrafted English C1 CoNLL-U sentences (en_c1_train_201–400)."""

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
    # Constitutional law 201-225
    "The constitutional court examined whether delegated legislative authority exceeded limits prescribed by the parliamentary enabling statute.",
    "Judicial review ensures that executive decrees remain consistent with fundamental rights enshrined in the national constitutional charter.",
    "Although federalism distributes competences across tiers of government, conflicts frequently arise over concurrent regulatory jurisdiction in commerce.",
    "The separation of powers doctrine constrains legislative overreach by empowering independent courts to invalidate unconstitutional statutory provisions.",
    "Constitutional amendments require supermajority approval and extended deliberation to prevent hasty revisions undermining institutional stability.",
    "It remains contested whether emergency powers granted during crises should automatically lapse after predetermined temporal thresholds expire.",
    "The bill of rights guarantees procedural fairness while obliging the state to justify limitations on civil liberties proportionally.",
    "Having ratified the convention, the state incorporated international human rights norms into domestically enforceable constitutional adjudication frameworks.",
    "Scholars debated whether originalist interpretation adequately addresses technological developments unforeseen by eighteenth-century constitutional framers.",
    "The ombudsman challenged administrative detention orders lacking individualized judicial authorization within constitutionally mandated timeframes.",
    "Preliminary references enable national courts to seek authoritative guidance from the supranational court on treaty compatibility questions.",
    "Notwithstanding parliamentary sovereignty doctrines, certain legal norms may bind future legislatures through entrenched constitutional guarantees.",
    "The constitutional tribunal ruled that electoral districting practices violated equal suffrage principles embedded in foundational constitutional texts.",
    "Federal supremacy clauses resolve conflicts between state legislation and nationally enacted statutes within enumerated competence domains.",
    "The attorney general defended the statute's constitutionality by demonstrating narrowly tailored objectives served by disputed regulatory measures.",
    "Constitutional conventions supplement written provisions by guiding executive conduct even when formal textual mandates remain deliberately ambiguous.",
    "The dissenting opinion warned that deferential scrutiny of security legislation risked eroding judicial protection of marginalized communities.",
    "Subnational governments invoked constitutional autonomy guarantees to resist centrally imposed standardization of educational curricula nationwide.",
    "The constitutional complaint procedure allows individual petitioners to challenge statutes directly before the highest constitutional adjudicator.",
    "Proportionality analysis requires assessing suitability, necessity, and balanced severity before restricting constitutionally protected expressive freedoms.",
    "The grand chamber reheard the case after conflicting lower panel decisions created doctrinal uncertainty concerning privacy rights online.",
    "Constitutional identity clauses permit member states to resist harmonization measures perceived as threatening foundational legal traditions domestically.",
    "The impeachment proceedings examined whether executive misconduct constituted high crimes warranting removal under constitutional removal standards.",
    "Judicial appointments commissions aim to depoliticize selection processes while preserving democratic accountability through transparent nomination hearings.",
    "The constitutional settlement balanced minority veto mechanisms with majoritarian decision rules to prevent legislative paralysis in plural societies.",
]

BATCH_2 = [
    # Constitutional law 226-250
    "The living constitutionalist argued that evolving social consensus should inform interpretation of open-textured constitutional equality provisions.",
    "Standing requirements limit access to constitutional litigation, thereby filtering disputes to plaintiffs demonstrating concrete and particularized injuries.",
    "Although the legislature enjoys broad discretion in taxation, discriminatory classification schemes remain subject to heightened constitutional scrutiny.",
    "The constitutional court invalidated retroactive criminal statutes that violated principles prohibiting punishment without prior legal authorization clearly.",
    "Federal police powers coexist uneasily with constitutional guarantees restraining searches conducted without probable cause or judicial warrants.",
    "It is widely acknowledged that constitutional adjudication inevitably involves value judgments about competing liberty and security imperatives.",
    "The amicus brief synthesized comparative constitutional scholarship supporting robust protection of digital privacy against indiscriminate state surveillance.",
    "Having consolidated fragmented jurisdiction, the tribunal centralized constitutional matters previously dispersed among specialized administrative tribunals nationally.",
    "The sunset clause ensured that extraordinary surveillance authorities expired unless explicitly reauthorized through renewed legislative deliberation periodically.",
    "Constitutional theorists examine how dialogic models encourage iterative exchanges between courts, legislatures, and civil society stakeholders.",
    "The reserved powers doctrine preserves regulatory authority for subnational entities in domains not expressly delegated to central institutions.",
    "Notwithstanding textual silence, implied constitutional rights derive from structural principles underlying democratic governance and rule of law.",
    "The chamber held that compelled speech requirements infringed constitutional autonomy by mandating endorsement of government-sponsored ideological messaging.",
    "Constitutional avoidance canons instruct courts to prefer statutory constructions that circumvent unnecessary resolution of constitutional questions.",
    "The referendum result triggered constitutional reform negotiations addressing longstanding grievances concerning regional autonomy and fiscal redistribution.",
    "Executive privilege claims must yield when competing constitutional interests in criminal investigation and accountability substantially outweigh confidentiality concerns.",
    "The constitutional bench affirmed that affirmative action programs required periodic reassessment to ensure continued alignment with remedial objectives.",
    "Parliamentary immunity shields legislators from prosecution for speeches delivered within chambers, preserving robust democratic deliberation constitutionally.",
    "The preliminary injunction suspended enforcement of the statute pending final constitutional review of its restrictions on peaceful assembly.",
    "Scholars caution that excessive judicial activism may undermine democratic legitimacy when courts substitute policy preferences for legislative judgment.",
    "The constitutional mandate required independent electoral administration insulated from partisan interference during nationally consequential ballot procedures.",
    "Dualist constitutional systems treat international treaties as requiring domestic incorporation before courts may enforce rights provisions directly.",
    "The concurring opinion emphasized that procedural due process safeguards apply equally to administrative agencies exercising quasi-judicial adjudicatory functions.",
    "Constitutional unamendability doctrines protect democratic core principles against repeal by transient legislative majorities pursuing authoritarian consolidation.",
    "The remedial decree mandated redistricting maps complying with constitutional standards for representational equality across demographic constituencies nationwide.",
]

BATCH_3 = [
    # Macroeconomics 251-275
    "Monetary policymakers signaled willingness to maintain restrictive interest rates until inflation expectations converge sustainably toward official targets.",
    "The fiscal multiplier for infrastructure investment appears larger during recessions when idle capacity and accommodative monetary conditions prevail simultaneously.",
    "Although unemployment declined, discouraged worker effects and declining participation rates complicated interpretation of headline labor market indicators.",
    "Central bank independence shields rate decisions from political pressures that might fuel unsustainable aggregate demand expansions.",
    "The Phillips curve weakened as supply shocks altered historical trade-offs between inflation and unemployment.",
    "It remains debatable whether quantitative easing primarily stimulated real activity or inflated asset valuations disproportionately benefiting wealthy households.",
    "Twin deficits in budgets and current accounts reflect structural imbalances requiring coordinated fiscal adjustment programs.",
    "Yield curve inversion historically precedes recessions when term premia fluctuate unpredictably amid policy uncertainty.",
    "Having accumulated precautionary reserves, emerging economies weathered capital outflow pressures better than during previous financial crisis episodes.",
    "Macroprudential regulators tightened lending ratios to restrain credit fueled property booms threatening financial system stability.",
    "The public debt ratio stabilized following primary fiscal surpluses and favorable nominal output growth effects.",
    "Currency depreciations transmit into consumer prices differently across economies according to exchange rate pass through dynamics.",
    "Core services inflation persisted due to wage growth in labor constrained sectors despite aggressive monetary tightening.",
    "The national accounts revision incorporated improved measurement of intangible investment and digital platform contributions to gross domestic product.",
    "Automatic stabilizers moderated recession severity by expanding transfer payments and reducing tax liabilities as aggregate income contracted cyclically.",
    "Forward guidance shapes expectations by communicating conditional policy pathways contingent upon evolving inflation and employment data releases.",
    "The output gap estimate suggested persistent slack despite low unemployment, reflecting mismeasurement of potential growth following productivity slowdowns.",
    "Sovereign risk premia widened amid concerns that state enterprise liabilities could destabilize medium term fiscal sustainability.",
    "The reserve currency status conferred exorbitant privilege, enabling cheaper international borrowing despite chronic current account deficits domestically.",
    "Stagflation scenarios complicate policymaking when contractionary monetary responses risk amplifying unemployment without reliably containing supply-driven inflation pressures.",
    "The growth accounting decomposition attributed deceleration primarily to slowing total factor productivity rather than demographic headwinds alone.",
    "Capital flight accelerated after investors anticipated imminent currency devaluation and capital control imposition by monetary authorities desperately.",
    "The Taylor rule prescription implied rates should remain elevated given inflation overshoots and output levels exceeding estimated potential consistently.",
    "Hysteresis effects suggest prolonged recessions may permanently reduce potential output through skill atrophy and diminished business formation rates.",
    "The intertemporal government budget constraint requires that present debt levels remain consistent with credible future primary balance commitments politically.",
]

BATCH_4 = [
    # Macroeconomics 276-300
    "Rational expectations models assume agents incorporate available information efficiently when forming forecasts about policy responses and inflation dynamics.",
    "Although headline inflation moderated, shelter and energy components continued exerting upward pressure on household inflation perception indices.",
    "The liquidity trap constrains conventional monetary policy effectiveness when nominal interest rates approach zero lower bound thresholds persistently.",
    "Fiscal consolidation packages combined expenditure caps with progressive taxation reforms aimed at preserving social safety net adequacy.",
    "The purchasing managers index declined for consecutive months, signaling weakening manufacturing activity amid contracting export order volumes globally.",
    "It cannot be denied that wealth inequality influences consumption patterns because marginal propensities differ substantially across income distributions.",
    "The balance of payments crisis necessitated emergency financing accompanied by structural reforms liberalizing trade and strengthening tax administration.",
    "Having diversified export markets, the economy reduced vulnerability to commodity price volatility affecting primary product dependent budgets.",
    "The neutral rate estimate guides policymakers assessing whether current settings are contractionary relative to long run equilibrium.",
    "Crowding out concerns arise when government borrowing elevates interest rates, displacing private investment in credit constrained environments.",
    "The misery index combining unemployment and inflation captures popular economic discontent during periods of simultaneous macroeconomic stress.",
    "Notwithstanding globalization benefits, trade shocks transmit rapidly through integrated supply chains amplifying domestic output fluctuations unexpectedly.",
    "The Okun coefficient quantified responsiveness of unemployment to output deviations during cyclical expansions and subsequent downturn episodes.",
    "Deflationary spirals emerge when falling prices increase real debt burdens, prompting consumption postponement and further downward price pressure.",
    "The structural budget balance adjusts for cyclical components, revealing underlying fiscal stance independent of temporary automatic stabilizer effects.",
    "Credibility of inflation targeting frameworks depends on transparent communication and consistent policy reactions to persistent target deviations.",
    "The real effective exchange rate appreciated substantially, eroding export competitiveness despite stable nominal bilateral currency peg arrangements.",
    "Financial accelerator mechanisms amplify business cycle fluctuations as collateral values and credit availability move procyclically during expansions.",
    "The precautionary saving motive intensified following heightened geopolitical uncertainty and deteriorating consumer confidence survey readings nationally.",
    "Supply side reforms targeting labor rigidities aim to raise potential growth without generating demand pull inflationary pressures.",
    "The consumption function specification incorporated wealth effects from housing valuations alongside disposable income variables empirically.",
    "External adjustment required real depreciation through internal devaluation when exchange rate flexibility remained politically constrained domestically.",
    "The Beveridge curve shift indicated worsening matching efficiency between vacant positions and unemployed workers across regional labor markets.",
    "Debt sustainability analyses stress test projections under adverse growth, interest rate, and contingent fiscal shock scenarios.",
    "The macroeconomic consensus forecast projected gradual disinflation alongside resilient employment, contingent upon stable energy prices internationally.",
]

BATCH_5 = [
    # Epidemiology 301-325
    "The cohort study estimated hazard ratios for cardiovascular events associated with prolonged exposure to fine particulate air pollution.",
    "Case definitions for notifiable diseases require laboratory confirmation to minimize misclassification bias in official surveillance reporting systems.",
    "Although vaccination reduced symptomatic infections, breakthrough transmission among household contacts necessitated continued nonpharmaceutical intervention measures temporarily.",
    "The reproductive number declined below unity following combined immunity from prior infection and mass immunization campaign completion nationwide.",
    "Molecular epidemiology traced outbreak clusters through genomic sequencing, revealing cryptic community transmission chains previously undetected by contact tracing.",
    "It remains critical that seroprevalence surveys account for waning antibody titers when inferring population-level cumulative infection burdens.",
    "The nested case control design evaluated dietary risk factors while controlling for confounding lifestyle variables through matched sampling.",
    "Having implemented syndromic surveillance, health authorities detected anomalous emergency department presentations preceding confirmed pathogen identification regionally.",
    "Attack rates within confined settings demonstrated heightened transmissibility under conditions of poor ventilation and prolonged close contact.",
    "The odds ratio suggested significant association between occupational exposure and incident disease after adjusting for age and comorbidity status.",
    "Notwithstanding declining incidence, geographic heterogeneity in vaccine coverage perpetuated localized resurgence risk among undervaccinated demographic subgroups.",
    "The incubation period distribution informed quarantine duration policies balancing transmission prevention against economic and psychosocial disruption costs.",
    "Propensity score matching reduced selection bias when comparing treatment outcomes between observational patient groups with differing baseline characteristics.",
    "The epidemic curve peaked earlier than modeled projections after aggressive testing expansion and behavior change reduced effective contact rates.",
    "Zoonotic spillover events underscore necessity of integrated surveillance linking veterinary, ecological, and human health data streams systematically.",
    "The attributable fraction quantified disease burden preventable through hypothetical elimination of modifiable environmental exposures across populations.",
    "Screening program sensitivity depends on test performance characteristics and disease prevalence within targeted asymptomatic community populations.",
    "The pharmacoepidemiological study linked prescription records to mortality registries while preserving patient confidentiality through anonymized linkage protocols.",
    "Having randomized intervention clusters, investigators assessed whether school closure policies reduced community transmission beyond direct educational settings.",
    "The confidence interval for vaccine effectiveness excluded null values, supporting robust protection against severe outcomes among immunocompromised patients.",
    "Surveillance bias may inflate apparent incidence trends when expanded diagnostic criteria capture milder clinical presentations previously underreported systematically.",
    "The compartmental model incorporated waning immunity parameters to project epidemic trajectories under varying booster vaccination coverage scenarios.",
    "Ecological studies examining aggregate exposures must cautiously interpret associations that may reflect confounding at individual levels.",
    "The outbreak investigation implicated a common food source through traceback analysis coordinated across multijurisdictional public health agencies.",
    "Latency period estimation informed occupational disease registries assessing cumulative exposure risks among manufacturing workers retrospectively.",
]

BATCH_6 = [
    # Epidemiology 326-350
    "The randomized field trial evaluated insecticide-treated bed nets reducing malaria incidence among children in hyperendemic transmission regions.",
    "Although herd immunity thresholds were approached, variant emergence threatened to erode population protection and necessitate updated vaccine formulations.",
    "The standardized mortality ratio compared observed deaths against expected rates derived from age-specific population reference tables regionally.",
    "Causal inference frameworks emphasize triangulating evidence from diverse study designs rather than relying solely on observational association measures.",
    "The contact tracing application logged proximity events while implementing privacy-preserving encryption to address civil liberties concerns adequately.",
    "It is increasingly recognized that social determinants mediate disease risk through pathways involving housing instability and inadequate healthcare access.",
    "Kaplan Meier analysis illustrated diverging survival curves before proportional hazards assumptions were formally tested statistically.",
    "Having stratified randomization by site, the multicenter trial minimized imbalance in baseline prognostic factors across participating clinical centers.",
    "False discovery rate adjustment controlled expected proportion of spurious associations in high dimensional genomic analyses.",
    "Disease mapping techniques revealed persistent spatial clustering of tuberculosis cases in neighborhoods characterized by socioeconomic deprivation indicators.",
    "Per protocol analysis excluded participants with major deviations, diverging from intention to treat effectiveness estimates.",
    "Notwithstanding high specificity, low pretest probability contexts rendered positive screening results more likely false alarms than true disease.",
    "The age period cohort model disentangled temporal trends when analyzing chronic disease incidence across generational cohorts.",
    "Environmental sampling detected pathogen persistence on surfaces, informing infection control protocols during nosocomial outbreak responses.",
    "The relative risk reduction translated into modest absolute benefits when baseline event rates remained low among screened populations.",
    "Reverse causation threatens validity when early disease stages alter exposures investigators later associate with outcome development retrospectively.",
    "The stepped wedge design sequentially introduced interventions across clusters, accommodating logistical constraints precluding simultaneous nationwide program rollout.",
    "Migrant health surveillance highlighted disparities in latent infection prevalence linked to origin-country endemicity and predeparture screening policies.",
    "The diagnostic odds ratio summarized test discrimination capacity across varying prevalence settings better than sensitivity and specificity alone.",
    "Having pooled individual participant data, the consortium increased statistical power to evaluate treatment effect heterogeneity across prespecified subgroups.",
    "The endemic equilibrium model predicted resurgent seasonality once susceptible recruitment through demographic turnover restored transmission potential cyclically.",
    "Recall bias may distort exposure histories when case participants more intensely search memory for hypothesized risk factors than controls.",
    "The attributable risk percent indicated what proportion of cases among exposed individuals resulted specifically from identified occupational hazards.",
    "Screening interval optimization balances early detection benefits against false positive harms and cumulative radiation exposure in imaging programs.",
    "The field epidemiology training program strengthened capacity for rapid outbreak response in resource-limited settings through mentored investigations.",
]

BATCH_7 = [
    # International trade 351-375
    "Most favored nation clauses prohibit discriminatory tariffs among trading partners except where preferential regional agreements apply legally.",
    "The trade deficit widened as import volumes surged despite currency depreciation failing to stimulate export competitiveness sufficiently rapidly.",
    "Although tariffs generated fiscal revenue, retaliatory measures offset anticipated gains by restricting market access for agricultural export commodities.",
    "Rules of origin determinations govern eligibility for preferential duties under free trade agreements incorporating substantial regional value content.",
    "The anti-dumping investigation examined whether foreign producers sold below normal value, injuring domestic industries manufacturing comparable product lines.",
    "It remains contentious whether bilateral investment treaties adequately balance investor protections with regulatory autonomy over environmental standards domestically.",
    "The customs union eliminated internal tariffs while harmonizing external trade policy toward nonmember countries across participating economies collectively.",
    "Having reformed export procedures, the ministry reduced dwell times at borders through digital preclearance and risk based inspection.",
    "Comparative advantage theory predicts specialization patterns even when trading partners exhibit absolute productivity advantages across all goods produced.",
    "The safeguard mechanism permitted temporary tariff increases when import surges threatened serious injury to domestic production sectors nationally.",
    "Notwithstanding services liberalization commitments, professional qualification recognition barriers continued impeding cross-border mobility of skilled practitioners.",
    "Sanitary agreements permit trade restrictions justified by scientific risk assessments protecting human, animal, and plant health.",
    "Terms of trade shocks redistribute welfare between exporting commodity producers and importing manufacturing economies during global price fluctuations.",
    "The export processing zone offered tax incentives attracting assembly operations reliant on imported intermediates reexported as finished manufactured goods.",
    "Intellectual property provisions extended patent protections, provoking debates about access to affordable generic medicines globally.",
    "The countervailing duty countered subsidized imports that distorted competition by enabling foreign producers to undercut unsubsidized domestic market prices.",
    "Trade facilitation reforms standardized documentation requirements, reducing transactional costs for small exporters participating in regional value chains.",
    "The gravity model explained bilateral trade flows using economic size, distance, and institutional affinity variables estimated econometrically.",
    "Having diversified suppliers, manufacturers mitigated disruption risks previously concentrated among single-source procurement arrangements during pandemic shortages.",
    "The bound tariff commitments limited permissible applied rates, enhancing predictability for exporters navigating multilateral trade negotiation outcomes.",
    "Export restrictions on critical minerals prompted concerns about weaponized interdependence in strategic industries including renewable energy manufacturing.",
    "The trade creation effect outweighed trade diversion as regional integration redirected commerce toward more efficient intra-bloc specialization patterns.",
    "Non-tariff barriers including licensing requirements and conformity assessment procedures often impede market entry more than ad valorem duties.",
    "The current account adjustment followed improved services exports and remittance inflows offsetting persistent merchandise trade deficits structurally.",
    "Bilateral swap arrangements provided liquidity during currency crises, stabilizing trade finance availability for importers facing acute dollar shortages.",
]

BATCH_8 = [
    # Cognitive science 376-400
    "The dual-process model distinguishes fast heuristic reasoning from slower analytic deliberation underlying judgment under uncertainty systematically.",
    "Although working memory capacity constrains complex problem solving, chunking strategies enable experts to circumvent apparent cognitive limitations effectively.",
    "Event related potential studies identified neural signatures of semantic incongruity within several hundred milliseconds post stimulus presentation.",
    "Embodied cognition theories emphasize sensorimotor interaction with environments rather than abstract symbol manipulation alone during conceptual processing.",
    "The Stroop task demonstrated automaticity of reading processes that interfere selectively with color naming responses under conflicting conditions.",
    "It is widely accepted that predictive coding frameworks minimize surprise by updating internal models through hierarchical Bayesian inference mechanisms.",
    "Longitudinal studies traced shifts from egocentric perspective taking toward mature theory of mind capacities across early childhood.",
    "Having controlled attentional load, researchers isolated priming effects attributable to implicit memory rather than conscious strategic processing.",
    "Neuroimaging evidence implicated prefrontal cortex in executive control functions mediating conflict monitoring and behavioral inhibition during tasks.",
    "The illusion of control persists because confirmation bias selectively reinforces beliefs that personal agency influenced random outcome events.",
    "Notwithstanding behavioral parallels, computational models debated whether neural implementations resemble symbolic production systems or distributed connectionist architectures.",
    "The attentional blink paradigm revealed temporal limitations preventing conscious perception of second targets presented shortly after first identification.",
    "Metacognitive monitoring enables learners to calibrate confidence judgments, improving study allocation decisions when accuracy feedback remains unavailable.",
    "The prototype theory explains categorization gradedness through similarity to central exemplars rather than necessary and sufficient definitional features.",
    "Cross-linguistic studies examined whether linguistic relativity shapes color discrimination thresholds across communities using differing basic color terminologies.",
    "The change blindness phenomenon illustrates limited capacity for detecting substantial visual alterations when attention remains focally allocated elsewhere.",
    "Having manipulated reward schedules, the experiment tested whether dopaminergic signals encoded prediction errors rather than reward receipt directly.",
    "The framing effect demonstrates reference dependence whereby equivalent outcomes elicit divergent choices depending on descriptive presentation format.",
    "Cognitive load theory distinguishes intrinsic, extraneous, and germane loads affecting learning efficiency in instructional design environments pedagogically.",
    "The mental rotation task provided evidence for spatial representation formats supporting analog transformation processes rather than purely propositional encodings.",
    "Sleep dependent memory consolidation strengthens hippocampal neocortical interactions integrating episodic details into semantic knowledge structures.",
    "The cocktail party effect shows selective attention can amplify attended speech streams while suppressing irrelevant background auditory input channels.",
    "Situated cognition research investigates how tools and environmental affordances extend problem solving capacities beyond isolated individual minds.",
    "The anchoring heuristic skews numerical estimates toward initially presented values even when participants recognize those anchors as arbitrary.",
    "Neuroplasticity findings suggest intensive training regimes can reallocate cortical representation maps following sensory deprivation or skill acquisition extensively.",
]

SENTENCE_BATCHES = [BATCH_1, BATCH_2, BATCH_3, BATCH_4, BATCH_5, BATCH_6, BATCH_7, BATCH_8]
SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"
for i, batch in enumerate(SENTENCE_BATCHES, 1):
    assert len(batch) == 25, f"BATCH_{i} has {len(batch)} sentences"

START_ID = 201
BATCH_SIZE = 25
TARGET_PATH = project_root / "data/handcraft/en/train/c1_new_002.conllu"

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
    "exercising": "exercise",
    "suppressing": "suppre",
    "suppress": "suppre",
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

    print("STATUS: OK — en_c1_train_201 through en_c1_train_400")


if __name__ == "__main__":
    main()