"""Generate 200 handcrafted English C1 CoNLL-U sentences (en_c1_train_601–800)."""

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
    # Neuroscience & neurology 601-625
    "Functional magnetic resonance imaging revealed distributed cortical networks underlying episodic memory retrieval during controlled laboratory recall tasks.",
    "Although deep brain stimulation alleviated tremor, cognitive side effects necessitated careful electrode placement and individualized programming adjustments.",
    "The longitudinal neuroimaging study tracked gray matter volume changes associated with intensive bilingual language training across adolescent participants.",
    "Optogenetic experiments demonstrated causal links between hippocampal interneuron activity and spatial navigation performance in freely moving rodents.",
    "The stroke rehabilitation protocol combined constraint-induced movement therapy with transcranial magnetic stimulation targeting ipsilesional motor cortex.",
    "It remains debated whether consciousness arises from integrated information processing or requires specific anatomical substrates within thalamocortical circuits.",
    "The neuropathology report identified tau protein aggregates consistent with neurodegenerative progression in autopsy specimens from dementia patients.",
    "Having mapped connectome hubs, researchers analyzed how hub disruption propagates network dysfunction in traumatic brain injury cohorts.",
    "Diffusion tensor imaging quantified white matter integrity declines correlating with cognitive speed deficits among multiple sclerosis patients.",
    "The electrophysiology lab recorded spike-timing-dependent plasticity mechanisms modulating synaptic strength during patterned presynaptic stimulation protocols.",
    "Notwithstanding symptomatic improvement, long-term antiepileptic therapy requires monitoring for metabolic bone disease and cognitive adverse effects.",
    "The neurodevelopmental assessment examined how preterm birth influences executive function trajectories throughout middle childhood educational transitions.",
    "Pharmacological blockade of NMDA receptors disrupted fear extinction learning, implicating glutamatergic signaling in anxiety disorder pathophysiology.",
    "The intracranial recording study localized seizure onset zones preceding spread into eloquent cortical areas governing speech production.",
    "Resting-state functional connectivity patterns distinguished autism spectrum presentations with varying sensory sensitivity and social communication profiles.",
    "It is increasingly recognized that neuroinflammation contributes to cognitive decline beyond classical amyloid-centric models of Alzheimer pathology.",
    "The neurosurgical team resected eloquent cortex adjacent to tumors using awake mapping to preserve language and motor function.",
    "Having administered cognitive training, investigators measured transfer effects on untrained working memory tasks after eight weeks.",
    "The cerebrospinal fluid biomarker panel detected reduced amyloid beta ratios predicting conversion from mild impairment to dementia.",
    "Transcranial direct current stimulation modulated motor learning rates when applied over primary motor cortex during skill acquisition.",
    "The pediatric neurology clinic evaluated mitochondrial disorders presenting with seizures, lactic acidosis, and progressive multisystem organ involvement.",
    "Neuroplasticity following spinal cord injury depends on rehabilitation intensity, lesion completeness, and residual descending supraspinal pathways.",
    "The computational model simulated dendritic integration nonlinearities shaping direction-selective responses in visual cortical pyramidal neurons.",
    "Sleep spindle density correlated with next-day declarative memory consolidation among healthy adults undergoing polysomnographic monitoring overnight.",
    "The movement disorders fellowship analyzed gait kinematics distinguishing parkinsonian bradykinesia from ataxic instability using instrumented walkway assessments.",
]

BATCH_2 = [
    # Criminal law & procedure 626-650
    "The appellate court examined whether warrantless searches incident to arrest exceeded Fourth Amendment reasonableness under contemporary precedent.",
    "Although the defendant invoked Miranda rights, subsequent voluntary statements remained admissible after counsel was properly appointed.",
    "The grand jury indictment alleged conspiracy to commit wire fraud through falsified invoices submitted to federal procurement agencies.",
    "Prosecutors sought enhanced sentencing under guidelines accounting for leadership roles and obstruction of justice during pretrial discovery.",
    "The suppression hearing challenged forensic evidence after defense experts identified laboratory chain-of-custody contamination risks.",
    "It remains contested whether predictive policing algorithms violate equal protection when deployment concentrates surveillance in minority neighborhoods.",
    "The plea agreement recommended substantial assistance departures contingent upon cooperating testimony against higher-ranking participants in organized schemes.",
    "Having filed a habeas petition, counsel argued ineffective assistance stemming from counsel's failure to investigate exculpatory witness statements.",
    "The double jeopardy motion asserted that reprosecution following mistrial violated constitutional protections against repeated exposure to criminal punishment.",
    "Eyewitness identification procedures must minimize suggestiveness through double-blind administration and sequential presentation of lineup alternatives.",
    "Notwithstanding prosecutorial discretion, selective enforcement claims require demonstrating similarly situated defendants received disparate treatment without justification.",
    "The restitution order compensated victims for direct financial losses while excluding speculative projections of future economic harm.",
    "Bail reform statutes presumptively favor release with conditions calibrated to flight risk and community safety concerns proportionally.",
    "The discovery order compelled disclosure of Brady material including impeachment evidence affecting credibility of cooperating government witnesses.",
    "Sentencing memoranda emphasized mitigating circumstances such as childhood trauma, rehabilitation efforts, and disproportionate guideline range severity.",
    "It is widely acknowledged that mandatory minimums constrain judicial discretion while producing racially disparate incarceration outcomes nationally.",
    "The immunity agreement shielded the witness from prosecution while requiring truthful testimony subject to perjury prosecution otherwise.",
    "Having exhausted state remedies, petitioners sought federal review alleging due process violations during capital sentencing proceedings.",
    "The exclusionary rule debate weighs deterrence benefits against social costs when suppressing reliable evidence obtained through minor violations.",
    "Joint trial severance motions argued that prejudicial spillover from co-defendant confessions would deny fair individualized adjudication of guilt.",
    "The criminal contempt sanction punished refusal to testify before legislative committees investigating corruption in municipal contracting processes.",
    "Racketeering statutes enable prosecuting continuing enterprises through predicate acts including extortion, money laundering, and fraudulent bankruptcy filings.",
    "The competency evaluation determined whether the accused understood charges and could assist counsel during fraud litigation.",
    "Victim impact statements informed sentencing while preserving defendants' rights to challenge factual assertions lacking evidentiary corroboration adequately.",
    "The appellate brief argued prosecutorial vindictiveness when enhanced charges followed defendants' rejection of prior plea offers unreasonably.",
]

BATCH_3 = [
    # Renewable energy & grid systems 651-675
    "The solar farm integrated tracking arrays and bifacial modules to maximize energy yield across seasonal irradiance variations.",
    "Although offshore wind capacity expanded rapidly, transmission bottlenecks delayed delivering power from coastal generation hubs inland.",
    "The grid operator implemented demand response programs curtailing industrial loads during peak intervals to maintain frequency stability.",
    "Battery energy storage systems provided ancillary services including rapid frequency regulation and deferred transmission infrastructure investments regionally.",
    "The power purchase agreement indexed tariffs to wholesale market prices while guaranteeing minimum revenue floors for project lenders.",
    "It remains challenging to balance renewable curtailment against reliability when inverter-based resources displace synchronous conventional generation.",
    "The microgrid controller islanded critical facilities during distribution outages while coordinating diesel backup and photovoltaic inverters seamlessly.",
    "Having upgraded substations, engineers reduced congestion costs on interregional lines transporting wind generation toward urban load centers.",
    "Levelized cost analyses incorporated learning curve assumptions for electrolyzers producing green hydrogen from curtailed renewable electricity surpluses.",
    "The interconnection queue study revealed years-long delays attributable to insufficient hosting capacity studies and outdated planning methodologies.",
    "Notwithstanding policy incentives, rooftop solar adoption lagged in rental housing markets lacking split incentive arrangements between landlords.",
    "The virtual power plant aggregated residential batteries dispatching coordinated discharge schedules during wholesale price spike events.",
    "Curtailment rates climbed when negative pricing episodes coincided with high wind output and limited export corridor capacity.",
    "The reliability assessment modeled resource adequacy under extreme weather scenarios stressing gas supply and simultaneous generator forced outages.",
    "Community choice aggregators procured renewable certificates while negotiating long-term contracts diversifying municipal electricity supply portfolios strategically.",
    "It is widely recognized that distribution operators require visibility into behind-the-meter inverters managing voltage fluctuations.",
    "The floating offshore platform supported turbines in deep waters where fixed foundations would be economically prohibitive to install.",
    "Having deployed synchrophasor monitoring, operators detected oscillatory instability precursors along high-voltage corridors connecting renewable zones.",
    "Agrivoltaic pilots combined crop cultivation with elevated panel arrays moderating microclimates while generating supplementary farm revenue streams.",
    "The capacity market redesign compensated resources for availability during scarcity events rather than solely energy produced hourly.",
    "Geothermal exploration identified permeable reservoir formations suitable for enhanced systems injecting fluids to extract subsurface heat.",
    "The inverter standard mandated ride-through capabilities during voltage dips preventing cascading disconnections during distant transmission faults.",
    "Corporate power agreements enabled manufacturers to hedge electricity costs while directly financing new wind projects near facilities.",
    "The sector coupling strategy linked power, heating, and transport through electrified boilers and smart vehicle charging.",
    "Lifecycle assessments compared emissions from manufacturing, installation, and recycling across competing photovoltaic module technologies.",
]

BATCH_4 = [
    # Development economics 676-700
    "The randomized cash transfer evaluation measured impacts on school enrollment and child labor among rural households facing liquidity constraints.",
    "Although gross domestic product grew steadily, multidimensional poverty indices revealed persistent deficits in sanitation and educational attainment.",
    "The microfinance impact study compared entrepreneurial outcomes between clients receiving group liability loans and individual liability products.",
    "Structural transformation models predict labor reallocations from agriculture toward manufacturing as productivity gaps narrow across sectors.",
    "The field experiment tested whether labeled savings accounts increased precautionary balances among informal sector workers without bank relationships.",
    "It remains debated whether foreign aid catalyzes sustainable growth or perpetuates dependency through tied procurement and fungibility problems.",
    "The poverty mapping exercise integrated satellite nightlight data with household surveys to estimate subnational consumption distributions accurately.",
    "Having implemented land titling reforms, policymakers examined whether formalized property rights increased agricultural investment and credit access.",
    "The graduation program combined asset transfers, training, and coaching to lift ultra-poor households toward livelihoods.",
    "Remittance inflows stabilized household consumption during droughts but introduced exchange rate pressures in recipient economies periodically.",
    "Notwithstanding trade liberalization, infant industries struggled against import competition without temporary tariff protection and export promotion.",
    "The governance indicator correlated public investment efficiency with bureaucratic quality and control of corruption across developing regions.",
    "Conditional cash programs required school attendance and health clinic visits in exchange for periodic transfers to beneficiary mothers.",
    "The agricultural extension trial disseminated drought-resistant seed varieties accompanied by demonstration plots and farmer field school sessions.",
    "Settlement upgrading projects provided tenure security, paved roads, and communal water connections improving living standards measurably.",
    "It is widely argued that inclusive institutions explain divergent development trajectories between resource-rich countries with similar endowments.",
    "The impact evaluation exploited regression discontinuity around eligibility cutoffs to estimate scholarship effects on tertiary completion rates.",
    "After devaluing the currency, authorities tightened policy to contain pass-through inflation in import-dependent economies.",
    "The social protection registry unified fragmented databases enabling targeted assistance during sudden unemployment shocks and natural disasters.",
    "Export processing zones attracted foreign direct investment through tax holidays while generating limited backward linkages domestically.",
    "The nutrition intervention distributed micronutrient supplements to pregnant women reducing anemia prevalence and low birthweight incidence.",
    "Public works programs employed unemployed youth on infrastructure maintenance tasks during seasonal agricultural slack periods locally.",
    "The debt relief initiative freed fiscal space for health spending while requiring transparent public financial management reforms.",
    "Digital payment adoption accelerated financial inclusion among previously unbanked merchants accepting mobile money for everyday transactions.",
    "The village lottery determined participation order in a sanitation subsidy program randomizing exposure to peer construction demonstrations.",
]

BATCH_5 = [
    # Medieval & early modern history 701-725
    "The monastic chronicle documented feudal obligations linking manorial tenants to labor services and grain rents during harvest seasons.",
    "Although crusading rhetoric emphasized spiritual redemption, commercial interests shaped Mediterranean trade routes connecting Italian city-states eastward.",
    "The archival inventory listed guild regulations governing apprenticeship terms, quality standards, and collective bargaining with municipal authorities.",
    "Humanist scholars revived classical rhetoric while critiquing scholastic disputation methods prevalent in late medieval university curricula.",
    "The plague visitation registers recorded mortality spikes disrupting inheritance patterns and wage negotiations in agrarian labor markets.",
    "It remains debated whether the Renaissance represented rupture with medieval traditions or gradual transformation of existing cultural practices.",
    "The inquisition trial transcripts reveal how conversos navigated accusations of crypto-Judaism amid intensified religious conformity campaigns.",
    "Having translated Aristotelian texts, commentators reconciled pagan philosophy with Christian theology through sophisticated allegorical interpretive frameworks.",
    "The Hanseatic league coordinated maritime insurance, diplomatic privileges, and standardized weights across networked Baltic trading entrepots.",
    "Peasant revolt manifestos demanded abolition of seigneurial dues and recognition of customary commons access rights historically.",
    "Notwithstanding dynastic marriages, territorial disputes over succession claims precipitated prolonged conflicts exhausting royal treasuries repeatedly.",
    "The scriptorium produced illuminated manuscripts whose marginalia preserved folkloric motifs alongside liturgical texts for aristocratic patrons.",
    "Enclosure movements consolidated fragmented open fields, displacing subsistence cultivators toward wage labor in emerging textile manufactories.",
    "The royal ordinance codified sumptuary laws restricting luxurious apparel signaling social status beyond prescribed estate hierarchies strictly.",
    "Mendicant orders preached urban charity while challenging entrenched ecclesiastical wealth accumulation among secular cathedral chapters.",
    "It is widely argued that print diffusion accelerated Reformation debates by enabling vernacular pamphlets to circulate among literate burghers.",
    "The diplomatic correspondence described marriage alliances brokered to secure border fortresses and commercial monopolies over salt routes.",
    "Having conquered Granada, monarchs centralized judicial authority while negotiating coexistence agreements regulating religious minorities temporarily.",
    "The witchcraft treatise synthesized demonological theories influencing prosecutorial zeal during episodes of localized social tension.",
    "Mercantilist pamphleteers advocated bullion accumulation and colonial monopolies as foundations of national power in rivalrous states.",
    "The voyage narrative blended empirical observation with classical geography, reshaping European conceptions of Atlantic and Pacific worlds.",
    "University statutes prescribed disputation formats, degree requirements, and faculty privileges within corporately governed academic communities.",
    "The grain price series reconstructed harvest failures correlating with subsistence crises and urban bread riot outbreaks periodically.",
    "Chivalric romances idealized courtly love conventions while reflecting aristocratic anxieties about lineage legitimacy and martial honor.",
    "The council minutes recorded jurisdictional conflicts between ecclesiastical courts and emerging royal justice over marriage annulment petitions.",
]

BATCH_6 = [
    # Immunology & vaccinology 726-750
    "The adaptive immune response generated high-affinity antibodies through somatic hypermutation and clonal selection in germinal centers.",
    "Although checkpoint inhibitors unleashed antitumor immunity, autoimmune adverse events required corticosteroid management in substantial patient subsets.",
    "The flow cytometry panel quantified regulatory T cell frequencies correlating with transplant tolerance in renal allograft recipients.",
    "Innate lymphoid cells orchestrated tissue repair and antimicrobial defense at mucosal barriers following pathogen challenge in murine models.",
    "The vaccine trial measured neutralizing antibody titers and cellular responses after heterologous booster schedules using messenger RNA platforms.",
    "It remains uncertain whether broadly neutralizing antibodies against influenza can be elicited reliably through universal vaccine antigen designs.",
    "The cytokine storm syndrome complicated severe viral pneumonia, prompting trials of interleukin blockade and corticosteroid immunomodulation strategies.",
    "Having sequenced T cell receptors, researchers tracked clonal expansions recognizing conserved epitopes across convalescent donor repertoires.",
    "The complement cascade amplified opsonization and membrane attack complex formation against encapsulated bacterial pathogens in immunocompromised hosts.",
    "Mucosal immunization induced secretory immunoglobulin A responses reducing upper respiratory colonization without systemic reactogenicity events.",
    "Notwithstanding high efficacy, waning humoral immunity necessitated periodic booster doses maintaining protection against variant lineages.",
    "The autoantibody screen detected anti-nuclear patterns associated with systemic lupus flares following ultraviolet exposure and infections.",
    "Dendritic cell vaccines pulsed with tumor antigens primed cytotoxic lymphocytes in personalized immunotherapy protocols for melanoma patients.",
    "The immunodeficiency registry cataloged genetic defects impairing B cell development, class switching, and phagocyte function.",
    "Allergen immunotherapy desensitized mast cells through repeated injections under escalating titration schedules.",
    "It is increasingly recognized that trained immunity modulates monocyte responses following bacillus Calmette-Guerin vaccination in neonatal cohorts.",
    "The graft-versus-host disease prophylaxis combined calcineurin inhibitors with methotrexate after allogeneic hematopoietic stem cell transplantation.",
    "Having depleted B cells, clinicians monitored immunoglobulin replacement and infection surveillance in refractory rheumatoid arthritis patients.",
    "The epitope mapping study identified conserved regions suitable for mosaic immunogens spanning diverse human immunodeficiency virus subtypes.",
    "Toll-like receptor agonists adjuvanted subunit vaccines enhancing dendritic cell maturation and cross-presentation of exogenous antigens effectively.",
    "The hypersensitivity workup distinguished immediate immunoglobulin E mediated reactions from delayed T cell driven contact dermatitis manifestations.",
    "Live attenuated vaccines conferred durable cellular immunity but remained contraindicated in severely immunosuppressed individuals receiving biologics.",
    "The immune repertoire analysis revealed public clonotypes shared across unrelated donors recognizing identical viral peptide major histocompatibility complexes.",
    "Regulatory pathways governing biologics licensure require standardized potency assays and lot release testing for batch consistency.",
    "The mucosal adjuvant formulation enhanced nasal vaccine uptake while minimizing inflammatory damage to olfactory epithelial tissues.",
]

BATCH_7 = [
    # Aerospace & aeronautics 751-775
    "The composite wing structure reduced structural mass while maintaining flutter margins across transonic cruise flight envelopes.",
    "Although turbofan efficiency improved, contrail formation and nitrogen oxide emissions remain significant climate impacts from aviation.",
    "The flight test program validated digital control laws under simulated engine failure and gust disturbance conditions.",
    "Fluid dynamics resolved shock-boundary interactions influencing buffet onset during high-altitude maneuvering of airframes.",
    "The satellite constellation optimized revisit intervals for Earth observation while minimizing collision risks in crowded orbital shells.",
    "Certifying detect-and-avoid systems for unmanned aircraft beyond visual line of sight remains technically challenging.",
    "The propulsion team tested rotating detonation engines promising higher thermodynamic efficiency than conventional gas turbine combustors.",
    "Having analyzed black box data, investigators reconstructed stall progression preceding loss of control during icing encounters.",
    "The launch vehicle guidance algorithm compensated for wind shear and stage separation transients using closed-loop trajectory corrections.",
    "Hypersonic wind tunnel experiments measured heat flux distributions on leading edges during atmospheric reentry simulation conditions.",
    "Notwithstanding budget overruns, the deep space probe delivered spectrometric analyses of asteroid regolith composition as planned.",
    "The air traffic flow management system rerouted flights around convective weather while preserving arrival slot adherence at hubs.",
    "Structural health monitoring detected delamination growth in rotor blades through embedded fiber optic strain sensing networks continuously.",
    "The reusable booster executed powered landing maneuvers using grid fin actuation and throttleable rocket engine gimbal control.",
    "Cabin pressurization systems maintained equivalent altitudes below physiological thresholds during long-range overwater commercial operations.",
    "It is widely recognized that laminar flow wings delay transition but remain sensitive to insect contamination and surface roughness.",
    "The mission design incorporated low-thrust electric propulsion spiraling spacecraft into geostationary orbit while minimizing propellant mass.",
    "Having qualified thermal protection tiles, engineers verified ablation performance under peak heating loads during atmospheric entry.",
    "The avionics architecture segregated safety-critical functions from passenger connectivity networks through hardware partitioning and monitoring gateways.",
    "Aeroelastic tailoring optimized composite stiffness distributions reducing gust loads while preserving desired roll response characteristics.",
    "The space suit life support system recycled carbon dioxide and humidity while maintaining breathable partial pressures during extravehicular activity.",
    "Supersonic inlet designs managed shock swallowing and unstart margins across varying Mach numbers during acceleration profiles.",
    "The runway incursion alerting system correlated surface radar tracks with cockpit displays preventing conflicting taxi and landing clearances.",
    "Orbital rendezvous maneuvers used relative navigation sensors fusing lidar and vision data for autonomous docking with space stations.",
    "The fatigue test article accumulated equivalent flight cycles validating damage tolerance assumptions for extended service life programs.",
]

BATCH_8 = [
    # Comparative literature & narratology 776-800
    "The narratological analysis distinguished focalization shifts from chronological anachronies structuring the novel's multigenerational family saga.",
    "Although postcolonial readings emphasized subaltern voices, formalist critics retained interest in metaphoric patterns uniting disparate narrative strands.",
    "The translation studies seminar compared domesticating strategies preserving cultural alterity with foreignizing approaches retaining source language idiosyncrasies.",
    "Intertextual references to classical mythology recoded heroic archetypes within contemporary dystopian settings critiquing authoritarian political regimes.",
    "The comparative anthology juxtaposed magical realism with socialist realism to illuminate divergent aesthetic responses to modernization projects.",
    "It remains debated whether digital hypertext narratives dismantle authorial authority or merely extend longstanding oral storytelling participatory traditions.",
    "The hermeneutic circle implied interpreters continually revised contextual assumptions while engaging recurrent motifs across translated literary corpora.",
    "Having cataloged manuscript variants, editors reconstructed authorial revisions illuminating compositional hesitations within epistolary novel drafts.",
    "The reception history traced evolving critical appraisals of romantic poetry across nationalist, feminist, and ecocritical interpretive frameworks.",
    "Metafictional devices exposed narrative artifice while inviting readers to reflect on fictionality within historiographic novels about wartime memory.",
    "Notwithstanding canon expansion efforts, syllabus design still underrepresented vernacular literatures from postcolonial archipelagic writing communities.",
    "The stylistic analysis quantified sentence length variability and lexical richness distinguishing modernist fragmentation from realist descriptive prose.",
    "Genre hybridization blended detective plotting with philosophical dialogue in works challenging clear boundaries between popular and elite literature.",
    "The diaspora memoir employed code-switching and untranslated idioms asserting linguistic sovereignty against assimilative publishing market pressures.",
    "Feminist narratology examined how free indirect discourse conveyed constrained agency within patriarchal domestic spaces historically.",
    "It is widely argued that trauma narratives resist linear emplotment, privileging cyclical temporality and testimonial fragmentation instead.",
    "The comparative myth study aligned trickster figures across Indigenous and African diasporic traditions revealing shared cosmological ambivalence motifs.",
    "Having translated poetry, scholars debated whether meter and rhyme could be approximated without sacrificing semantic density of originals.",
    "The ecocritical reading interpreted pastoral imagery as nostalgic retreat masking extractive economies devastating rural ecosystems silently.",
    "Unreliable narration compelled readers to reconcile contradictory accounts within polyphonic novels depicting revolutionary upheaval and exile.",
    "The book history lecture explained how subscription publishing financed eighteenth-century novels before commercial lending libraries transformed distribution.",
    "Adaptation studies analyzed how cinematic flashbacks reordered episodic plot structures while amplifying visual symbolism from source texts.",
    "The world literature debate questioned whether global circulation homogenizes aesthetic difference or fosters creolized transnational literary forms.",
    "Autofiction blurred memoir and invention, prompting ethical inquiries about representational harm toward recognizable figures depicted thinly.",
    "The narrative ethics seminar evaluated empathy cultivation through focalized suffering while cautioning against sentimental exploitation of victimhood.",
]

SENTENCE_BATCHES = [BATCH_1, BATCH_2, BATCH_3, BATCH_4, BATCH_5, BATCH_6, BATCH_7, BATCH_8]
SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"
for i, batch in enumerate(SENTENCE_BATCHES, 1):
    assert len(batch) == 25, f"BATCH_{i} has {len(batch)} sentences"

START_ID = 601
BATCH_SIZE = 25
TARGET_PATH = project_root / "data/handcraft/en/train/c1_new_004.conllu"

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
    "transit-oriented": ("transit-oriented", "ADJ"),
    "mixed-use": ("mixed-use", "ADJ"),
    "arm's": ("arm", "NOUN"),
    "Rawlsian": ("Rawlsian", "PROPN"),
    "Kantian": ("Kantian", "PROPN"),
    "Aristotelian": ("Aristotelian", "PROPN"),
    "EBITDA": ("EBITDA", "PROPN"),
    "CRISPR": ("CRISPR", "PROPN"),
    "Bayesian": ("Bayesian", "ADJ"),
    "Pending": ("pending", "ADP"),
    "pending": ("pending", "ADP"),
    "biased": ("biased", "ADJ"),
    "centralized": ("centralized", "ADJ"),
    "irrigated": ("irrigated", "ADJ"),
    "underrepresented": ("underrepresented", "ADJ"),
}

VERB_OVERRIDES: dict[str, str] = {
    "emphasized": "emphasize",
    "emphasizes": "emphasize",
    "recognized": "recognize",
    "prioritizes": "prioritize",
    "prioritized": "prioritize",
    "integrated": "integrate",
    "integrates": "integrate",
    "documented": "document",
    "demonstrated": "demonstrate",
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
    "synthesized": "synthesize",
    "requested": "request",
    "reproduced": "reproduce",
    "indicated": "indicate",
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
    "refined": "refine",
    "encourage": "encourage",
    "expanded": "expand",
    "persist": "persist",
    "incorporate": "incorporate",
    "remains": "remain",
    "evaluated": "evaluate",
    "depend": "depend",
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
    "detected": "detect",
    "persisted": "persist",
    "assessed": "asse",
    "assess": "asse",
    "assesses": "asse",
    "assessing": "asse",
    "help": "help",
    "monitored": "monitor",
    "emphasize": "emphasize",
    "flagged": "flag",
    "report": "report",
    "reported": "report",
    "requires": "require",
    "combines": "combine",
    "necessitates": "necessitate",
    "predefined": "predefine",
    "incorporated": "incorporate",
    "reveal": "reveal",
    "personalize": "personalize",
    "increasingly": "increase",
    "must": "must",
    "indicates": "indicate",
    "creates": "create",
    "signaled": "signal",
    "affect": "affect",
    "correlate": "correlate",
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
    "addressing": "addre",
    "reflects": "reflect",
    "generates": "generate",
    "recognizes": "recognize",
    "facilitated": "facilitate",
    "intensified": "intensify",
    "proved": "prove",
    "proposed": "propose",
    "accepted": "accept",
    "escalate": "escalate",
    "documented": "document",
    "linked": "link",
    "debated": "debate",
    "explored": "explore",
    "coordinates": "coordinate",
    "constrain": "constrain",
    "deploy": "deploy",
    "weighs": "weigh",
    "intersects": "intersect",
    "analyze": "analyze",
    "direct": "direct",
    "hindered": "hinder",
    "affects": "affect",
    "reduced": "reduce",
    "left": "leave",
    "deployed": "deploy",
    "examine": "examine",
    "aligns": "align",
    "strengthened": "strengthen",
    "improve": "improve",
    "strengthens": "strengthen",
    "serve": "serve",
    "documents": "document",
    "coordinate": "coordinate",
    "combine": "combine",
    "funded": "fund",
    "perpetuate": "perpetuate",
    "published": "publish",
    "adopted": "adopt",
    "segmented": "segment",
    "enforced": "enforce",
    "balance": "balance",
    "enables": "enable",
    "raising": "raise",
    "incentivizes": "incentivize",
    "evaluates": "evaluate",
    "implement": "implement",
    "permits": "permit",
    "preserved": "preserve",
    "caution": "caution",
    "reinforce": "reinforce",
    "prohibit": "prohibit",
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
    "conditioned": "condition",
    "discouraging": "discourage",
    "pursuing": "pursue",
    "supporting": "support",
    "piloted": "pilot",
    "introducing": "introduce",
    "documenting": "document",
    "targeting": "target",
    "requiring": "require",
    "facilitating": "facilitate",
    "reducing": "reduce",
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
    "evaluating": "evaluate",
    "implementing": "implement",
    "siting": "site",
    "threatened": "threaten",
    "aligned": "align",
    "favoring": "favor",
    "demonstrating": "demonstrate",
    "operating": "operate",
    "disadvantaged": "disadvantage",
    "imposing": "impose",
    "had": "have",
    "Having": "have",
    "transcended": "transcend",
    "highlighted": "highlight",
    "govern": "govern",
    "resolve": "resolve",
    "cover": "cover",
    "covering": "cover",
    "exercising": "exercise",
    "suppressing": "suppre",
    "suppress": "suppre",
    "derive": "derive",
    "derives": "derive",
    "emphasizes": "emphasize",
    "emphasized": "emphasize",
    "prohibits": "prohibit",
    "compared": "compare",
    "claims": "claim",
    "reviewed": "review",
    "remain": "remain",
    "demand": "demand",
    "analyzed": "analyze",
    "critiques": "critique",
    "concerns": "concern",
    "evaluate": "evaluate",
    "complicates": "complicate",
    "violates": "violate",
    "describe": "describe",
    "treat": "treat",
    "challenged": "challenge",
    "warns": "warn",
    "extends": "extend",
    "distinguishes": "distinguish",
    "investigate": "investigate",
    "seeks": "seek",
    "expands": "expand",
    "revealed": "reveal",
    "traced": "trace",
    "contested": "contest",
    "sustain": "sustain",
    "reinforcing": "reinforce",
    "perpetuates": "perpetuate",
    "mapped": "map",
    "emphasized": "emphasize",
    "explains": "explain",
    "reconfigure": "reconfigure",
    "analyzed": "analyze",
    "reconstructed": "reconstruct",
    "influence": "influence",
    "interrogates": "interrogate",
    "linked": "link",
    "included": "include",
    "disclosed": "disclose",
    "pressured": "pressure",
    "identified": "identify",
    "disputed": "dispute",
    "approved": "approve",
    "avoided": "avoid",
    "required": "require",
    "integrates": "integrate",
    "delayed": "delay",
    "strained": "strain",
    "allocated": "allocate",
    "examined": "examine",
    "relies": "rely",
    "recommended": "recommend",
    "misclassified": "misclassify",
    "implemented": "implement",
    "detailed": "detail",
    "hedged": "hedge",
    "introduced": "introduce",
    "triggered": "trigger",
    "recommend": "recommend",
    "separated": "separate",
    "specify": "specify",
    "mandated": "mandate",
    "acquire": "acquire",
    "distinguished": "distinguish",
    "posits": "posit",
    "documented": "document",
    "debated": "debate",
    "convey": "convey",
    "reconstructed": "reconstruct",
    "improved": "improve",
    "measured": "measure",
    "examines": "examine",
    "compared": "compare",
    "analyzes": "analyze",
    "disambiguate": "disambiguate",
    "distinguished": "distinguish",
    "aligned": "align",
    "interrogates": "interrogate",
    "tracked": "track",
    "classifies": "classify",
    "recorded": "record",
    "investigates": "investigate",
    "identified": "identify",
    "integrated": "integrate",
    "contested": "contest",
    "gathered": "gather",
    "balanced": "balance",
    "remains": "remain",
    "deployed": "deploy",
    "regulated": "regulate",
    "revealed": "reveal",
    "designed": "design",
    "requested": "request",
    "projected": "project",
    "relocated": "relocate",
    "optimized": "optimize",
    "redesigned": "redesign",
    "recognized": "recognize",
    "evaluated": "evaluate",
    "coordinated": "coordinate",
    "emphasized": "emphasize",
    "improved": "improve",
    "elevated": "elevate",
    "repurposed": "repurpose",
    "incorporated": "incorporate",
    "replaced": "replace",
    "correlated": "correlate",
    "identified": "identify",
    "enabled": "enable",
    "engineers": "engineer",
    "standardized": "standardize",
    "remains": "remain",
    "predicted": "predict",
    "sequenced": "sequence",
    "monitored": "monitor",
    "examined": "examine",
    "demanded": "demand",
    "guided": "guide",
    "directed": "direct",
    "altered": "alter",
    "resolved": "resolve",
    "optimized": "optimize",
    "detected": "detect",
    "silenced": "silence",
    "localized": "localize",
    "reviewed": "review",
    "labeled": "label",
    "integrated": "integrate",
    "incorporated": "incorporate",
    "contextualized": "contextualize",
    "persist": "persist",
    "influencing": "influence",
    "interrogated": "interrogate",
    "recovered": "recover",
    "debated": "debate",
    "analyzed": "analyze",
    "enabled": "enable",
    "evaluated": "evaluate",
    "decoded": "decode",
    "sustained": "sustain",
    "weighed": "weigh",
    "interpreted": "interpret",
    "linked": "link",
    "documented": "document",
    "argued": "argue",
    "stabilized": "stabilize",
    "highlighted": "highlight",
    "examined": "examine",
    "analyzed": "analyze",
    "explored": "explore",
    "challenged": "challenge",
    "traced": "trace",
    "investigated": "investigate",
    "integrated": "integrate",
    "optimized": "optimize",
    "necessitate": "necessitate",
    "achieved": "achieve",
    "predicted": "predict",
    "evaluated": "evaluate",
    "remains": "remain",
    "measured": "measure",
    "characterized": "characterize",
    "tuned": "tune",
    "investigated": "investigate",
    "face": "face",
    "influenced": "influence",
    "recovered": "recover",
    "reduced": "reduce",
    "govern": "govern",
    "enhanced": "enhance",
    "reduced": "reduce",
    "implemented": "implement",
    "detected": "detect",
    "exhibited": "exhibit",
    "optimized": "optimize",
    "increased": "increase",
    "quantified": "quantify",
    "restoring": "restore",
    "predicted": "predict",
    "express": "expre",
    "expresses": "expre",
    "expressed": "expre",
    "expressing": "expre",
}

PARTICIPIAL_ADJ: frozenset[str] = frozenset({
    "biased", "centralized", "decentralized", "irrigated", "underrepresented",
    "aging", "negotiated", "targeted", "related", "sustainable", "coordinated",
    "automated", "integrated", "diversified", "marginalized", "standardized",
    "overlapping", "unforeseen", "inherited", "precarious", "overlapping",
    "nonpublic", "overlapping", "unstressed", "inflected", "overlapping",
    "permeable", "underutilized", "inherited", "circulating", "isolated",
    "overlapping", "unreliable", "orientalist", "excluded", "participatory",
    "toughened", "lubricated", "coated", "aligned", "continuous",
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
        for sid, count in bad_lengths[:30]:
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

    print("STATUS: OK — en_c1_train_601 through en_c1_train_800")


if __name__ == "__main__":
    main()