"""Generate 100 handcrafted English C1 CoNLL-U sentences (en_c1_train_801–900)."""

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
    # Maritime law & ocean governance 801-825
    "The admiralty court adjudicated salvage claims arising from vessel collisions within congested international shipping lanes during winter storms.",
    "Although coastal states exercise territorial sovereignty, freedom of navigation principles constrain excessive regulatory interference with commercial transit.",
    "The exclusive economic zone treaty delineated seabed mining rights while preserving migratory fish stocks for sustainable regional fisheries management.",
    "Flag state enforcement obligations require periodic inspections ensuring compliance with safety and environmental standards aboard merchant vessels.",
    "It remains contested whether autonomous vessels qualify as ships under conventions governing collision liability and masters' duties at sea.",
    "The port state control regime detained carriers after discovering deficiencies in life-saving equipment and ballast water systems.",
    "Having charted disputed waters, negotiators proposed joint development zones balancing resource extraction with navigational security for littoral states.",
    "The marine pollution tribunal assessed liability for bunker fuel spills contaminating mangrove ecosystems and coastal aquaculture livelihoods severely.",
    "Submarine cable operators sought injunctive relief protecting infrastructure from anchoring damage near submarine canyon approaches along continental shelves.",
    "Notwithstanding customary law precedents, continental shelf delimitation disputes increasingly invoke geomorphological evidence from bathymetric survey datasets.",
    "The fisheries adjudication panel imposed quota penalties after verifying illegal transshipment activities beyond authorized high seas pocket allocations.",
    "Maritime lien priorities determine whether mortgage holders or bunker suppliers recover unpaid claims during judicial vessel arrest proceedings.",
    "The archipelagic baseline agreement recognized straight baseline segments enclosing historically integrated insular waters for internal navigation regulation.",
    "It is widely argued that climate-driven sea level rise will necessitate revising baselines governing territorial sea entitlements.",
    "The carriage contract incorporated Hague-Visby Rules limiting carrier liability for negligent stowage absent declared value clauses.",
    "Passage rights through international straits balance coastal regulatory authority against uninterrupted transit for nuclear-powered warships.",
    "The marine protected area designation restricted bottom trawling while permitting indigenous subsistence fishing under co-management agreements locally.",
    "Having ratified the tonnage convention, signatories harmonized measurement standards affecting harbor dues and canal transit fee assessments.",
    "The casualty investigation attributed grounding to ECDIS alarm misconfiguration compounded by fatigue among bridge watchkeeping officers overnight.",
    "Under the Nairobi convention, wreck removal obligations allocate costs to owners failing to cooperate with coastal authorities promptly.",
    "The marine insurance policy excluded war risks while covering general average contributions arising from voluntary cargo jettison during emergencies.",
    "Piracy prosecutions tested universal jurisdiction doctrines when apprehended suspects lacked nationality links to the prosecuting forum state.",
    "The offshore wind lease auction allocated seabed parcels subject to decommissioning bonds and fisheries mitigation commitments contractually.",
    "Deep seabed mining regulations require environmental impact assessments prior to exploitation licenses within international waters administered multilaterally.",
    "The maritime arbitration clause mandated London seat proceedings under institutional rules excluding appeals to domestic courts regarding awards.",
]

BATCH_2 = [
    # Paleontology & evolutionary biology 826-850
    "The fossil assemblage preserved transitional morphologies bridging aquatic sarcopterygians and early tetrapod limb prototypes in Devonian strata.",
    "Although molecular clock estimates diverged, integrative phylogenetics reconciled divergence dates using combined morphological and genomic datasets systematically.",
    "Taphonomic biases skew fossil records toward hard-shelled organisms, complicating inferences about Cambrian explosion diversity dynamics statistically.",
    "The paleobotanical survey documented gymnosperm radiation coinciding with aridification events reshaping continental interior floras during Permian intervals.",
    "It remains debated whether punctuated equilibria explain stasis in fossil lineages better than gradualist models emphasizing phyletic transformation.",
    "Biostratigraphic correlations aligned ammonite zones across basins, enabling precise chronostratigraphic calibration of sedimentary sequences regionally.",
    "Having extracted ancient collagen, researchers sequenced proteins supporting closer affinities among archosaur lineages than morphology alone suggested.",
    "The mass extinction horizon exhibited iridium anomalies coinciding with ejecta layers attributed to bolide impact scenarios globally.",
    "Convergent evolution produced analogous fin shapes in unrelated marine reptiles occupying comparable pelagic niches during Mesozoic oceans.",
    "Notwithstanding preservation limitations, dental microwear texture analysis reconstructed dietary shifts among hominin populations across Pleistocene landscapes.",
    "The cladistic parsimony analysis resolved polytomies by incorporating incomplete fossil taxa with known synapomorphies from cranial characters.",
    "Island biogeography models predicted dwarfism and gigantism patterns among insular proboscideans isolated after Pleistocene sea level regressions.",
    "Paleoenvironmental proxies inferred monsoonal intensification from oxygen isotope ratios preserved in foraminiferal tests throughout core records.",
    "It is increasingly recognized that horizontal gene transfer complicates reconstructing microbial phylogenies from single-gene marker approaches alone.",
    "The stratotype section defined boundary criteria separating stages within the international chronostratigraphic chart through voted GSSP proposals.",
    "Adaptive radiation following end-Cretaceous extinctions enabled placental mammals to diversify into vacant ecological roles across continents rapidly.",
    "Having modeled trait evolution, paleobiologists tested whether body size trends reflect Cope's rule across theropod dinosaur lineages.",
    "Sedimentary geochemistry traced anoxic event pulses linked to volcanic outgassing and marine productivity feedbacks during Oceanic Anoxic Events.",
    "The ontogenetic series distinguished juvenile ornamentation from adult sexual dimorphism in crested hadrosaurid specimens across museum collections.",
    "Phylogenetic niche conservatism constrained recolonization pathways as climate refugia shifted latitudinally during Quaternary glacial interglacial cycles repeatedly.",
    "Fossilized melanosomes inferred plumage coloration patterns in feathered dinosaurs using scanning electron microscopy and comparative extant bird spectra.",
    "The endosymbiotic hypothesis explains organelle origins through serial acquisitions of photosynthetic and respiratory bacterial partners in eukaryogenesis.",
    "Coprolite analyses revealed herbivorous diets incorporating seeds and foliage inferred from embedded plant cuticles within fossilized fecal matrices.",
    "Molecular phylogeography traced vicariance events separating sister clades isolated by rising seaways across formerly connected land bridges.",
    "The paleoecological network model simulated trophic cascades following megafaunal extinctions reshaping vegetation structure and fire regimes continentally.",
]

BATCH_3 = [
    # Quantum computing & information theory 851-875
    "The quantum algorithm exploited superposition and entanglement to factor integers faster than known classical subexponential methods theoretically.",
    "Although error correction thresholds were met, maintaining coherence beyond millisecond scales remains challenging for superconducting transmon qubit arrays.",
    "Quantum key distribution protocols detect eavesdropping through basis mismatch statistics violating Bell inequality bounds under realistic channel losses.",
    "The density matrix formalism generalized state descriptions accommodating mixed ensembles and decoherence induced by environmental coupling effects.",
    "It remains uncertain whether noisy intermediate-scale devices achieve quantum advantage without fault-tolerant logical qubits.",
    "Topological qubits promise enhanced stability through non-Abelian anyon braiding operations resistant to local perturbations in two-dimensional systems.",
    "Having calibrated pulse sequences carefully, experimentalists minimized cross-talk between neighboring gates on programmable trapped-ion quantum processors.",
    "The quantum channel capacity theorem bounds reliable communication rates accounting for entanglement assistance and memoryless noise models.",
    "Variational quantum eigensolvers approximated molecular ground states using parameterized circuits optimized through classical feedback loops iteratively.",
    "Notwithstanding hardware limitations, quantum simulation experiments modeled Hubbard Hamiltonians revealing antiferromagnetic ordering transitions on small lattices.",
    "The no-cloning theorem prohibits duplication of arbitrary quantum states, underpinning security proofs for quantum cryptography schemes.",
    "Quantum supremacy demonstrations compared sampling distributions from random circuits against classical simulations using tensor network contraction heuristics.",
    "Entanglement entropy quantified subsystem correlations scaling area laws in gapped ground states versus volume laws in critical systems.",
    "It is widely argued that quantum machine learning kernels may accelerate classification when data exhibit latent manifold structure.",
    "The surface code layout arranged stabilizer measurements detecting bit-flip syndromes for real-time error correction cycles.",
    "Adiabatic quantum computing interpolated Hamiltonians gradually to maintain ground state tracking while solving combinatorial optimization instances.",
    "Having implemented tomography protocols, researchers reconstructed density operators verifying entanglement witnesses for bipartite photonic polarization states.",
    "Quantum annealers encoded quadratic unconstrained binary optimization problems into Ising couplings mapped onto hardware connectivity graphs natively.",
    "The measurement problem debates whether wavefunction collapse is fundamental or emergent from decoherence in macroscopic apparatus interactions.",
    "Boson sampling experiments validated output distributions against permanents computed for multimode interference patterns at single-photon inputs.",
    "Logical qubit encoding concatenated physical qubits achieving suppressed error rates below fault-tolerant computation thresholds under syndrome extraction.",
    "Quantum error mitigation extrapolated noiseless expectation values using zero-noise extrapolation and probabilistic error cancellation techniques combined.",
    "The holographic principle conjectured boundary descriptions encoding bulk gravitational physics, inspiring tensor network representations of quantum states.",
    "Quantum random number generators harvested entropy certified by Bell test violations ensuring unpredictability for cryptographic seeding applications.",
    "Post-quantum cryptography standardized lattice-based schemes resisting attacks from Shor algorithms on large quantum computers.",
]

BATCH_4 = [
    # Archaeology & museology 876-900
    "The excavation trench exposed stratified occupational layers spanning Neolithic settlement phases through late Bronze Age metallurgical workshops sequentially.",
    "Although conservation treatments stabilized pigments, curators debated whether invasive cleaning compromised authenticity of mural fragments permanently.",
    "LiDAR surveys revealed ceremonial platforms concealed beneath tropical canopy, redirecting survey priorities toward upland interfluvial archaeological landscapes.",
    "The provenance research dossier documented ownership chains resolving gaps preceding museum acquisitions from wartime displacement and sales.",
    "It remains contested whether megalithic monuments functioned as burial sites, territorial markers, or astronomical observatories regionally.",
    "Zooarchaeological assemblages indicated shifting subsistence strategies incorporating domesticated livestock alongside hunted ungulates during agropastoral transitions.",
    "Having digitized catalogs, museums enabled comparative study of related ceramic vessels through interoperable digital repositories.",
    "The restoration ethics charter prioritized minimal intervention while ensuring stability for marble sculptures displayed in climate-controlled galleries.",
    "Rescue archaeology salvage recorded features threatened by highway construction using rapid stratigraphic sampling and photogrammetric documentation protocols.",
    "Notwithstanding colonial collecting histories, repatriation negotiations emphasized community stewardship and ceremonial reintegration of ancestral remains respectfully.",
    "Ceramic petrography distinguished locally tempered wares from imported amphorae signaling long-distance exchange networks across Mediterranean ports.",
    "The exhibition narrative contextualized ritual objects within indigenous cosmologies rather than presenting them as detached aesthetic curiosities exclusively.",
    "It is widely argued that community archaeology partnerships improve interpretive validity by incorporating descendant knowledge alongside stratigraphic evidence.",
    "The typological seriation ordered grave goods reflecting stylistic changes correlating with regional chronologies derived from radiocarbon calibration curves.",
    "Having analyzed starch residues, archaeobotanists inferred cultivation of tuber crops predating ceramic adoption in highland village settlements.",
    "Museum storage upgrades installed seismic bracing and humidity buffering systems protecting fragile textiles from fluctuating environmental degradation risks.",
    "The landscape archaeology project mapped field systems associated with villa estates revealing intensified agricultural production during imperial expansion phases.",
    "Underwater excavations documented shipwreck assemblages preserving trade ceramics that illuminate provisioning networks supplying frontier garrison settlements.",
    "Experimental archaeology replicated flint knapping techniques validating wear patterns observed on Palaeolithic toolkit assemblages from open-air workshops.",
    "The deaccessioning policy permitted transferring duplicate specimens to educational institutions while retaining holotype references for scholarly research.",
    "Isotopic analysis of skeletal remains inferred residential mobility patterns among pastoral communities traversing seasonal grazing corridors across steppes.",
    "The public archaeology outreach program engaged school groups in supervised sieving exercises at buffered excavation units near settlement peripheries.",
    "Conservation scientists applied multispectral imaging revealing erased inscriptions on papyrus fragments previously illegible under visible illumination alone.",
    "The UNESCO nomination dossier justified outstanding universal value through comparative analyses of fortified citadel architecture spanning multiple cultural phases.",
    "Post-excavation analysis integrated spatial statistics identifying activity zones within household compounds distinguished by artifact density distributions.",
]

SENTENCE_BATCHES = [BATCH_1, BATCH_2, BATCH_3, BATCH_4]
SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 100, f"Expected 100 sentences, got {len(SENTENCES)}"
for i, batch in enumerate(SENTENCE_BATCHES, 1):
    assert len(batch) == 25, f"BATCH_{i} has {len(batch)} sentences"

START_ID = 801
BATCH_SIZE = 25
TARGET_PATH = project_root / "data/handcraft/en/train/c1_new_005.conllu"

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
    "climate-driven": ("climate-driven", "ADJ"),
    "nuclear-powered": ("nuclear-powered", "ADJ"),
    "co-management": ("co-management", "NOUN"),
    "life-saving": ("life-saving", "ADJ"),
    "hard-shelled": ("hard-shelled", "ADJ"),
    "low-dimensional": ("low-dimensional", "ADJ"),
    "real-time": ("real-time", "ADJ"),
    "zero-noise": ("zero-noise", "ADJ"),
    "lattice-based": ("lattice-based", "ADJ"),
    "IIIF-compliant": ("IIIF-compliant", "ADJ"),
    "climate-controlled": ("climate-controlled", "ADJ"),
    "long-distance": ("long-distance", "ADJ"),
    "open-air": ("open-air", "ADJ"),
    "Hague-Visby": ("Hague-Visby", "PROPN"),
    "ECDIS": ("ECDIS", "PROPN"),
    "Nairobi": ("Nairobi", "PROPN"),
    "London": ("London", "PROPN"),
    "LiDAR": ("LiDAR", "PROPN"),
    "IIIF": ("IIIF", "PROPN"),
    "UNESCO": ("UNESCO", "PROPN"),
    "GSSP": ("GSSP", "PROPN"),
    "Cope's": ("Cope", "PROPN"),
    "Shor": ("Shor", "PROPN"),
    "Palaeolithic": ("Palaeolithic", "ADJ"),
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
    "adjudicated": "adjudicate",
    "arising": "arise",
    "exercise": "exercise",
    "delineated": "delineate",
    "preserving": "preserve",
    "ensuring": "ensure",
    "qualify": "qualify",
    "detained": "detain",
    "discovering": "discover",
    "charted": "chart",
    "proposed": "propose",
    "balancing": "balance",
    "contaminating": "contaminate",
    "protecting": "protect",
    "invoke": "invoke",
    "imposed": "impose",
    "verifying": "verify",
    "determine": "determine",
    "recover": "recover",
    "recognized": "recognize",
    "enclosing": "enclose",
    "revising": "revise",
    "limiting": "limit",
    "restricted": "restrict",
    "permitting": "permit",
    "ratified": "ratify",
    "harmonized": "harmonize",
    "attributed": "attribute",
    "allocate": "allocate",
    "excluded": "exclude",
    "covering": "cover",
    "allocated": "allocate",
    "mandated": "mandate",
    "excluding": "exclude",
    "preserved": "preserve",
    "bridging": "bridge",
    "diverged": "diverge",
    "reconciled": "reconcile",
    "skew": "skew",
    "complicating": "complicate",
    "reshaping": "reshape",
    "aligned": "align",
    "enabling": "enable",
    "extracted": "extract",
    "sequenced": "sequence",
    "exhibited": "exhibit",
    "produced": "produce",
    "occupying": "occupy",
    "reconstructed": "reconstruct",
    "resolved": "resolve",
    "incorporating": "incorporate",
    "predicted": "predict",
    "isolated": "isolate",
    "inferred": "infer",
    "defined": "define",
    "separating": "separate",
    "diversify": "diversify",
    "reflect": "reflect",
    "constrained": "constrain",
    "shifted": "shift",
    "explains": "explain",
    "simulated": "simulate",
    "exploited": "exploit",
    "factor": "factor",
    "maintaining": "maintain",
    "detect": "detect",
    "violating": "violate",
    "generalized": "generalize",
    "accommodating": "accommodate",
    "promise": "promise",
    "calibrated": "calibrate",
    "minimized": "minimize",
    "bounds": "bound",
    "accounting": "account",
    "approximated": "approximate",
    "modeled": "model",
    "revealing": "reveal",
    "prohibits": "prohibit",
    "quantified": "quantify",
    "scaling": "scale",
    "accelerate": "accelerate",
    "arranged": "arrange",
    "detecting": "detect",
    "interpolated": "interpolate",
    "implemented": "implement",
    "encoded": "encode",
    "debates": "debate",
    "validated": "validate",
    "computed": "compute",
    "concatenated": "concatenate",
    "achieving": "achieve",
    "extrapolated": "extrapolate",
    "conjectured": "conjecture",
    "encoding": "encode",
    "inspiring": "inspire",
    "harvested": "harvest",
    "certified": "certify",
    "standardized": "standardize",
    "resisting": "resist",
    "executed": "execute",
    "exposed": "expose",
    "spanning": "span",
    "stabilized": "stabilize",
    "compromised": "compromise",
    "concealed": "conceal",
    "redirecting": "redirect",
    "resolving": "resolve",
    "functioned": "function",
    "indicated": "indicate",
    "digitized": "digitize",
    "prioritized": "prioritize",
    "recorded": "record",
    "emphasized": "emphasize",
    "distinguished": "distinguish",
    "signaling": "signal",
    "contextualized": "contextualize",
    "presenting": "present",
    "ordered": "order",
    "reflecting": "reflect",
    "correlating": "correlate",
    "derived": "derive",
    "predating": "predate",
    "installed": "install",
    "mapped": "map",
    "illuminate": "illuminate",
    "replicated": "replicate",
    "validating": "validate",
    "observed": "observe",
    "permitted": "permit",
    "transferring": "transfer",
    "retaining": "retain",
    "traversing": "traverse",
    "engaged": "engage",
    "applied": "apply",
    "justified": "justify",
    "integrated": "integrate",
    "identifying": "identify",
    "used": "use",
    "lacked": "lack",
    "failing": "fail",
    "administered": "administer",
    "administering": "administer",
    "requiring": "require",
    "subject": "subject",
    "arrest": "arrest",
    "arising": "arise",
    "governing": "govern",
    "necessitate": "necessitate",
    "emphasizing": "emphasize",
    "emphasize": "emphasize",
    "explain": "explain",
    "complicates": "complicate",
    "reconstructing": "reconstruct",
    "separating": "separate",
    "linked": "link",
    "traced": "trace",
    "separated": "separate",
    "compared": "compare",
    "compared": "compare",
    "compared": "compare",
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
    "congested", "integrated", "transitional", "concealed", "tempered",
    "detached", "fortified", "buffered", "illegible", "parameterized",
    "mixed", "authorized", "substandard", "disputed", "stratified",
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
        print(f"Processing batch {batch_num}/4 (en_c1_train_{start:03d}–{end:03d})...")

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

    print("STATUS: OK — en_c1_train_801 through en_c1_train_900")


if __name__ == "__main__":
    main()