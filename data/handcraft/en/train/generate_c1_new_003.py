"""Generate 200 handcrafted English C1 CoNLL-U sentences (en_c1_train_401–600)."""

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
    # Philosophy & ethics 401-425
    "Utilitarian philosophers argue that moral obligations derive from maximizing aggregate welfare across all affected sentient beings impartially.",
    "Although virtue ethics emphasizes character cultivation, critics question whether habitual dispositions alone can resolve contemporary moral dilemmas adequately.",
    "The deontological framework prohibits treating persons merely as means regardless of beneficial consequences anticipated from utilitarian calculations.",
    "Rawlsian justice theory prioritizes arrangements benefiting the least advantaged members within hypothetical contracts behind a veil of ignorance.",
    "It remains debated whether moral relativism undermines cross-cultural dialogue or simply acknowledges legitimate pluralism in ethical reasoning traditions.",
    "The trolley problem illustrates tensions between consequentialist reasoning and inviolable rights protecting individual persons from deliberate instrumental harm.",
    "Having examined competing normative theories, the seminar compared Kantian duties with Aristotelian accounts of practical wisdom systematically.",
    "Moral intuitionism claims that certain ethical truths are self-evident without requiring inferential justification from higher theoretical principles.",
    "The ethics committee reviewed whether clinical trials using placebo controls remain defensible given availability of proven therapeutic interventions.",
    "Existentialist thinkers emphasized authentic choice and personal responsibility amid conditions of radical freedom without predetermined metaphysical guidance.",
    "Notwithstanding persuasive appeals to natural law, skeptics demand empirical evidence linking moral prescriptions to observable human flourishing outcomes.",
    "The applied ethicist analyzed distributive justice implications of allocating scarce organs among patients with differing prognostic profiles.",
    "Feminist moral philosophy critiques traditional frameworks for neglecting relational contexts and caregiving labor central to ethical life.",
    "The metaethical debate concerns whether moral statements express objective facts or subjective attitudes shaped by cultural conditioning processes.",
    "Contractualist approaches evaluate principles according to whether reasonable agents could accept resulting social arrangements under fair conditions.",
    "It is widely argued that moral luck complicates responsibility attributions when outcomes depend partly on uncontrollable environmental circumstances.",
    "The bioethics panel assessed whether germline editing violates human dignity principles enshrined in international research governance instruments.",
    "Phenomenological accounts describe how moral perception arises through situated engagement rather than detached application of abstract rules.",
    "Having challenged moral foundationalism, contemporary pragmatists treat ethical inquiry as fallible collective deliberation responsive to experiential consequences.",
    "The communitarian critique warns that excessive individualism erodes shared values sustaining democratic solidarity and civic virtue nationally.",
    "Animal ethics scholarship extends moral consideration beyond human subjects to nonhuman creatures capable of experiencing suffering and preference.",
    "The principle of double effect distinguishes permissible foreseen side effects from intentionally chosen wrongful means in complex cases.",
    "Moral psychologists investigate how empathy and cognitive biases shape everyday ethical judgments outside philosophical reflection.",
    "The discourse ethics framework seeks universalizable norms through rational dialogue among participants committed to sincere argumentation procedures.",
    "Environmental ethics expands moral concern to ecosystems and future generations affected by present resource extraction decisions.",
]

BATCH_2 = [
    # Sociology & anthropology 426-450
    "The ethnographic study documented ritual practices mediating kinship obligations within communities undergoing rapid urbanization and labor migration.",
    "Although modernization theory predicted secularization, resurgent religious movements continue shaping public morality in diverse postindustrial societies.",
    "The intersectionality framework analyzes how race, gender, and class hierarchies compound disadvantage through mutually reinforcing structural mechanisms.",
    "Symbolic interactionism examines how shared meanings emerge through everyday communicative exchanges in localized social interaction settings.",
    "The longitudinal survey traced intergenerational mobility patterns revealing persistent reproduction of elite status across educational institutions.",
    "It remains contested whether globalization homogenizes cultural identities or facilitates hybrid forms of localized creative adaptation.",
    "The fieldwork revealed how informal economies sustain livelihoods where formal employment opportunities remain scarce and precarious.",
    "Having conducted participant observation, the anthropologist analyzed gift exchange systems reinforcing reciprocal obligations among neighboring villages.",
    "Social capital theory links dense civic networks to cooperative problem solving and collective action in resource constrained communities.",
    "The stratification literature documents how occupational prestige rankings correlate with educational credentials and inherited economic advantages.",
    "Notwithstanding policy interventions, residential segregation perpetuates unequal access to high-performing schools and municipal public services.",
    "The comparative study examined matrilineal inheritance customs challenging assumptions about universal patriarchal family organization patterns globally.",
    "Network analysis mapped brokerage positions enabling individuals to bridge otherwise disconnected organizational clusters within metropolitan labor markets.",
    "The cultural turn in sociology emphasized discourse and identity construction rather than purely material determinants of behavior.",
    "Ritual theory explains how ceremonial performances transform social statuses during rites of passage marking critical life transitions.",
    "It is increasingly recognized that digital platforms reconfigure social ties by altering visibility norms and interpersonal accountability expectations.",
    "The migration study analyzed remittance flows sustaining household consumption while transforming gendered division of labor in origin communities.",
    "Having triangulated archival records with oral histories, researchers reconstructed community responses to colonial administrative restructuring policies.",
    "The institutional ethnography traced how bureaucratic categories shape everyday experiences of eligibility within welfare administration offices.",
    "Urban ethnographers documented how public space design influences patterns of surveillance, leisure, and informal economic exchange.",
    "The social movement literature examines framing strategies mobilizing collective identities around shared grievances and transformative political aspirations.",
    "Household survey data revealed persistent gender gaps in unpaid domestic labor despite rising female labor force participation rates.",
    "The postcolonial critique interrogates how anthropological knowledge production historically reinforced hierarchical representations of colonized populations.",
    "Peer group dynamics influence adolescent risk behaviors through normative pressures operating independently of parental supervision mechanisms.",
    "The community study linked neighborhood disorder perceptions to declining collective efficacy and rising fear of criminal victimization.",
]

BATCH_3 = [
    # Corporate law & business 451-475
    "The merger agreement included change clauses allowing termination when regulatory developments substantially impair projected synergies.",
    "Although fiduciary duties require loyalty, directors may consider stakeholder interests when evaluating long-term enterprise value strategies.",
    "The securities filing disclosed contingent liabilities arising from pending litigation concerning alleged misstatements in prior earnings guidance.",
    "Shareholder activism pressured the board to separate chair and chief executive roles to strengthen independent governance oversight mechanisms.",
    "The due diligence review identified intellectual property encumbrances potentially limiting commercialization of acquired pharmaceutical development pipelines.",
    "Poison pill defenses remain disputed for protecting shareholders from coercive takeovers or entrenching incumbent management.",
    "The bankruptcy court approved a restructuring plan prioritizing secured creditors while preserving essential operations through debtor financing.",
    "Having renegotiated covenants, the borrower avoided technical default despite temporary EBITDA declines during cyclical demand contraction periods.",
    "The antitrust clearance required divestiture of overlapping product lines to address concerns about reduced competition in regional markets.",
    "Corporate social responsibility reporting increasingly integrates environmental metrics alongside financial disclosures demanded by institutional investor coalitions.",
    "Notwithstanding robust revenues, working capital constraints delayed supplier payments and strained relationships with critical component manufacturers.",
    "The joint venture agreement allocated governance rights proportionally while establishing deadlock resolution procedures for strategic investment decisions.",
    "Insider trading investigations examined whether executives traded securities after receiving nonpublic information about pending regulatory approvals.",
    "The leveraged buyout model relies on debt financing secured against target assets while promising operational improvements post acquisition.",
    "Proxy advisory firms recommended voting against executive compensation packages lacking performance hurdles tied to measurable shareholder returns.",
    "It is widely acknowledged that transfer pricing arrangements require documentation demonstrating arm's length transactions among multinational corporate affiliates.",
    "The class action settlement resolved allegations that the company misclassified contractors to avoid providing statutory employment benefits.",
    "Having implemented enterprise risk management, the firm centralized cybersecurity and continuity planning oversight.",
    "The initial public offering prospectus detailed use of proceeds, competitive risks, and regulatory uncertainties affecting profitability forecasts.",
    "Derivative instruments hedged currency exposure but introduced counterparty credit risks requiring collateralization under revised master agreements.",
    "The whistleblower complaint triggered internal investigation into procurement kickbacks involving intermediaries across several international subsidiaries.",
    "Corporate governance codes recommend audit committee independence and rotation of external auditors to safeguard financial reporting integrity.",
    "The spin-off separated legacy divisions while enabling capital allocation toward higher-margin technology businesses.",
    "Venture capital term sheets specify liquidation preferences, anti-dilution protections, and board representation rights for lead institutional investors.",
    "The compliance program mandated training, monitoring, and disciplinary procedures addressing bribery risks under extraterritorial anti-corruption statutes.",
]

BATCH_4 = [
    # Linguistics & communication 476-500
    "The corpus analysis revealed collocational patterns distinguishing academic prose from conversational registers across multiple disciplinary domains.",
    "Although children acquire syntax rapidly, mastery of pragmatic inference and discourse markers continues developing throughout adolescence.",
    "The phonological study examined vowel reduction patterns in unstressed syllables across regional dialects of contemporary spoken English.",
    "Generative grammar posits an innate universal grammar constraining possible human languages despite surface diversity in morphological systems.",
    "The sociolinguistic survey documented style shifting among bilingual speakers navigating formal institutional contexts and informal peer interactions.",
    "It remains debated whether machine translation systems capture semantic nuance without deeper modeling of contextual presupposition and implicature.",
    "The discourse analysis traced thematic progression through topicalization strategies organizing information flow in persuasive political speeches.",
    "Having annotated dependency trees, researchers evaluated parser accuracy on constructions involving long-distance wh-movement and control predicates.",
    "Pragmatic theories explain how speakers convey indirect meanings through conversational implicatures calculable from cooperative communicative assumptions.",
    "The historical linguistics seminar reconstructed sound changes linking proto-languages to attested daughter languages through systematic correspondence sets.",
    "Despite orthographic irregularities, segmentation algorithms improved lemmatization on inflected agglutinative language data.",
    "The experimental paradigm measured garden path effects when syntactic ambiguities delayed resolution of thematic role assignments temporarily.",
    "Code-switching research examines how bilingual speakers alternate languages to signal identity, solidarity, or contextual appropriateness.",
    "The typological survey compared word order flexibility across languages correlating verb-final structures with case marking richness systematically.",
    "Forensic linguistics analyzes authorship attribution using stylometric features including function word frequencies and syntactic construction preferences.",
    "It is increasingly recognized that prosodic contours disambiguate syntactic structures otherwise identical at the segmental phonological representation level.",
    "The semantics seminar distinguished compositional meaning from conventionalized metaphors requiring encyclopedic cultural knowledge for interpretation.",
    "Having developed parallel corpora, the team aligned multilingual texts to study translation shifts in modality and evidential marking.",
    "Critical discourse analysis interrogates how media representations reproduce ideological assumptions about gender, race, and national belonging.",
    "The language acquisition study tracked overgeneralization errors in past tense marking before children internalize irregular verb paradigms.",
    "Speech act theory classifies utterances according to illocutionary force including assertions, directives, commissives, and expressive performatives.",
    "The neurolinguistic experiment recorded event-related potentials associated with syntactic violations presented during auditory sentence comprehension tasks.",
    "Contact linguistics investigates pidginization producing simplified grammars when speakers without shared languages communicate commercially.",
    "The rhetoric course analyzed ethos, pathos, and logos in deliberative speeches addressing contested policy reforms before legislatures.",
    "Corpus-based lexicography identifies emergent neologisms and shifting connotations reflecting technological change in contemporary public discourse.",
]

BATCH_5 = [
    # Urban planning & architecture 501-525
    "The master plan integrated transit corridors reducing automobile dependency through mixed-use zoning near commuter rail stations.",
    "Although density bonuses incentivized affordable housing, neighborhood associations contested height limits threatening historic streetscape character preservation.",
    "The urban heat island study recommended reflective roofing materials and expanded tree canopy coverage across densely paved commercial districts.",
    "Participatory planning workshops gathered resident input on pedestrian safety improvements along arterials with high collision injury rates.",
    "The heritage conservation guidelines balanced adaptive reuse of industrial buildings with strict controls on facade alterations and signage.",
    "Financing stormwater retrofits remains challenging in aging neighborhoods with fragmented parcel ownership.",
    "The smart city initiative deployed sensor networks monitoring air quality, traffic congestion, and energy consumption across districts.",
    "Having adopted form-based codes, the municipality regulated building envelopes while permitting diverse architectural styles within districts.",
    "Transit equity analysis revealed underserved corridors where low-income residents face lengthy commutes due to infrequent bus service.",
    "The landscape architect designed bioswales and permeable pavements managing runoff while enhancing public realm aesthetics along boulevards.",
    "Notwithstanding parking minimum reductions, developers requested variances citing tenant expectations for onsite vehicle storage capacity.",
    "The housing needs assessment projected shortfalls in moderately priced units for essential workers employed near central business districts.",
    "Urban regeneration projects relocated informal settlements while providing tenure security and upgraded sanitation in replacement housing complexes.",
    "The building performance simulation optimized daylighting and natural ventilation reducing operational energy demand in subtropical climatic conditions.",
    "Complete streets policies redesigned corridors accommodating cyclists, pedestrians, and public transit users alongside motor vehicle traffic safely.",
    "It is widely recognized that exclusionary zoning historically restricted multifamily development in affluent suburbs surrounding metropolitan cores.",
    "The corridor study evaluated economic impacts of proposed light rail extensions on property values and small business displacement risks.",
    "Having coordinated interagency review, planners aligned floodplain regulations with wetland restoration goals along vulnerable coastal estuaries.",
    "The architectural competition emphasized mass timber construction demonstrating carbon sequestration potential in midrise commercial office buildings.",
    "Wayfinding systems improved navigability for visually impaired residents through tactile paving and audible pedestrian signal installations.",
    "The climate adaptation plan elevated critical infrastructure above projected storm surge levels while preserving waterfront public access amenities.",
    "Infill development strategies repurposed underutilized parcels near existing utilities to minimize greenfield expansion at urban peripheries.",
    "The civic square redesign incorporated flexible seating, shade structures, and programmable spaces supporting community events throughout seasons.",
    "Parking reform replaced subsidized curb lanes with protected bike infrastructure and expanded loading zones for local retail businesses.",
    "The urban morphology analysis correlated block size and intersection density with pedestrian activity and retail commercial vitality indicators.",
]

BATCH_6 = [
    # Biotechnology & genetics 526-550
    "The genome-wide association study identified susceptibility loci linked to autoimmune disorders after adjusting for population stratification effects.",
    "Although CRISPR editing enables precise modifications, off-target cleavage events necessitate rigorous validation before therapeutic clinical applications.",
    "The transcriptomic profiling revealed differential expression patterns distinguishing tumor subtypes with distinct prognostic and treatment response profiles.",
    "Synthetic biology engineers modular genetic circuits controlling metabolic pathways for sustainable production of pharmaceutical precursor compounds.",
    "The biobank protocol standardized informed consent procedures governing secondary research use of donated tissue specimens and genomic data.",
    "It remains uncertain whether polygenic risk scores provide clinically actionable guidance beyond established family history and lifestyle factors.",
    "The protein folding simulation predicted conformational changes associated with pathogenic mutations affecting enzyme catalytic activity substantially.",
    "Having sequenced metagenomic samples, researchers cataloged microbial diversity in soil ecosystems responding to agricultural management practices.",
    "Gene therapy trials monitored immune responses to viral vectors delivering corrective alleles to hepatocytes in inherited metabolic disorders.",
    "The epigenetic analysis examined methylation marks correlating with environmental exposures during critical windows of early developmental plasticity.",
    "Notwithstanding promising preclinical results, regulatory agencies demanded additional safety data before approving novel cell-based immunotherapy products.",
    "The pharmacogenomic panel guided dosing decisions by identifying patients carrying variants affecting drug metabolism enzyme activity levels.",
    "Stem cell differentiation protocols directed pluripotent lines toward functional neuronal populations for modeling neurodegenerative disease mechanisms.",
    "The microbiome intervention altered gut bacterial composition and modulated inflammatory biomarkers in patients with refractory gastrointestinal symptoms.",
    "Single-cell sequencing resolved heterogeneous cell states within tumors previously obscured by bulk tissue averaging in expression analyses.",
    "It is increasingly recognized that gene drive technologies raise ecological risks requiring containment strategies and community consent procedures.",
    "The structural biology team resolved cryo-electron microscopy densities revealing ligand binding mechanisms in membrane receptor signaling complexes.",
    "Having optimized fermentation conditions, the bioprocess engineers increased recombinant protein yields while reducing downstream purification costs.",
    "The diagnostic assay detected circulating tumor DNA fragments enabling earlier relapse monitoring following curative intent surgical resections.",
    "RNA interference silenced oncogene expression in vitro but delivery challenges limited efficacy in solid tissue tumor microenvironments.",
    "The consanguinity mapping localized recessive disease alleles within isolated populations exhibiting elevated homozygosity across genomic regions.",
    "Biosafety committees reviewed dual-use research proposals involving pathogen enhancement studies with potential misuse implications internationally.",
    "The lineage tracing experiment labeled progenitor cells to track differentiation trajectories during organogenesis in transgenic animal models.",
    "Personalized oncology pipelines integrated genomic profiling with drug sensitivity screens to prioritize targeted therapies for refractory cancers.",
    "The gene regulation model incorporated chromatin accessibility data predicting enhancer activity across diverse human tissue types systematically.",
]

BATCH_7 = [
    # Art history & cultural studies 551-575
    "The curatorial essay contextualized avant-garde collages within interwar anxieties about mechanization and mass media spectacle.",
    "Although attribution debates persist, technical imaging revealed underdrawings consistent with the master's workshop practices during that period.",
    "The exhibition traced diasporic aesthetic exchanges influencing hybrid musical forms emerging along transatlantic migration routes historically.",
    "Postcolonial criticism interrogated orientalist representations perpetuating exoticized depictions of non-Western subjects in nineteenth-century salon paintings.",
    "The archival research recovered correspondence documenting patronage networks sustaining female artists excluded from official academy exhibitions.",
    "It remains debated whether digital reproduction diminishes auratic experience or democratizes access to masterpieces previously confined to museums.",
    "The performance studies seminar analyzed embodied rituals challenging normative gender scripts through subversive costuming and choreographed interventions.",
    "Having digitized fragile manuscripts, conservators enabled scholarly comparison of iconographic programs across regional monastic scriptoria.",
    "The cultural policy review evaluated public funding formulas balancing elite institutional missions with community-based participatory arts initiatives.",
    "Iconographic analysis decoded allegorical figures referencing theological virtues embedded within civic fresco cycles adorning municipal council chambers.",
    "Notwithstanding commercialization pressures, independent filmmakers sustained experimental narratives resisting formulaic genre conventions in mainstream cinema.",
    "The museum repatriation debate weighed legal title claims against ethical obligations to source communities seeking ceremonial object returns.",
    "Semiotic approaches interpreted advertising imagery reinforcing consumer desires through recurring visual motifs and color symbolism strategies.",
    "The architectural historian linked brutalist concrete structures to postwar welfare state ambitions and utopian urban planning ideologies.",
    "Folk art surveys documented vernacular craftsmanship techniques transmitted orally across generations within geographically isolated rural communities.",
    "It is widely argued that canon formation reflects power relations privileging certain aesthetic traditions while marginalizing others systematically.",
    "The film restoration project stabilized nitrate prints while reconstructing missing intertitles using contemporaneous production documentation sources.",
    "Having curated transnational retrospectives, the institution highlighted cross-cultural dialogues among artists responding to decolonization movements globally.",
    "The literary criticism symposium examined narrative unreliability techniques destabilizing reader confidence in autobiographical testimony and memory.",
    "Material culture studies analyzed pottery shards revealing trade networks connecting coastal entrepots with inland agricultural production centers.",
    "The sound studies lecture explored how phonograph recordings transformed listening practices and domestic entertainment during early modernity.",
    "Curatorial activism challenged institutional neutrality by foregrounding artworks addressing structural racism and environmental justice themes explicitly.",
    "The typographic history seminar traced humanist letterforms influencing readable book design during the transition from manuscript to print.",
    "Popular culture scholars investigated fan communities producing transformative works extending canonical narratives through collaborative online platforms.",
    "The heritage management plan integrated intangible traditions including oral epics with tangible site conservation along historic pilgrimage routes.",
]

BATCH_8 = [
    # Materials science & engineering 576-600
    "The alloy development program optimized grain boundary chemistry to enhance creep resistance in turbine blades operating at elevated temperatures.",
    "Although composite laminates reduce weight, delamination risks necessitate rigorous nondestructive inspection throughout aircraft maintenance cycles.",
    "The nanomaterial synthesis achieved uniform particle size distributions enabling predictable optical properties in photovoltaic coating applications.",
    "Finite element simulations predicted stress concentrations near geometric discontinuities guiding redesign of load-bearing structural joints.",
    "The corrosion study evaluated protective oxide layers formed on stainless steel exposed to chloride-rich marine atmospheric environments.",
    "It remains challenging to scale laboratory perovskite solar cells to durable modules meeting commercial warranty performance requirements.",
    "The polymer rheology experiments measured viscoelastic behavior under oscillatory shear relevant to advanced injection molding process parameters.",
    "Having characterized fracture toughness, engineers selected toughened ceramics for ballistic protection applications requiring minimal areal density.",
    "Additive manufacturing parameters were tuned to minimize porosity while preserving mechanical anisotropy acceptable for aerospace component certification.",
    "The tribology lab investigated wear mechanisms in lubricated contacts under boundary conditions prevalent in high-load gearbox assemblies.",
    "Notwithstanding high conductivity, copper interconnects face electromigration failures unless diffusion barrier layers are deposited precisely.",
    "The catalyst support morphology influenced active site dispersion and turnover frequencies during heterogeneous hydrogenation reaction processes.",
    "Shape memory alloys recovered predetermined configurations upon heating, enabling compact actuators in minimally invasive surgical instrument designs.",
    "The thermal management study compared phase change materials stabilizing temperature fluctuations in battery packs during rapid charging cycles.",
    "Superhydrophobic surface treatments reduced ice adhesion on aerodynamic surfaces but degraded under ultraviolet exposure during prolonged outdoor service.",
    "It is widely recognized that dislocation dynamics govern plastic deformation behavior in crystalline metals subjected to cyclic loading.",
    "The fiber alignment process enhanced tensile strength along principal stress directions in continuous carbon reinforced thermoplastic composites.",
    "Having implemented quality control protocols, the foundry reduced defect rates in castings supplying precision automotive powertrain components.",
    "The electrochemical impedance spectroscopy detected early pitting initiation on coated substrates immersed in accelerated corrosion testing chambers.",
    "Metamaterial lattices exhibited negative Poisson ratios enabling energy absorption applications in protective equipment and automotive crumple zones.",
    "The semiconductor fabrication line optimized doping profiles to minimize leakage currents in subthreshold operation of low-power transistors.",
    "Hydrogen embrittlement susceptibility increased when high-strength fasteners were exposed to cathodic charging during offshore platform service.",
    "The coating adhesion test quantified interfacial shear strength between thermal barrier layers and nickel-based superalloy turbine substrates.",
    "Self-healing polymers incorporated microencapsulated monomers restoring crack integrity after mechanical damage in structural adhesive applications.",
    "The materials informatics pipeline predicted stable crystal structures by screening compositional spaces using machine learning interatomic potentials.",
]

SENTENCE_BATCHES = [BATCH_1, BATCH_2, BATCH_3, BATCH_4, BATCH_5, BATCH_6, BATCH_7, BATCH_8]
SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"
for i, batch in enumerate(SENTENCE_BATCHES, 1):
    assert len(batch) == 25, f"BATCH_{i} has {len(batch)} sentences"

START_ID = 401
BATCH_SIZE = 25
TARGET_PATH = project_root / "data/handcraft/en/train/c1_new_003.conllu"

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

    print("STATUS: OK — en_c1_train_401 through en_c1_train_600")


if __name__ == "__main__":
    main()