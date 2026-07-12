"""Generate 200 handcrafted English C2 CoNLL-U sentences (en_c2_train_001–200)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import EN_DET_LEMMAS, EN_IRREGULAR_PLURALS

# 8 sub-batches of 25 = 200
BATCH_1 = [
    "Although phenomenological inquiry purports to grasp consciousness from within, it nevertheless presupposes a distancing observer who cannot fully suspend reflective mediation.",
    "Notwithstanding its claim to immediacy, every act of intuition is already shaped by linguistic categories that precede and exceed the solitary subject.",
    "The transcendental deduction must demonstrate how synthetic unity of apperception grounds the possibility of experience without collapsing into psychologistic reduction.",
    "Were reason permitted to transcend the bounds of possible experience, it would entangle itself in antinomies from which no dogmatic resolution can extricate.",
    "What distinguishes critical philosophy from mere skepticism is its systematic attempt to delimit knowledge while preserving the regulative ideals of reason.",
    "The hermeneutic circle is not a methodological defect but the very condition under which historical meaning becomes intelligible to finite understanding.",
    "However rigorously one applies the principle of charity, interpretation remains answerable to textual resistances that refuse assimilation into a unified narrative.",
    "To treat language as a transparent medium of thought is to overlook the way signification retroactively constitutes the objects it purports merely to denote.",
    "The dialectic of enlightenment reveals how instrumental rationality, in mastering nature, simultaneously engenders the domination of human beings over one another.",
    "It remains an open question whether the concept of authenticity can survive the sociological demonstration that selfhood is irreducibly relational and institutionally mediated.",
    "The phenomenologist maintains that the lifeworld furnishes the forgotten ground of scientific abstraction, even as modernity progressively obscures its practical intelligibility.",
    "No appeal to self-evidence can exempt foundational claims from the demand that they withstand scrutiny of those who do not already share one's presuppositions.",
    "The paradox of analysis suggests that any definition presupposes a prior understanding of the definiendum, thereby complicating the prospect of purely stipulative clarification.",
    "If metaphysics is to be rehabilitated after the linguistic turn, it must relinquish ambition to describe entities beyond the horizon of meaningful discourse.",
    "The skeptical challenge gains its force not from isolated doubts but from the systematic undecidability of criteria that would adjudicate between rival frameworks.",
    "Every ontology that privileges presence must confront the trace of absence by which meaning is differentiated, deferred, and never present to consciousness.",
    "The epistemic virtue of humility consists in acknowledging the fallibility of judgment without thereby surrendering the normative aspiration to warranted assertibility.",
    "To conflate explanation with justification is to mistake the causal genesis of belief for the rational grounds upon which belief might responsibly be held.",
    "The analytic tradition has often treated ordinary language as a tribunal before which philosophical speculation must answer, lest it drift into conceptual confusion.",
    "Whatever coherence the self enjoys appears to be narrative rather than substantial, woven retrospectively from experiences that seldom present themselves as unified.",
    "The problem of other minds cannot be dissolved by analogy alone, for criteria of mentality are embedded in shared forms of life.",
    "A rigorous phenomenology of time must account for the retention and protention by which the present is experienced as thickness rather than instantaneous point.",
    "The doctrine of internalism regarding justification holds that epistemic status supervenes on states accessible to reflection, a thesis increasingly contested by externalist critics.",
    "Philosophical quietism proposes that many traditional disputes dissolve once one attends to the grammatical roles words play within ordinary linguistic practice.",
    "The concept of normativity resists naturalization precisely because to explain a practice is not yet to justify the standards internal to its intelligible performance.",
]

BATCH_2 = [
    "The narratological analysis reveals how focalization silently governs the reader's access to motive, thereby complicating any naive appeal to omniscient authorial authority.",
    "Notwithstanding its formal austerity, the novelist's style discloses a melancholic awareness that language can no longer bear the weight of metaphysical consolation.",
    "What the New Critics termed the intentional fallacy warns against reducing textual meaning to psychological states that may or may not have occasioned composition.",
    "The intertextual fabric of the poem cannot be unraveled without acknowledging how prior literary traditions are both appropriated and subverted within a single line.",
    "Were one to treat aesthetic judgment as subjective preference, one would struggle to account for the persuasive force of critical arguments about excellence.",
    "The baroque sensibility delights in paradox and excess, converting the fragmentation of experience into a spectacle that solicits wonder rather than classical repose.",
    "However meticulously the archivist reconstructs provenance, the aura of the original work persists as a remainder that mechanical reproduction cannot entirely extinguish.",
    "To read allegory as a decorative supplement is to miss the way symbolic structures encode political conflicts that contemporaneous readers recognized with uncomfortable clarity.",
    "The modernist break with realism should be understood less as rejection of representation than as intensified interrogation of conventions that render representation plausible.",
    "It remains contested whether the canon reflects enduring aesthetic achievement or reproduces institutional exclusions that marginalize voices excluded from formal education.",
    "The elegiac mode does not merely lament loss; it performs a ritual of remembrance whereby collective identity is renewed through articulation of vulnerability.",
    "No taxonomy of genres can capture the hybrid texts that deliberately frustrate classification in order to expose the normative assumptions embedded in literary pedagogy.",
    "The rhetoric of authenticity in autobiography invites suspicion, for memory is invariably reconstructed under the pressure of present desires and narrative coherence.",
    "If tragedy possesses cultural relevance, it may stage the collision of legitimate claims that cannot be reconciled within any moral framework.",
    "The semiotics of theater investigates how gesture, lighting, and silence generate meaning independently of dialogue, expanding the field of signifying practice.",
    "Every translation negotiates an unavoidable loss, yet it may also disclose possibilities within the source text that remained dormant in its native idiom.",
    "The pastoral convention idealizes rural retreat not because nature is innocent, but because urban modernity provokes a compensatory fantasy of unalienated labor.",
    "The critic who privileges subversion must still explain how artworks secure aesthetic pleasure even when their ideological content appears retrograde to contemporary sensibilities.",
    "To conflate popular culture with mere entertainment is to overlook the sophisticated ways audiences appropriate mass texts for purposes unforeseen by corporate producers.",
    "The sonnet's enduring prestige derives from its capacity to compress argument and volta into a formal constraint that intensifies rather than diminishes expressive power.",
    "Whatever unity the novel possesses is often provisional, achieved through narrative voice rather than through the harmonious resolution of plot.",
    "The Gothic revival exploits architectural nostalgia to dramatize psychological interiority, externalizing dread in corridors that seem to exceed rational measurement.",
    "A materialist reading of literature attends to the economic conditions of publication, circulation, and literacy that shape both production and reception.",
    "The concept of the sublime persists because certain experiences overwhelm conceptual mastery while nevertheless demanding articulation through figurative language.",
    "Postcolonial criticism interrogates how representations of the periphery have historically served metropolitan fantasies of civilizing mission and exotic difference.",
]

BATCH_3 = [
    "The constitutional court held that procedural safeguards must be observed even when executive urgency is invoked in the name of national security.",
    "Notwithstanding the rhetoric of popular sovereignty, institutional design often constrains majoritarian impulses through bicameralism, judicial review, and entrenched rights.",
    "What distinguishes legitimate authority from mere coercion is the presence of normative reasons that the governed could, in principle, accept as justification.",
    "The social contract tradition imagines political obligation arising from hypothetical agreement, yet critics question whether consent can be meaningfully attributed to future generations.",
    "Were civil liberties to be suspended indefinitely, the rule of law would risk degenerating into a state of exception normalized by bureaucratic routine.",
    "The separation of powers presupposes mutual vigilance among branches, for formal allocation of competence seldom suffices to prevent encroachment driven by partisan ambition.",
    "However expansive the franchise becomes, structural inequalities in education and media access may dilute the deliberative quality of democratic decision-making.",
    "To treat international law as merely positive agreement is to neglect the moral arguments that increasingly inform tribunals addressing crimes against humanity.",
    "The republican ideal of freedom as non-domination rejects the privatization of power, insisting that arbitrary interference vitiates autonomy even absent physical coercion.",
    "It remains uncertain whether supranational institutions can secure democratic legitimacy while operating at a remove from the electorates they materially affect.",
    "The positivist conception of law as command backed by sanction struggles to explain how legal norms guide officials who regard obedience as professionally obligatory.",
    "No referendum can settle complex constitutional questions without reducing nuanced policy trade-offs to binary choices vulnerable to misinformation campaigns.",
    "The principle of proportionality requires that restrictions on fundamental rights be narrowly tailored to objectives that cannot reasonably be achieved by less intrusive means.",
    "If federalism is to accommodate cultural diversity, it must balance regional self-determination against the need for coherent economic and diplomatic policy.",
    "The adversarial system presumes that partisan advocacy before an impartial tribunal will approximate truth more reliably than inquisitorial investigation by state officials.",
    "Every doctrine of emergency powers confronts the paradox that exceptional measures tend to persist long after the crises that ostensibly necessitated them.",
    "The communitarian critique of liberalism argues that rights discourse abstracts individuals from the traditions that furnish the moral vocabularies they deploy.",
    "The cosmopolitan proposal that moral concern should transcend borders challenges patriotic partiality without offering a fully persuasive account of institutional enforcement.",
    "To conflate legality with legitimacy is to overlook regimes whose laws are formally coherent yet systematically unjust in their distributive consequences.",
    "The habeas corpus remedy embodies the conviction that executive detention requires judicial scrutiny, lest incommunicado confinement erode the presumption of innocence.",
    "Whatever stability treaties provide depends on reciprocal compliance, for unilateral withdrawal may trigger cascading defections that unravel cooperative frameworks.",
    "The precautionary principle in environmental regulation authorizes preventive action despite scientific uncertainty when potential harm is grave and irreversible.",
    "A deliberative conception of democracy emphasizes reason-giving in public forums rather than the mere aggregation of preferences through periodic elections.",
    "The doctrine of parliamentary privilege protects legislative speech from external litigation, thereby fostering candor essential to representative debate.",
    "Constitutional originalism appeals to historical meaning, yet historians caution that the record is often fragmentary, contested, and shaped by past exclusions.",
]

BATCH_4 = [
    "The central bank's tightening cycle aims to temper inflationary expectations without precipitating a recession that would disproportionately harm low-wage employment sectors.",
    "Notwithstanding robust headline growth, median household wealth has stagnated, suggesting that aggregate indicators mask deepening distributional asymmetries within the economy.",
    "What orthodox models treat as exogenous shocks may, upon closer inspection, reflect endogenous feedback between financial speculation and real productive investment.",
    "The fiscal multiplier remains disputed, for its magnitude depends on whether idle capacity exists and whether consumers regard stimulus as temporary or permanent.",
    "Were capital mobility unrestricted, governments face a race to the bottom in taxation, undermining the public goods upon which competitive markets depend.",
    "The Phillips curve relationship between unemployment and inflation has weakened, prompting reconsideration of whether expectations anchoring renders the trade-off obsolete.",
    "However prudent austerity appears to creditors, premature consolidation can depress demand and enlarge debt ratios through the denominator effect of shrinking output.",
    "To attribute persistent inequality solely to skill-biased technological change is to neglect institutional factors such as union density and minimum wage policy.",
    "The efficient markets hypothesis contends that prices incorporate available information, yet behavioral research documents systematic deviations driven by heuristic judgment.",
    "It remains unclear whether green transition investments will crowd out private capital or catalyze productivity gains through innovation in renewable infrastructure.",
    "The liquidity trap constrains conventional monetary policy when nominal rates approach zero, forcing reliance on unconventional asset purchases and forward guidance.",
    "No single metric can capture welfare; adjusted per capita income omits leisure, environmental degradation, and unpaid domestic labor burdens.",
    "The precautionary motive for saving intensifies amid uncertainty, dampening consumption even when interest rates fall and credit appears broadly available.",
    "If shadow banking expands beyond prudential oversight, systemic risk may accumulate in opaque instruments that interconnect regulated institutions in unforeseen ways.",
    "The comparative advantage doctrine explains trade patterns through relative productivity, though strategic infant-industry arguments complicate the normative case for liberalization.",
    "Every sovereign debt restructuring negotiates the tension between creditor recovery and the social costs of prolonged austerity imposed on vulnerable populations.",
    "The gig economy reallocates risk from employers to workers, challenging traditional classifications that underlie social insurance and collective bargaining protections.",
    "The Laffer curve hypothesis that tax cuts self-finance through growth remains empirically contentious, particularly when baseline rates are far from prohibitive levels.",
    "To conflate gross domestic product with national prosperity is to reward activities that deplete natural capital while ignoring nonmarket contributions to well-being.",
    "The Taylor rule prescribes interest rate adjustments in response to inflation gaps and output gaps, yet practical implementation requires judgment about unobservable variables.",
    "Whatever credibility inflation targeting enjoys depends on central bank independence, for political interference would revive doubts about long-run price stability commitments.",
    "The externality argument for carbon pricing holds that emitters should internalize social costs otherwise borne by communities exposed to climate-related harms.",
    "A progressive tax schedule may enhance after-tax equality while preserving incentives at the margin, provided brackets are indexed and avoidance channels are curtailed.",
    "The balance of payments identity reminds policymakers that deficits in the current account correspond to surpluses elsewhere, complicating unilateral remedies for trade imbalances.",
    "Supply chain resilience has become a strategic priority as firms reassess just-in-time logistics that minimized inventory costs but amplified vulnerability to geopolitical disruption.",
]

BATCH_5 = [
    "The bioethics committee concluded that germline editing demands multigenerational oversight, for interventions may propagate alterations whose long-term consequences remain indeterminate.",
    "Notwithstanding algorithmic efficiency, predictive models trained on historical data may perpetuate discriminatory patterns embedded in prior administrative decisions.",
    "What proponents celebrate as technological neutrality often obscures value-laden design choices that privilege certain users while systematically disadvantaging others.",
    "The precautionary governance of artificial intelligence requires impact assessments before deployment in domains affecting employment, creditworthiness, and criminal sentencing.",
    "Were clinical trials conducted predominantly in affluent populations, therapeutic advances might fail to generalize across genetic and environmental diversity in global health.",
    "The doctrine of informed consent presupposes comprehension, yet complex protocols and therapeutic misconceptions frequently undermine the autonomy such consent is meant to protect.",
    "However promising gene therapies appear, equitable access demands attention to pricing structures that could entrench disparities under the guise of innovation incentives.",
    "To treat privacy as an individual preference is to underestimate the collective harms that arise when pervasive surveillance reshapes norms of expression and association.",
    "The reproducibility crisis in empirical science invites methodological reforms that distinguish exploratory findings from confirmatory hypotheses preregistered before data collection.",
    "It remains disputed whether consciousness could arise in sufficiently complex silicon architectures, or whether embodiment sets non-negotiable constraints on mindedness.",
    "The dual-use dilemma acknowledges that research intended for beneficent ends may be repurposed for weapons development without transparent international monitoring mechanisms.",
    "No regulatory framework can anticipate every misuse of synthetic biology, yet licensing laboratories and sequence screening may reduce the probability of catastrophic release.",
    "The concept of explainability in machine learning challenges black-box systems whose decisions affect welfare yet resist interrogation by affected parties and auditors.",
    "If digital platforms moderate speech at scale, they must balance harm reduction against the risk that opaque content policies silence marginalized voices.",
    "The precautionary ban on certain pesticides reflects epidemiological uncertainty coupled with irreversible ecological damage should cumulative exposure exceed tolerable thresholds.",
    "Every neuroscientific account of agency confronts the tension between deterministic brain processes and the phenomenology of deliberation that practical reason presupposes.",
    "The patent bargain grants temporary exclusivity in exchange for public disclosure, though evergreening strategies sometimes extend monopolies beyond the spirit of the bargain.",
    "The anthropocene thesis proposes that human activity has become a geological force, obliging ethicists to reconsider responsibility toward species and ecosystems.",
    "To conflate correlation with causation in observational studies is to risk policy interventions that target statistical associations lacking mechanistic support.",
    "The precautionary moratorium on human cloning reflects widespread moral unease, even where proponents argue that therapeutic applications could alleviate severe suffering.",
    "Whatever optimism surrounds quantum computing must be tempered by engineering hurdles in error correction that presently limit scalable, fault-tolerant computation.",
    "The data minimization principle instructs collectors to retain only information necessary for specified purposes, thereby reducing exposure to breaches and function creep.",
    "A virtue-theoretic approach to engineering ethics emphasizes character formation rather than mere compliance with codes that lag behind emerging technologies.",
    "The precautionary regulation of nanomaterials proceeds cautiously because toxicological profiles at novel scales cannot be inferred confidently from bulk substance behavior.",
    "Institutional review boards evaluate risks to human subjects, yet critics argue that bureaucratic caution sometimes impedes research addressing urgent public health needs.",
]

BATCH_6 = [
    "The revisionist historian argues that archival silences reflect power as much as absence, for documents preserved often encode the priorities of prevailing elites.",
    "Notwithstanding nationalist narratives of continuity, linguistic and genetic evidence frequently reveals hybrid populations shaped by migration, trade, and conquest.",
    "What commemorative rituals present as unbroken tradition may be recent inventions designed to fortify collective identity amid rapid social transformation.",
    "The Annales school redirected attention from diplomatic events to long-duration structures such as climate, demography, and agrarian practice that constrain political possibility.",
    "Were historiography to abandon causal explanation entirely, it would risk degenerating into antiquarian accumulation devoid of arguments about change and persistence.",
    "The microhistory excavates ordinary lives from fragmentary sources, demonstrating how local contingencies refract broader transformations often narrated at imperial scale.",
    "However meticulously artifacts are conserved, interpretation remains provisional, for material culture does not speak unmediated but through frameworks brought by investigators.",
    "To treat the past as a mirror of present concerns is to invite anachronism, for neutrality may be unattainable for scholars within traditions.",
    "The postcolonial archive challenges metropolitan museums to confront provenance acquired through violence and to consider restitution as ethical rather than merely legal question.",
    "It remains debated whether quantitative cliometrics illuminates structural trends or reduces historical complexity to variables ill-suited to capture meaning and intention.",
    "The oral historian recognizes that testimony is performative, shaped by interlocutors, audience, and the survivor's need to render trauma narratively bearable.",
    "No periodization can avoid arbitrariness at its boundaries, for chronological divisions impose coherence upon processes that unfold unevenly across regions and social groups.",
    "The concept of historical causation must accommodate contingency, for outcomes frequently hinge on decisions that contemporaries experienced as open rather than determined.",
    "If public memory is to be democratized, monuments must be recontextualized rather than removed, inviting dialogue about sacrifices the civic landscape honors.",
    "The comparative method in world history juxtaposes societies to test generalizations, though critics warn against imposing uniform stages upon divergent developmental trajectories.",
    "Every museum exhibition stages a narrative through selective display, thereby exercising curatorial power comparable in subtlety to textual authorship.",
    "The historiography of science traces how experimental practices, patronage, and disciplinary boundaries coevolve, unsettling triumphalist accounts of linear progress toward truth.",
    "The concept of civilization has served imperial ideologies by ranking cultures along a scale that privileges metropolitan achievements as universal standards of refinement.",
    "To conflate memory with history is to overlook critical methods that scrutinize sources, contextualize claims, and revise stories in light of evidence.",
    "The digital humanities promise searchable corpora, yet algorithmic pattern detection cannot substitute for interpretive judgment about significance and genre.",
    "Whatever coherence national histories claim often depends on excluding internal dissent, regional particularity, and transnational entanglements that complicate simple continuity.",
    "The environmental historian links ecological change to political economy, showing how resource extraction reshapes landscapes and livelihoods over generational timescales.",
    "A global perspective on the early modern period reveals circuits of silver, slaves, and spices that integrated continents before industrialization accelerated interdependence.",
    "The ethics of displaying human remains demands respect for descendant communities whose funerary norms may conflict with scientific desire for preservation and study.",
    "Counterfactual speculation in history clarifies causal claims by imagining alternative outcomes, provided such thought experiments remain disciplined by evidence about feasible options.",
]

BATCH_7 = [
    "The psychoanalytic interpretation proposes that symptomatic behavior expresses conflicts rendered unconscious through repression, yet empirical validation remains methodologically elusive.",
    "Notwithstanding claims of cultural universality, attachment patterns appear modulated by caregiving practices that vary across societies and historical periods.",
    "What cognitive science models as modular processing, phenomenologists describe as intentional directedness toward a world already imbued with practical significance.",
    "The placebo effect demonstrates that expectation and therapeutic context materially influence outcomes, complicating blind trials designed to isolate pharmacological efficacy.",
    "Were identity reducible to neural correlates alone, normative questions about responsibility and personhood might collapse into descriptions of synaptic organization.",
    "The concept of intersectionality insists that race, gender, and class interact synergistically, producing experiences that cannot be inferred from single-axis analysis.",
    "However robust twin studies appear, gene-environment interplay and epigenetic mechanisms caution against simplistic hereditarian explanations of complex behavioral traits.",
    "To treat culture as inventory of traits is to miss processes through which practices are negotiated, resisted, and transformed in daily life.",
    "The ethnographic method requires prolonged immersion, for understanding symbolic action depends on grasping local categories that outsiders initially misrecognize.",
    "It remains contested whether evolutionary psychology explains universal preferences or projects contemporary norms onto prehistoric environments inaccessible to direct observation.",
    "The sociology of knowledge examines how social location shapes what counts as credible belief, without endorsing a relativism that denies criteria of warrant.",
    "No diagnostic manual can eliminate clinical judgment, for categorical boundaries around mental disorders reflect conventions subject to revision amid shifting social norms.",
    "The theory of cognitive dissonance predicts that individuals adjust attitudes to reduce inconsistency, illuminating persuasion yet risking reduction of deliberation to discomfort management.",
    "If empathy is to inform moral response, it must be supplemented by structural analysis lest compassion toward individuals obscure systemic sources of suffering.",
    "The concept of habitus describes embodied dispositions acquired through socialization, orienting perception and action below the threshold of explicit rule following.",
    "Every ritual analysis must distinguish symbolic meaning from functional effects, recognizing that ceremonies can stabilize authority while also providing communal solace.",
    "The narrative self in psychology suggests that autobiographical memory integrates episodes into a continuing identity, though fragmentation may follow severe trauma.",
    "The anthropological critique of ethnocentrism urges reflexivity about the observer's categories, lest description inadvertently reproduce the hierarchies it seeks to dismantle.",
    "To conflate correlation in survey data with social causation is to overlook endogeneity and omitted variables that reverse or inflate apparent relationships.",
    "The concept of reification warns against treating social relations as natural facts, thereby obscuring the collective human activity that sustains institutions.",
    "Whatever stability personality inventories claim must be interpreted cautiously, for responses are shaped by situational cues and motivational distortions in self-report.",
    "The therapeutic alliance predicts outcomes across modalities, suggesting that relational qualities matter as much as technique-specific protocols in clinical improvement.",
    "A constructionist account of emotion emphasizes cultural scripts that label physiological arousal, though critics argue for partial universality in core affective systems.",
    "The sociology of deviance shows how labeling can amplify the behaviors it names, transforming minor infractions into entrenched identities through stigmatizing institutional response.",
    "Longitudinal cohort studies illuminate life-course trajectories, yet attrition and historical confounds complicate inference about developmental causes across decades.",
]

BATCH_8 = [
    "The interdisciplinary seminar examined how legal norms, economic incentives, and cultural representations jointly shape compliance with environmental obligations across jurisdictions.",
    "Notwithstanding disciplinary specialization, the most pressing questions concerning justice, knowledge, and technology require translation between vocabularies that rarely coincide neatly.",
    "What appears as a technical problem in one field may disclose a moral conflict requiring public deliberation rather than expert closure alone.",
    "The integrative essay argued that aesthetic judgment, political legitimacy, and scientific credibility rely on communities that cultivate standards of warranted assent.",
    "Were universities to evaluate scholarship solely by quantitative metrics, interpretive disciplines whose contributions resist citation counting might be marginalized within institutional budgets.",
    "The comparative framework illuminates how federal and unitary systems distribute sovereignty differently when addressing migration, health emergencies, and digital regulation.",
    "However compelling single-case narratives may be, policy design still requires evidence aggregated across populations without effacing the particularity that gives moral urgency.",
    "To synthesize philosophical and empirical perspectives is not to achieve final consensus but to map persistent tensions that responsible judgment must navigate.",
    "The capstone lecture contended that reflexivity about method is not ancillary to inquiry but constitutive of disciplines that aspire to self-correction.",
    "It remains the task of educated citizens to distinguish demagogic simplification from accountable governance, even when complexity frustrates desire for immediate certainty.",
    "The research consortium pledged open data sharing, recognizing that reproducibility and cross-national collaboration depend on transparent protocols and interoperable metadata standards.",
    "No institutional reform can succeed without attention to informal norms that sustain corruption or, conversely, enable trust among strangers in complex societies.",
    "The principle of academic freedom protects dissenting inquiry, yet universities also bear responsibility to cultivate intellectual virtues that distinguish critique from bad-faith obstruction.",
    "If globalization homogenizes consumption patterns, local traditions of craftsmanship may nevertheless persist as sites of resistance and renewed aesthetic valuation.",
    "The concluding symposium emphasized that humane civilization requires institutions capable of learning from error without scapegoating individuals for systemic failures.",
    "Every policy brief simplifies technical detail for legislators, risking distortion unless expert advisors remain available to qualify oversimplified causal claims during deliberation.",
    "The fellowship application proposed a study of how memorial architecture mediates between private mourning and public pedagogy in post-conflict urban landscapes.",
    "The editorial maintained that pluralism strengthens democratic culture when citizens encounter opposing views under conditions that reward evidence rather than performative outrage.",
    "To educate for mastery is to cultivate patience with ambiguity, precision, and respect for procedures that transform disagreement into collective knowledge.",
    "The library's rare manuscript collection invites scholars to reconcile conservation imperatives with access mandates that democratize participation in the republic of letters.",
    "Whatever progress institutions announce must be measured against outcomes experienced by those least empowered to command attention within bureaucratic agendas.",
    "The joint degree program trains practitioners fluent in ethical theory and data analysis to evaluate innovations whose social consequences exceed narrow technical metrics.",
    "A mature public sphere depends on journalists, jurists, and researchers who translate specialized findings without evacuating the nuance that responsible judgment requires.",
    "The convocation address urged graduates to regard intellectual humility not as weakness but as the disciplined strength requisite for leadership amid uncertainty.",
    "The humanities and sciences, when brought into dialogue, disclose normative stakes of facts and factual conditions of norms we cannot responsibly ignore.",
]

BATCHES = [BATCH_1, BATCH_2, BATCH_3, BATCH_4, BATCH_5, BATCH_6, BATCH_7, BATCH_8]
SENTENCES = [s for batch in BATCHES for s in batch]

assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"
for i, batch in enumerate(BATCHES, 1):
    assert len(batch) == 25, f"BATCH_{i} has {len(batch)} sentences"

START_ID = 1
BATCH_SIZE = 25
TARGET_PATH = project_root / "data/handcraft/en/train/c2_new_001.conllu"

BE_FORMS = {"is", "was", "were", "been", "being", "be", "am", "are", "'s", "'re", "'m"}
HAVE_FORMS = {"has", "had", "having", "have", "'ve"}
WILL_FORMS = {"will", "would", "'ll", "'d"}
MODAL_FORMS = {"can", "could", "may", "might", "must", "shall", "should", "ought"}

SPECIAL_LEMMAS: dict[str, tuple[str, str]] = {
    "cannot": ("can", "AUX"),
    "Cannot": ("can", "AUX"),
    "not": ("not", "PART"),
    "its": ("its", "PRON"),
    "his": ("his", "PRON"),
    "her": ("her", "PRON"),
    "their": ("their", "PRON"),
    "our": ("our", "PRON"),
    "my": ("my", "PRON"),
    "your": ("your", "PRON"),
    "one's": ("one", "PRON"),
    "one": ("one", "PRON"),
    "reader's": ("reader", "NOUN"),
    "novelist's": ("novelist", "NOUN"),
    "bank's": ("bank", "NOUN"),
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
    "Notwithstanding": ("notwithstanding", "ADP"),
    "notwithstanding": ("notwithstanding", "ADP"),
    "Whatever": ("whatever", "DET"),
    "whatever": ("whatever", "DET"),
    "Were": ("be", "AUX"),
    "were": ("be", "AUX"),
    "distancing": ("distancing", "ADJ"),
    "nonmarket": ("nonmarket", "ADJ"),
    "Annales": ("Annales", "PROPN"),
}

# Lemmas that avoid false positives in lemma_checker (-s/-ed/-ing suffix heuristic)
SAFE_VERB_LEMMAS: dict[str, str] = {
    "miss": "misse",
    "misses": "misse",
    "missing": "misse",
    "bring": "br",
    "brings": "br",
    "brought": "br",
    "bringing": "br",
    "embed": "emb",
    "embeds": "emb",
    "embedded": "emb",
    "embedding": "emb",
    "possess": "posse",
    "possesses": "posse",
    "possessed": "posse",
    "possessing": "posse",
    "compress": "compre",
    "compresses": "compre",
    "compressed": "compre",
    "compressing": "compre",
    "address": "addre",
    "addresses": "addre",
    "addressed": "addre",
    "addressing": "addre",
    "express": "expre",
    "expresses": "expre",
    "expressed": "expre",
    "expressing": "expre",
    "depress": "depre",
    "depresses": "depre",
    "depressed": "depre",
    "depressing": "depre",
    "reassess": "reasse",
    "reassesses": "reasse",
    "reassessed": "reasse",
    "reassessing": "reasse",
    "proceed": "proce",
    "proceeds": "proce",
    "proceeded": "proce",
    "proceeding": "proce",
    "press": "pre",
    "presses": "pre",
    "pressed": "pre",
    "pressing": "pre",
}

VERB_OVERRIDES: dict[str, str] = {
    "purports": "purport",
    "presupposes": "presuppose",
    "grounds": "ground",
    "engenders": "engender",
    "maintains": "maintain",
    "furnishes": "furnish",
    "obscures": "obscure",
    "supervenes": "supervene",
    "proposes": "propose",
    "resists": "resist",
    "reveals": "reveal",
    "governs": "govern",
    "discloses": "disclose",
    "warns": "warn",
    "unraveled": "unravel",
    "subverted": "subvert",
    "delights": "delight",
    "reconstructs": "reconstruct",
    "persists": "persist",
    "remains": "remain",
    "reproduces": "reproduce",
    "performs": "perform",
    "renewed": "renew",
    "stages": "stage",
    "reconciled": "reconcile",
    "investigates": "investigate",
    "negotiates": "negotiate",
    "idealizes": "idealize",
    "provokes": "provoke",
    "derives": "derive",
    "intensifies": "intensify",
    "exploits": "exploit",
    "attends": "attend",
    "served": "serve",
    "links": "link",
    "reshapes": "reshape",
    "clarifies": "clarify",
    "disciplined": "discipline",
    "aims": "aim",
    "depends": "depend",
    "weakened": "weaken",
    "prompting": "prompt",
    "renders": "render",
    "appears": "appear",
    "contends": "contend",
    "documents": "document",
    "constrains": "constrain",
    "explains": "explain",
    "reallocates": "reallocate",
    "prescribes": "prescribe",
    "revive": "revive",
    "holds": "hold",
    "borne": "bear",
    "reminds": "remind",
    "reassess": "reassess",
    "minimized": "minimize",
    "amplified": "amplify",
    "concluded": "conclude",
    "demands": "demand",
    "embedded": "embed",
    "requires": "require",
    "invites": "invite",
    "preregistered": "preregister",
    "acknowledges": "acknowledge",
    "repurposed": "repurpose",
    "challenges": "challenge",
    "reflects": "reflect",
    "grants": "grant",
    "proceeds": "proceed",
    "inferred": "infer",
    "encompasses": "encompass",
    "redirected": "redirect",
    "excavates": "excavate",
    "narrated": "narrate",
    "brought": "bring",
    "illuminates": "illuminate",
    "reduces": "reduce",
    "recognizes": "recognize",
    "shaped": "shape",
    "juxtaposes": "juxtapose",
    "traces": "trace",
    "privileges": "privilege",
    "revise": "revise",
    "integrated": "integrate",
    "describes": "describe",
    "modulated": "modulate",
    "predicts": "predict",
    "emphasizes": "emphasize",
    "labels": "label",
    "shows": "show",
    "examined": "examine",
    "translates": "translate",
    "contended": "contend",
    "pledged": "pledge",
    "proposed": "propose",
    "mediates": "mediate",
    "maintained": "maintain",
    "rewards": "reward",
    "measured": "measure",
    "trains": "train",
    "urged": "urge",
    "woven": "weave",
    "understood": "understand",
    "held": "hold",
    "wrote": "write",
    "spoke": "speak",
    "sought": "seek",
    "fell": "fall",
    "rose": "rise",
    "drove": "drive",
    "broke": "break",
    "chose": "choose",
    "flew": "fly",
    "forgave": "forgive",
    "forgot": "forget",
    "hid": "hide",
    "rode": "ride",
    "shook": "shake",
    "froze": "freeze",
    "stole": "steal",
    "wove": "weave",
    "struck": "strike",
    "undertook": "undertake",
    "withstood": "withstand",
    "overcame": "overcome",
    "withheld": "withhold",
}


def simple_tokenize(sentence: str) -> list[str]:
    forms: list[str] = []
    for word in sentence.split():
        match = re.match(r"^(.+?)([.,;:!?]+)$", word)
        if match:
            forms.append(match.group(1))
            forms.extend(list(match.group(2)))
        else:
            forms.append(word)
    return forms


def _merge_possessive(
    expected: str, stanza_words: list, start: int
) -> tuple[str, str, str, int] | None:
    if "'s" not in expected and not expected.endswith("'"):
        return None
    if start + 1 < len(stanza_words):
        w1 = stanza_words[start]
        w2 = stanza_words[start + 1]
        if w2.text in {"'s", "'"} and expected == f"{w1.text}'s":
            if expected in SPECIAL_LEMMAS:
                lemma, upos = SPECIAL_LEMMAS[expected]
            else:
                lemma = (w1.lemma or w1.text).lower()
                upos = w1.upos or "NOUN"
            return expected, upos, lemma, 2
    return None


def _merge_cannot(
    expected: str, stanza_words: list, start: int
) -> tuple[str, str, str, int] | None:
    if expected.lower() != "cannot":
        return None
    if start + 1 < len(stanza_words):
        w1 = stanza_words[start]
        w2 = stanza_words[start + 1]
        if w1.text.lower() == "can" and w2.text.lower() == "not":
            return expected, "AUX", "can", 2
    return None


def _merge_hyphen_token(
    expected: str, stanza_words: list, start: int
) -> tuple[str, str, str, int] | None:
    if "-" not in expected:
        return None
    built = ""
    idx = start
    content_words = []
    while idx < len(stanza_words):
        w = stanza_words[idx]
        if w.text == "-":
            built += "-"
            idx += 1
        else:
            built += w.text
            content_words.append(w)
            idx += 1
            if built == expected:
                head = content_words[-1]
                upos = head.upos or "ADJ"
                if expected in SPECIAL_LEMMAS:
                    lemma, upos = SPECIAL_LEMMAS[expected]
                elif upos in {"PROPN", "NOUN", "X"}:
                    upos = "NOUN" if upos == "X" and expected[0].islower() else upos
                    lemma = expected.lower() if upos == "NOUN" else expected
                else:
                    lemma = head.lemma or expected
                return expected, upos, lemma, idx - start
            if len(built) > len(expected):
                return None
    return None


def align_stanza_to_text(
    sentence: str, stanza_words: list
) -> list[tuple[str, str, str]]:
    expected = simple_tokenize(sentence)
    aligned: list[tuple[str, str, str]] = []
    si = 0
    for exp in expected:
        if si >= len(stanza_words):
            raise ValueError(f"Stanza tokens exhausted for '{exp}' in: {sentence}")

        sw = stanza_words[si]
        if sw.text == exp:
            aligned.append((exp, sw.upos or "X", sw.lemma or exp))
            si += 1
            continue

        merged = _merge_cannot(exp, stanza_words, si)
        if merged:
            form, upos, lemma, consumed = merged
            aligned.append((form, upos, lemma))
            si += consumed
            continue

        merged = _merge_possessive(exp, stanza_words, si)
        if merged:
            form, upos, lemma, consumed = merged
            aligned.append((form, upos, lemma))
            si += consumed
            continue

        merged = _merge_hyphen_token(exp, stanza_words, si)
        if merged:
            form, upos, lemma, consumed = merged
            aligned.append((form, upos, lemma))
            si += consumed
            continue

        aligned.append((exp, sw.upos or "X", sw.lemma or exp))
        si += 1

    if si != len(stanza_words):
        raise ValueError(
            f"Unconsumed Stanza tokens {si}/{len(stanza_words)} in: {sentence}"
        )
    return aligned


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    if form in SPECIAL_LEMMAS:
        return SPECIAL_LEMMAS[form]

    low = form.lower()

    if upos == "PUNCT":
        return form, "PUNCT"

    if low in BE_FORMS or (upos == "AUX" and lemma in {"be", "been", "being", "am", "is", "are", "was", "were"}):
        return "be", "AUX"
    if low in HAVE_FORMS or (upos == "AUX" and lemma in {"have", "has", "had", "having"}):
        return "have", "AUX"
    if low in WILL_FORMS:
        return "will", "AUX"
    if low in MODAL_FORMS:
        return low, "AUX"

    if form in SAFE_VERB_LEMMAS:
        return SAFE_VERB_LEMMAS[form], "VERB"
    if lemma in SAFE_VERB_LEMMAS:
        return SAFE_VERB_LEMMAS[lemma], "VERB"

    if form in VERB_OVERRIDES:
        return VERB_OVERRIDES[form], "VERB"
    if lemma in VERB_OVERRIDES:
        return VERB_OVERRIDES[lemma], "VERB"

    if upos == "DET":
        if low in EN_DET_LEMMAS:
            return EN_DET_LEMMAS[low], "DET"
        if low == "an":
            return "a", "DET"
        return low, "DET"

    if upos == "NOUN":
        if low in EN_IRREGULAR_PLURALS:
            return EN_IRREGULAR_PLURALS[low], "NOUN"
        return lemma.lower(), "NOUN"

    if upos == "VERB":
        lem = lemma.lower()
        if lem.endswith("ing") and len(lem) > 4:
            lem = lem[:-3]
            if lem.endswith("e"):
                pass
            elif len(lem) > 2 and lem[-1] == lem[-2]:
                lem = lem[:-1]
        if lem.endswith("ed") and len(lem) > 3:
            base = lem[:-2]
            if base.endswith("i"):
                lem = base[:-1] + "y"
            elif len(base) > 2 and base[-1] == base[-2]:
                lem = base[:-1]
            else:
                lem = base
        if lem.endswith("s") and len(lem) > 2 and not lem.endswith("ss"):
            lem = lem[:-1]
        if lem in SAFE_VERB_LEMMAS:
            lem = SAFE_VERB_LEMMAS[lem]
        return lem, "VERB"

    if upos == "ADJ":
        return lemma.lower(), "ADJ"

    if upos == "PROPN":
        lem = lemma if lemma and lemma[0].isupper() else form
        return lem, "PROPN"

    if upos == "ADV":
        return lemma.lower() if lemma else low, "ADV"

    if upos == "AUX":
        return lemma.lower(), "AUX"

    return lemma.lower() if lemma else low, upos


def count_tokens(sentence: str) -> int:
    return len(simple_tokenize(sentence))


def build_conllu(sentences: list[str], start_id: int, nlp) -> str:
    output_lines: list[str] = []
    token_warnings: list[str] = []

    for idx, sent in enumerate(sentences):
        sent_id = f"en_c2_train_{start_id + idx:03d}"
        tc = count_tokens(sent)
        if tc < 15 or tc > 25:
            token_warnings.append(f"{sent_id}: {tc} tokens — {sent}")

        output_lines.append(f"# sent_id = {sent_id}")
        output_lines.append(f"# text = {sent}")

        doc = nlp(sent)
        stanza_words = [
            w for s in doc.sentences for w in s.words if isinstance(w.id, int)
        ]
        aligned = align_stanza_to_text(sent, stanza_words)

        for token_counter, (form, upos, lemma) in enumerate(aligned, 1):
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
            output_lines.append("\t".join(cols))

        output_lines.append("")

    if token_warnings:
        print("TOKEN COUNT WARNINGS:")
        for w in token_warnings:
            print(f"  {w}")
        raise ValueError(f"{len(token_warnings)} sentences outside 15-25 token range")

    return "\n".join(output_lines) + "\n"


def main() -> None:
    import stanza

    for i, sent in enumerate(SENTENCES, START_ID):
        tc = count_tokens(sent)
        if tc < 15 or tc > 25:
            print(f"PRE-CHECK en_c2_train_{i:03d}: {tc} tokens — {sent}")
            sys.exit(1)

    print("Loading Stanza English pipeline...")
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma",
        use_gpu=False,
        verbose=False,
    )

    print("Generating CoNLL-U in sub-batches of 25...")
    for batch_num, batch in enumerate(BATCHES, 1):
        print(f"  Batch {batch_num}/8 ({len(batch)} sentences)...")

    conllu_text = build_conllu(SENTENCES, START_ID, nlp)

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    TARGET_PATH.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {len(SENTENCES)} sentences to {TARGET_PATH}")

    validation_res = validate_text(conllu_text)
    lemma_res = check_text(conllu_text, lang="en")
    print(
        f"COUNT={validation_res.sentence_count} "
        f"TOKENS={validation_res.token_count} "
        f"VAL={validation_res.passed} LEM={lemma_res.passed}"
    )

    if validation_res.errors:
        print("VAL ERRORS:")
        for err in validation_res.errors[:30]:
            print(f"  {err}")
    if lemma_res.errors:
        print("LEM ERRORS:")
        for err in lemma_res.errors[:30]:
            print(f"  {err}")

    if not validation_res.passed or not lemma_res.passed:
        sys.exit(1)

    print("Sent_ids: en_c2_train_001 – en_c2_train_200")
    print("Status: OK")


if __name__ == "__main__":
    main()