"""Generate en_c1_val_001–100 in data/handcraft/en/val/c1.conllu."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text

# Import normalization pipeline from English C1 train generator.
_train_gen_path = project_root / "data/handcraft/en/train/generate_c1_new_001.py"
_spec = importlib.util.spec_from_file_location("en_c1_train_gen", _train_gen_path)
_en_gen = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_en_gen)

process_sentence = _en_gen.process_sentence

SENTENCES = [
    # Journalism & media ethics (001-025)
    "The investigative reporter documented how editorial independence eroded when conglomerates consolidated regional newsrooms nationwide.",
    "Although press freedom indices improved modestly, journalists still faced intimidation while covering corruption in municipal procurement.",
    "It remains disputed whether algorithmic newsfeeds amplify polarization without adequate transparency regarding ranking criteria.",
    "The ombudsman concluded that anonymous sourcing should meet stricter verification standards before publication in sensitive investigations.",
    "Having reviewed internal correspondence, editors acknowledged that advertorial content blurred boundaries between journalism and marketing.",
    "The ethics council recommended mandatory disclosure whenever commentators receive compensation from organizations they routinely criticize publicly.",
    "Notwithstanding declining circulation, public broadcasters maintained investigative units scrutinizing regulatory capture in telecommunications markets.",
    "The defamation suit alleged that selectively edited footage misrepresented statements made during an off-the-record briefing.",
    "Whistleblowers supplying leaked documents insisted that redactions protected vulnerable sources without obscuring systemic misconduct.",
    "The parliamentary inquiry examined whether state advertising disproportionately favored outlets aligned with incumbent political narratives.",
    "It is widely argued that local journalism requires sustainable funding models beyond advertising revenue and philanthropic grants.",
    "The fact-checking consortium debunked viral claims while explaining methodological limitations inherent in rapid response verification.",
    "Although subscription models expanded, paywalls risked excluding low-income readers from essential civic information during elections.",
    "The union negotiated protections ensuring reporters cannot be reassigned arbitrarily for pursuing stories disfavored by advertisers.",
    "Pending judicial review, the court order prohibited further dissemination of materials obtained through allegedly unlawful surveillance.",
    "The documentary filmmaker balanced narrative pacing with evidentiary rigor when reconstructing events lacking comprehensive archival footage.",
    "Critics maintained that sensational headlines undermined public understanding of complex policy debates concerning climate adaptation.",
    "The publisher implemented corrections protocols requiring prominent retractions whenever factual errors affected reputations materially.",
    "It cannot be denied that foreign influence operations exploit social platforms to sow distrust in electoral institutions.",
    "The editorial board endorsed electoral reform while acknowledging legitimate disagreement among constitutional scholars regarding implementation.",
    "Having secured protective orders, the reporter continued covering organized crime despite credible threats against family members.",
    "The media literacy curriculum teaches students to evaluate source credibility and recognize manipulative framing techniques online.",
    "Although live broadcasts increased engagement, producers cautioned against prioritizing spectacle over substantive policy interrogation.",
    "The regulatory filing revealed undisclosed partnerships between influencers and pharmaceutical companies promoting unapproved therapeutic products.",
    "Scholars of communication analyzed how metaphorical framing shapes public receptivity toward contentious biomedical research initiatives.",
    # Urban planning & architecture (026-050)
    "The master plan prioritizes transit-oriented development while preserving historic districts vulnerable to speculative redevelopment pressures.",
    "Although zoning reforms permitted denser housing, neighborhood associations contested height limits affecting sightlines and sunlight access.",
    "It remains unclear whether adaptive reuse incentives will offset demolition costs for structurally sound but underutilized industrial buildings.",
    "The architect proposed permeable streetscapes integrating stormwater management with pedestrian safety improvements in flood-prone corridors.",
    "Having consulted residents, planners revised corridor designs to include protected cycling lanes and shaded waiting areas.",
    "The heritage commission denied permits after concluding that proposed facade alterations compromised the building's architectural integrity.",
    "Notwithstanding ambitious sustainability targets, embodied carbon in new construction continues outpacing operational efficiency gains substantially.",
    "The housing authority allocated vouchers toward mixed-income developments located near employment centers and reliable public transit.",
    "Urban heat island mitigation strategies combine reflective roofing, expanded tree canopy, and restrictions on impervious surface coverage.",
    "The feasibility study examined whether modular construction could accelerate affordable unit delivery without sacrificing durability standards.",
    "It is increasingly recognized that parking minimums inflate development costs while encouraging car dependency in compact cities.",
    "The landscape architect integrated bioswales and native plantings to restore ecological connectivity across fragmented suburban parcels.",
    "Although vacancy rates declined downtown, peripheral neighborhoods still experienced disinvestment and deteriorating commercial streetscapes.",
    "The municipal code now requires life-cycle assessments comparing renovation alternatives before approving large-scale demolition permits.",
    "Pending environmental review, the developer suspended excavation after archaeologists identified culturally significant subsurface artifacts.",
    "The transit agency coordinated last-mile solutions linking rail stations with micromobility hubs and secure bicycle storage.",
    "Critics argued that luxury condominium projects displaced longtime renters without providing adequate relocation assistance or compensation.",
    "The participatory budgeting process allocated funds toward playground renovations prioritized by parents in underserved school catchments.",
    "It cannot be overlooked that informal settlements lack secure tenure despite incremental infrastructure improvements.",
    "The structural engineer certified that seismic retrofits satisfied updated building codes without compromising occupant accessibility requirements.",
    "Having adopted new zoning codes, the municipality encouraged walkable mixed use blocks replacing car oriented strip commercial corridors.",
    "The smart city pilot deployed sensor networks monitoring air quality while addressing privacy concerns through aggregated anonymized reporting.",
    "Although public plazas revitalized civic life, maintenance budgets rarely covered ongoing upkeep of fountains and landscaping features.",
    "The regional coalition harmonized building standards facilitating cross-border cooperation on wildfire-resistant construction in interface zones.",
    "Scholars of urban studies debate whether innovation districts generate inclusive employment or primarily benefit highly credentialed professionals.",
    # Corporate governance & finance (051-075)
    "The board committee strengthened clawback provisions linking executive compensation to audited performance metrics and risk management outcomes.",
    "Although shareholder activism increased, dual-class share structures still insulated founders from accountability regarding strategic missteps.",
    "It remains contentious whether environmental disclosures should be subject to the same assurance standards as financial statements.",
    "The audit firm rotated lead partners after identifying material weaknesses in internal controls over revenue recognition procedures.",
    "Having restructured debt, the company maintained liquidity while negotiating covenant relief with institutional lenders during market volatility.",
    "The proxy statement detailed insider transactions requiring independent director approval under strengthened conflict of interest policies.",
    "Notwithstanding robust earnings, the firm faced scrutiny over transfer pricing arrangements shifting profits toward low-tax jurisdictions.",
    "The derivatives desk implemented position limits after stress tests revealed concentration risks exceeding board-approved risk appetite thresholds.",
    "Insider trading investigations examined whether executives traded securities ahead of undisclosed clinical trial results affecting valuations.",
    "The credit rating agency downgraded the issuer citing deteriorating interest coverage ratios and weakening free cash flow generation.",
    "It is broadly accepted that independent board chairs enhance oversight when separating leadership roles from chief executive functions.",
    "The venture fund diversified portfolios across sectors while maintaining disciplined valuation discipline during speculative market exuberance.",
    "Although liquidity improved, pension underfunding persisted due to conservative discount assumptions and lengthening life expectancies.",
    "The compliance officer mandated enhanced due diligence for correspondent banking relationships operating in jurisdictions with weak enforcement.",
    "Pending regulatory approval, the merger partners agreed to divest overlapping business lines to address antitrust concerns raised publicly.",
    "The forensic accountants traced fraudulent journal entries through subsidiary ledgers concealed within complex intercompany reconciliation schedules.",
    "Critics maintained that stock buybacks prioritized short-term earnings per share over productive capital expenditures and workforce development.",
    "The stewardship code encouraged institutional investors to engage constructively on governance matters rather than divesting passively.",
    "It cannot be assumed that passive index funds automatically exercise voting rights influencing corporate environmental commitments meaningfully.",
    "The restructuring plan allocated proceeds toward debt reduction while preserving research investments critical to long-term competitiveness.",
    "Having adopted IFRS reporting, the group harmonized segment disclosures facilitating cross-border comparability for multinational analysts.",
    "The fintech startup obtained banking licenses subject to capital requirements and operational resilience testing by supervisory authorities.",
    "Although default rates remained low, analysts warned that variable rate exposures could strain highly leveraged commercial property borrowers.",
    "The sovereign wealth fund published ethical investment guidelines excluding companies with documented labor violations in supply chains.",
    "Scholars of corporate law analyze how stakeholder governance models balance fiduciary duties with broader societal impact considerations.",
    # Philosophy, law & technology (076-100)
    "The philosopher argued that moral responsibility presupposes capacities for reflective deliberation absent in certain neurodevelopmental conditions.",
    "Although machine learning systems outperform clinicians, epistemic justification for algorithmic recommendations remains philosophically contested.",
    "It remains unresolved whether consciousness entails integrated information structures distinguishable from sophisticated behavioral mimicry alone.",
    "The appellate court held that compelled decryption implicates Fifth Amendment protections when testimonial acts reveal incriminating knowledge.",
    "Having surveyed jurisprudential traditions, the comparative lawyer identified convergent principles governing contractual good faith obligations.",
    "The bioethicist cautioned that germline editing raises intergenerational justice concerns extending beyond immediate therapeutic beneficiaries.",
    "Notwithstanding encryption safeguards, lawful access frameworks must balance investigative needs with protections against authoritarian overreach.",
    "The tribunal affirmed that customary international law obligations bind states even absent ratification of corresponding treaty instruments.",
    "Legal realists emphasized how institutional incentives shape judicial outcomes beyond formalistic application of canonical interpretive methods.",
    "The cognitive scientist demonstrated that attentional bottlenecks constrain parallel processing during demanding perceptual discrimination tasks.",
    "It is frequently asserted that personal identity persists through psychological continuity despite gradual replacement of physical constituents.",
    "The constitutional scholar examined whether emergency powers require sunset clauses preventing normalization of exceptional executive authority.",
    "Although predictive policing reduced certain crime categories, civil liberties advocates documented disparate impacts on marginalized communities.",
    "The intellectual historian traced how enlightenment concepts of autonomy informed contemporary debates about data protection rights.",
    "Pending legislative deliberation, the privacy regulator issued guidance clarifying consent requirements for cross-context behavioral advertising.",
    "The metaphysician defended a modal realist account explaining counterfactual dependence through relationships among possible world structures.",
    "Critics maintained that large language models reproduce linguistic patterns without genuine understanding of referential meaning in utterances.",
    "The arbitration panel enforced forum selection clauses while scrutinizing procedural fairness guarantees available to weaker contractual parties.",
    "It cannot be refuted that evidentiary standards in administrative proceedings differ substantively from criminal trials requiring proof beyond doubt.",
    "The phenomenologist described how embodied intentionality structures perceptual experience prior to explicit conceptual categorization processes.",
    "Having modeled adversarial examples, researchers demonstrated vulnerability of vision classifiers to imperceptible perturbations causing misclassification.",
    "The antitrust authority challenged exclusivity agreements foreclosing rival access to essential digital distribution platforms nationwide.",
    "Although open access mandates expanded, subscription journals retained prestige advantages influencing hiring and promotion decisions.",
    "The restorative justice facilitator convened dialogues prioritizing repair over retribution when addressing harm within school communities.",
    "Scholars of legal philosophy debate whether artificial agents could bear rights independently of human creators and institutional frameworks.",
]

assert len(SENTENCES) == 100, f"Expected 100 sentences, got {len(SENTENCES)}"

START_ID = 1
ID_PREFIX = "en_c1_val"
TARGET_PATH = project_root / "data/handcraft/en/val/c1.conllu"


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

    for idx, sent in enumerate(SENTENCES):
        sent_id = f"{ID_PREFIX}_{START_ID + idx:03d}"
        block, n_tokens = process_sentence(sent, sent_id, nlp)
        output_lines.extend(block)
        token_counts.append((sent_id, n_tokens))
        if (idx + 1) % 25 == 0:
            print(f"  Processed {idx + 1}/100...")

    conllu_text = "\n".join(output_lines) + "\n"
    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    TARGET_PATH.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {len(SENTENCES)} sentences to {TARGET_PATH}")

    bad_lengths = [(sid, n) for sid, n in token_counts if n < 12 or n > 20]
    if bad_lengths:
        print(f"Token length violations ({len(bad_lengths)}):")
        for sid, count in bad_lengths:
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

    print("STATUS: OK — en_c1_val_001 through en_c1_val_100")


if __name__ == "__main__":
    main()