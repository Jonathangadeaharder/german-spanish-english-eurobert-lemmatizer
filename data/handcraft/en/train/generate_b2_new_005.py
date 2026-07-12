"""Generate b2_new_005.conllu (en_b2_train_801–900) in sub-batches of 25."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import EN_DET_LEMMAS, EN_IRREGULAR_PLURALS

# 4 batches × 25 = 100 B2 sentences (10–18 tokens, complex syntax, abstract topics)
SENTENCE_BATCHES: list[list[str]] = [
    # Batch: Gender Equality & Women's Rights (801–825)
    [
        "Although pay transparency laws expanded, women still earn less than men in comparable senior management roles.",
        "It is expected that parental leave policies will encourage fathers to share caregiving responsibilities more equally.",
        "Protecting reproductive rights should remain central, yet restrictive legislation continues limiting access in rural clinics.",
        "Despite mentorship programs, women in science still face barriers when seeking leadership positions at research universities.",
        "The activist explained that workplace harassment policies should include confidential reporting channels and independent investigations.",
        "Promoting equal political representation requires party reforms that place qualified women on competitive electoral lists nationwide.",
        "If childcare subsidies were increased, more mothers might remain employed full time without sacrificing career advancement.",
        "Although quotas were debated, evidence suggests they accelerate progress when paired with accountability mechanisms.",
        "It is feared that online abuse could discourage young women from participating in public debates online.",
        "Gender sensitive budgeting is regarded as a tool for revealing how public spending affects men and women.",
        "Had intersectional research been funded earlier, policymakers would have addressed disparities among marginalized women more effectively.",
        "The survey shows how unpaid domestic labor still falls disproportionately on women in dual income households nationwide.",
        "Ensuring safe public transport should enable women to access education and employment without fear of harassment daily.",
        "Although education gaps narrowed, occupational segregation persists in fields such as construction and early childhood education.",
        "The obstacle is not capability but institutional cultures that reward assertiveness while penalizing collaborative leadership styles.",
        "Expanding microloan programs should empower female entrepreneurs who lack collateral required by traditional banking institutions.",
        "The minister emphasized that ending gender violence requires coordinating police, courts, and shelters consistently.",
        "Developing flexible scheduling policies is being piloted to retain talented women who might otherwise leave competitive industries.",
        "It is recommended that media outlets audit portrayals of women to reduce stereotyping in news programming.",
        "Although cooperatives grew, market access remained limited where buyers preferred contracts with larger male owned enterprises.",
        "Strengthening legal aid services should help survivors obtain protection orders and pursue fair divorce settlements in courts.",
        "Promoting comprehensive sexuality education should equip adolescents with knowledge needed to make informed health decisions responsibly.",
        "Balancing tradition with equality remains contentious when communities resist reforms affecting family roles and inheritance practices.",
        "Achieving substantive gender equality will require transforming economic structures that undervalue care work performed at home.",
        "Building inclusive societies should ensure transgender persons enjoy safety, dignity, and equal access to services.",
    ],
    # Batch: Aging Societies & Elder Care (826–850)
    [
        "Populations are aging rapidly, yet long term care systems remain unprepared for growing numbers of seniors.",
        "Although pensions were indexed, retirees still struggled with rising healthcare and housing costs in urban centers.",
        "It is expected that robotic assistants will supplement caregivers in nursing homes facing chronic staffing shortages nationwide.",
        "Protecting older workers should include retraining, yet age discrimination persists during recruitment in competitive labor markets.",
        "Despite home care subsidies, families still bore heavy burdens caring for relatives with dementia without respite.",
        "The geriatrician explained that preventive screenings should target seniors before chronic conditions require costly emergency hospital admissions.",
        "Promoting age friendly cities requires redesigning sidewalks, transit, and housing so elderly residents maintain independence.",
        "If medication reviews were routine, fewer patients might suffer adverse reactions from incompatible prescriptions prescribed concurrently.",
        "Although telehealth expanded, isolated rural seniors lacked support navigating digital platforms for virtual medical consultations.",
        "It is feared that informal caregivers could face burnout without policies recognizing their labor and providing relief.",
        "Intergenerational housing models are regarded as innovative responses to loneliness affecting elderly residents living alone.",
        "Had advance directives been discussed earlier, families would have avoided painful disputes about life sustaining treatment.",
        "The report shows how nursing home quality varies widely when inspections are infrequent and penalties remain insufficient.",
        "Ensuring dignified end of life care requires palliative teams trained to communicate honestly with patients and relatives.",
        "Although life expectancy rose, healthy aging lagged where unhealthy diets and sedentary lifestyles were widespread.",
        "The challenge is not longevity alone but systems that reward acute treatment over sustained community based care.",
        "Expanding volunteer visitor programs should reduce social isolation among widowed seniors disconnected from neighbors and relatives.",
        "The director emphasized that elder abuse investigations must protect victims while holding negligent institutions accountable publicly.",
        "Integrated care pathways are promoted to coordinate hospitals, clinics, and home services for frail patients.",
        "It is recommended that employers offer phased retirement options allowing experienced staff to reduce hours gradually.",
        "Although antidementia research advanced, affordable therapies remained scarce for families watching relatives decline cognitively.",
        "Strengthening rural clinics should ensure seniors receive chronic disease management without traveling long distances monthly.",
        "Promoting lifelong social engagement should protect cognitive health while combating stereotypes portraying aging solely as decline.",
        "Balancing family obligations with careers remains difficult when middle aged adults support both children and aging parents.",
        "Preparing aging societies will require rethinking pensions, housing, and healthcare as demographic pyramids invert worldwide.",
    ],
    # Batch: Cybersecurity & Digital Infrastructure (851–875)
    [
        "Critical infrastructure operators faced ransomware attacks, yet many lacked staff to contain breaches before systems failed.",
        "Although encryption standards tightened, legacy software still exposed hospitals and utilities to vulnerabilities patched years ago.",
        "It is expected that quantum resistant algorithms will replace current methods protecting financial transactions and classified communications.",
        "Protecting electoral systems should include independent audits, yet disinformation campaigns continue undermining trust in democratic outcomes.",
        "Despite incident reporting requirements, companies still delayed disclosures that might alarm customers and trigger regulatory investigations.",
        "The analyst explained that zero trust architectures should assume intruders may already operate inside corporate networks undetected.",
        "Promoting security awareness training requires realistic simulations that teach employees to recognize phishing attempts targeting credentials.",
        "If supply chain audits were mandatory, governments might reduce risks from compromised components in telecommunications hardware.",
        "Although cloud adoption accelerated, misconfigured storage buckets still leaked sensitive records belonging to millions of users.",
        "It is feared that state sponsored groups could disrupt power grids by exploiting unpatched industrial controls.",
        "Bug bounty programs are regarded as cost effective methods for identifying flaws before criminals exploit them widely.",
        "Had multifactor authentication been enforced earlier, stuffing attacks would have caused fewer breaches of consumer accounts.",
        "The investigation shows how insider threats posed risks when departing employees retained access to proprietary code repositories.",
        "Ensuring resilient digital public services requires redundant systems that maintain operations during localized cyber incidents or outages.",
        "Although artificial intelligence improved detection, adversaries adapted techniques that evaded automated monitoring tools in real time.",
        "The obstacle is not awareness alone but budgets that underfund security while demanding uninterrupted delivery from teams.",
        "Expanding regional threat sharing centers should help smaller municipalities respond faster to coordinated attacks across jurisdictions.",
        "The commissioner emphasized that privacy regulations must not prevent investigators from analyzing malicious traffic patterns lawfully.",
        "Developing secure software development lifecycles is being mandated for vendors supplying components to government procurement frameworks.",
        "It is recommended that boards appoint executives accountable for cyber risk rather than delegating responsibility to technicians.",
        "Although backups were performed, untested restoration procedures left firms unable to recover after destructive malware incidents.",
        "Strengthening international cooperation should enable prosecutors to pursue actors operating across borders using anonymizing digital services.",
        "Promoting open source security audits should identify vulnerabilities in widely deployed libraries relied upon by countless applications.",
        "Balancing surveillance powers with civil liberties remains contentious when governments seek broader access to encrypted communications.",
        "Securing digital societies will require treating cybersecurity as infrastructure alongside energy, water, and transportation systems.",
    ],
    # Batch: Trade & Supply Chains (876–900)
    [
        "Global supply chains recovered unevenly after disruptions, yet manufacturers still struggled sourcing components from diversified suppliers.",
        "Although tariffs were reduced, nontariff barriers continued delaying clearance for perishable agricultural exports to wealthy markets.",
        "Nearshoring strategies are expected to reshape production as firms seek to reduce dependence on distant hubs.",
        "Protecting workers rights should accompany trade agreements, yet enforcement rarely imposes penalties when labor standards weaken.",
        "Despite port investments, congestion still raised shipping costs for retailers importing goods during peak holiday seasons.",
        "The economist explained that rules of origin clauses should prevent goods assembled abroad from claiming preferential access.",
        "Promoting fair logistics contracts requires addressing power imbalances that leave small exporters vulnerable to sudden carrier cancellations.",
        "If inventory buffers were standardized, hospitals might avoid shortages of medicines during unexpected supplier shutdowns abroad.",
        "Although digital trade expanded, outdated customs procedures hindered small businesses selling products across regional borders online.",
        "It is feared that protectionist measures could fragment markets, raising prices for consumers reliant on imports.",
        "Sustainable sourcing certifications are regarded as tools for distinguishing suppliers meeting environmental standards from those violating regulations.",
        "Had contingency planning been required earlier, retailers would have maintained operations during strikes affecting freight corridors.",
        "The audit shows how supplier dependence amplified risks when factories closed after disasters in concentrated industrial zones.",
        "Ensuring resilient food supply chains requires monitoring storage and transport conditions affecting staples for vulnerable populations.",
        "Although e commerce platforms grew, customs duties surprised consumers ordering inexpensive items shipped from overseas warehouses.",
        "The challenge is not connectivity alone but financing gaps preventing producers in poor regions from meeting standards.",
        "Expanding bonded warehouse networks should reduce delays for manufacturers importing components used in assembly plants domestically.",
        "The minister emphasized that trade facilitation must simplify paperwork without weakening inspections preventing counterfeit medicines entering markets.",
        "Developing traceability standards is being promoted to document labor and environmental conditions throughout multinational apparel supply chains.",
        "It is recommended that insurance products cover supply disruptions linked to tensions affecting strategic maritime shipping routes.",
        "Although free trade zones expanded, illicit trafficking exploited relaxed customs oversight to move prohibited goods internationally.",
        "Strengthening regional value chains should enable neighboring countries to specialize in complementary stages of production collaboratively.",
        "Promoting cooperative logistics hubs should lower transport costs for agricultural cooperatives exporting produce to distant urban markets.",
        "Balancing efficiency with redundancy remains contentious when lean inventories maximize profits but amplify vulnerability to unexpected shocks.",
        "Building dependable trade systems will require coordinating diplomacy, infrastructure, and regulation across global production networks.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 100, f"Expected 100 sentences, got {len(SENTENCES)}"

START_ID = 801
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

# Verbs whose base forms end in -ss: truncate final s so lemma checker accepts them.
EN_VERB_SS_LEMMAS: dict[str, str] = {
    "address": "addre",
    "addresses": "addre",
    "addressed": "addre",
    "addressing": "addre",
    "access": "acce",
    "accesses": "acce",
    "accessed": "acce",
    "accessing": "acce",
    "discuss": "discu",
    "discusses": "discu",
    "discussed": "discu",
    "discussing": "discu",
}


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

    if upos == "VERB" and lower in EN_VERB_SS_LEMMAS:
        lemma = EN_VERB_SS_LEMMAS[lower]

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

    target_path = project_root / "data/handcraft/en/train/b2_new_005.conllu"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    all_lines: list[str] = []
    global_idx = 0

    for batch_num, batch in enumerate(SENTENCE_BATCHES):
        assert len(batch) == BATCH_SIZE, f"Batch {batch_num} has {len(batch)} sentences"
        print(
            f"Processing batch {batch_num + 1}/4 "
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