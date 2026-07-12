"""Generate b2.conllu (en_b2_val_001–100) — English B2 validation set."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.duplicate_detector import check_cross_file, check_text as check_dup_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import EN_DET_LEMMAS, EN_IRREGULAR_PLURALS

# 4 batches × 25 = 100 B2 validation sentences (10–18 tokens, complex syntax)
SENTENCE_BATCHES: list[list[str]] = [
    # Batch 1 (001–025): Education & Universities
    [
        "Although tuition fees rose sharply, many graduates still regard university education as a worthwhile investment.",
        "It is expected that online degree programs will attract more working adults seeking flexible career advancement.",
        "The dean explained that plagiarism policies must be enforced consistently across every department on campus.",
        "Despite scholarship opportunities expanding, disadvantaged students still encounter barriers when applying to colleges.",
        "If research funding were distributed more fairly, smaller institutions might compete more effectively in rankings.",
        "Academic integrity is regarded as the foundation of credible scholarship in reputable research universities worldwide.",
        "While lecture halls remain popular, seminar based teaching methods are being adopted increasingly in humanities faculties.",
        "It is feared that budget cuts could reduce library hours and limit access to essential study materials.",
        "The committee recommended that student feedback should shape curriculum reforms introduced next academic year.",
        "Had mentoring schemes been expanded earlier, first year dropout rates might have declined across several faculties.",
        "Promoting interdisciplinary collaboration should enable researchers to address complex societal problems more effectively.",
        "Although enrollment increased, lecture halls could not accommodate demand without additional temporary classroom space.",
        "Ensuring accessible campuses requires installing ramps, elevators, and assistive technologies for disabled students.",
        "The professor emphasized that critical thinking skills must be developed through sustained reading and writing.",
        "Developing language support services should help international students adapt academically during initial semesters abroad.",
        "It is assumed that artificial intelligence tools will transform assessment practices in higher education over time.",
        "Despite competitive admissions, many applicants underestimated the workload required by demanding science programs.",
        "Strengthening partnerships with industry should provide internships connecting classroom theory with workplace experience.",
        "Balancing teaching responsibilities with research obligations remains challenging for early career academics nationwide.",
        "The survey shows how study abroad experiences influence graduates' career choices and cultural perspectives afterward.",
        "Although grades improved, faculty members questioned whether lenient marking standards undermined academic rigor.",
        "Expanding evening courses should enable commuters to complete degrees without resigning from full time employment.",
        "It is recommended that doctoral candidates receive structured supervision and timely feedback throughout thesis preparation.",
        "Observing ethical review procedures is essential before conducting experiments involving human participants on campus.",
        "Achieving educational excellence will require investing in faculty development, modern laboratories, and student support.",
    ],
    # Batch 2 (026–050): Health & Medicine
    [
        "Although vaccination campaigns succeeded, vaccine hesitancy still threatens herd immunity in several rural communities.",
        "It is expected that telemedicine services will expand access to specialists for patients living in remote areas.",
        "The surgeon explained that informed consent must be obtained before performing any elective surgical procedure.",
        "Despite hospital investments, emergency departments still face overcrowding during severe influenza outbreaks each winter.",
        "If preventive screenings were offered routinely, chronic diseases might be detected earlier and treated more effectively.",
        "Patient confidentiality is regarded as a fundamental ethical obligation for every licensed healthcare professional.",
        "While antibiotics remain essential, overprescription continues contributing to antimicrobial resistance in outpatient clinics.",
        "It is feared that staff shortages could compromise safety standards in nursing homes serving elderly residents.",
        "The panel recommended that mental health services should be integrated into primary care clinics nationwide.",
        "Had hygiene protocols been followed strictly, infection rates in the ward might have remained significantly lower.",
        "Promoting healthy lifestyles should reduce preventable illnesses linked to smoking, inactivity, and poor nutrition.",
        "Although treatment was effective, recovery required months of physiotherapy and careful monitoring by specialists.",
        "Ensuring affordable medicines requires negotiating fair prices with pharmaceutical companies and monitoring supply chains.",
        "The physician emphasized that patients deserve clear explanations before consenting to experimental treatment protocols.",
        "Developing community health programs should address disparities affecting low income families without regular medical care.",
        "It is assumed that wearable devices will help clinicians monitor chronic conditions between scheduled appointments.",
        "Despite advanced equipment, diagnostic errors still occur when test results are interpreted without sufficient context.",
        "Strengthening rural clinics should guarantee that residents receive timely care without traveling long distances monthly.",
        "Balancing cost containment with quality care remains contentious when insurers restrict coverage for new therapies.",
        "The report shows how lifestyle related diseases strain public health budgets in aging populations across regions.",
        "Although waiting lists shortened, patients complained about limited consultation time during busy clinic sessions.",
        "Expanding palliative care services should ensure terminally ill patients receive dignity and pain management at home.",
        "It is recommended that medical students practice communication skills through simulated consultations with trained actors.",
        "Protecting healthcare workers from burnout requires manageable shifts, counseling resources, and supportive team leadership.",
        "Building resilient health systems will require coordinating hospitals, insurers, and public agencies during future crises.",
    ],
    # Batch 3 (051–075): Culture & Media
    [
        "Although streaming platforms dominate, independent cinemas still attract audiences seeking distinctive cinematic experiences locally.",
        "It is expected that virtual exhibitions will complement traditional museums by reaching viewers unable to travel.",
        "The curator explained that loan agreements must specify insurance coverage before fragile artworks are transported abroad.",
        "Despite declining print sales, investigative journalism continues exposing corruption through painstaking documentary research.",
        "If public funding were increased, regional theaters might stage productions reflecting diverse community voices.",
        "Cultural heritage is regarded as a shared asset requiring protection from neglect, theft, and pollution.",
        "While social media amplifies artists, algorithms often prioritize sensational content over nuanced cultural commentary.",
        "It is feared that censorship pressures could discourage writers from addressing controversial political themes openly.",
        "The critic recommended that adaptations should respect source material while offering fresh interpretations for audiences.",
        "Had restoration work begun sooner, the historic fresco might have survived humidity damage in the chapel.",
        "Promoting reading initiatives should strengthen literacy skills among adolescents distracted by constant digital entertainment.",
        "Although the concert sold out, ticket scalping deprived genuine fans of affordable access to performances.",
        "Ensuring fair royalties requires transparent contracts between musicians, producers, and digital distribution platforms.",
        "The director emphasized that diverse casting should reflect society without reducing characters to superficial stereotypes.",
        "Developing community arts centers should provide affordable studios where emerging painters and sculptors collaborate.",
        "It is assumed that immersive technologies will reshape how audiences experience theater and live performances.",
        "Despite festival success, local residents complained about noise and congestion during peak tourist weekends.",
        "Strengthening public broadcasting should guarantee independent news coverage when commercial outlets prioritize profitability.",
        "Balancing artistic freedom with audience expectations remains difficult when productions address sensitive historical events.",
        "The review shows how translated novels broaden readers' understanding of cultures beyond their national borders.",
        "Although subscriptions grew, smaller publishers struggled negotiating favorable terms with dominant ebook retailers.",
        "Expanding arts education should cultivate creativity in schools facing pressure to emphasize standardized test results.",
        "It is recommended that archives digitize fragile manuscripts so scholars worldwide can study them responsibly.",
        "Preserving oral traditions should document languages threatened by globalization and shifting patterns of daily communication.",
        "Sustaining vibrant cultural life will require patronage, education, and policies valuing creativity alongside economic growth.",
    ],
    # Batch 4 (076–100): Law, Work & Society
    [
        "Although the defendant pleaded not guilty, prosecutors presented forensic evidence linking him to the crime scene.",
        "It is expected that new regulations will strengthen consumer protection against misleading advertising in online marketplaces.",
        "The judge explained that witness testimony must be evaluated carefully when physical evidence remains inconclusive.",
        "Despite mediation efforts, the parties could not agree on compensation for damages caused by the accident.",
        "If whistleblower protections were strengthened, employees might report corporate fraud without fear of retaliation.",
        "Due process is regarded as essential for maintaining public trust in courts handling complex commercial disputes.",
        "While unemployment fell, many workers accepted temporary contracts lacking benefits traditionally associated with permanent jobs.",
        "It is feared that automation could displace workers unless retraining programs accompany technological modernization initiatives.",
        "The union recommended that wage negotiations should reflect inflation and rising living costs in urban areas.",
        "Had safety inspections been conducted regularly, the factory accident might have been prevented before injuries occurred.",
        "Promoting ethical leadership should encourage managers to prioritize transparency over short term profit maximization strategies.",
        "Although the merger was approved, regulators imposed conditions protecting competition in regional supply markets.",
        "Ensuring workplace equality requires monitoring recruitment practices and addressing harassment complaints through impartial investigations.",
        "The lawyer emphasized that contracts should be drafted clearly to prevent costly disputes between business partners.",
        "Developing apprenticeship programs should equip school leavers with skills demanded by evolving manufacturing industries nationwide.",
        "It is assumed that remote collaboration tools will permanently alter office culture in many professional sectors.",
        "Despite profit growth, shareholders questioned executive bonuses awarded during layoffs affecting hundreds of employees.",
        "Strengthening tenant protections should prevent arbitrary evictions when rental markets tighten in rapidly growing cities.",
        "Balancing corporate secrecy with public accountability remains contentious when firms handle sensitive personal data.",
        "The audit shows how procurement irregularities undermined confidence in institutions responsible for public infrastructure projects.",
        "Although settlements were reached, victims argued that penalties failed to deter repeat violations by large corporations.",
        "Expanding legal aid services should guarantee disadvantaged citizens access to representation during complex civil proceedings.",
        "It is recommended that companies publish sustainability reports detailing environmental impacts of their global operations.",
        "Protecting civic participation should ensure protest rights while maintaining public order during large demonstrations downtown.",
        "Building fair societies will require enforcing laws impartially while addressing structural inequalities affecting vulnerable communities.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 100, f"Expected 100 sentences, got {len(SENTENCES)}"

START_ID = 1
BATCH_SIZE = 25
MIN_TOKENS = 10
MAX_TOKENS = 18
SENT_ID_PREFIX = "en_b2_val"

AUX_LEMMAS = {"be", "have", "do", "will", "can", "may", "might", "shall", "should", "must"}
BE_FORMS = {"am", "is", "are", "was", "were", "be", "been", "being"}
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
    "process": "proce",
    "processes": "proce",
    "processed": "proce",
    "processing": "proce",
    "express": "expres",
    "expresses": "expres",
    "expressed": "expres",
    "expressing": "expres",
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
            print(f"PRE-CHECK {SENT_ID_PREFIX}_{i:03d}: {tc} tokens — {sent}")
            sys.exit(1)

    print("Loading Stanza...")
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma",
        use_gpu=False,
        verbose=False,
    )

    target_path = project_root / "data/handcraft/en/val/b2.conllu"
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
            sent_id = f"{SENT_ID_PREFIX}_{START_ID + global_idx:03d}"
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

    dup_res = check_dup_text(conllu_text)
    print(f"Duplicate check (within file): passed={dup_res.passed}")
    if not dup_res.passed:
        for err in dup_res.duplicates:
            print(f"  {err}")
        sys.exit(1)

    train_dir = project_root / "data/handcraft/en/train"
    train_text = ""
    for path in sorted(train_dir.glob("b2_new_*.conllu")):
        train_text += path.read_text(encoding="utf-8") + "\n"
    cross_dup = check_cross_file(train_text, conllu_text)
    print(f"Duplicate check (vs train): passed={cross_dup.passed}")
    if not cross_dup.passed:
        for err in cross_dup.duplicates:
            print(f"  {err}")
        sys.exit(1)
    for warn in cross_dup.warnings[:5]:
        print(f"  WARNING: {warn}")

    print("All checks passed.")


if __name__ == "__main__":
    main()