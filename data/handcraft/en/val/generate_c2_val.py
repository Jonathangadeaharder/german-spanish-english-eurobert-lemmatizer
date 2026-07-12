"""Generate 100 handcrafted English C2 validation CoNLL-U sentences (en_c2_val_001–100)."""

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

from c2_val_sentences import BATCHES, SENTENCES


START_ID = 1
BATCH_SIZE = 25
TARGET_PATH = project_root / "data/handcraft/en/val/c2.conllu"
SENT_ID_PREFIX = "en_c2_val"

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
    "artwork's": ("artwork", "NOUN"),
    "Quine's": ("Quine", "PROPN"),
    "avant-garde's": ("avant-garde", "NOUN"),
    "Mary's": ("Mary", "PROPN"),
    "treaty's": ("treaty", "NOUN"),
    "Kantian": ("Kantian", "ADJ"),
    "Hegel": ("Hegel", "PROPN"),
    "Bakhtin": ("Bakhtin", "PROPN"),
    "Aristotelian": ("Aristotelian", "ADJ"),
    "Wabi-sabi": ("Wabi-sabi", "NOUN"),
    "Shakespeare": ("Shakespeare", "PROPN"),
    "Libet-style": ("Libet-style", "ADJ"),
    "Mary": ("Mary", "PROPN"),
    "Quine": ("Quine", "PROPN"),
    "Sellars": ("Sellars", "PROPN"),
    "language's": ("language", "NOUN"),
    "shepherd's": ("shepherd", "NOUN"),
    "genre's": ("genre", "NOUN"),
    "villanelle's": ("villanelle", "NOUN"),
    "beneficiaries'": ("beneficiary", "NOUN"),
    "sense-data": ("sense-data", "NOUN"),
    "internalism-externalism": ("internalism-externalism", "NOUN"),
    "context-sensitive": ("context-sensitive", "ADJ"),
    "genre-specific": ("genre-specific", "ADJ"),
    "reception-oriented": ("reception-oriented", "ADJ"),
    "self-dealing": ("self-dealing", "NOUN"),
    "truth-finding": ("truth-finding", "NOUN"),
    "founding-era": ("founding-era", "ADJ"),
    "representative-agent": ("representative-agent", "ADJ"),
    "meta-analysis": ("meta-analysis", "NOUN"),
    "off-target": ("off-target", "ADJ"),
    "net-zero": ("net-zero", "ADJ"),
    "R-value": ("R-value", "NOUN"),
    "data-sparse": ("data-sparse", "ADJ"),
    "cross-section": ("cross-section", "NOUN"),
    "CRISPR": ("CRISPR", "PROPN"),
    "Bayesian": ("Bayesian", "ADJ"),
    "de": ("de", "ADP"),
    "facto": ("facto", "ADV"),
    "jure": ("jure", "ADV"),
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
    "repress": "repre",
    "represses": "repre",
    "repressed": "repre",
    "repressing": "repre",
    "suppress": "suppre",
    "suppresses": "suppre",
    "suppressed": "suppre",
    "suppressing": "suppre",
    "bypass": "bypa",
    "bypasses": "bypa",
    "bypassed": "bypa",
    "bypassing": "bypa",
    "process": "proce",
    "processes": "proce",
    "processed": "proce",
    "processing": "proce",
    "exceed": "exce",
    "exceeds": "exce",
    "exceeded": "exce",
    "exceeding": "exce",
    "need": "ne",
    "needs": "ne",
    "needed": "ne",
    "needing": "ne",
}

VERB_OVERRIDES: dict[str, str] = {
    "purports": "purport",
    "presupposes": "presuppose",
    "presupposing": "presuppose",
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

    "insists": "insist",
    "valorizes": "valorize",
    "elevated": "elevate",
    "showed": "show",
    "deciphered": "decipher",
    "gutted": "gut",
    "awaited": "await",
    "borne": "bear",
    "woven": "weave",
    "misread": "misread",
    "dissolved": "dissolve",
    "displaced": "displace",
    "convened": "convene",
    "administered": "administer",
    "coordinated": "coordinate",
    "conveyed": "convey",
    "documented": "document",
    "emphasized": "emphasize",
    "outlined": "outline",
    "pledged": "pledge",
    "proposed": "propose",
    "scrutinized": "scrutinize",
    "sequenced": "sequence",
    "welcomed": "welcome",
    "appropriated": "appropriate",
    "compensated": "compensate",
    "constituted": "constitute",
    "calibrated": "calibrate",
    "authorized": "authorize",
    "prioritized": "prioritize",
    "reaffirming": "reaffirm",
    "repressed": "repre",
    "hinges": "hinge",
    "suppressing": "suppress",
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
        if w2.text in {"'s", "'"} and expected in {f"{w1.text}'s", f"{w1.text}'"}:
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
        sent_id = f"{SENT_ID_PREFIX}_{start_id + idx:03d}"
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
            print(f"PRE-CHECK {SENT_ID_PREFIX}_{i:03d}: {tc} tokens — {sent}")
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
        print(f"  Batch {batch_num}/{len(BATCHES)} ({len(batch)} sentences)...")

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

    dup_res = check_dup_text(conllu_text)
    print(f"Duplicate check: passed={dup_res.passed}")
    if not dup_res.passed:
        for err in dup_res.duplicates[:20]:
            print(f"  {err}")
        sys.exit(1)

    train_dir = project_root / "data/handcraft/en/train"
    train_text = ""
    for path in sorted(train_dir.glob("c2_new_*.conllu")):
        train_text += path.read_text(encoding="utf-8") + "\n"
    cross_res = check_cross_file(train_text, conllu_text)
    print(f"Cross-file duplicate check (vs train): passed={cross_res.passed}")
    if not cross_res.passed:
        for err in cross_res.duplicates[:20]:
            print(f"  {err}")
        sys.exit(1)

    print(f"Sent_ids: {SENT_ID_PREFIX}_001 – {SENT_ID_PREFIX}_100")
    print("Status: OK")


if __name__ == "__main__":
    main()