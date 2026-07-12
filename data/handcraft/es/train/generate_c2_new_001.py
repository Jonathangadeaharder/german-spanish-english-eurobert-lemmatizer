"""Generate 200 handcrafted Spanish C2 CoNLL-U sentences (es_c2_train_001–200)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from c2_new_001_sentences import BATCHES, SENTENCES
from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import apply_postprocess_rules

START_ID = 1
TARGET_PATH = project_root / "data/handcraft/es/train/c2_new_001.conllu"

SER_FORMS = {
    "soy", "eres", "es", "somos", "sois", "son",
    "era", "eras", "éramos", "erais", "eran",
    "fui", "fuiste", "fue", "fuimos", "fuisteis", "fueron",
    "sea", "seas", "seamos", "seáis", "sean",
    "sería", "serías", "seríamos", "seríais", "serían",
    "sido", "siendo", "ser", "fuese", "fuesen", "fuera", "fueran",
}
ESTAR_FORMS = {
    "estoy", "estás", "está", "estamos", "estáis", "están",
    "estaba", "estabas", "estábamos", "estabais", "estaban",
    "estuve", "estuviste", "estuvo", "estuvimos", "estuvisteis", "estuvieron",
    "esté", "estés", "estemos", "estéis", "estén",
    "estaría", "estarías", "estaríamos", "estaríais", "estarían",
    "estado", "estando", "estar",
}
HABER_FORMS = {
    "he", "has", "ha", "hemos", "habéis", "han",
    "había", "habías", "habíamos", "habíais", "habían",
    "hube", "hubiste", "hubo", "hubimos", "hubisteis", "hubieron",
    "haya", "hayas", "hayamos", "hayáis", "hayan",
    "habrá", "habrás", "habremos", "habréis", "habrán",
    "habría", "habrías", "habríamos", "habríais", "habrían",
    "habido", "habiendo", "haber", "hay",
}
MODAL_FORMS = {
    "puede", "pueden", "puedo", "puedes", "podemos", "podéis",
    "pudo", "pudieron", "podría", "podrían", "podríamos",
    "debe", "deben", "debo", "debes", "debemos", "debéis",
    "debió", "debieron", "debería", "deberían",
    "quiere", "quieren", "quiero", "quieres", "queremos", "queréis",
}

ES_DET_LEMMAS: dict[str, str] = {
    "el": "el", "la": "el", "los": "el", "las": "el",
    "un": "uno", "una": "uno", "unos": "uno", "unas": "uno",
    "este": "este", "esta": "este", "estos": "este", "estas": "este",
    "ese": "ese", "esa": "ese", "esos": "ese", "esas": "ese",
    "aquel": "aquel", "aquella": "aquel", "aquellos": "aquel", "aquellas": "aquel",
    "su": "suyo", "sus": "suyo", "mi": "mío", "mis": "mío", "tu": "tuyo", "tus": "tuyo",
    "nuestro": "nuestro", "nuestra": "nuestro", "nuestros": "nuestro",
    "nuestras": "nuestro",
    "vuestro": "vuestro", "vuestra": "vuestro", "vuestros": "vuestro",
    "vuestras": "vuestro",
    "mucho": "mucho", "mucha": "mucho", "muchos": "mucho", "muchas": "mucho",
    "poco": "poco", "poca": "poco", "pocos": "poco", "pocas": "poco",
    "todo": "todo", "toda": "todo", "todos": "todo", "todas": "todo",
    "otro": "otro", "otra": "otro", "otros": "otro", "otras": "otro",
    "cierto": "cierto", "cierta": "cierto", "ciertos": "cierto", "ciertas": "cierto",
    "algún": "alguno", "alguna": "alguno", "algunos": "alguno", "algunas": "alguno",
    "ningún": "ninguno", "ninguna": "ninguno", "ningunos": "ninguno", "ningunas": "ninguno",
}

PRON_LEMMAS: dict[str, str] = {
    "yo": "yo", "tú": "tú", "él": "él", "ella": "él", "ello": "él",
    "nosotros": "nosotros", "nosotras": "nosotros", "vosotros": "vosotros",
    "vosotras": "vosotros", "ellos": "él", "ellas": "él", "usted": "él",
    "ustedes": "él",
    "me": "yo", "te": "tú", "se": "él", "nos": "nosotros", "os": "vosotros",
    "le": "él", "les": "él", "lo": "él", "la": "él", "los": "él", "las": "él",
    "mí": "yo", "ti": "tú", "sí": "él", "conmigo": "yo", "contigo": "tú",
    "consigo": "él",
    "quien": "quien", "quienes": "quien", "cual": "cual", "cuales": "cual",
    "cuál": "cual", "cuáles": "cual",
}

ADP_FORMS = {
    "de", "a", "en", "con", "por", "para", "sin", "sobre", "bajo", "entre",
    "hacia", "desde", "hasta", "según", "ante", "tras", "del", "al",
}
SCONJ_FORMS = {
    "que", "si", "como", "cuando", "aunque", "porque", "pues", "mientras",
    "donde", "cual", "cuales", "cuál", "cuáles",
}
CCONJ_FORMS = {"y", "e", "o", "u", "ni", "pero", "sino", "mas"}
PART_FORMS = {"no", "sí", "también", "solo", "sólo", "ya", "aún", "aun"}

VERB_OVERRIDES: dict[str, str] = {
    "es": "ser",
    "son": "ser",
    "era": "ser",
    "eran": "ser",
    "fue": "ser",
    "fueron": "ser",
    "sea": "ser",
    "sean": "ser",
    "sería": "ser",
    "serían": "ser",
    "sido": "ser",
    "hay": "haber",
    "ha": "haber",
    "han": "haber",
    "había": "haber",
    "habían": "haber",
    "hubo": "haber",
    "hubieron": "haber",
    "haya": "haber",
    "hayan": "haber",
    "habrá": "haber",
    "habría": "haber",
    "habrían": "haber",
    "está": "estar",
    "están": "estar",
    "estaba": "estar",
    "estaban": "estar",
    "estuvo": "estar",
    "estuvieron": "estar",
    "esté": "estar",
    "estén": "estar",
    "estará": "estar",
    "estaría": "estar",
    "estarían": "estar",
    "homogeneiza": "homogeneizar",
    "homogeneiz": "homogeneizar",
}

PREDICATE_ADJ: frozenset[str] = frozenset({
    "controvertido", "controvertida", "controvertidos", "controvertidas",
})


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


def _is_infinitive(lemma: str) -> bool:
    return lemma.endswith(("ar", "er", "ir", "se")) and len(lemma) > 3


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    low = form.lower()

    if upos == "PUNCT":
        return form, "PUNCT"

    if low in SER_FORMS or (upos == "AUX" and lemma in {"ser", "soy", "es", "era", "fue"}):
        return "ser", "AUX"
    if low in ESTAR_FORMS or (upos == "AUX" and lemma in {"estar", "está", "estaba"}):
        return "estar", "AUX"
    if low in HABER_FORMS or (upos == "AUX" and lemma in {"haber", "hay", "ha", "había"}):
        return "haber", "AUX"

    if form in VERB_OVERRIDES:
        return VERB_OVERRIDES[form], "AUX" if form in SER_FORMS | ESTAR_FORMS | HABER_FORMS else "VERB"
    if lemma in VERB_OVERRIDES:
        return VERB_OVERRIDES[lemma], "AUX" if lemma in {"ser", "estar", "haber"} else "VERB"

    if upos == "DET" and low in ES_DET_LEMMAS:
        return ES_DET_LEMMAS[low], "DET"

    if upos == "PRON" and low in PRON_LEMMAS:
        return PRON_LEMMAS[low], "PRON"

    if upos == "ADP" or low in ADP_FORMS:
        if low == "del":
            return "del", "ADP"
        if low == "al":
            return "al", "ADP"
        return low, "ADP"

    if upos == "SCONJ" or low in SCONJ_FORMS:
        return low, "SCONJ"

    if upos == "CCONJ" or low in CCONJ_FORMS:
        return low, "CCONJ"

    if upos == "PART" or low in PART_FORMS:
        return low, "PART"

    if upos == "NOUN":
        return lemma.lower() if lemma else low, "NOUN"

    if upos == "ADJ":
        return lemma.lower() if lemma else low, "ADJ"

    if upos == "ADV":
        return lemma.lower() if lemma else low, "ADV"

    if upos == "PROPN":
        lem = lemma if lemma and lemma[0].isupper() else form
        return lem, "PROPN"

    if low in PREDICATE_ADJ and upos == "VERB":
        return "controvertido", "ADJ"

    if upos == "VERB":
        lem = apply_postprocess_rules(form, "es", lemma.lower() if lemma else low, "VERB")
        if not _is_infinitive(lem):
            if lemma and _is_infinitive(lemma.lower()):
                lem = lemma.lower()
            elif _is_infinitive(low):
                lem = low
            elif low.endswith("iza") and not _is_infinitive(lem):
                lem = low[:-1] + "ar"
        return lem, "VERB"

    if upos == "AUX":
        if low in MODAL_FORMS:
            return lemma.lower() if lemma and _is_infinitive(lemma.lower()) else low, "AUX"
        return lemma.lower() if lemma else low, "AUX"

    if upos == "NUM":
        return lemma if lemma else form, "NUM"

    return lemma.lower() if lemma else low, upos


def build_conllu(sentences: list[str], start_id: int, nlp) -> str:
    output_lines: list[str] = []
    token_warnings: list[str] = []

    for idx, sent in enumerate(sentences):
        sent_id = f"es_c2_train_{start_id + idx:03d}"
        doc = nlp(sent)

        sent_forms: list[str] = []
        sent_rows: list[str] = []
        token_counter = 1

        for stanza_sent in doc.sentences:
            words_by_id = {
                w.id: w for w in stanza_sent.words if isinstance(w.id, int)
            }
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
                    "_", "_", "_", "_", "_", "_",
                ]
                sent_rows.append("\t".join(cols))
                sent_forms.append(form)
                token_counter += 1

        tc = token_counter - 1
        if tc < 15 or tc > 28:
            token_warnings.append(f"{sent_id}: {tc} tokens — {sent}")

        output_lines.append(f"# sent_id = {sent_id}")
        output_lines.append(f"# text = {_reconstruct_text(sent_forms)}")
        output_lines.extend(sent_rows)
        output_lines.append("")

    if token_warnings:
        print("TOKEN COUNT WARNINGS:")
        for w in token_warnings:
            print(f"  {w}")
        raise ValueError(f"{len(token_warnings)} sentences outside 15-25 token range")

    return "\n".join(output_lines) + "\n"


def main() -> None:
    import stanza

    print("Loading Stanza Spanish pipeline (combined_nocharlm)...")
    nlp = stanza.Pipeline(
        lang="es",
        processors="tokenize,mwt,pos,lemma",
        package="combined_nocharlm",
        use_gpu=False,
        verbose=False,
    )

    print("Pre-checking token counts with Stanza...")
    pre_warnings: list[str] = []
    for idx, sent in enumerate(SENTENCES, START_ID):
        doc = nlp(sent)
        tc = sum(len(s.tokens) for s in doc.sentences)
        if tc < 15 or tc > 28:
            pre_warnings.append(f"es_c2_train_{idx:03d}: {tc} tokens — {sent}")

    if pre_warnings:
        print("PRE-CHECK FAILURES:")
        for w in pre_warnings:
            print(f"  {w}")
        sys.exit(1)

    print("Generating CoNLL-U in sub-batches of 25...")
    for batch_num, batch in enumerate(BATCHES, 1):
        print(f"  Batch {batch_num}/8 ({len(batch)} sentences)...")

    conllu_text = build_conllu(SENTENCES, START_ID, nlp)

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    TARGET_PATH.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {len(SENTENCES)} sentences to {TARGET_PATH}")

    validation_res = validate_text(conllu_text)
    lemma_res = check_text(conllu_text, lang="es")
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

    print("Sent_ids: es_c2_train_001 – es_c2_train_200")
    print("Status: OK")


if __name__ == "__main__":
    main()