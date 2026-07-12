"""Generate 100 handcrafted Spanish C1 CoNLL-U sentences (es_c1_train_801–900)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import es_reflexive_lemma

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1_new_005_sentences import SENTENCE_BATCHES, SENTENCES  # noqa: E402

START_ID = 801
BATCH_SIZE = 25
TARGET_PATH = project_root / "data/handcraft/es/train/c1_new_005.conllu"

SER_FORMS = {
    "soy", "eres", "es", "somos", "sois", "son",
    "era", "eras", "éramos", "eran", "fue", "fueron",
    "sea", "seas", "sean", "sido", "siendo", "ser",
}
ESTAR_FORMS = {
    "estoy", "estás", "está", "estamos", "estáis", "están",
    "estaba", "estabas", "estábamos", "estaban",
    "estuvo", "estuvieron", "esté", "estén", "estado", "estando", "estar",
}
HABER_FORMS = {
    "he", "has", "ha", "hemos", "habéis", "han", "hay", "había", "habías",
    "habíamos", "habían", "hubo", "hubieron", "haya", "hayan",
    "habría", "habrían", "habrá", "habrán", "habiendo", "habido", "haber",
}
PODER_FORMS = {
    "puedo", "puedes", "puede", "podemos", "podéis", "pueden",
    "podía", "podían", "pudo", "pudieron", "pueda", "puedan",
    "podría", "podrían", "podrá", "podrán", "poder",
}
DEBER_FORMS = {
    "debo", "debes", "debe", "debemos", "debéis", "deben",
    "debía", "debían", "deba", "deban", "debería", "deberían", "deber",
}
AUX_LEMMAS = {"ser", "estar", "haber", "poder", "deber"}

EL_DET = {"el", "la", "los", "las", "lo"}
UNO_DET = {"un", "una", "unos", "unas"}
MIO_DET = {"mi", "mis", "mío", "mía", "míos", "mías"}
SU_DET = {"su", "sus", "suyo", "suya", "suyos", "suyas"}
ESTE_DET = {"este", "esta", "estos", "estas"}
ESE_DET = {"ese", "esa", "esos", "esas"}
AQUEL_DET = {"aquel", "aquella", "aquellos", "aquellas"}
OTRO_DET = {"otro", "otra", "otros", "otras"}
TODO_DET = {"todo", "toda", "todos", "todas"}
CUANTO_DET = {"cuanto", "cuanta", "cuantos", "cuantas", "cuánto", "cuánta", "cuántos", "cuántas"}
QUE_DET = {"qué", "que"}

SCONJ_WORDS = {
    "que", "aunque", "si", "como", "cuando", "donde", "mientras", "porque",
    "pues", "ya", "antes", "después", "hasta", "desde", "según", "salvo",
}
CCONJ_WORDS = {"y", "o", "u", "e", "ni", "pero", "sino"}
ADP_WORDS = {
    "a", "ante", "bajo", "cabe", "con", "contra", "de", "del", "desde",
    "durante", "en", "entre", "hacia", "hasta", "mediante", "para", "por",
    "según", "sin", "so", "sobre", "tras", "al", "versus", "vía",
}
PART_WORDS = {"no", "sí"}
PRON_WORDS = {
    "yo", "tú", "él", "ella", "ello", "nosotros", "nosotras", "vosotros",
    "vosotras", "ellos", "ellas", "usted", "ustedes", "me", "te", "se", "le",
    "lo", "la", "los", "las", "nos", "os", "les", "mí", "ti", "sí", "consigo",
}

SPECIAL_LEMMAS: dict[str, tuple[str, str]] = {
    "NFT": ("NFT", "PROPN"),
    "STEM": ("STEM", "PROPN"),
    "on-line": ("on-line", "ADV"),
    "del": ("del", "ADP"),
    "al": ("al", "ADP"),
    "Pendiente": ("pendiente", "ADP"),
    "pendiente": ("pendiente", "ADP"),
    "No": ("no", "PART"),
    "no": ("no", "PART"),
    "Sí": ("sí", "PART"),
    "sí": ("sí", "PART"),
    "cada": ("cada", "DET"),
    "mismo": ("mismo", "DET"),
    "misma": ("mismo", "DET"),
    "mismos": ("mismo", "DET"),
    "mismas": ("mismo", "DET"),
    "cualquier": ("cualquier", "DET"),
    "ningún": ("ninguno", "DET"),
    "ninguna": ("ninguno", "DET"),
    "ningunos": ("ninguno", "DET"),
    "ningunas": ("ninguno", "DET"),
    "varios": ("vario", "DET"),
    "varias": ("vario", "DET"),
    "mucho": ("mucho", "ADV"),
    "mucha": ("mucho", "ADV"),
    "muchos": ("mucho", "ADV"),
    "muchas": ("mucho", "ADV"),
    "poco": ("poco", "ADV"),
    "poca": ("poco", "ADV"),
    "pocos": ("poco", "ADV"),
    "pocas": ("poco", "ADV"),
    "más": ("más", "ADV"),
    "menos": ("menos", "ADV"),
    "muy": ("muy", "ADV"),
    "tan": ("tan", "ADV"),
    "tanto": ("tanto", "ADV"),
    "tanta": ("tanto", "ADV"),
    "tantos": ("tanto", "ADV"),
    "tantas": ("tanto", "ADV"),
    "bien": ("bien", "ADV"),
    "mal": ("mal", "ADV"),
    "cómo": ("cómo", "ADV"),
    "cuándo": ("cuándo", "ADV"),
    "dónde": ("dónde", "ADV"),
    "porqué": ("porqué", "NOUN"),
    "por": ("por", "ADP"),
    "para": ("para", "ADP"),
    "con": ("con", "ADP"),
    "sin": ("sin", "ADP"),
    "sobre": ("sobre", "ADP"),
    "entre": ("entre", "ADP"),
    "hacia": ("hacia", "ADP"),
    "desde": ("desde", "ADP"),
    "hasta": ("hasta", "ADP"),
    "durante": ("durante", "ADP"),
    "mediante": ("mediante", "ADP"),
    "según": ("según", "ADP"),
    "contra": ("contra", "ADP"),
    "ante": ("ante", "ADP"),
    "bajo": ("bajo", "ADP"),
    "tras": ("tras", "ADP"),
    "y": ("y", "CCONJ"),
    "o": ("o", "CCONJ"),
    "u": ("o", "CCONJ"),
    "e": ("y", "CCONJ"),
    "ni": ("ni", "CCONJ"),
    "pero": ("pero", "CCONJ"),
    "sino": ("sino", "CCONJ"),
    "aunque": ("aunque", "SCONJ"),
    "si": ("si", "SCONJ"),
    "que": ("que", "SCONJ"),
    "como": ("como", "SCONJ"),
    "cuando": ("cuando", "SCONJ"),
    "donde": ("donde", "ADV"),
    "mientras": ("mientras", "SCONJ"),
    "porque": ("porque", "SCONJ"),
    "pues": ("pues", "SCONJ"),
    "ya": ("ya", "ADV"),
    "a": ("a", "ADP"),
    "de": ("de", "ADP"),
    "en": ("en", "ADP"),
}

VERB_OVERRIDES: dict[str, str] = {
    "monitorearon": "monitorear",
    "monitorea": "monitorear",
    "monitoreó": "monitorear",
    "monitorear": "monitorear",
    "monitor": "monitorear",
    "monitorean": "monitorear",
}

PREDICATE_ADJ: frozenset[str] = frozenset({
    "adecuado", "adecuada", "adecuados", "adecuadas",
    "suficiente", "suficientes", "necesario", "necesaria", "necesarios", "necesarias",
})


def simple_tokenize(sentence: str) -> list[str]:
    forms: list[str] = []
    for word in sentence.split():
        match = re.match(r"^(.+?)([.,;:!?¿¡]+)$", word)
        if match:
            forms.append(match.group(1))
            forms.extend(list(match.group(2)))
        else:
            forms.append(word)
    return forms


def count_tokens(sentence: str) -> int:
    return len(simple_tokenize(sentence))


def _reconstruct_text(forms: list[str]) -> str:
    punct_prefixes = ".,;:!?¿¡\"')"
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


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    """Apply handcrafted Spanish lemma/UPOS rules (es_test conventions)."""
    if form in SPECIAL_LEMMAS:
        return SPECIAL_LEMMAS[form]

    low = form.lower()

    if upos == "PUNCT":
        return form, "PUNCT"

    if low in SER_FORMS:
        return "ser", "AUX"
    if low in ESTAR_FORMS:
        return "estar", "AUX"
    if low in HABER_FORMS:
        return "haber", "AUX"
    if low in PODER_FORMS:
        return "poder", "AUX"
    if low in DEBER_FORMS:
        return "deber", "AUX"

    if form in VERB_OVERRIDES:
        return VERB_OVERRIDES[form], "VERB"
    if lemma in VERB_OVERRIDES:
        return VERB_OVERRIDES[lemma], "VERB"

    if low in SCONJ_WORDS and low not in {"como", "donde"}:
        if low == "que" and upos in {"PRON", "DET"}:
            pass
        else:
            return low, "SCONJ"
    if low in CCONJ_WORDS:
        return low if low != "e" else "y", "CCONJ"
    if low in PART_WORDS:
        return low, "PART"
    if low in ADP_WORDS or (upos == "ADP" and low not in PRON_WORDS):
        if low == "del":
            return "del", "ADP"
        if low == "al":
            return "al", "ADP"
        return low, "ADP"

    if low in EL_DET or form in {"El", "La", "Los", "Las", "Lo"}:
        return "el", "DET"
    if low in UNO_DET:
        return "uno", "DET"
    if low in MIO_DET:
        return "mío", "DET"
    if low in SU_DET:
        return "suyo", "DET"
    if low in ESTE_DET:
        return "este", "DET"
    if low in ESE_DET:
        return "ese", "DET"
    if low in AQUEL_DET:
        return "aquel", "DET"
    if low in OTRO_DET:
        return "otro", "DET"
    if low in TODO_DET:
        return "todo", "DET"
    if low in CUANTO_DET:
        return "cuanto", "DET"
    if low in QUE_DET and upos in {"DET", "PRON"}:
        return "que", "DET" if upos == "DET" else "que"

    if form in {"Ella", "Ellos", "Ellas"} or low in {"ella", "ellos", "ellas"}:
        return "él", "PRON"
    if low == "se":
        return "él", "PRON"
    if low == "le" or low == "les":
        return "él", "PRON"
    if low in {"yo", "tú", "nosotros", "nosotras", "vosotros", "vosotras", "usted", "ustedes"}:
        return low, "PRON"
    if form == "Él":
        return "él", "PRON"
    if low == "él":
        return "él", "PRON"

    if low in PREDICATE_ADJ and upos == "VERB":
        adj_lemma = {
            "adecuado": "adecuado", "adecuada": "adecuado",
            "adecuados": "adecuado", "adecuadas": "adecuado",
            "suficiente": "suficiente", "suficientes": "suficiente",
            "necesario": "necesario", "necesaria": "necesario",
            "necesarios": "necesario", "necesarias": "necesario",
        }[low]
        return adj_lemma, "ADJ"

    if upos == "VERB":
        lem = (lemma or form).lower()
        reflexive = es_reflexive_lemma(low)
        if reflexive is not None:
            lem = reflexive
        if form in VERB_OVERRIDES:
            lem = VERB_OVERRIDES[form]
        elif lemma in VERB_OVERRIDES:
            lem = VERB_OVERRIDES[lemma]
        return lem, "VERB"

    if upos == "ADJ" and lemma:
        return lemma.lower(), "ADJ"

    if upos == "NOUN" and lemma:
        return lemma.lower(), "NOUN"

    if upos == "PROPN":
        lem = lemma if lemma else form
        if lem and lem[0].islower():
            lem = lem[0].upper() + lem[1:]
        return lem, "PROPN"

    if upos == "ADV":
        return (lemma or form).lower(), "ADV"

    if upos == "AUX":
        if lemma in AUX_LEMMAS:
            return lemma, "AUX"
        return (lemma or form).lower(), "VERB"

    if upos == "DET":
        if low in EL_DET:
            return "el", "DET"
        if low in UNO_DET:
            return "uno", "DET"

    return lemma or form, upos


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

    bad_lengths = [(i + 1, count_tokens(s)) for i, s in enumerate(SENTENCES)
                   if count_tokens(s) < 12 or count_tokens(s) > 20]
    if bad_lengths:
        print(f"PRE-RUN token length violations ({len(bad_lengths)}):")
        for num, count in bad_lengths[:30]:
            print(f"  sentence {num}: {count} tokens — {SENTENCES[num - 1]}")
        sys.exit(1)

    print("Loading Stanza Spanish pipeline...")
    nlp = stanza.Pipeline(
        lang="es",
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
        print(f"Processing batch {batch_num}/4 (es_c1_train_{start:03d}–{end:03d})...")

        for sent in batch:
            sent_id = f"es_c1_train_{START_ID + global_idx:03d}"
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
    lem = check_text(conllu_text, lang="es")
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

    print("STATUS: OK — es_c1_train_801 through es_c1_train_900")


if __name__ == "__main__":
    main()