"""Generate b1_new_003.conllu (es_b1_train_401–600) with Stanza + validation."""

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
from b1_new_003_sentences import SENTENCE_BATCHES, SENTENCES  # noqa: E402


START_ID = 401
BATCH_SIZE = 25
MIN_TOKENS = 8
MAX_TOKENS = 15

SER_FORMS = {
    "es", "son", "era", "eran", "fue", "fui", "fuiste", "fueron", "sería", "serían",
    "será", "serán", "sido", "siendo", "sea", "seas", "sean", "fuese", "fuesen",
    "eres", "somos", "sois", "fui", "fuimos", "fuisteis", "soy",
}
ESTAR_FORMS = {
    "está", "están", "estaba", "estaban", "estuvo", "estuve", "estará", "estarán",
    "estén", "estemos", "estoy", "estás", "estamos", "estáis", "estado", "estando",
}
HABER_FORMS = {
    "ha", "han", "había", "habían", "hubo", "hubiera", "hubieran", "haya", "hayan",
    "he", "has", "hemos", "habéis", "habido", "habiendo",
}
HAY_FORMS = {"hay", "habrá", "habría", "habrían"}

SPECIAL_LEMMAS: dict[str, tuple[str, str]] = {
    "yo": ("yo", "PRON"), "Yo": ("yo", "PRON"),
    "tú": ("tú", "PRON"), "Tú": ("tú", "PRON"),
    "él": ("él", "PRON"), "Él": ("él", "PRON"),
    "ella": ("él", "PRON"), "Ella": ("él", "PRON"),
    "ellos": ("él", "PRON"), "ellas": ("él", "PRON"),
    "me": ("yo", "PRON"), "Me": ("yo", "PRON"),
    "te": ("tú", "PRON"),
    "ti": ("tú", "PRON"),
    "se": ("él", "PRON"),
    "nos": ("nosotros", "PRON"), "Nos": ("nosotros", "PRON"),
    "os": ("vosotros", "PRON"),
    "le": ("él", "PRON"), "les": ("él", "PRON"),
    "lo": ("él", "PRON"), "la": ("él", "PRON"), "los": ("él", "PRON"), "las": ("él", "PRON"),
    "nosotros": ("nosotros", "PRON"), "Nosotros": ("nosotros", "PRON"),
    "vosotros": ("vosotros", "PRON"),
    "usted": ("él", "PRON"), "ustedes": ("él", "PRON"),
    "quien": ("quien", "PRON"), "Quién": ("quien", "PRON"), "quién": ("quien", "PRON"),
    "cual": ("cual", "PRON"), "Cuál": ("cual", "PRON"), "cuál": ("cual", "PRON"),
    "que": ("que", "PRON"), "Qué": ("que", "PRON"), "qué": ("que", "PRON"),
    "algo": ("algo", "PRON"), "nada": ("nada", "PRON"), "alguien": ("alguien", "PRON"),
    "nadie": ("nadie", "PRON"),
    "el": ("el", "DET"), "la": ("el", "DET"), "los": ("el", "DET"), "las": ("el", "DET"),
    "El": ("el", "DET"), "La": ("el", "DET"), "Los": ("el", "DET"), "Las": ("el", "DET"),
    "un": ("uno", "DET"), "una": ("uno", "DET"), "unos": ("uno", "DET"), "unas": ("uno", "DET"),
    "Un": ("uno", "DET"), "Una": ("uno", "DET"),
    "mi": ("mío", "DET"), "mis": ("mío", "DET"), "Mi": ("mío", "DET"), "Mis": ("mío", "DET"),
    "tu": ("tuyo", "DET"), "tus": ("tuyo", "DET"),
    "su": ("suyo", "DET"), "sus": ("suyo", "DET"),
    "nuestro": ("nuestro", "DET"), "nuestra": ("nuestro", "DET"),
    "nuestros": ("nuestro", "DET"), "nuestras": ("nuestro", "DET"),
    "vuestro": ("vuestro", "DET"), "vuestra": ("vuestro", "DET"),
    "este": ("este", "DET"), "esta": ("este", "DET"), "estos": ("este", "DET"), "estas": ("este", "DET"),
    "ese": ("ese", "DET"), "esa": ("ese", "DET"), "esos": ("ese", "DET"), "esas": ("ese", "DET"),
    "aquel": ("aquel", "DET"), "aquella": ("aquel", "DET"),
    "mucho": ("mucho", "DET"), "muchos": ("mucho", "DET"), "mucha": ("mucho", "DET"), "muchas": ("mucho", "DET"),
    "poco": ("poco", "DET"), "pocos": ("poco", "DET"), "poca": ("poco", "DET"), "pocas": ("poco", "DET"),
    "todo": ("todo", "DET"), "toda": ("todo", "DET"), "todos": ("todo", "DET"), "todas": ("todo", "DET"),
    "cada": ("cada", "DET"), "Cada": ("cada", "DET"),
    "otro": ("otro", "DET"), "otra": ("otro", "DET"), "otros": ("otro", "DET"), "otras": ("otro", "DET"),
    "mismo": ("mismo", "DET"), "misma": ("mismo", "DET"), "mismos": ("mismo", "DET"), "mismas": ("mismo", "DET"),
    "varios": ("vario", "DET"), "varias": ("vario", "DET"),
    "ciertos": ("cierto", "DET"), "ciertas": ("cierto", "DET"),
    "numerosos": ("numeroso", "DET"), "numerosas": ("numeroso", "DET"),
    "distintas": ("distinto", "DET"), "distintos": ("distinto", "DET"),
    "diversas": ("diverso", "DET"),
    "del": ("del", "ADP"), "al": ("al", "ADP"),
    "de": ("de", "ADP"), "en": ("en", "ADP"), "a": ("a", "ADP"), "con": ("con", "ADP"),
    "por": ("por", "ADP"), "para": ("para", "ADP"), "sin": ("sin", "ADP"), "sobre": ("sobre", "ADP"),
    "entre": ("entre", "ADP"), "hasta": ("hasta", "ADP"), "desde": ("desde", "ADP"), "según": ("según", "ADP"),
    "ante": ("ante", "ADP"), "bajo": ("bajo", "ADP"), "tras": ("tras", "ADP"),
    "y": ("y", "CCONJ"), "o": ("o", "CCONJ"), "ni": ("ni", "CCONJ"), "pero": ("pero", "CCONJ"),
    "aunque": ("aunque", "SCONJ"), "Aunque": ("aunque", "SCONJ"),
    "si": ("si", "SCONJ"), "Si": ("si", "SCONJ"),
    "que": ("que", "SCONJ"),
    "como": ("como", "SCONJ"), "cuando": ("cuando", "SCONJ"), "mientras": ("mientras", "SCONJ"),
    "porque": ("porque", "SCONJ"),
    "donde": ("donde", "ADV"), "dónde": ("donde", "ADV"), "Dónde": ("donde", "ADV"),
    "adonde": ("adonde", "ADV"), "Adónde": ("adonde", "ADV"),
    "no": ("no", "ADV"), "sí": ("sí", "ADV"),
    "muy": ("muy", "ADV"), "más": ("más", "ADV"), "menos": ("menos", "ADV"),
    "ya": ("ya", "ADV"), "aún": ("aún", "ADV"), "aun": ("aun", "ADV"),
    "solo": ("solo", "ADV"), "sólo": ("solo", "ADV"),
    "también": ("también", "ADV"), "además": ("además", "ADV"),
    "hoy": ("hoy", "ADV"), "ayer": ("ayer", "ADV"), "mañana": ("mañana", "ADV"),
    "siempre": ("siempre", "ADV"), "nunca": ("nunca", "ADV"), "jamás": ("jamás", "ADV"),
    "aquí": ("aquí", "ADV"), "allí": ("allí", "ADV"), "ahora": ("ahora", "ADV"),
    "antes": ("antes", "ADV"), "después": ("después", "ADV"),
    "bien": ("bien", "ADV"), "mal": ("mal", "ADV"),
    "cómo": ("cómo", "ADV"), "por qué": ("por qué", "ADV"),
    "Lo": ("él", "PRON"),
}

SCONJ_WORDS = {
    "que", "aunque", "si", "como", "cuando", "mientras", "porque", "pues",
    "donde", "según", "aun", "sino",
}
CCONJ_WORDS = {"y", "o", "ni", "pero", "sino"}
ADP_WORDS = {
    "de", "en", "a", "con", "por", "para", "sin", "sobre", "entre", "hasta",
    "desde", "según", "ante", "bajo", "tras", "del", "al", "hacia", "mediante",
}


IRREGULAR_PARTICIPLES = {
    "hecho": "hacer",
    "dicho": "decir",
    "visto": "ver",
    "escrito": "escribir",
    "abierto": "abrir",
    "cubierto": "cubrir",
    "muerto": "morir",
    "roto": "romper",
    "puesto": "poner",
    "vuelto": "volver",
}


def _participle_to_infinitive(lemma: str) -> str | None:
    lower = lemma.lower()
    if lower in IRREGULAR_PARTICIPLES:
        return IRREGULAR_PARTICIPLES[lower]
    for part_suffix, inf_suffix in (
        ("ados", "ar"), ("adas", "ar"), ("ado", "ar"), ("ada", "ar"),
        ("idos", "er"), ("idas", "er"), ("ido", "er"), ("ida", "er"),
        ("idos", "ir"), ("idas", "ir"), ("ido", "ir"), ("ida", "ir"),
    ):
        if lower.endswith(part_suffix) and len(lower) > len(part_suffix) + 2:
            stem = lower[: -len(part_suffix)]
            candidate = stem + inf_suffix
            if candidate.endswith(("ar", "er", "ir")):
                return candidate
    return None


def _aux_lemma(form: str) -> tuple[str, str]:
    fl = form.lower()
    if fl in SER_FORMS:
        return "ser", "AUX"
    if fl in ESTAR_FORMS:
        return "estar", "AUX"
    if fl in HABER_FORMS:
        return "haber", "AUX"
    if fl in HAY_FORMS:
        return "haber", "VERB"
    return fl, "AUX"


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    """Apply handcraft lemma/UPOS conventions per es_test.conllu."""
    if form in SPECIAL_LEMMAS:
        lemma, upos = SPECIAL_LEMMAS[form]

    fl = form.lower()

    aux_lemma, aux_upos = _aux_lemma(form)
    if fl in SER_FORMS | ESTAR_FORMS | HABER_FORMS:
        lemma, upos = aux_lemma, aux_upos
    elif fl in HAY_FORMS:
        lemma, upos = "haber", "VERB"

    if upos == "VERB":
        lemma = lemma.lower() if lemma else form.lower()
        reflexive = es_reflexive_lemma(lemma)
        if reflexive is not None:
            lemma = reflexive
        participle = _participle_to_infinitive(lemma)
        if participle is not None:
            lemma = participle
        if not any(lemma.endswith(e) for e in ("ar", "er", "ir", "se", "ír")):
            if fl.endswith(("ar", "er", "ir", "arse", "erse", "irse")):
                lemma = fl

    if upos == "NOUN" and lemma:
        lemma = lemma.lower()
        if lemma.endswith("es") and fl.endswith("es") and len(lemma) > 3:
            lemma = lemma[:-2] if lemma.endswith("iones") else lemma[:-1]
        elif lemma.endswith("s") and fl.endswith("s") and not lemma.endswith("ss"):
            lemma = lemma[:-1]

    if upos == "ADJ" and lemma:
        lemma = lemma.lower()
        if lemma.endswith("mente"):
            pass
        elif lemma.endswith("os") and fl.endswith("os"):
            lemma = lemma[:-1] + "o" if lemma.endswith("ivos") else lemma[:-1]
        elif lemma.endswith("as") and fl.endswith("as"):
            lemma = lemma[:-1] + "o"
        elif lemma.endswith("a") and fl.endswith("a") and not lemma.endswith("ía"):
            lemma = lemma[:-1] + "o"

    if upos == "PROPN" and lemma:
        lemma = lemma[0].upper() + lemma[1:] if lemma else lemma

    if upos == "PUNCT":
        lemma = form

    if fl in {"del", "al"}:
        upos, lemma = "ADP", fl

    if fl in SCONJ_WORDS and not (fl == "como" and upos == "ADP"):
        if fl != "donde" or upos != "ADV":
            upos = "SCONJ"
            lemma = fl
    elif fl in CCONJ_WORDS:
        upos, lemma = "CCONJ", fl
    elif fl in ADP_WORDS:
        upos, lemma = "ADP", fl if fl in {"del", "al"} else fl

    if fl == "se" and upos in {"PRON", "PART"}:
        upos, lemma = "PRON", "él"

    if form in SPECIAL_LEMMAS:
        lemma, upos = SPECIAL_LEMMAS[form]

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


def count_tokens(sentence: str) -> int:
    return len(re.findall(r"[\w']+|[^\w\s]", sentence))


def main() -> None:
    import stanza

    for i, sent in enumerate(SENTENCES, START_ID):
        tc = count_tokens(sent)
        if tc < MIN_TOKENS or tc > MAX_TOKENS:
            print(f"PRE-CHECK es_b1_train_{i:03d}: {tc} tokens — {sent}")
            sys.exit(1)

    print("Loading Stanza...")
    stanza.download("es", processors="tokenize,mwt,pos,lemma", verbose=False)
    nlp = stanza.Pipeline(
        lang="es",
        processors="tokenize,mwt,pos,lemma",
        use_gpu=False,
        verbose=False,
    )

    target_path = project_root / "data/handcraft/es/train/b1_new_003.conllu"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    all_lines: list[str] = []
    global_idx = 0

    for batch_num, batch in enumerate(SENTENCE_BATCHES):
        assert len(batch) == BATCH_SIZE, f"Batch {batch_num} has {len(batch)} sentences"
        print(
            f"Processing batch {batch_num + 1}/8 "
            f"(sentences {START_ID + global_idx}–{START_ID + global_idx + BATCH_SIZE - 1})"
        )
        for sent in batch:
            sent_id = f"es_b1_train_{START_ID + global_idx:03d}"
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

            all_lines.append(f"# sent_id = {sent_id}")
            all_lines.append(f"# text = {_reconstruct_text(sent_forms)}")
            all_lines.extend(sent_rows)
            all_lines.append("")
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

    lemma_res = check_text(conllu_text, lang="es")
    print(f"Lemma check: passed={lemma_res.passed}")
    if not lemma_res.passed:
        for err in lemma_res.errors:
            print(f"  {err}")
        sys.exit(1)

    print("All checks passed.")


if __name__ == "__main__":
    main()