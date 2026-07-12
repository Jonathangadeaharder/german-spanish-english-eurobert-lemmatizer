"""Generate all Spanish validation CoNLL-U files (es_*_val_001–100 × 6 levels)."""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lemmatizer.data.conllu_validator import validate_file, validate_text
from lemmatizer.data.duplicate_detector import check_cross_file, check_text as check_dup_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import es_reflexive_lemma

STANZA_DIR = project_root / ".stanza_resources"

LEVELS: dict[str, dict] = {
    "a1": {"min": 3, "max": 8, "module": "a1_val_sentences", "out": "a1.conllu"},
    "a2": {"min": 5, "max": 12, "module": "a2_val_sentences", "out": "a2.conllu"},
    "b1": {"min": 8, "max": 15, "module": "b1_val_sentences", "out": "b1.conllu"},
    "b2": {"min": 10, "max": 18, "module": "b2_val_sentences", "out": "b2.conllu"},
    "c1": {"min": 12, "max": 20, "module": "c1_val_sentences", "out": "c1.conllu"},
    "c2": {"min": 15, "max": 28, "module": "c2_val_sentences", "out": "c2.conllu"},
}

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
MODAL_LEMMAS = {"poder", "deber", "querer", "soler", "saber"}

SPECIAL_LEMMAS: dict[str, tuple[str, str]] = {
    "yo": ("yo", "PRON"), "Yo": ("yo", "PRON"),
    "tú": ("tú", "PRON"), "Tú": ("tú", "PRON"),
    "él": ("él", "PRON"), "Él": ("él", "PRON"),
    "ella": ("él", "PRON"), "Ella": ("él", "PRON"),
    "ellos": ("él", "PRON"), "ellas": ("él", "PRON"),
    "me": ("yo", "PRON"), "Me": ("yo", "PRON"),
    "te": ("tú", "PRON"), "ti": ("tú", "PRON"),
    "se": ("él", "PRON"), "Se": ("él", "PRON"),
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
    "ante": ("ante", "ADP"), "bajo": ("bajo", "ADP"), "tras": ("tras", "ADP"), "hacia": ("hacia", "ADP"),
    "y": ("y", "CCONJ"), "o": ("o", "CCONJ"), "ni": ("ni", "CCONJ"), "pero": ("pero", "CCONJ"),
    "aunque": ("aunque", "SCONJ"), "Aunque": ("aunque", "SCONJ"),
    "si": ("si", "SCONJ"), "Si": ("si", "SCONJ"),
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
    "cómo": ("cómo", "ADV"), "Cómo": ("cómo", "ADV"),
    "cuándo": ("cuando", "ADV"), "Cuándo": ("cuando", "ADV"),
    "cómo": ("cómo", "ADV"),
    "¿": ("¿", "PUNCT"), "?": ("?", "PUNCT"), ".": (".", "PUNCT"), ",": (",", "PUNCT"),
    "¡": ("¡", "PUNCT"), "!": ("!", "PUNCT"), ";": (";", "PUNCT"), ":": (":", "PUNCT"),
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

_PARTICIPLE_FIXES: dict[str, str] = {
    "expandido": "expandir", "expandida": "expandir",
    "exigido": "exigir", "exigida": "exigir",
    "descrito": "describir", "descrita": "describir",
    "continuado": "continuar", "continuada": "continuar",
    "publicado": "publicar", "publicada": "publicar",
    "documentado": "documentar", "documentada": "documentar",
    "implementado": "implementar", "implementada": "implementar",
    "distorsionado": "distorsionar", "distorsionada": "distorsionar",
}

VERB_LEMMA_MAP: dict[str, str] = {
    "oigo": "oir", "oímos": "oir", "oír": "oir", "oye": "oir", "oyen": "oir",
    "envío": "enviar", "envio": "enviar", "envía": "enviar", "enviamos": "enviar",
    "leo": "leer", "lee": "leer", "leemos": "leer", "leen": "leer", "lees": "leer",
    "veo": "ver", "ve": "ver", "vemos": "ver", "ven": "ver", "ves": "ver",
    "doy": "dar", "da": "dar", "damos": "dar", "dan": "dar",
    "voy": "ir", "va": "ir", "vamos": "ir", "van": "ir", "vas": "ir",
    "soy": "ser", "es": "ser", "son": "ser", "eres": "ser", "somos": "ser",
    "estoy": "estar", "está": "estar", "están": "estar", "estamos": "estar",
    "tengo": "tener", "tiene": "tener", "tenemos": "tener", "tienen": "tener",
    "hago": "hacer", "hace": "hacer", "hacemos": "hacer", "hacen": "hacer",
    "pongo": "poner", "pone": "poner", "ponemos": "poner",
    "digo": "decir", "dice": "decir", "decimos": "decir", "dicen": "decir",
    "puedo": "poder", "puede": "poder", "podemos": "poder", "pueden": "poder",
    "quiero": "querer", "quiere": "querer", "queremos": "querer", "quieren": "querer",
    "sé": "saber", "sabe": "saber", "sabemos": "saber", "saben": "saber",
    "vengo": "venir", "viene": "venir", "venimos": "venir", "vienen": "venir",
    "salgo": "salir", "sale": "salir", "salimos": "salir", "salen": "salir",
    "juego": "jugar", "juega": "jugar", "jugamos": "jugar", "juegan": "jugar",
    "pienso": "pensar", "piensa": "pensar", "pensamos": "pensar",
    "encuentro": "encontrar", "encuentra": "encontrar", "encontramos": "encontrar",
    "como": "comer", "come": "comer", "comemos": "comer", "comen": "comer", "comes": "comer",
    "compro": "comprar", "compra": "comprar", "compramos": "comprar",
    "cocino": "cocinar", "cocina": "cocinar", "cocinamos": "cocinar",
    "monto": "montar", "monta": "montar", "montamos": "montar",
    "empaco": "empacar", "empaca": "empacar",
    "hiervo": "hervir", "hierve": "hervir",
    "limpia": "limpiar", "limpio": "limpiar", "limpiamos": "limpiar",
    "oriente": "orientar", "orienta": "orientar", "orientan": "orientar",
    "recontextualizan": "recontextualizar", "recontextualiza": "recontextualizar",
    "recontextual": "recontextualizar",
}


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

    if fl in SER_FORMS | ESTAR_FORMS | HABER_FORMS:
        lemma, upos = _aux_lemma(form)
    elif fl in HAY_FORMS:
        lemma, upos = "haber", "VERB"

    if upos == "VERB":
        lemma = lemma.lower() if lemma else form.lower()
        if form in VERB_LEMMA_MAP:
            lemma = VERB_LEMMA_MAP[form]
        elif fl in VERB_LEMMA_MAP:
            lemma = VERB_LEMMA_MAP[fl]
        elif lemma in VERB_LEMMA_MAP:
            lemma = VERB_LEMMA_MAP[lemma]
        if lemma in _PARTICIPLE_FIXES:
            lemma = _PARTICIPLE_FIXES[lemma]
        if fl in _PARTICIPLE_FIXES:
            lemma = _PARTICIPLE_FIXES[fl]
        reflexive = es_reflexive_lemma(lemma)
        if reflexive is not None:
            lemma = reflexive
        if not any(lemma.endswith(e) for e in ("ar", "er", "ir", "se", "ír")):
            if fl.endswith(("ar", "er", "ir", "arse", "erse", "irse")):
                lemma = fl
            elif form in VERB_LEMMA_MAP:
                lemma = VERB_LEMMA_MAP[form]
            elif fl in VERB_LEMMA_MAP:
                lemma = VERB_LEMMA_MAP[fl]

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

    if upos == "AUX" and lemma in MODAL_LEMMAS:
        upos = "VERB"

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
    tokens: list[str] = []
    for word in sentence.split():
        word_work = word
        while word_work and word_work[0] in "¿¡":
            tokens.append(word_work[0])
            word_work = word_work[1:]
        if not word_work:
            continue
        match = re.match(r"^(.+?)([.,;:!?]+)$", word_work)
        if match:
            tokens.append(match.group(1))
            tokens.extend(list(match.group(2)))
        else:
            tokens.append(word_work)
    return len(tokens)


def sentence_to_conllu(sent_id: str, sent: str, nlp) -> list[str]:
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
                str(token_counter), form, lemma, upos,
                "_", "_", "_", "_", "_", "_",
            ]
            sent_rows.append("\t".join(cols))
            sent_forms.append(form)
            token_counter += 1

    lines = [f"# sent_id = {sent_id}", f"# text = {_reconstruct_text(sent_forms)}"]
    lines.extend(sent_rows)
    lines.append("")
    return lines


def load_sentences(module_name: str) -> list[str]:
    mod = importlib.import_module(module_name)
    return list(mod.SENTENCES)


def generate_level(level: str, cfg: dict, nlp) -> tuple[Path, str]:
    sentences = load_sentences(cfg["module"])
    assert len(sentences) == 100, f"{level}: expected 100 sentences, got {len(sentences)}"

    for i, sent in enumerate(sentences, 1):
        tc = count_tokens(sent)
        if tc < cfg["min"] or tc > cfg["max"]:
            raise ValueError(
                f"{level} sentence {i:03d}: {tc} tokens (need {cfg['min']}–{cfg['max']}): {sent}"
            )

    out_path = Path(__file__).resolve().parent / cfg["out"]
    all_lines: list[str] = []

    for i, sent in enumerate(sentences, 1):
        sent_id = f"es_{level}_val_{i:03d}"
        all_lines.extend(sentence_to_conllu(sent_id, sent, nlp))

    conllu_text = "\n".join(all_lines) + "\n"
    out_path.write_text(conllu_text, encoding="utf-8")
    return out_path, conllu_text


def main() -> None:
    import stanza

    os.environ["STANZA_RESOURCES_DIR"] = str(STANZA_DIR)
    STANZA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Stanza (es)...")
    stanza.download(
        "es",
        package="default_fast",
        processors="tokenize,mwt,pos,lemma",
        verbose=False,
    )
    nlp = stanza.Pipeline(
        lang="es",
        package="default_fast",
        processors="tokenize,mwt,pos,lemma",
        use_gpu=False,
        verbose=False,
        dir=str(STANZA_DIR),
    )

    train_dir = project_root / "data/handcraft/es/train"
    train_text = ""
    for path in sorted(train_dir.glob("*_new_*.conllu")):
        train_text += path.read_text(encoding="utf-8") + "\n"

    results: list[dict] = []

    for level, cfg in LEVELS.items():
        print(f"\n=== Generating {level.upper()} validation ===")
        out_path, conllu_text = generate_level(level, cfg, nlp)
        print(f"Wrote {out_path}")

        val_res = validate_text(conllu_text)
        print(
            f"Validate: sentences={val_res.sentence_count}, "
            f"tokens={val_res.token_count}, passed={val_res.passed}"
        )
        if not val_res.passed:
            for err in val_res.errors[:10]:
                print(f"  {err}")
            sys.exit(1)

        lemma_res = check_text(conllu_text, lang="es")
        print(f"Lemma check: passed={lemma_res.passed}")
        if not lemma_res.passed:
            for err in lemma_res.errors[:10]:
                print(f"  {err}")
            sys.exit(1)

        dup_res = check_dup_text(conllu_text)
        print(f"Duplicate check (within file): passed={dup_res.passed}")
        if not dup_res.passed:
            for err in dup_res.duplicates[:5]:
                print(f"  {err}")
            sys.exit(1)

        cross_dup = check_cross_file(train_text, conllu_text)
        print(f"Duplicate check (vs train): passed={cross_dup.passed}")
        if not cross_dup.passed:
            for err in cross_dup.duplicates[:5]:
                print(f"  {err}")
            sys.exit(1)
        for warn in cross_dup.warnings[:3]:
            print(f"  WARNING: {warn}")

        results.append({
            "level": level,
            "file": str(out_path.name),
            "sentences": val_res.sentence_count,
            "tokens": val_res.token_count,
        })

    print("\n=== SUMMARY ===")
    total_sents = 0
    total_tokens = 0
    for r in results:
        print(f"  {r['file']}: {r['sentences']} sentences, {r['tokens']} tokens")
        total_sents += r["sentences"]
        total_tokens += r["tokens"]
    print(f"  TOTAL: {total_sents} sentences, {total_tokens} tokens")
    print("All checks passed.")


if __name__ == "__main__":
    main()