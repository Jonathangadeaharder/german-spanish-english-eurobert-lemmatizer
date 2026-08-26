from pathlib import Path


def _make_sentence(lang: str, words: list[str], lemmas: list[str], upos: list[str]) -> dict:
    return {"lang": lang, "words": words, "lemmas": lemmas, "upos": upos}


def _parse_token_line(line: str) -> tuple[str, str, str] | None:
    """Parse a token line, returning (form, lemma, pos) or None to skip."""
    cols = line.split("\t")
    if len(cols) < 4:
        return None
    token_id = cols[0]
    if "-" in token_id or "." in token_id:
        return None
    form, lemma, pos = cols[1], cols[2], cols[3]
    if not form or not lemma or lemma == "_":
        return None
    return form, lemma, pos


def read_conllu(path: str, lang: str):
    sentences = []
    words = []
    lemmas = []
    upos = []

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line:
            if words:
                sentences.append(_make_sentence(lang, words, lemmas, upos))
                words = []
                lemmas = []
                upos = []
            continue

        if line.startswith("#"):
            continue

        parsed = _parse_token_line(line)
        if parsed is None:
            continue

        form, lemma, pos = parsed
        words.append(form)
        lemmas.append(lemma)
        upos.append(pos)

    if words:
        sentences.append(_make_sentence(lang, words, lemmas, upos))

    return sentences
