from pathlib import Path


def read_conllu(path: str, lang: str):
    sentences = []
    words = []
    lemmas = []
    upos = []

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line:
            if words:
                sentences.append(
                    {
                        "lang": lang,
                        "words": words,
                        "lemmas": lemmas,
                        "upos": upos,
                    }
                )
                words = []
                lemmas = []
                upos = []
            continue

        if line.startswith("#"):
            continue

        cols = line.split("\t")
        if len(cols) < 4:
            continue

        token_id = cols[0]

        if "-" in token_id:
            continue

        if "." in token_id:
            continue

        form = cols[1]
        lemma = cols[2]
        pos = cols[3]

        if not form or not lemma or lemma == "_":
            continue

        words.append(form)
        lemmas.append(lemma)
        upos.append(pos)

    if words:
        sentences.append(
            {
                "lang": lang,
                "words": words,
                "lemmas": lemmas,
                "upos": upos,
            }
        )

    return sentences
