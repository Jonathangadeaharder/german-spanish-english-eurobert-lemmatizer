"""Annotate subtitle sentences with stanza/spaCy → CoNLL-U.

For sv/ar/zh: uses stanza (Stanford NLP) which outputs native
CoNLL-U with UPOS + lemma + dependencies.
For de/en/es/fr/nl: uses spaCy large models.

Quality filters: length, garble detection, deduplication.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from lemmatizer.data.vtt_parser import load_subtitle_sentences

# Languages and their annotator backends.
STANZA_LANGS = {"sv", "ar", "zh"}
SPACY_LANGS = {"de", "en", "es", "fr", "nl"}

# Quality thresholds.
MIN_TOKENS = 3
MAX_TOKENS = 50
MIN_ALPHA_RATIO = 0.4  # At least 40% alphanumeric tokens.

# CoNLL-U column indices.
COL_FORM = 1
COL_LEMMA = 2
COL_UPOS = 3


def _token_count(text: str) -> int:
    # For CJK text (no spaces), count characters as tokens.
    if not text:
        return 0
    words = text.split()
    if len(words) <= 1 and any("\u4e00" <= c <= "\u9fff" for c in text):
        return len(text)
    return len(words)


def _alpha_ratio(text: str) -> float:
    total = len(text)
    if total == 0:
        return 0.0
    # CJK characters are alphanumeric in Python's isalnum().
    alpha = sum(c.isalnum() or c.isspace() for c in text)
    return alpha / total


def filter_sentences(sentences: list[str]) -> list[str]:
    """Remove sentences that are too short, too long, or garbled."""
    seen: set[str] = set()
    result: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        tc = _token_count(s)
        if tc < MIN_TOKENS or tc > MAX_TOKENS:
            continue
        if _alpha_ratio(s) < MIN_ALPHA_RATIO:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(s)
    return result


def annotate_with_stanza(sentences: list[str], lang: str, nlp=None) -> list[str]:
    """Annotate sentences with stanza → CoNLL-U strings.

    Returns a list of CoNLL-U sentence blocks (one per input sentence
    that was successfully annotated).
    """
    if nlp is None:
        import stanza

        try:
            # mwt is not available for all languages; use only
            # tokenize,pos,lemma (lemma requires pos).
            nlp = stanza.Pipeline(
                lang=lang,
                processors="tokenize,pos,lemma",
                verbose=False,
                use_gpu=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load stanza pipeline for {lang}: {exc}") from exc

    results: list[str] = []
    for i, sent in enumerate(sentences):
        try:
            doc = nlp(sent)
            for j, sentence in enumerate(doc.sentences):
                lines: list[str] = []
                lines.append(f"# sent_id = subtitle-{lang}-{i:05d}-{j}")
                lines.append(f"# text = {sent}")
                for word in sentence.words:
                    form = word.text
                    lemma = word.lemma or form
                    upos = word.upos or "X"
                    tid = word.id
                    head = word.head if word.head else "0"
                    deprel = word.deprel or "_"
                    # 10-column CoNLL-U.
                    cols = [
                        str(tid),
                        form,
                        lemma,
                        upos,
                        "_",
                        "_",
                        str(head),
                        deprel,
                        "_",
                        "_",
                    ]
                    lines.append("\t".join(cols))
                lines.append("")
                results.append("\n".join(lines))
        except Exception as exc:
            print(
                f"  stanza: failed sentence {i}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
    return results


def annotate_with_spacy(sentences: list[str], lang: str, nlp=None) -> list[str]:
    """Annotate sentences with spaCy → CoNLL-U strings."""
    spacy_models = {
        "de": "de_core_news_lg",
        "en": "en_core_web_lg",
        "es": "es_core_news_lg",
        "fr": "fr_core_news_lg",
        "nl": "nl_core_news_lg",
    }
    if nlp is None:
        import spacy

        model = spacy_models[lang]
        try:
            nlp = spacy.load(model)
        except Exception as exc:
            raise RuntimeError(f"Failed to load spaCy model {model}: {exc}") from exc

    results: list[str] = []
    # Batch-process for efficiency on large subtitle datasets.
    docs = list(nlp.pipe(sentences))
    for i, (sent, doc) in enumerate(zip(sentences, docs, strict=True)):
        try:
            lines: list[str] = []
            lines.append(f"# sent_id = subtitle-{lang}-{i:05d}")
            lines.append(f"# text = {sent}")
            for k, token in enumerate(doc, start=1):
                form = token.text
                lemma = token.lemma_ or form
                upos = token.pos_ or "X"
                # token.head.i is absolute (0-based) within the doc;
                # the doc is a single sentence, so +1 gives the
                # 1-based head index, root → 0.
                if token.head == token:
                    head = "0"
                else:
                    head = str(token.head.i + 1)
                deprel = token.dep_ or "_"
                cols = [
                    str(k),
                    form,
                    lemma,
                    upos,
                    "_",
                    "_",
                    head,
                    deprel,
                    "_",
                    "_",
                ]
                lines.append("\t".join(cols))
            lines.append("")
            results.append("\n".join(lines))
        except Exception as exc:
            print(
                f"  spacy: failed sentence {i}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
    return results


def process_language(
    lang: str,
    subtitle_paths: list[Path],
    output_path: Path,
    nlp=None,
) -> int:
    """Full pipeline: parse VTTs → filter → annotate → write CoNLL-U.

    Returns the number of sentences written.
    """
    all_sentences: list[str] = []
    for p in subtitle_paths:
        if p.exists():
            all_sentences.extend(load_subtitle_sentences(p))

    filtered = filter_sentences(all_sentences)
    print(
        f"  {lang}: {len(all_sentences)} raw → {len(filtered)} after filter",
        flush=True,
    )
    if not filtered:
        return 0

    if lang in STANZA_LANGS:
        annotated = annotate_with_stanza(filtered, lang, nlp=nlp)
    else:
        annotated = annotate_with_spacy(filtered, lang, nlp=nlp)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(annotated), encoding="utf-8")
    print(f"  {lang}: wrote {len(annotated)} sentences to {output_path}")
    return len(annotated)


if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else "sv"
    sub_dir_env = os.getenv("SUBTITLE_DIR")
    if not sub_dir_env:
        print(
            "Error: SUBTITLE_DIR env var must point to the subtitle directory.",
            file=sys.stderr,
        )
        sys.exit(1)
    sub_dir = Path(sub_dir_env)
    gold_dir = Path("data/gold")

    # Map language to subtitle files (VTT + SRT).
    vtt_map = {
        "sv": [
            "taskmaster_sweden.vtt",
            "Midnight_Sun.srt",
            "False_Trail.srt",
        ],
        "ar": [
            "lmaktoub.vtt",
            "Ramy_Youssef__Feelings.srt",
            "Amira___Sam.srt",
            "_Pillar_of_Fire__The_Jew_Returns_-_The_Arab_Awakens_1895-1920.srt",
            "The_Purple_Rose_of_Cairo.srt",
        ],
        "zh": [
            "Crouching_Tiger__Hidden_Dragon.srt",
        ],
        "de": ["nicos_weg.vtt", "jojo_sucht_das_glueck.vtt"],
        "en": ["extra_english.vtt"],
        "es": ["extra_spanish.vtt"],
        "fr": ["extra_french.vtt"],
        "nl": ["extra_dutch.vtt"],
    }

    files = [sub_dir / f for f in vtt_map.get(lang, [])]
    output = gold_dir / lang / "subtitle_sentences.conllu"
    count = process_language(lang, files, output)
    print(f"Done: {count} sentences for {lang}")
