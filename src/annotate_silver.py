import json
from collections import Counter
from pathlib import Path

import spacy

from edit_trees import EXCLUDE_UPOS, apply_edit_label, make_edit_label
from language_assets import SPACY_MODELS, normalize_lang

UD_UPOS = {
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
}

MIN_SENTENCE_LENGTH = 2
MAX_SENTENCE_LENGTH = 128


def load_gold_lexicon(lang: str) -> dict[str, dict[str, str]]:
    path = Path(f"artifacts/gold_lexicon_{lang}.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_gold_lexicon(lang: str) -> dict[str, dict[str, str]]:
    from conllu_reader import read_conllu

    lexicon: dict[str, dict[str, str]] = {}
    for split in ("train", "dev"):
        path = Path(f"data/gold/{lang}/{split}.conllu")
        if not path.exists():
            continue
        for sent in read_conllu(str(path), lang=lang):
            for w, lem, u in zip(sent["words"], sent["lemmas"], sent["upos"], strict=True):
                if u in EXCLUDE_UPOS:
                    continue
                entry = lexicon.setdefault(w, {})
                if u not in entry:
                    entry[u] = lem
    return lexicon


def annotate_sentence(nlp, text: str) -> dict[str, object] | None:
    doc = nlp(text.strip())
    words = [token.text for token in doc if not token.is_space]
    lemmas = [token.lemma_ for token in doc if not token.is_space]
    upos = [token.pos_ for token in doc if not token.is_space]

    if not words or len(words) != len(lemmas) or len(words) != len(upos):
        return None

    return {
        "text": text.strip(),
        "words": words,
        "lemmas": lemmas,
        "upos": upos,
    }


def validate_and_correct(
    ann: dict,
    gold_lexicon: dict[str, dict[str, str]],
    lang: str,
) -> tuple[dict, list[str]]:
    words = ann["words"]
    lemmas = list(ann["lemmas"])
    upos_tags = ann["upos"]
    issues: list[str] = []

    if len(words) != len(lemmas) or len(words) != len(upos_tags):
        issues.append("length_mismatch")
        return ann, issues

    if len(words) < MIN_SENTENCE_LENGTH:
        issues.append("too_short")
        return ann, issues

    if len(words) > MAX_SENTENCE_LENGTH:
        issues.append("too_long")
        return ann, issues

    corrections = 0
    for i, (w, u) in enumerate(zip(words, upos_tags, strict=True)):
        if not w.strip():
            issues.append(f"empty_word@{i}")
            continue

        if u not in UD_UPOS:
            issues.append(f"invalid_upos@{i}:{u}")
            continue

        if not lemmas[i].strip():
            issues.append(f"empty_lemma@{i}")
            continue

        gold_entry = gold_lexicon.get(w)
        if gold_entry:
            gold_lemma = gold_entry.get(u)
            if gold_lemma is None and len(gold_entry) == 1:
                gold_lemma = next(iter(gold_entry.values()))
            if gold_lemma is not None and gold_lemma != lemmas[i]:
                issues.append(
                    f"lemma_disagree@{i}: word={w} spacy={lemmas[i]} gold={gold_lemma} upos={u}"
                )
                lemmas[i] = gold_lemma
                corrections += 1

    for i, (w, lem, u) in enumerate(zip(words, lemmas, upos_tags, strict=True)):
        if u in EXCLUDE_UPOS:
            continue
        label = make_edit_label(w, lem, u)
        reconstructed = apply_edit_label(w, label)
        if reconstructed != lem:
            issues.append(
                f"roundtrip_fail@{i}: word={w} lemma={lem} "
                f"label={label} reconstructed={reconstructed}"
            )

    corrected = dict(ann)
    corrected["lemmas"] = lemmas
    if corrections:
        corrected["gold_corrections"] = corrections

    return corrected, issues


def main():
    lang = normalize_lang()
    model_name = SPACY_MODELS.get(lang)
    if not model_name:
        print(f"No spaCy model for language '{lang}'")
        return

    print(f"Loading spaCy model: {model_name}")
    nlp = spacy.load(model_name)

    gold_lexicon_path = Path(f"artifacts/gold_lexicon_{lang}.json")
    if gold_lexicon_path.exists():
        print(f"Loading gold lexicon from {gold_lexicon_path}")
        gold_lexicon = json.loads(gold_lexicon_path.read_text(encoding="utf-8"))
    else:
        print("Building gold lexicon from UD treebanks ...")
        gold_lexicon = build_gold_lexicon(lang)
        gold_lexicon_path.parent.mkdir(parents=True, exist_ok=True)
        gold_lexicon_path.write_text(json.dumps(gold_lexicon, ensure_ascii=False), encoding="utf-8")
        print(f"Saved gold lexicon ({len(gold_lexicon)} entries) to {gold_lexicon_path}")

    raw_paths: list[Path] = []

    existing_path = Path(f"data/silver/{lang}/{lang}_lemma_silver.jsonl")
    if existing_path.exists():
        raw_paths.append(existing_path)

    generic_path = Path(f"data/silver/lemma_{lang}_raw.jsonl")
    if generic_path.exists():
        raw_paths.append(generic_path)

    targeted_path = Path(f"data/silver/targeted/targeted_{lang}_raw.jsonl")
    if targeted_path.exists():
        raw_paths.append(targeted_path)

    if not raw_paths:
        print(f"No raw silver files found for language '{lang}'")
        return

    output_dir = Path("data/silver/annotated")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{lang}_silver_annotated.jsonl"
    report_path = output_dir / f"{lang}_validation_report.json"

    total_rows = 0
    total_annotated = 0
    total_skipped = 0
    total_corrections = 0
    issue_categories: Counter = Counter()
    all_issues: list[str] = []
    lemma_disagree_samples: list[str] = []
    roundtrip_fail_samples: list[str] = []

    with output_path.open("w", encoding="utf-8") as out:
        for raw_path in raw_paths:
            print(f"Processing {raw_path} ...")
            with raw_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "annotated_sentences" in row and row["annotated_sentences"]:
                        for ann in row["annotated_sentences"]:
                            if (
                                len(ann.get("words", []))
                                == len(ann.get("lemmas", []))
                                == len(ann.get("upos", []))
                            ):
                                corrected, issues = validate_and_correct(ann, gold_lexicon, lang)
                                for iss in issues:
                                    cat = iss.split("@")[0]
                                    issue_categories[cat] += 1
                                    all_issues.append(iss)
                                    if cat == "lemma_disagree" and len(lemma_disagree_samples) < 50:
                                        lemma_disagree_samples.append(iss)
                                    if cat == "roundtrip_fail" and len(roundtrip_fail_samples) < 50:
                                        roundtrip_fail_samples.append(iss)
                                if "length_mismatch" in issues or "too_short" in issues:
                                    total_skipped += 1
                                    continue
                                corrections = corrected.get("gold_corrections", 0)
                                total_corrections += corrections
                                out.write(json.dumps(corrected, ensure_ascii=False) + "\n")
                                total_annotated += 1
                        total_rows += 1
                        continue

                    sentences = row.get("sentences", [])
                    if not sentences:
                        continue

                    for sentence in sentences:
                        ann = annotate_sentence(nlp, sentence)
                        if ann is None:
                            total_skipped += 1
                            continue
                        corrected, issues = validate_and_correct(ann, gold_lexicon, lang)
                        for iss in issues:
                            cat = iss.split("@")[0]
                            issue_categories[cat] += 1
                            all_issues.append(iss)
                            if cat == "lemma_disagree" and len(lemma_disagree_samples) < 50:
                                lemma_disagree_samples.append(iss)
                            if cat == "roundtrip_fail" and len(roundtrip_fail_samples) < 50:
                                roundtrip_fail_samples.append(iss)
                        if "length_mismatch" in issues or "too_short" in issues:
                            total_skipped += 1
                            continue
                        corrections = corrected.get("gold_corrections", 0)
                        total_corrections += corrections
                        out.write(json.dumps(corrected, ensure_ascii=False) + "\n")
                        total_annotated += 1

                    total_rows += 1

    report = {
        "lang": lang,
        "total_raw_rows": total_rows,
        "total_annotated": total_annotated,
        "total_skipped": total_skipped,
        "total_gold_corrections": total_corrections,
        "issue_categories": dict(issue_categories.most_common()),
        "lemma_disagree_samples": lemma_disagree_samples[:30],
        "roundtrip_fail_samples": roundtrip_fail_samples[:30],
        "pass_rate": round(total_annotated / max(1, total_annotated + total_skipped), 4),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Validation Report ({lang}) ===")
    print(f"Annotated: {total_annotated}")
    print(f"Skipped:   {total_skipped}")
    print(f"Gold corrections: {total_corrections}")
    print(f"Pass rate: {report['pass_rate']:.4f}")
    print(f"Issue categories: {dict(issue_categories.most_common())}")
    print(f"Full report: {report_path}")
    print(f"\nWrote {total_annotated} annotated sentences from {total_rows} rows to {output_path}")


if __name__ == "__main__":
    main()
