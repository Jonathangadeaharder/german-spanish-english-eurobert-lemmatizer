import json
from collections import Counter, defaultdict
from pathlib import Path

from conllu_reader import read_conllu
from edit_trees import apply_edit_label, make_edit_label

MIN_LABEL_COUNT = 2
TOP_TREE_COUNT = 300

LANGS = ["de", "es", "en"]

INPUT_FILES = {
    "de": [
        "data/gold/de/train.conllu",
        "data/gold/de/dev.conllu",
    ],
    "es": [
        "data/gold/es/train.conllu",
        "data/gold/es/dev.conllu",
    ],
    "en": [
        "data/gold/en/train.conllu",
        "data/gold/en/dev.conllu",
    ],
}

OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    counter = Counter()
    examples = {}
    lexicon_counts = {lang: defaultdict(Counter) for lang in LANGS}
    upos_counts = Counter()
    exceptions = []

    for lang in LANGS:
        for path in INPUT_FILES[lang]:
            sentences = read_conllu(path, lang=lang)

            for sent in sentences:
                for word, lemma, upos in zip(
                    sent["words"], sent["lemmas"], sent["upos"], strict=True
                ):
                    if upos and upos != "_":
                        upos_counts[upos] += 1

                    lexicon_counts[lang][word][lemma] += 1

                    base_label = make_edit_label(word, lemma)
                    reconstructed = apply_edit_label(word, base_label)

                    if reconstructed != lemma:
                        exceptions.append(
                            {
                                "lang": lang,
                                "word": word,
                                "lemma": lemma,
                                "base_label": base_label,
                                "reason": "reconstruction_failed",
                            }
                        )
                        continue

                    full_label = f"{lang}::{base_label}"

                    counter[full_label] += 1

                    if full_label not in examples:
                        examples[full_label] = {
                            "lang": lang,
                            "word": word,
                            "lemma": lemma,
                            "base_label": base_label,
                        }

    labels = ["UNKNOWN"]

    for label, count in counter.most_common():
        if count >= MIN_LABEL_COUNT:
            labels.append(label)

    for lang in LANGS:
        for base in ["IDENTITY", "LOWERCASE"]:
            label = f"{lang}::{base}"
            if label not in labels:
                labels.append(label)

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {str(i): label for label, i in label2id.items()}
    upos_labels = sorted(upos_counts)
    upos_label2id = {label: i for i, label in enumerate(upos_labels)}
    upos_id2label = {str(i): label for label, i in upos_label2id.items()}

    edit_trees = {
        label: {
            "label": label,
            "count": counter.get(label, 0),
            "example": examples.get(label),
        }
        for label in labels
    }

    lexicon = {}

    for lang in LANGS:
        lang_lexicon = {}

        for word, lemma_counts in lexicon_counts[lang].items():
            winner = sorted(
                lemma_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[0][0]
            lang_lexicon[word] = winner

        lexicon[lang] = lang_lexicon

    (OUT_DIR / "label2id.json").write_text(
        json.dumps(label2id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUT_DIR / "id2label.json").write_text(
        json.dumps(id2label, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUT_DIR / "upos_label2id.json").write_text(
        json.dumps(upos_label2id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUT_DIR / "upos_id2label.json").write_text(
        json.dumps(upos_id2label, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUT_DIR / "edit_trees.json").write_text(
        json.dumps(edit_trees, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUT_DIR / "lexicon.json").write_text(
        json.dumps(lexicon, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUT_DIR / "exceptions.json").write_text(
        json.dumps(exceptions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    top_trees = []
    for label, _count in counter.most_common():
        if label in labels and label != "UNKNOWN":
            top_trees.append(label)
        if len(top_trees) >= TOP_TREE_COUNT:
            break

    reduced_labels = ["UNKNOWN"] + top_trees
    for lang in LANGS:
        for base in ["IDENTITY", "LOWERCASE"]:
            label = f"{lang}::{base}"
            if label not in reduced_labels:
                reduced_labels.append(label)

    reduced_label2id = {label: i for i, label in enumerate(reduced_labels)}
    reduced_id2label = {str(i): label for label, i in reduced_label2id.items()}

    (OUT_DIR / "label2id_top300.json").write_text(
        json.dumps(reduced_label2id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUT_DIR / "id2label_top300.json").write_text(
        json.dumps(reduced_id2label, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUT_DIR / "top_edit_trees.json").write_text(
        json.dumps(top_trees, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Created {len(labels)} multilingual labels")
    print(f"Created {len(reduced_labels)} reduced labels (top {TOP_TREE_COUNT})")
    print(f"Created {len(upos_labels)} UPOS labels")
    print("Top labels:")
    for label, count in counter.most_common(30):
        print(count, label)


if __name__ == "__main__":
    main()
