"""Build the Chinese exception lexicon from gold treebank data.

Chinese lemma = surface form for 99.33% of gold test tokens. The only
deviations are closed-class: 们-pronouns (他们→他), fused adverb/negation+
verb (不是→是, 都是→是). Neural lemmatization is the wrong tool — identity
plus a small exception lexicon solves lemma to ~99.3%.

Usage:
    uv run python -m lemmatizer.data.zh_lexicon
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from lemmatizer.data.conllu import read_conllu

GOLD_TRAIN = "data/gold/zh/train.conllu"
OUT_PATH = "artifacts/lemma_zh/exceptions.json"


def build_exceptions(conllu_path: str = GOLD_TRAIN) -> dict[str, str]:
    """Scan train.conllu for (form, lemma) where form != lemma.

    Returns a dict mapping form → lemma (majority vote for ambiguous forms).
    """
    form_lemma_counts: dict[str, Counter] = defaultdict(Counter)
    total_tokens = 0
    identity_tokens = 0

    for sent in read_conllu(conllu_path, lang="zh"):
        for form, lemma in zip(sent["words"], sent["lemmas"], strict=True):
            total_tokens += 1
            if form == lemma:
                identity_tokens += 1
            else:
                form_lemma_counts[form][lemma] += 1

    exceptions: dict[str, str] = {}
    for form, counts in form_lemma_counts.items():
        # Deterministic tie-break: sort by count desc, then lemma alpha.
        best_lemma = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        exceptions[form] = best_lemma

    if total_tokens == 0:
        print("WARNING: no tokens found in CoNLL-U file")
        return {}

    print(f"Total tokens: {total_tokens}")
    print(f"Identity (form==lemma): {identity_tokens} ({identity_tokens / total_tokens:.4f})")
    print(f"Exception forms: {len(exceptions)}")
    print("Sample exceptions:")
    for form, lemma in sorted(exceptions.items())[:10]:
        print(f"  {form} → {lemma}")

    return exceptions


def main() -> None:
    exceptions = build_exceptions()
    out = Path(OUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(exceptions, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nSaved {len(exceptions)} exceptions to {out}")


if __name__ == "__main__":
    main()
