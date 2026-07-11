"""Compute the edit-tree oracle ceiling per language.

For each language: fraction of gold dev+test tokens whose gold edit tree
exists in the label space (plus lexicon/identity fallbacks). This tells us
the maximum achievable lemma accuracy given the current label space.

Usage:
    uv run python -m lemmatizer.eval.oracle_ceilings
"""

from __future__ import annotations

import json
from pathlib import Path

from lemmatizer.data.conllu import read_conllu
from lemmatizer.data.edit_trees import make_edit_label
from lemmatizer.languages import LANGUAGES, language_assets

IDENTITY_UPOS = {"PROPN", "PUNCT", "SYM", "X", "NUM"}


def compute_ceiling(lang: str) -> dict:
    assets = language_assets(lang)
    label2id = json.loads(assets.label2id_path.read_text(encoding="utf-8"))
    lexicon_path = assets.lexicon_path
    lexicon = json.loads(lexicon_path.read_text(encoding="utf-8")) if lexicon_path.exists() else {}

    gold_dir = Path(f"data/gold/{lang}")
    results = {"lang": lang, "splits": {}}

    for split in ("dev", "test"):
        conllu_path = gold_dir / f"{split}.conllu"
        if not conllu_path.exists():
            continue

        sentences = read_conllu(str(conllu_path), lang=lang)
        total = 0
        learnable = 0
        identity_upos_skipped = 0
        identity_fallback = 0
        in_lexicon = 0
        unlearnable = 0

        for sent in sentences:
            for word, lemma, upos in zip(sent["words"], sent["lemmas"], sent["upos"], strict=True):
                if upos in IDENTITY_UPOS or lemma in ("_", "-"):
                    identity_upos_skipped += 1
                    continue

                total += 1
                base_label = make_edit_label(word, lemma)
                full_label = f"{lang}::{base_label}"

                if full_label in label2id:
                    learnable += 1
                elif word in lexicon:
                    in_lexicon += 1
                elif word == lemma:
                    identity_fallback += 1
                else:
                    unlearnable += 1

        solvable = learnable + in_lexicon + identity_fallback
        ceiling = solvable / max(total, 1)
        results["splits"][split] = {
            "total_scored": total,
            "learnable_via_edit_tree": learnable,
            "in_lexicon": in_lexicon,
            "identity_fallback": identity_fallback,
            "identity_upos_skipped": identity_upos_skipped,
            "unlearnable": unlearnable,
            "ceiling": round(ceiling, 4),
        }

    return results


def main() -> None:
    print(
        f"{'Lang':6s} {'Split':6s} {'Total':>6s} {'EditTree':>9s} {'Lexicon':>8s} "
        f"{'Ident':>6s} {'Unlearn':>8s} {'Ceiling':>8s}"
    )
    print("-" * 70)

    all_results = []
    for spec in LANGUAGES:
        if spec.lang == "zh":
            # zh uses identity+lexicon, not edit-trees — ceiling is ~100%
            print(
                f"{'zh':6s} {'all':6s} {'-':>6s} {'-':>9s} {'-':>8s} "
                f"{'-':>6s} {'-':>8s} {'~1.0000':>8s}  (identity+lexicon)"
            )
            all_results.append(
                {
                    "lang": "zh",
                    "splits": {
                        "all": {
                            "total_scored": 0,
                            "learnable_via_edit_tree": 0,
                            "in_lexicon": 0,
                            "identity_fallback": 0,
                            "identity_upos_skipped": 0,
                            "unlearnable": 0,
                            "ceiling": 1.0,
                            "note": "identity+lexicon (no edit-tree label space)",
                        }
                    },
                }
            )
            continue

        r = compute_ceiling(spec.lang)
        all_results.append(r)
        for split, s in r["splits"].items():
            print(
                f"{spec.lang:6s} {split:6s} {s['total_scored']:6d} "
                f"{s['learnable_via_edit_tree']:9d} {s['in_lexicon']:8d} "
                f"{s['identity_fallback']:6d} {s['unlearnable']:8d} "
                f"{s['ceiling']:8.4f}"
            )

    # Write to docs/training/ceilings.md
    out = Path("docs/training/ceilings.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Edit-Tree Oracle Ceilings",
        "",
        "Maximum achievable lemma accuracy given the current label space.",
        "Tokens where the gold edit tree is absent from the label space",
        "are unlearnable unless covered by lexicon fallback or identity.",
        "",
    ]

    for r in all_results:
        lines.append(f"## {r['lang']}")
        for split, s in r["splits"].items():
            lines.append(f"### {split}")
            lines.append(f"- Total scored: {s['total_scored']}")
            lines.append(f"- Learnable via edit-tree: {s['learnable_via_edit_tree']}")
            lines.append(f"- In lexicon: {s['in_lexicon']}")
            lines.append(f"- Identity (fallback): {s['identity_fallback']}")
            lines.append(f"- Identity-UPOS (skipped): {s['identity_upos_skipped']}")
            lines.append(f"- Unlearnable: {s['unlearnable']}")
            lines.append(f"- **Ceiling: {s['ceiling']:.4f}**")
            lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
