"""Download UD treebanks.

Driven by `LANGUAGES` registry — adding a language = one `LanguageSpec`
entry, and `fetch-ud` immediately downloads it. No per-lang dict here.
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

from lemmatizer.languages import LANGUAGES, LanguageSpec

SPLITS = ["train", "dev", "test"]


def download_treebank(s: LanguageSpec, out_root: Path = Path("data/gold")) -> None:
    """Download one language's UD CoNLL-U train/dev/test into data/gold/<lang>/."""
    out_dir = out_root / s.lang
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        filename = f"{s.ud_prefix}-ud-{split}.conllu"
        url = (
            f"https://raw.githubusercontent.com/UniversalDependencies/"
            f"{s.ud_repo}/master/{filename}"
        )
        out_path = out_dir / f"{split}.conllu"
        print(f"Downloading {url} -> {out_path}")
        urlretrieve(url, out_path)


def main() -> None:
    """Download all registered languages' treebanks."""
    for s in LANGUAGES:
        download_treebank(s)


if __name__ == "__main__":
    main()
