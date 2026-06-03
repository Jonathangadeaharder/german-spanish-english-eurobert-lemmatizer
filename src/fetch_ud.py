from pathlib import Path
from urllib.request import urlretrieve


TREEBANKS = {
    "de": {
        "repo": "UD_German-GSD",
        "prefix": "de_gsd",
    },
    "es": {
        "repo": "UD_Spanish-AnCora",
        "prefix": "es_ancora",
    },
    "en": {
        "repo": "UD_English-EWT",
        "prefix": "en_ewt",
    },
}


SPLITS = ["train", "dev", "test"]


def download_treebank(lang: str):
    spec = TREEBANKS[lang]
    repo = spec["repo"]
    prefix = spec["prefix"]

    out_dir = Path("data/gold") / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
      # Use the official UD GitHub mirror.
        filename = f"{prefix}-ud-{split}.conllu"
        url = f"https://raw.githubusercontent.com/UniversalDependencies/{repo}/master/{filename}"
        out_path = out_dir / f"{split}.conllu"
        print(f"Downloading {url} -> {out_path}")
        urlretrieve(url, out_path)


def main():
    for lang in ["de", "es", "en"]:
        download_treebank(lang)


if __name__ == "__main__":
    main()
