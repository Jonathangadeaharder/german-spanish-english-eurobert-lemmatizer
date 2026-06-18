import json
from pathlib import Path

from make_dataset import load_silver_sentences


def test_load_silver_sentences_skips_invalid_rows(tmp_path):
    path = tmp_path / "de_lemma_silver.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "annotated_sentences": [
                            {
                                "words": ["Die", "Kinder"],
                                "lemmas": ["der", "Kind"],
                                "upos": ["DET", "NOUN"],
                            },
                            {
                                "words": ["invalid"],
                                "lemmas": [],
                                "upos": ["X"],
                            },
                        ]
                    }
                )
            ]
        ),
        encoding="utf-8",
    )

    rows = load_silver_sentences(Path(path))

    assert rows == [
        {
            "words": ["Die", "Kinder"],
            "lemmas": ["der", "Kind"],
            "upos": ["DET", "NOUN"],
        }
    ]
