from datasets import Dataset

from lemmatizer.eval.evaluate import filter_test_rows_for_lang


def test_filter_test_rows_for_lang_filters_multilingual_dataset():
    rows = Dataset.from_list(
        [
            {"lang": "de", "words": ["Haus"]},
            {"lang": "zh", "words": ["的"]},
            {"lang": "zh", "words": ["人"]},
        ]
    )

    filtered = filter_test_rows_for_lang(rows, "zh")

    assert len(filtered) == 2
    assert filtered[0]["words"] == ["的"]


def test_filter_test_rows_for_lang_leaves_unlabeled_dataset():
    rows = Dataset.from_list([{"words": ["Haus"]}])

    assert filter_test_rows_for_lang(rows, "zh") is rows
