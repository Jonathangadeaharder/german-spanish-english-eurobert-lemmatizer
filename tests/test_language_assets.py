from language_assets import language_assets, normalize_lang, split_files_for_lang


def test_normalize_lang_accepts_aliases():
    assert normalize_lang("english") == "en"
    assert normalize_lang("deutsch") == "de"
    assert normalize_lang("español") == "es"


def test_language_assets_are_language_scoped(monkeypatch):
    monkeypatch.delenv("ARTIFACTS_DIR", raising=False)
    monkeypatch.delenv("DATASET_PATH", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    monkeypatch.delenv("MERGED_DIR", raising=False)
    monkeypatch.delenv("ONNX_DIR", raising=False)
    monkeypatch.delenv("WEB_MODEL_DIR", raising=False)
    monkeypatch.delenv("TOKENIZER_DIR", raising=False)

    assets = language_assets("en")

    assert str(assets.artifacts_dir) == "artifacts/lemma_en"
    assert str(assets.dataset_path) == "data/processed/eurobert_lemma_en_dataset"
    assert str(assets.output_dir) == "runs/eurobert-lemma-en-210m-lora"
    assert str(assets.web_model_dir) == "web/model/lemma_en"


def test_split_files_for_lang_returns_single_language():
    files = split_files_for_lang("es")

    assert files == {
        "train": "data/gold/es/train.conllu",
        "validation": "data/gold/es/dev.conllu",
        "test": "data/gold/es/test.conllu",
    }
