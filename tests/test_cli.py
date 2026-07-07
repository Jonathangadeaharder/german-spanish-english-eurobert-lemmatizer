from typer.testing import CliRunner

from lemmatizer.cli import app

runner = CliRunner()


def test_cli_exposes_primary_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    command_names = {
        command.name or command.callback.__name__.replace("_", "-")
        for command in app.registered_commands
    }
    assert {
        "fetch-ud",
        "build-labels",
        "make-dataset",
        "prepare",
        "train",
        "evaluate",
        "evaluate-cefr",
        "export-onnx",
        "package-web",
    }.issubset(command_names)


def test_train_rejects_unknown_lang():
    result = runner.invoke(app, ["train", "--lang", "xx"])
    assert result.exit_code != 0


def test_export_onnx_unsupported_lang_exits_nonzero():
    result = runner.invoke(app, ["export-onnx", "--lang", "de"])
    assert result.exit_code == 1


def test_train_forwards_lang_to_env(monkeypatch):
    """The train command must set LEMMA_LANG so backend trainers and the
    dataset builder resolve the target language."""
    import os

    captured = {}

    def fake_run(spec, opts):
        captured["lemma_lang"] = os.environ.get("LEMMA_LANG")

    monkeypatch.setattr("lemmatizer.train.mlx_multitask.run", fake_run)

    result = runner.invoke(app, ["train", "--lang", "de", "--checkpoint", "stub"])
    assert result.exit_code == 0, result.output
    assert captured["lemma_lang"] == "de"
