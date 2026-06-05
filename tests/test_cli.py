from typer.testing import CliRunner

from cli import app

runner = CliRunner()


def test_cli_exposes_primary_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    command_names = {
        command.name or command.callback.__name__.replace("_", "-")
        for command in app.registered_commands
    }
    assert {"prepare", "train", "evaluate", "merge", "export-onnx", "package-web"}.issubset(
        command_names
    )
