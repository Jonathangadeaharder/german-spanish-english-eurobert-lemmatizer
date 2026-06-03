from cli import app


def test_cli_exposes_primary_commands():
    command_names = {
        command.name or command.callback.__name__.replace("_", "-")
        for command in app.registered_commands
    }

    assert {"prepare", "train", "evaluate", "merge", "export-onnx", "package-web"}.issubset(
        command_names
    )
