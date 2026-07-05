"""Typer CLI for the lemmatizer.

Commands dispatch to MLX trainers (one canonical stack per language family):
- de/en/es/fr/sv → lemmatizer.train.mlx_multitask
- ar             → lemmatizer.train.train_byt5
- zh             → lemmatizer.train.zh_bio

Eval and packaging are backend-agnostic (ONNX/LoRA/merged resolved by EvalContext).
"""
from __future__ import annotations

import os

import typer

from lemmatizer.languages import LANGS

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="EuroBERT lemmatizer CLI (MLX training, multi-backend eval)",
)


def _set_env(**values: object) -> None:
    for key, value in values.items():
        if value is None:
            continue
        os.environ[key] = (
            "1" if value is True else "0" if value is False else str(value)
        )


@app.command()
def fetch_ud() -> None:
    """Download UD treebanks."""
    from lemmatizer.data.ud import main

    main()


@app.command()
def build_labels() -> None:
    """Build label2id / id2label / edit_trees / gold_lexicon artifacts."""
    from lemmatizer.data.build_labels import main

    main()


@app.command()
def make_dataset() -> None:
    """Build the multilingual EuroBERT token-classification dataset."""
    from lemmatizer.data.dataset import main

    main()


@app.command()
def prepare() -> None:
    """Run build-labels + make-dataset in sequence."""
    from lemmatizer.data.build_labels import main as build_labels_main
    from lemmatizer.data.dataset import main as make_dataset_main

    build_labels_main()
    make_dataset_main()


@app.command()
def train(
    lang: str = typer.Option(..., help=f"Language: {', '.join(LANGS)}"),
) -> None:
    """Train the MLX model for one language.

    Dispatches by language family:
      ar → ByT5 encoder + lemma classifier
      zh → BERT BIO-POS (openmed-backed)
      de/en/es/fr/sv → EuroBERT/ScandiBERT multitask (upos + lemma)
    """
    if lang not in LANGS:
        raise typer.BadParameter(
            f"Unknown language '{lang}'. Supported: {', '.join(LANGS)}"
        )

    if lang == "ar":
        from lemmatizer.train.train_byt5 import main
    elif lang == "zh":
        from lemmatizer.train.zh_bio import main
    else:
        from lemmatizer.train.mlx_multitask import main

    main()


@app.command()
def evaluate(
    model_dir: str | None = typer.Option(None, help="Model directory"),
    eval_limit: int | None = typer.Option(None, help="Limit total test rows"),
    batch_size: int | None = typer.Option(None, help="Evaluation batch size"),
) -> None:
    """Treebank evaluation (UPOS + lemma accuracy)."""
    _set_env(
        MODEL_DIR=model_dir,
        EVAL_LIMIT=eval_limit,
        EVAL_BATCH_SIZE=batch_size,
    )
    from lemmatizer.eval.evaluate import main

    main()


@app.command("evaluate-cefr")
def evaluate_cefr(
    batch_size: int | None = typer.Option(None, help="Evaluation batch size"),
) -> None:
    """CEFR-level evaluation via the shared EvalContext."""
    _set_env(EVAL_BATCH_SIZE=batch_size)
    from lemmatizer.eval.evaluate_cefr import main

    main()


@app.command("export-onnx")
def export_onnx(
    lang: str = typer.Option(..., help=f"Language: {', '.join(LANGS)}"),
    model_dir: str | None = typer.Option(None, help="MLX checkpoint dir"),
    onnx_dir: str | None = typer.Option(None, help="ONNX output directory"),
    lexicon_dir: str | None = typer.Option(None, help="Lexicon artifacts dir"),
) -> None:
    """Export an MLX-trained model to ONNX for the web runtime.

    Currently only Arabic (ByT5) has an MLX→ONNX bridge.
    de/en/es/fr/sv require a future bridge (see plan non-goals).
    """
    if lang not in LANGS:
        raise typer.BadParameter(
            f"Unknown language '{lang}'. Supported: {', '.join(LANGS)}"
        )
    _set_env(
        MODEL_DIR=model_dir,
        ONNX_DIR=onnx_dir,
        LEXICON_DIR=lexicon_dir,
    )

    if lang == "ar":
        from lemmatizer.export.byt5_onnx import main

        main()
    else:
        typer.echo(
            f"MLX→ONNX bridge for '{lang}' is not implemented yet. "
            "Only Arabic has a working bridge today."
        )
        raise typer.Exit(code=1)


@app.command("package-web")
def package_web(
    artifacts_dir: str | None = typer.Option(None, help="Artifacts directory"),
    onnx_dir: str | None = typer.Option(None, help="ONNX directory"),
    web_model_dir: str | None = typer.Option(None, help="Browser bundle directory"),
) -> None:
    """Copy artifacts + ONNX into the web/model/ browser bundle."""
    _set_env(
        ARTIFACTS_DIR=artifacts_dir,
        ONNX_DIR=onnx_dir,
        WEB_MODEL_DIR=web_model_dir,
    )
    from lemmatizer.export.package_web import main

    main()
