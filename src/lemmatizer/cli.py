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

from lemmatizer.languages import LANGUAGES, lang_codes, spec
from lemmatizer.train import TrainOptions, train_language

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="EuroBERT lemmatizer CLI (MLX training, multi-backend eval)",
)


def _set_env(**values: object) -> None:
    for key, value in values.items():
        if value is None:
            continue
        os.environ[key] = "1" if value is True else "0" if value is False else str(value)


@app.command("fetch-ud")
def fetch_ud(
    lang: str | None = typer.Option(
        None, help=f"Language code (default: all of {', '.join(lang_codes())})"
    ),
) -> None:
    """Download UD treebanks. One language, or all registered languages."""
    from lemmatizer.data.ud import download_treebank

    if lang is None:
        for s in LANGUAGES:
            download_treebank(s)
    else:
        try:
            download_treebank(spec(lang))
        except ValueError as e:
            raise typer.BadParameter(str(e)) from e


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
    lang: str = typer.Option(..., help=f"Language: {', '.join(lang_codes())}"),
    checkpoint: str = typer.Option(
        "",
        help="Base model dir/id. Empty only valid for ByT5 family (ar); "
        "MULTITASK/ZH_BIO families require a checkpoint path.",
    ),
    output_dir: str = typer.Option("", help="Output dir (default: per-family)"),
    epochs: float = typer.Option(0.0, help="Epochs (0 = baseline eval only)"),
    batch_size: int = typer.Option(64, help="Batch size"),
    lr: float = typer.Option(2e-5, help="Learning rate"),
    lora_rank: int = typer.Option(8, help="LoRA rank (0 = full finetune)"),
    lora_alpha: float = typer.Option(16.0, help="LoRA alpha"),
    curriculum: bool = typer.Option(False, help="Enable curriculum sampling"),
    max_train_rows: int = typer.Option(0, help="Cap train rows (0 = all)"),
    max_val_rows: int = typer.Option(0, help="Cap val rows (0 = all)"),
    unfreeze_encoder: bool = typer.Option(False, help="Unfreeze all encoder layers"),
    unfreeze_last_n: int = typer.Option(0, help="Unfreeze last N encoder layers"),
    grad_accum: int = typer.Option(1, help="Gradient accumulation steps"),
    warmup: float = typer.Option(0.06, help="Warmup fraction (0-1)"),
    upos_weight: float = typer.Option(1.0, help="UPOS loss weight (default 1.0)"),
) -> None:
    """Train the MLX model for one language.

    Dispatches to the canonical trainer by `lang`'s family (registry-driven):
      ar → ByT5 encoder + lemma classifier
      zh → BERT BIO-POS (openmed-backed)
      de/en/es/fr/sv → EuroBERT/ScandiBERT multitask (upos + lemma)
    """
    opts = TrainOptions(
        checkpoint=checkpoint,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        curriculum=curriculum,
        max_train_rows=max_train_rows,
        max_val_rows=max_val_rows,
        unfreeze_encoder=unfreeze_encoder,
        unfreeze_last_n=unfreeze_last_n,
        grad_accum=grad_accum,
        warmup=warmup,
        upos_weight=upos_weight,
    )
    # Forward the target language to backend trainers and the dataset builder,
    # which resolve per-language artifacts via LEMMA_LANG. spec()/train_language
    # also normalize via this env when the CLI's --lang isn't propagated.
    _set_env(LEMMA_LANG=lang)
    try:
        train_language(lang, opts)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


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


@app.command("cefr-eval")
def cefr_eval(
    lang: str = typer.Option(..., help=f"Language: {', '.join(lang_codes())} or 'all'"),
    batch_size: int = typer.Option(8, min=1, help="Evaluation batch size"),
) -> None:
    """CEFR vocabulary eval gate (>90% lemma + UPOS, nonzero exit on fail)."""
    if lang != "all":
        try:
            resolved = spec(lang)  # raises ValueError on unknown lang
        except ValueError as e:
            raise typer.BadParameter(str(e)) from e
        lang = resolved.lang  # normalize aliases (e.g. "english" -> "en")
    from lemmatizer.eval.cefr_eval import main

    raise typer.Exit(code=main(["--lang", lang, "--batch-size", str(batch_size)]))


@app.command("export-onnx")
def export_onnx(
    lang: str = typer.Option(..., help=f"Language: {', '.join(lang_codes())}"),
    model_dir: str | None = typer.Option(None, help="MLX checkpoint dir"),
    onnx_dir: str | None = typer.Option(None, help="ONNX output directory"),
    lexicon_dir: str | None = typer.Option(None, help="Lexicon artifacts dir"),
) -> None:
    """Export an MLX-trained model to ONNX for the web runtime.

    Currently only Arabic (ByT5) has an MLX→ONNX bridge.
    de/en/es/fr/sv require a future bridge (see plan non-goals).
    """
    try:
        s = spec(lang)  # raises ValueError on unknown lang
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    _set_env(
        MODEL_DIR=model_dir,
        ONNX_DIR=onnx_dir,
        LEXICON_DIR=lexicon_dir,
    )

    if s.lang == "ar":
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
