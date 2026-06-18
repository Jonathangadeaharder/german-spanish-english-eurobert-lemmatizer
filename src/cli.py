from __future__ import annotations

import os

import typer

from config import apply_env, load_profile

app = typer.Typer(add_completion=False, no_args_is_help=True, help="EuroBERT lemmatizer CLI")


def _load_profile(profile: str | None) -> None:
    if not profile:
        return

    apply_env(load_profile(profile))


def _set_env(**values: object) -> None:
    for key, value in values.items():
        if value is None:
            continue
        os.environ[key] = "1" if value is True else "0" if value is False else str(value)


@app.command()
def fetch_ud(profile: str | None = typer.Option(None, help="Optional config profile")) -> None:
    _load_profile(profile)
    from fetch_ud import main

    main()


@app.command()
def build_labels(profile: str | None = typer.Option(None, help="Optional config profile")) -> None:
    _load_profile(profile)
    from build_labels import main

    main()


@app.command()
def make_dataset(profile: str | None = typer.Option(None, help="Optional config profile")) -> None:
    _load_profile(profile)
    from make_dataset import main

    main()


@app.command()
def prepare(profile: str | None = typer.Option(None, help="Optional config profile")) -> None:
    _load_profile(profile)
    from build_labels import main as build_labels_main
    from make_dataset import main as make_dataset_main

    build_labels_main()
    make_dataset_main()


@app.command()
def train(
    profile: str | None = typer.Option(None, help="Optional config profile"),
    output_dir: str | None = typer.Option(None, help="Training output directory"),
    resume_from: str | None = typer.Option(None, help="Checkpoint path or auto"),
    max_steps: int | None = typer.Option(None, help="Maximum training steps"),
    epochs: float | None = typer.Option(None, help="Number of training epochs"),
    learning_rate: float | None = typer.Option(None, help="Learning rate"),
    warmup_ratio: float | None = typer.Option(None, help="Warmup ratio"),
    save_total_limit: int | None = typer.Option(None, help="Maximum retained checkpoints"),
    batch_size: int | None = typer.Option(None, help="Per-device train batch size"),
    eval_batch_size: int | None = typer.Option(None, help="Per-device eval batch size"),
    bf16: bool | None = typer.Option(None, help="Enable bf16"),
    fp16: bool | None = typer.Option(None, help="Enable fp16"),
    group_by_length: bool | None = typer.Option(None, help="Group batches by sequence length"),
) -> None:
    _load_profile(profile)
    _set_env(
        OUTPUT_DIR=output_dir,
        TRAIN_RESUME_FROM=resume_from,
        TRAIN_MAX_STEPS=max_steps,
        TRAIN_EPOCHS=epochs,
        TRAIN_LEARNING_RATE=learning_rate,
        TRAIN_WARMUP_RATIO=warmup_ratio,
        TRAIN_SAVE_TOTAL_LIMIT=save_total_limit,
        TRAIN_BATCH_SIZE=batch_size,
        TRAIN_EVAL_BATCH_SIZE=eval_batch_size,
        TRAIN_BF16=bf16,
        TRAIN_FP16=fp16,
        TRAIN_GROUP_BY_LENGTH=group_by_length,
    )
    from train import main

    main()


@app.command()
def evaluate(
    profile: str | None = typer.Option(None, help="Optional config profile"),
    model_dir: str | None = typer.Option(None, help="Model directory"),
    eval_limit: int | None = typer.Option(None, help="Limit total test rows"),
    per_lang_limit: int | None = typer.Option(None, help="Limit rows per language"),
    batch_size: int | None = typer.Option(None, help="Evaluation batch size"),
) -> None:
    _load_profile(profile)
    _set_env(
        MODEL_DIR=model_dir,
        EVAL_LIMIT=eval_limit,
        EVAL_PER_LANG_LIMIT=per_lang_limit,
        EVAL_BATCH_SIZE=batch_size,
    )
    from evaluate import main

    main()


@app.command()
def merge(
    profile: str | None = typer.Option(None, help="Optional config profile"),
    adapter_dir: str | None = typer.Option(None, help="LoRA adapter directory"),
    merged_dir: str | None = typer.Option(None, help="Merged model output directory"),
) -> None:
    _load_profile(profile)
    _set_env(ADAPTER_DIR=adapter_dir, MERGED_DIR=merged_dir)
    from merge_lora import main

    main()


@app.command("export-onnx")
def export_onnx(
    profile: str | None = typer.Option(None, help="Optional config profile"),
    merged_dir: str | None = typer.Option(None, help="Merged model directory"),
    onnx_dir: str | None = typer.Option(None, help="ONNX output directory"),
) -> None:
    _load_profile(profile)
    _set_env(MERGED_DIR=merged_dir, ONNX_DIR=onnx_dir)
    from export_onnx import main

    main()


@app.command("package-web")
def package_web(
    profile: str | None = typer.Option(None, help="Optional config profile"),
    artifacts_dir: str | None = typer.Option(None, help="Artifacts directory"),
    merged_dir: str | None = typer.Option(None, help="Merged model directory"),
    onnx_dir: str | None = typer.Option(None, help="ONNX output directory"),
    web_model_dir: str | None = typer.Option(None, help="Browser bundle directory"),
) -> None:
    _load_profile(profile)
    _set_env(
        ARTIFACTS_DIR=artifacts_dir,
        MERGED_DIR=merged_dir,
        ONNX_DIR=onnx_dir,
        WEB_MODEL_DIR=web_model_dir,
    )
    from package_web_model import main

    main()


@app.command("benchmark-mps")
def benchmark_mps(
    profile: str | None = typer.Option(None, help="Optional config profile"),
    configs: str | None = typer.Option(None, help="Comma-separated benchmark configs"),
    steps: int | None = typer.Option(None, help="Training steps per benchmark run"),
    train_limit: int | None = typer.Option(None, help="Training rows per benchmark run"),
    eval_limit: int | None = typer.Option(None, help="Eval rows per benchmark run"),
    learning_rate: float | None = typer.Option(None, help="Learning rate"),
    warmup_ratio: float | None = typer.Option(None, help="Warmup ratio"),
) -> None:
    _load_profile(profile)
    _set_env(
        MPS_BENCH_CONFIGS=configs,
        MPS_BENCH_STEPS=steps,
        MPS_BENCH_TRAIN_LIMIT=train_limit,
        MPS_BENCH_EVAL_LIMIT=eval_limit,
        MPS_BENCH_LEARNING_RATE=learning_rate,
        MPS_BENCH_WARMUP_RATIO=warmup_ratio,
    )
    from benchmark_mps import main

    main([])


@app.command()
def pipeline(profile: str | None = typer.Option(None, help="Optional config profile")) -> None:
    _load_profile(profile)
    from build_labels import main as build_labels_main
    from evaluate import main as evaluate_main
    from export_onnx import main as export_main
    from make_dataset import main as make_dataset_main
    from merge_lora import main as merge_main
    from package_web_model import main as package_main
    from train import main as train_main

    build_labels_main()
    make_dataset_main()
    train_main()
    evaluate_main()
    merge_main()
    export_main()
    package_main()
