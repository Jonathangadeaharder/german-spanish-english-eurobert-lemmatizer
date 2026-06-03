import argparse
import json
import os
import platform
import subprocess
import sys
import time
from math import isfinite
from pathlib import Path
from statistics import median

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.2")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "1.0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


MODEL_ID = "EuroBERT/EuroBERT-210m"
TOKENIZER_DIR = "artifacts/tokenizer"
DATASET_PATH = "data/processed/eurobert_multilingual_lemma_dataset"
LABEL2ID_PATH = "artifacts/label2id.json"
UPOS_LABEL2ID_PATH = "artifacts/upos_label2id.json"
OUTPUT_PATH = Path("artifacts/mps_benchmark.json")
RUNS_DIR = Path("runs/mps-bench")
LANGUAGE_TOKENS = ["[LANG_DE]", "[LANG_ES]", "[LANG_EN]"]


def env_str(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else value


def env_int(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def env_float(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def env_bool(name, default):
    value = os.getenv(name)

    if value is None or value == "":
        return default

    return value.lower() in {"1", "true", "yes", "on"}


def env_list(name, default, cast=str):
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def trainable_language_token_indices(tokenizer):
    return [tokenizer.convert_tokens_to_ids(token) for token in LANGUAGE_TOKENS]


def warn_if_rosetta():
    if platform.machine() == "arm64":
        return

    print(
        "Warning: benchmark is running under x86_64. Results will not reflect a native arm64 MPS run.",
        file=sys.stderr,
    )


def base_configs():
    return [
        {
            "name": "fp32_bs4_nogroup",
            "batch_size": 4,
            "bf16": False,
            "fp16": False,
            "group_by_length": False,
            "gradient_checkpointing": False,
            "prefer_metal": False,
            "fast_math": False,
        },
        {
            "name": "bf16_bs4_group",
            "batch_size": 4,
            "bf16": True,
            "fp16": False,
            "group_by_length": True,
            "gradient_checkpointing": False,
            "prefer_metal": False,
            "fast_math": False,
        },
        {
            "name": "bf16_bs8_group",
            "batch_size": 8,
            "bf16": True,
            "fp16": False,
            "group_by_length": True,
            "gradient_checkpointing": False,
            "prefer_metal": False,
            "fast_math": False,
        },
        {
            "name": "bf16_bs12_group",
            "batch_size": 12,
            "bf16": True,
            "fp16": False,
            "group_by_length": True,
            "gradient_checkpointing": False,
            "prefer_metal": False,
            "fast_math": False,
        },
        {
            "name": "bf16_bs16_group",
            "batch_size": 16,
            "bf16": True,
            "fp16": False,
            "group_by_length": True,
            "gradient_checkpointing": False,
            "prefer_metal": False,
            "fast_math": False,
        },
        {
            "name": "bf16_bs16_group_gc",
            "batch_size": 16,
            "bf16": True,
            "fp16": False,
            "group_by_length": True,
            "gradient_checkpointing": True,
            "prefer_metal": False,
            "fast_math": False,
        },
    ]


def matrix_configs():
    batch_sizes = env_list("MPS_BENCH_BATCH_SIZES", [], int)
    modes = env_list("MPS_BENCH_FLAGS", ["baseline"])
    workers = env_list("MPS_BENCH_DATALOADER_WORKERS_LIST", [], int)
    group_by_length = env_bool("MPS_BENCH_GROUP_BY_LENGTH", True)
    gradient_checkpointing = env_bool("MPS_BENCH_GRADIENT_CHECKPOINTING", False)

    if not batch_sizes:
        return []

    if workers:
        worker_values = workers
    else:
        worker_values = [None]

    configs = []

    for batch_size in batch_sizes:
        for mode in modes:
            for worker_count in worker_values:
                config = {
                    "name": f"bf16_bs{batch_size}_group_{mode}",
                    "batch_size": batch_size,
                    "bf16": True,
                    "fp16": False,
                    "group_by_length": group_by_length,
                    "gradient_checkpointing": gradient_checkpointing,
                    "prefer_metal": mode == "prefer_metal",
                    "fast_math": mode == "fast_math",
                }

                if worker_count is not None:
                    config["dataloader_num_workers"] = worker_count
                    config["name"] = f"{config['name']}_w{worker_count}"

                configs.append(config)

    return configs


def selected_base_configs():
    matrix = matrix_configs()

    if matrix:
        return matrix

    requested = os.getenv("MPS_BENCH_CONFIGS", "").strip()
    configs = base_configs()

    if not requested:
        return configs

    wanted = {name.strip() for name in requested.split(",") if name.strip()}
    return [config for config in configs if config["name"] in wanted]


def runtime_defaults_from_env():
    return {
        "steps": env_int("MPS_BENCH_STEPS", 12),
        "train_limit": env_int("MPS_BENCH_TRAIN_LIMIT", 1024),
        "eval_limit": env_int("MPS_BENCH_EVAL_LIMIT", 256),
        "learning_rate": float(env_str("MPS_BENCH_LEARNING_RATE", "2e-4")),
        "warmup_ratio": float(env_str("MPS_BENCH_WARMUP_RATIO", "0.06")),
        "float32_matmul_precision": env_str("MPS_BENCH_FLOAT32_MATMUL_PRECISION", "high"),
        "dataloader_num_workers": env_int("MPS_BENCH_DATALOADER_NUM_WORKERS", 2),
        "eval_accumulation_steps": env_int("MPS_BENCH_EVAL_ACCUMULATION_STEPS", 8),
        "gradient_accumulation_steps": env_int("MPS_BENCH_GRADIENT_ACCUMULATION_STEPS", 1),
        "seed": env_int("MPS_BENCH_SEED", 42),
        "full_epochs": env_float("MPS_BENCH_FULL_EPOCHS", 3.0),
        "repeats": env_int("MPS_BENCH_REPEATS", 1),
    }


def run_child(config_json):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.2")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "1.0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = json.loads(config_json)
    env_flags = {
        "PYTORCH_MPS_PREFER_METAL": "1" if config.get("prefer_metal") else None,
        "PYTORCH_MPS_FAST_MATH": "1" if config.get("fast_math") else None,
    }

    for key, value in env_flags.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    import torch
    from datasets import load_from_disk
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from multitask_model import (
        EuroBertForUposLemma,
        EuroBertUposLemmaConfig,
        MultiTaskDataCollator,
        compute_multitask_metrics,
    )
    from runtime_utils import MPSMemoryCleanupCallback, cleanup_torch_mps

    def load_json_local(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def compute_metrics(eval_pred):
        metrics = compute_multitask_metrics(eval_pred)
        return {
            "token_accuracy": metrics["joint_accuracy"],
            **metrics,
        }

    def configure_runtime():
        precision = config.get("float32_matmul_precision", runtime_defaults["float32_matmul_precision"])

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(precision)

    def warn_if_rosetta_local():
        if platform.machine() != "arm64":
            print(
                "Warning: benchmark child is x86_64. Use native arm64 Python for final numbers.",
                file=sys.stderr,
            )

    runtime_defaults = runtime_defaults_from_env()
    set_seed(runtime_defaults["seed"])
    warn_if_rosetta_local()
    configure_runtime()

    label2id = load_json_local(LABEL2ID_PATH)
    upos_label2id = load_json_local(UPOS_LABEL2ID_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)

    config = EuroBertUposLemmaConfig(
        base_model_name_or_path=MODEL_ID,
        upos_label2id=upos_label2id,
        lemma_label2id=label2id,
    )

    model = EuroBertForUposLemma.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["upos_classifier", "lemma_classifier"],
        trainable_token_indices=trainable_language_token_indices(tokenizer),
    )
    model = get_peft_model(model, lora_config)

    if config.get("gradient_checkpointing"):
        if hasattr(model.model.config, "use_cache"):
            model.model.config.use_cache = False
        if hasattr(model.model, "gradient_checkpointing_enable"):
            model.model.gradient_checkpointing_enable()
        if hasattr(model.model, "enable_input_require_grads"):
            model.model.enable_input_require_grads()

    dataset = load_from_disk(DATASET_PATH)
    train_limit = min(runtime_defaults["train_limit"], len(dataset["train"]))
    eval_limit = min(runtime_defaults["eval_limit"], len(dataset["validation"]))

    train_dataset = dataset["train"].select(range(train_limit))
    eval_dataset = dataset["validation"].select(range(eval_limit))
    data_collator = MultiTaskDataCollator(tokenizer=tokenizer)

    output_dir = RUNS_DIR / config["name"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    bf16 = bool(config.get("bf16"))
    fp16 = bool(config.get("fp16"))

    if bf16 and fp16:
        raise ValueError("Benchmark config cannot enable both bf16 and fp16")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=runtime_defaults["learning_rate"],
        warmup_ratio=runtime_defaults["warmup_ratio"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=runtime_defaults["gradient_accumulation_steps"],
        num_train_epochs=1,
        max_steps=runtime_defaults["steps"],
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="no",
        evaluation_strategy="no",
        report_to="none",
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=config.get(
            "dataloader_num_workers", runtime_defaults["dataloader_num_workers"]
        ),
        dataloader_pin_memory=False,
        eval_accumulation_steps=runtime_defaults["eval_accumulation_steps"],
        group_by_length=bool(config.get("group_by_length")),
        length_column_name="length",
        gradient_checkpointing=bool(config.get("gradient_checkpointing")),
        seed=runtime_defaults["seed"],
        disable_tqdm=True,
        label_names=["labels", "upos_labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MPSMemoryCleanupCallback()],
    )

    started_at = time.perf_counter()

    try:
        train_result = trainer.train()
        cleanup_torch_mps("benchmark_post_train")
        eval_result = trainer.evaluate()
        cleanup_torch_mps("benchmark_post_eval")
        status = "ok"
        error = None
    except Exception as exc:  # noqa: BLE001 - benchmark harness must capture failures
        status = "error"
        train_result = None
        eval_result = None
        cleanup_torch_mps("benchmark_error")
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    elapsed = time.perf_counter() - started_at

    memory = {}
    if hasattr(torch, "mps"):
        for attr in ["current_allocated_memory", "driver_allocated_memory"]:
            fn = getattr(torch.mps, attr, None)

            if callable(fn):
                try:
                    memory[attr] = int(fn())
                except Exception:
                    continue

    result = {
        "config": config,
        "status": status,
        "elapsed_seconds": round(elapsed, 3),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "float32_matmul_precision": getattr(torch, "get_float32_matmul_precision", lambda: "unknown")(),
        },
        "train_examples": train_limit,
        "eval_examples": eval_limit,
        "memory": memory,
    }

    if train_result is not None:
        result["train_metrics"] = {
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in train_result.metrics.items()
        }

    if eval_result is not None:
        result["eval_metrics"] = {
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in eval_result.items()
        }

    if error is not None:
        result["error"] = error

    print(json.dumps(result, ensure_ascii=False))


def run_child_process(config):
    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.2")
    env.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "1.0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    if config.get("prefer_metal"):
        env["PYTORCH_MPS_PREFER_METAL"] = "1"
    else:
        env.pop("PYTORCH_MPS_PREFER_METAL", None)

    if config.get("fast_math"):
        env["PYTORCH_MPS_FAST_MATH"] = "1"
    else:
        env.pop("PYTORCH_MPS_FAST_MATH", None)

    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(config)]
    completed = subprocess.run(cmd, capture_output=True, text=True, env=env)

    payload = None
    if completed.stdout.strip():
        last_line = completed.stdout.strip().splitlines()[-1]
        try:
            payload = json.loads(last_line)
        except json.JSONDecodeError:
            payload = {
                "status": "error",
                "error": {
                    "type": "InvalidJSON",
                    "message": last_line,
                },
            }

    if payload is None:
        payload = {
            "status": "error",
            "error": {
                "type": "MissingOutput",
                "message": completed.stderr.strip() or "child process produced no JSON output",
            },
        }

    payload["returncode"] = completed.returncode
    payload["stdout"] = completed.stdout
    payload["stderr"] = completed.stderr
    return payload


def choose_best_config(results):
    successful = [r for r in results if r.get("status") == "ok" and not r.get("rejection_reasons")]

    if not successful:
        return None

    def sort_key(result):
        return (
            float(result.get("estimated_full_train_seconds", float("inf"))),
            -float(result.get("median_train_samples_per_second", 0.0)),
            -float(result.get("median_train_steps_per_second", 0.0)),
            float(result.get("median_train_samples_per_second", 0.0)),
            -float(result.get("median_eval_token_accuracy", 0.0)),
            float(result.get("median_train_loss", float("inf"))),
        )

    return sorted(successful, key=sort_key)[0]


def config_signature(config):
    filtered = {
        key: value
        for key, value in config.items()
        if key not in {"name", "repeat"}
    }
    return json.dumps(filtered, sort_keys=True)


def _finite_metric(value):
    if value is None:
        return False

    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def rejection_reasons(result, runtime_defaults, full_train_examples):
    reasons = []

    if result.get("returncode", 0) != 0:
        reasons.append(f"returncode={result.get('returncode')}")

    runtime = result.get("runtime", {})
    if not runtime.get("mps_available"):
        reasons.append("mps_unavailable")

    if runtime.get("machine") != "arm64":
        reasons.append(f"non_arm64_runtime={runtime.get('machine')}")

    error = result.get("error")
    if error:
        error_type = error.get("type", "Error")
        reasons.append(error_type)

    train_metrics = result.get("train_metrics", {})
    eval_metrics = result.get("eval_metrics", {})

    for key in ["train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss"]:
        if key in train_metrics and not _finite_metric(train_metrics.get(key)):
            reasons.append(f"non_finite_{key}")

    for key in ["eval_loss", "eval_token_accuracy"]:
        if key in eval_metrics and not _finite_metric(eval_metrics.get(key)):
            reasons.append(f"non_finite_{key}")

    memory = result.get("memory", {})
    driver_memory = memory.get("driver_allocated_memory")
    if isinstance(driver_memory, (int, float)):
        driver_gib = float(driver_memory) / (1024**3)
        if driver_gib > env_float("MPS_BENCH_MAX_DRIVER_GIB", 52.0):
            reasons.append(f"driver_memory_gib={driver_gib:.2f}")

    samples_per_second = train_metrics.get("train_samples_per_second")
    if _finite_metric(samples_per_second) and float(samples_per_second) > 0:
        total_examples = full_train_examples * runtime_defaults["full_epochs"]
        result["estimated_full_train_seconds"] = round(total_examples / float(samples_per_second), 2)

    return sorted(set(reasons))


def aggregate_group(results, runtime_defaults, full_train_examples):
    first = results[0]
    config = dict(first["config"])
    summary = {
        "config": config,
        "run_count": len(results),
        "successful_runs": 0,
        "failed_runs": 0,
        "status": "rejected",
        "rejection_reasons": [],
        "runs": results,
    }

    successful = []
    all_reasons = []

    for result in results:
        reasons = rejection_reasons(result, runtime_defaults, full_train_examples)
        result["rejection_reasons"] = reasons
        if reasons:
            all_reasons.extend(reasons)
        else:
            successful.append(result)

    summary["successful_runs"] = len(successful)
    summary["failed_runs"] = len(results) - len(successful)
    summary["rejection_reasons"] = sorted(set(all_reasons))

    if not successful:
        return summary

    summary["status"] = "ok" if len(successful) == len(results) else "partial"

    def collect(path, default=None):
        values = []
        for result in successful:
            current = result
            for part in path:
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(part)
            if _finite_metric(current):
                values.append(float(current))
        return median(values) if values else default

    summary["median_elapsed_seconds"] = collect(["elapsed_seconds"])
    summary["median_train_runtime"] = collect(["train_metrics", "train_runtime"])
    summary["median_train_samples_per_second"] = collect(["train_metrics", "train_samples_per_second"])
    summary["median_train_steps_per_second"] = collect(["train_metrics", "train_steps_per_second"])
    summary["median_train_loss"] = collect(["train_metrics", "train_loss"])
    summary["median_eval_runtime"] = collect(["eval_metrics", "eval_runtime"])
    summary["median_eval_samples_per_second"] = collect(["eval_metrics", "eval_samples_per_second"])
    summary["median_eval_steps_per_second"] = collect(["eval_metrics", "eval_steps_per_second"])
    summary["median_eval_loss"] = collect(["eval_metrics", "eval_loss"])
    summary["median_eval_token_accuracy"] = collect(["eval_metrics", "eval_token_accuracy"])
    summary["train_metrics"] = {
        "train_runtime": summary["median_train_runtime"],
        "train_samples_per_second": summary["median_train_samples_per_second"],
        "train_steps_per_second": summary["median_train_steps_per_second"],
        "train_loss": summary["median_train_loss"],
    }
    summary["eval_metrics"] = {
        "eval_runtime": summary["median_eval_runtime"],
        "eval_samples_per_second": summary["median_eval_samples_per_second"],
        "eval_steps_per_second": summary["median_eval_steps_per_second"],
        "eval_loss": summary["median_eval_loss"],
        "eval_token_accuracy": summary["median_eval_token_accuracy"],
    }
    summary["best_run"] = max(
        successful,
        key=lambda result: float(result.get("train_metrics", {}).get("train_steps_per_second", 0.0)),
    )

    samples_per_second = summary["median_train_samples_per_second"]
    if _finite_metric(samples_per_second) and float(samples_per_second) > 0:
        total_examples = full_train_examples * runtime_defaults["full_epochs"]
        summary["estimated_full_train_seconds"] = round(total_examples / float(samples_per_second), 2)

    return summary


def parent_main():
    warn_if_rosetta()
    results = []
    runtime_defaults = runtime_defaults_from_env()

    base = selected_base_configs()
    full_train_examples = 0

    try:
        from datasets import load_from_disk

        dataset = load_from_disk(DATASET_PATH)
        full_train_examples = len(dataset["train"])
    except Exception:
        full_train_examples = 0

    repeats = max(1, runtime_defaults["repeats"])

    for config in base:
        for repeat_index in range(repeats):
            run_config = dict(config)
            if repeats > 1:
                run_config["repeat"] = repeat_index + 1
                run_config["name"] = f"{config['name']}_r{repeat_index + 1}"
            print(f"Running benchmark: {run_config['name']}")
            result = run_child_process(run_config)
            results.append(result)

    grouped = {}
    for result in results:
        signature = config_signature(result["config"])
        grouped.setdefault(signature, []).append(result)

    aggregate_results = [
        aggregate_group(group, runtime_defaults, full_train_examples) for group in grouped.values()
    ]

    best = choose_best_config(aggregate_results)
    tuned_results = []

    if best is not None and not os.getenv("MPS_BENCH_SKIP_TUNED_RUNS", "").strip():
        best_base = dict(best["config"])
        for suffix, flag_name in [("prefer_metal", "prefer_metal"), ("fast_math", "fast_math")]:
            if best_base.get(flag_name):
                continue
            tuned_config = dict(best_base)
            tuned_config["name"] = f"{best_base['name']}_{suffix}"
            tuned_config[flag_name] = True
            print(f"Running benchmark: {tuned_config['name']}")
            tuned_results.append(run_child_process(tuned_config))

    all_results = results + tuned_results
    grouped = {}
    for result in all_results:
        signature = config_signature(result["config"])
        grouped.setdefault(signature, []).append(result)

    aggregate_results = [
        aggregate_group(group, runtime_defaults, full_train_examples) for group in grouped.values()
    ]
    best = choose_best_config(aggregate_results)

    report = {
        "machine": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "defaults": runtime_defaults,
        "runs": all_results,
        "aggregates": aggregate_results,
        "selected_base_run": best,
        "recommendation": None,
    }

    if best is not None:
        best_metrics = best.get("best_run", {}).get("train_metrics", {})
        best_eval = best.get("best_run", {}).get("eval_metrics", {})
        report["recommendation"] = {
            "name": best["config"]["name"],
            "batch_size": best["config"]["batch_size"],
            "group_by_length": best["config"].get("group_by_length", False),
            "gradient_checkpointing": best["config"].get("gradient_checkpointing", False),
            "dataloader_num_workers": best["config"].get("dataloader_num_workers"),
            "prefer_metal": best["config"].get("prefer_metal", False),
            "fast_math": best["config"].get("fast_math", False),
            "median_train_samples_per_second": best.get("median_train_samples_per_second"),
            "median_train_steps_per_second": best.get("median_train_steps_per_second"),
            "median_eval_token_accuracy": best.get("median_eval_token_accuracy"),
            "estimated_full_train_seconds": best.get("estimated_full_train_seconds"),
            "best_run_train_samples_per_second": best_metrics.get("train_samples_per_second"),
            "best_run_train_steps_per_second": best_metrics.get("train_steps_per_second"),
            "best_run_eval_token_accuracy": best_eval.get("eval_token_accuracy", best_eval.get("token_accuracy")),
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["recommendation"], ensure_ascii=False, indent=2) if report["recommendation"] else "{}")
    print(f"Saved benchmark report to {OUTPUT_PATH}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", help="run a single benchmark config as a child process")
    args = parser.parse_args(argv)

    if args.child:
        run_child(args.child)
        return

    parent_main()


if __name__ == "__main__":
    main()
