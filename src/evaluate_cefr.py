from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer

from edit_trees import apply_edit_label
from language_assets import language_assets
from multitask_model import EuroBertForUposLemma, EuroBertUposLemmaConfig

MAX_LENGTH = 256
CEFR_DATA_DIR = Path("data/cefr_eval")
VOCAB_INVENTORY_PATH = Path("artifacts/vocab/canonical_inventory.json")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def env_int(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def load_cefr_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error"):
                continue
            for sentence in row.get("sentences", []):
                rows.append(
                    {
                        "lang": row["lang"],
                        "level": row["level"],
                        "term": row["term"],
                        "sentence": sentence,
                    }
                )
    return rows


def find_term_index(words: list[str], term: str) -> int | None:
    term_lower = term.lower()
    for idx, word in enumerate(words):
        if word.lower() == term_lower:
            return idx
    return None


def resolve_base_model_source(model_dir: str) -> str:
    adapter_config = Path(model_dir) / "adapter_config.json"
    if adapter_config.exists():
        data = load_json(adapter_config)
        base_model = data.get("base_model_name_or_path")
        if isinstance(base_model, str) and base_model.strip():
            return base_model
    return "EuroBERT/EuroBERT-210m"


def main():
    assets = language_assets()
    model_dir = os.getenv("MODEL_DIR", str(assets.output_dir))
    data_dir = Path(os.getenv("CEFR_EVAL_DIR", str(CEFR_DATA_DIR)))
    path = data_dir / f"{assets.lang}.jsonl"
    eval_limit = env_int("EVAL_LIMIT", 0)
    batch_size = max(1, env_int("EVAL_BATCH_SIZE", 8))

    tokenizer = AutoTokenizer.from_pretrained(str(assets.tokenizer_dir), trust_remote_code=True)
    device = get_device()

    # Load label mappings from the MODEL's config (not the artifacts), because
    # the artifacts may have been regenerated with different label IDs after
    # training (FR has 1383/1384 mismatches). The config's lemma_id2label is
    # the ground truth the model was trained with.
    model_config_path = Path(model_dir) / "config.json"
    if model_config_path.exists():
        model_config = load_json(model_config_path)
        lemma_label2id = model_config.get("lemma_label2id", load_json(assets.label2id_path))
        # Convert string IDs to int (JSON keys are strings)
        lemma_label2id = {k: int(v) for k, v in lemma_label2id.items()}
        upos_label2id = model_config.get("upos_label2id", load_json(assets.upos_label2id_path))
        upos_label2id = {k: int(v) for k, v in upos_label2id.items()}
        # Build id2label from the config's label2id (inverted)
        id2label = {str(v): k for k, v in lemma_label2id.items()}
    else:
        lemma_label2id = load_json(assets.label2id_path)
        upos_label2id = load_json(assets.upos_label2id_path)
        id2label = load_json(assets.id2label_path)

    base_model_source = resolve_base_model_source(model_dir)
    config = EuroBertUposLemmaConfig(
        base_model_name_or_path=base_model_source,
        upos_label2id=upos_label2id,
        lemma_label2id=lemma_label2id,
    )

    base_model = EuroBertForUposLemma.from_pretrained(
        base_model_source,
        config=config,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    if Path(model_dir, "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        model = EuroBertForUposLemma.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    model.eval()

    rows = load_cefr_rows(path)
    if eval_limit > 0:
        rows = rows[: min(eval_limit, len(rows))]

    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    sample_errors = defaultdict(list)
    # id2label was set above from the model's config (or artifacts as fallback).

    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            words_batch = []
            metadata = []
            for row in batch:
                words = row["sentence"].split()
                idx = find_term_index(words, row["term"])
                if idx is None:
                    continue
                words_batch.append(words)
                metadata.append((row, idx))

            if not words_batch:
                continue

            encoded = tokenizer(
                words_batch,
                is_split_into_words=True,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            outputs = model(**{k: v.to(device) for k, v in encoded.items()})
            lemma_logits = outputs.logits[1].detach().cpu().numpy()

            for batch_index, (row, idx) in enumerate(metadata):
                word_ids = encoded.word_ids(batch_index=batch_index)
                if idx < 0 or idx >= len(row["sentence"].split()):
                    continue
                words = row["sentence"].split()
                predicted = None
                for token_idx, word_id in enumerate(word_ids):
                    if word_id == idx:
                        label_id = int(np.argmax(lemma_logits[batch_index][token_idx]))
                        predicted = label_id
                        break

                level = row["level"]
                lemma_label = (
                    id2label.get(str(predicted), "UNKNOWN") if predicted is not None else "UNKNOWN"
                )
                stats[level]["total"] += 1
                predicted_lemma = (
                    apply_edit_label(words[idx], lemma_label)
                    if lemma_label not in {"UNKNOWN", "IDENTITY", "LOWERCASE"}
                    else words[idx].lower()
                )
                correct = (
                    predicted_lemma is not None and predicted_lemma.lower() == row["term"].lower()
                )
                if correct:
                    stats[level]["correct"] += 1
                elif len(sample_errors[level]) < 8:
                    sample_errors[level].append(
                        {
                            "term": row["term"],
                            "level": level,
                            "sentence": row["sentence"],
                            "predicted": lemma_label,
                            "predicted_lemma": predicted_lemma,
                        }
                    )

    report = {"lang": assets.lang, "levels": {}}
    for level in ["A1", "A2", "B1", "B2", "C1"]:
        total = stats[level]["total"]
        correct = stats[level]["correct"]
        lo, hi = wilson_interval(correct, total)
        report["levels"][level] = {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "wilson_95_ci": [round(lo, 4), round(hi, 4)],
            "examples": sample_errors[level],
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
