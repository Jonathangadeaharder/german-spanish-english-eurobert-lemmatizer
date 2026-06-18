from __future__ import annotations

import csv
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from language_assets import LANGUAGE_NAMES, vocab_levels_root
from lmstudio_client import LMStudioClient, parse_sentence_lines

DEFAULT_OUTPUT_DIR = Path("data/silver/de")
LMSTUDIO_MODEL = "gemma-4-e2b-it-qat"
LMSTUDIO_URL = "http://127.0.0.1:1234"


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return default if value is None or value == "" else Path(value)


@dataclass(frozen=True)
class Seed:
    level: str
    lemma: str


@dataclass(frozen=True)
class Job:
    level: str
    lemma: str
    prompt: str


def load_german_seeds() -> list[Seed]:
    root = vocab_levels_root() / LANGUAGE_NAMES["de"]
    seeds: list[Seed] = []
    for level in ["A1", "A2", "B1", "B2", "C1"]:
        path = root / f"{level}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                lemma = (row.get("German_Lemma") or "").strip()
                if lemma and " " not in lemma:
                    seeds.append(Seed(level=level, lemma=lemma))
    return seeds


def make_prompt(level: str, lemma: str, count: int) -> str:
    return (
        f"Schreibe {count} natürliche deutsche Sätze auf CEFR-{level}-Niveau. "
        f"Jeder Satz muss das Wort '{lemma}' oder eine passende Flexionsform davon enthalten. "
        "Gib nur die Sätze, genau eine Zeile pro Satz, ohne Nummerierung und ohne Erklärung."
    )


def load_spacy():
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "spaCy is required. Install it and download de_core_news_lg with:\n"
            "python -m spacy download de_core_news_lg"
        ) from exc

    try:
        return spacy.load("de_core_news_lg")
    except OSError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "Missing spaCy model de_core_news_lg. Install it with:\n"
            "python -m spacy download de_core_news_lg"
        ) from exc


def annotate_sentence(nlp, text: str) -> dict[str, object] | None:
    doc = nlp(text.strip())
    words = [token.text for token in doc if not token.is_space]
    lemmas = [token.lemma_ for token in doc if not token.is_space]
    upos = [token.pos_ for token in doc if not token.is_space]

    if not words or len(words) != len(lemmas):
        return None

    return {
        "text": text.strip(),
        "words": words,
        "lemmas": lemmas,
        "upos": upos,
    }


def generate_job(client: LMStudioClient, job: Job, sentences_per_seed: int) -> dict[str, object]:
    raw = client.chat(
        make_prompt(job.level, job.lemma, sentences_per_seed),
        system_prompt=(
            "Du erzeugst kurze deutsche Trainingssätze. Antworte nur mit den Sätzen, "
            "ohne Erklärungen, ohne Markdown, ohne Nummerierung."
        ),
    )
    return {
        "level": job.level,
        "lemma": job.lemma,
        "raw_output": raw,
        "sentences": parse_sentence_lines(raw),
    }


def main():
    limit = env_int("SILVER_LIMIT", 0)
    sentences_per_seed = env_int("SILVER_SENTENCES_PER_SEED", 1)
    parallel = max(1, env_int("SILVER_PARALLEL", 4))
    output_dir = env_path("SILVER_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "de_lemma_silver.jsonl"

    seeds = load_german_seeds()
    if limit > 0:
        seeds = seeds[:limit]

    client = LMStudioClient(
        base_url=os.getenv("LMSTUDIO_URL", LMSTUDIO_URL),
        model=os.getenv("LMSTUDIO_MODEL", LMSTUDIO_MODEL),
    )
    nlp = load_spacy()
    write_lock = threading.Lock()

    existing = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                existing.add((row.get("level"), row.get("lemma")))

    jobs = [
        Job(
            level=seed.level,
            lemma=seed.lemma,
            prompt=make_prompt(seed.level, seed.lemma, sentences_per_seed),
        )
        for seed in seeds
        if (seed.level, seed.lemma) not in existing
    ]

    with output_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(generate_job, client, job, sentences_per_seed): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                row = {
                    "level": job.level,
                    "lemma": job.lemma,
                    "sentences": [],
                    "raw_output": "",
                }
                try:
                    generated = future.result()
                    row.update(generated)
                except Exception as exc:  # noqa: BLE001 - keep batch progress
                    row["error"] = str(exc)

                annotated = []
                for sentence in row.get("sentences", []):
                    result = annotate_sentence(nlp, sentence)
                    if result is not None:
                        annotated.append(result)

                row["annotated_sentences"] = annotated
                with write_lock:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()

    print(f"Wrote German silver data to {output_path}")


if __name__ == "__main__":
    main()
