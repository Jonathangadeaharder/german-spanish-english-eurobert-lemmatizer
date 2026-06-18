from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from language_assets import LANGUAGE_NAMES, normalize_lang
from lmstudio_client import LMStudioClient, parse_sentence_lines
from vocab_inventory import LEVELS, load_vocab_entries

DEFAULT_OUTPUT_DIR = Path("data/silver")
SYSTEM_PROMPT = (
    "You generate concise natural language training sentences. "
    "Return only sentence lines, no commentary."
)


@dataclass(frozen=True)
class SilverJob:
    lang: str
    level: str
    seed: str
    prompt: str


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def sentence_prompt(lang: str, level: str, entry: str, count: int) -> str:
    language_name = LANGUAGE_NAMES[lang]
    return (
        f"Write {count} natural {language_name} sentences at CEFR {level} level. "
        f"Each sentence must contain the vocabulary item '{entry}' or a natural inflected "
        "form of it. Return only one sentence per line. Do not number the lines."
    )


def generate_job(client: LMStudioClient, job: SilverJob) -> dict[str, object]:
    text = client.chat(job.prompt, system_prompt=SYSTEM_PROMPT)
    return {
        "lang": job.lang,
        "level": job.level,
        "seed": job.seed,
        "sentences": parse_sentence_lines(text),
        "raw_output": text,
    }


def main():
    lang = normalize_lang()
    lmstudio_url = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234")
    model = os.getenv("LMSTUDIO_MODEL", "gemma-4-e2b-it-qat")
    sentences_per_entry = env_int("SILVER_SENTENCES_PER_ENTRY", 5)
    parallel = max(1, env_int("SILVER_PARALLEL", 4))
    limit = env_int("SILVER_LIMIT", 0)
    output_dir = Path(os.getenv("SILVER_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"lemma_{lang}_raw.jsonl"

    entries = [
        entry
        for entry in load_vocab_entries()
        if entry.lang == lang and entry.level in LEVELS and entry.canonical
    ]
    if limit > 0:
        entries = entries[:limit]

    client = LMStudioClient(base_url=lmstudio_url, model=model)
    write_lock = threading.Lock()
    seen_keys = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen_keys.add((row.get("level"), row.get("seed")))

    jobs = []
    for entry in entries:
        key = (entry.level, entry.entry)
        if key in seen_keys:
            continue
        jobs.append(
            SilverJob(
                lang=lang,
                level=entry.level,
                seed=entry.entry,
                prompt=sentence_prompt(lang, entry.level, entry.entry, sentences_per_entry),
            )
        )

    print(f"Generating {len(jobs)} {lang} silver batches with parallel={parallel}")

    with output_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(generate_job, client, job): job for job in jobs}

            for future in as_completed(futures):
                job = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001 - preserve batch progress
                    row = {
                        "lang": job.lang,
                        "level": job.level,
                        "seed": job.seed,
                        "sentences": [],
                        "raw_output": "",
                        "error": str(exc),
                    }

                with write_lock:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()

    print(f"Wrote raw silver sentences to {output_path}")


if __name__ == "__main__":
    main()
