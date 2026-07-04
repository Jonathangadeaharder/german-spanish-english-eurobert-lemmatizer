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

DEFAULT_OUTPUT_DIR = Path("data/cefr_eval")
SYSTEM_PROMPT = (
    "You generate concise natural language training sentences. "
    "Return only sentence lines, no commentary."
)


@dataclass(frozen=True)
class EvalJob:
    lang: str
    level: str
    term: str
    prompt: str


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def sentence_prompt(lang: str, level: str, term: str, count: int) -> str:
    language_name = LANGUAGE_NAMES[lang]
    return (
        f"Write {count} natural {language_name} sentences at CEFR {level} level. "
        f"Each sentence MUST contain an INFLECTED form of the word '{term}' — "
        f"NOT the dictionary form itself. Use conjugated verbs, plural nouns, "
        f"declined adjectives, or case-marked forms. The sentence must contain "
        f"a word that, when lemmatized, yields '{term}'. "
        f"Return only one sentence per line. Do not number the lines."
    )


def generate_job(client: LMStudioClient, job: EvalJob) -> dict[str, object]:
    text = client.chat(job.prompt, system_prompt=SYSTEM_PROMPT)
    return {
        "lang": job.lang,
        "level": job.level,
        "term": job.term,
        "sentences": parse_sentence_lines(text),
        "raw_output": text,
    }


def main():
    lang = normalize_lang()
    lmstudio_url = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234")
    model = os.getenv("LMSTUDIO_MODEL", "gemma-4-e2b-it-qat")
    parallel = max(1, env_int("SILVER_PARALLEL", 4))
    per_level_limit = env_int("CEFR_EVAL_LIMIT_PER_LEVEL", 100)
    sentences_per_term = env_int("CEFR_EVAL_SENTENCES_PER_TERM", 1)
    output_dir = Path(os.getenv("CEFR_EVAL_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{lang}.jsonl"

    entries = [
        entry
        for entry in load_vocab_entries()
        if entry.lang == lang and entry.level in LEVELS and entry.canonical
    ]

    by_level: dict[str, list] = {level: [] for level in LEVELS}
    for entry in entries:
        by_level[entry.level].append(entry)

    jobs = []
    for level in LEVELS:
        level_entries = by_level[level][:per_level_limit]
        for entry in level_entries:
            jobs.append(
                EvalJob(
                    lang=lang,
                    level=entry.level,
                    term=entry.entry,
                    prompt=sentence_prompt(lang, entry.level, entry.entry, sentences_per_term),
                )
            )

    client = LMStudioClient(base_url=lmstudio_url, model=model)
    write_lock = threading.Lock()
    seen = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen.add((row.get("level"), row.get("term")))

    todo = [job for job in jobs if (job.level, job.term) not in seen]
    print(f"Generating {len(todo)} CEFR eval jobs for {lang} with parallel={parallel}")

    with output_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(generate_job, client, job): job for job in todo}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001 - preserve progress
                    row = {
                        "lang": job.lang,
                        "level": job.level,
                        "term": job.term,
                        "sentences": [],
                        "raw_output": "",
                        "error": str(exc),
                    }
                with write_lock:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()

    print(f"Wrote CEFR eval sentences to {output_path}")


if __name__ == "__main__":
    main()
