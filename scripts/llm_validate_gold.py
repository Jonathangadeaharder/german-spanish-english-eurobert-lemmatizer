#!/usr/bin/env python3
"""LLM-powered CoNLL-U gold file validator using skainet GLM-5.2.

Uses httpx directly (no pydantic-ai dependency required).

4 endpoint variants:
  1. internal-direct  — chat.model.tngtech.com, GLM-5.2, direct
  2. internal-batch   — chat.model.tngtech.com, GLM-5.2, TNG batch API
  3. external-glm52   — external.model.tngtech.com, GLM-5.2, direct
  4. external-tee      — external.model.tngtech.com, GLM-5.2-TEE, direct

Timeout + retry strategy:
  - Connect timeout: 30s, read timeout: 180s, write timeout: 30s, pool timeout: 30s
  - Exponential backoff with jitter on retryable failures (429, 500-504, timeout, connect error)
  - Non-retryable errors (400, 401, 403, 404) fail immediately
  - Parse failures (invalid JSON from LLM) retried up to MAX_RETRIES
  - Partial results saved after each file to survive crashes
  - Failed batches tracked and reported in summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

GOLD_DIR = Path(__file__).resolve().parent.parent / "data" / "gold"
REPORT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "llm_validation"
LANGUAGES = ["de", "en", "es", "fr", "nl", "sv", "zh", "ar"]

ENDPOINTS: dict[str, dict[str, Any]] = {
    "internal-direct": {
        "label": "Skainet Internal (GLM-5.2, direct)",
        "base_url": "https://chat.model.tngtech.com/v1",
        "model": "zai-org/GLM-5.2",
        "batch": False,
    },
    "internal-batch": {
        "label": "Skainet Internal (GLM-5.2, batch API)",
        "base_url": "https://chat.model.tngtech.com/v1",
        "batch_api_url": "https://chat.model.tngtech.com/v1/batches",
        "model": "zai-org/GLM-5.2",
        "batch": True,
    },
    "external-glm52": {
        "label": "Skainet External (GLM-5.2, direct)",
        "base_url": "https://external.model.tngtech.com/v1",
        "model": "zai-org/GLM-5.2",
        "batch": False,
    },
    "external-tee": {
        "label": "Skainet External (GLM-5.2-TEE, direct)",
        "base_url": "https://external.model.tngtech.com/v1",
        "model": "zai-org/GLM-5.2-TEE",
        "batch": False,
    },
}

CONCURRENCY = {"internal-direct": 3, "internal-batch": 1, "external-glm52": 5, "external-tee": 5}
BATCH_POLL_INTERVAL = 3.0
BATCH_POLL_TIMEOUT = 900.0
BATCH_CHUNK_SIZE = 50
SENTENCES_PER_LLM_CALL = 5
TEST_SENTENCES_PER_LANG = 3

MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0
RETRY_MAX_DELAY = 60.0
RETRY_JITTER = 1.5

HTTP_TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=180.0,
    write=30.0,
    pool=30.0,
)
BATCH_HTTP_TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=60.0,
    write=30.0,
    pool=30.0,
)
POLL_HTTP_TIMEOUT = httpx.Timeout(
    connect=15.0,
    read=30.0,
    write=10.0,
    pool=15.0,
)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

LEMMA_RULES: dict[str, str] = {
    "de": (
        "German: NOUN lemmas capitalized; VERB infinitives (-en/-n); "
        "ADJ base form; PUNCT lemma=form; modern spelling (ss not ss after "
        "short vowels, but ss and ss are correct after long vowels); "
        "contracted preps: am->an, im->in, zum->zu, zur->zu"
    ),
    "en": (
        "English: VERB base form (go not goes/going/went); "
        "NOUN singular; ADJ positive; PUNCT lemma=form"
    ),
    "es": (
        "Spanish: VERB infinitive (-ar/-er/-ir/-se); "
        "NOUN singular; ADJ masc singular; PUNCT lemma=form"
    ),
    "fr": (
        "French: VERB infinitive (-er/-ir/-re/-oir); past participles "
        "(-e, -u, -i) and present participles (-ant) NOT valid; "
        "NOUN singular; ADJ masc singular; PUNCT lemma=form"
    ),
    "nl": (
        "Dutch: VERB infinitive (usually -en; spelen, waarborgen); "
        "singular present NOT valid verb lemma; NOUN singular; "
        "compound lemmas single word no underscores; PUNCT lemma=form"
    ),
    "sv": (
        "Swedish: VERB infinitive (usually -a; spela, rekommendera); "
        "present (-ar/-er), past (-ade/-de), past participle (-ad/-t) "
        "NOT valid; NOUN singular indefinite; ADJ base; PUNCT lemma=form"
    ),
    "zh": "Chinese: lemma equals form for ALL tokens (no inflection)",
    "ar": (
        "Arabic: lemmas in Arabic script; PUNCT lemma=form; "
        "pronoun lemmas match form; question particle hal=PART not VERB"
    ),
}


# --- Models ---


class TokenValidation(BaseModel):
    line_number: int = Field(description="Line number in the CoNLL-U file")
    form: str = Field(description="Token form (column 2)")
    current_lemma: str = Field(description="Current lemma in gold file")
    upos: str = Field(description="UPOS tag (column 4)")
    is_correct: bool = Field(description="True if lemma is correct")
    suggested_lemma: str | None = Field(default=None, description="Correct lemma if wrong")
    reason: str = Field(description="Brief explanation")


class BatchValidation(BaseModel):
    validations: list[TokenValidation] = Field(description="One entry per token")


# --- Data classes ---


@dataclass
class Token:
    line_number: int
    token_id: str
    form: str
    lemma: str
    upos: str


@dataclass
class Sentence:
    sent_id: str
    text: str
    tokens: list[Token] = field(default_factory=list)
    start_line: int = 0


@dataclass
class BatchResult:
    """Result of validating one batch of sentences."""

    validation: BatchValidation | None = None
    error: str | None = None
    attempts: int = 0
    elapsed: float = 0.0


# --- CoNLL-U parsing ---


def parse_conllu(path: Path) -> list[Sentence]:
    sentences: list[Sentence] = []
    current: Sentence | None = None
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("# sent_id"):
            if current is not None:
                sentences.append(current)
            sid = line.split("=", 1)[1].strip() if "=" in line else ""
            current = Sentence(sent_id=sid, text="", start_line=i)
            continue
        if line.startswith("# text"):
            if current is not None:
                current.text = line.split("=", 1)[1].strip() if "=" in line else ""
            continue
        if not line.strip():
            if current is not None:
                sentences.append(current)
                current = None
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 4:
            continue
        tid = cols[0]
        if "-" in tid or "." in tid:
            continue
        if current is None:
            current = Sentence(sent_id="", text="", start_line=i)
        current.tokens.append(
            Token(line_number=i, token_id=tid, form=cols[1], lemma=cols[2], upos=cols[3])
        )
    if current is not None:
        sentences.append(current)
    return sentences


# --- Prompt building ---


def build_system_prompt(lang: str) -> str:
    rules = LEMMA_RULES.get(lang, "Standard: lemma is the dictionary form.")
    return (
        f"You are a lemmatization expert validating CoNLL-U gold data for '{lang}'.\n\n"
        f"{rules}\n\n"
        "You receive batches of sentences with tokens (line_number, form, "
        "current_lemma, upos). For EACH token, check if current_lemma is the "
        "correct dictionary lemma for that form+UPOS. Return a validation entry "
        "for EVERY token (including correct ones with is_correct=true). For "
        "wrong lemmas, set is_correct=false and give suggested_lemma. "
        "Trust the UPOS tag; only validate the lemma. Be strict but not pedantic.\n\n"
        "Respond with ONLY a JSON object in this exact format (no markdown):\n"
        '{"validations": [{"line_number": 0, "form": "...", '
        '"current_lemma": "...", "upos": "...", "is_correct": true, '
        '"suggested_lemma": null, "reason": "..."}, ...]}'
    )


def build_user_prompt(sentences: list[Sentence]) -> str:
    parts: list[str] = []
    for sent in sentences:
        parts.append(f"--- {sent.sent_id} ---")
        parts.append(f"Text: {sent.text}")
        parts.append("Tokens:")
        for tok in sent.tokens:
            parts.append(
                f"  line={tok.line_number} form={tok.form} lemma={tok.lemma} upos={tok.upos}"
            )
        parts.append("")
    return "\n".join(parts)


# --- API helpers ---


def resolve_api_key() -> str:
    for var in ("SKAINET_API_KEY", "API_KEY", "SKAINET_EXTERNAL_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val
    raise RuntimeError("No API key. Set SKAINET_API_KEY or API_KEY env var.")


def build_chat_request(
    endpoint: dict[str, Any], lang: str, sentences: list[Sentence]
) -> dict[str, Any]:
    return {
        "model": endpoint["model"],
        "messages": [
            {"role": "system", "content": build_system_prompt(lang)},
            {"role": "user", "content": build_user_prompt(sentences)},
        ],
        "temperature": 0.0,
        "max_tokens": 16384,
    }


def strip_json_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        start = 1
        end = len(lines)
        if lines[-1].strip().startswith("```"):
            end = -1
        content = "\n".join(lines[start:end])
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        content = match.group(0)
    return content.strip()


def parse_llm_response(content: str) -> BatchValidation | None:
    content = strip_json_fences(content)
    try:
        return BatchValidation.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  WARN: parse failed: {e}", file=sys.stderr)
        print(f"  Content preview: {content[:300]}", file=sys.stderr)
        return None


def compute_backoff(attempt: int) -> float:
    """Exponential backoff with jitter: base * 2^attempt + random(0, jitter)."""
    delay = RETRY_BASE_DELAY * (2**attempt)
    delay = min(delay, RETRY_MAX_DELAY)
    delay += random.uniform(0, RETRY_JITTER)
    return delay


def is_retryable_status(status: int) -> bool:
    return status in RETRYABLE_STATUS


def is_retryable_exception(exc: Exception) -> bool:
    """Check if an exception is worth retrying (network/timeout errors)."""
    if isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.PoolTimeout,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.WriteTimeout,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return is_retryable_status(exc.response.status_code)
    if isinstance(exc, (OSError, ConnectionError, asyncio.TimeoutError)):
        return True
    return False


# --- Direct API validation with retry ---


async def validate_direct_httpx(
    client: httpx.AsyncClient,
    endpoint: dict[str, Any],
    api_key: str,
    lang: str,
    sentences: list[Sentence],
    supports_response_format: bool,
) -> BatchResult:
    """Validate a batch of sentences via direct HTTP API call with full retry logic."""
    result = BatchResult()
    request = build_chat_request(endpoint, lang, sentences)
    url = f"{endpoint['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "tng/eurobert-lemmatizer",
    }

    for attempt in range(MAX_RETRIES):
        result.attempts = attempt + 1
        start_time = time.time()

        try:
            req = dict(request)
            if supports_response_format:
                req["response_format"] = {"type": "json_object"}

            resp = await client.post(url, json=req, headers=headers, timeout=HTTP_TIMEOUT)
            result.elapsed += time.time() - start_time

            # Rate limited — retry with backoff
            if resp.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    wait = compute_backoff(attempt)
                    print(
                        f"  429 rate limited (attempt {attempt + 1}/{MAX_RETRIES}), "
                        f"waiting {wait:.1f}s...",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(wait)
                    continue
                result.error = f"Rate limited after {MAX_RETRIES} attempts"
                return result

            # response_format not supported — strip and retry immediately
            if resp.status_code == 400 and "response_format" in resp.text:
                print("  response_format not supported, retrying without...", file=sys.stderr)
                supports_response_format = False
                continue

            # Other retryable server errors
            if is_retryable_status(resp.status_code):
                if attempt < MAX_RETRIES - 1:
                    wait = compute_backoff(attempt)
                    print(
                        f"  HTTP {resp.status_code} (attempt {attempt + 1}/{MAX_RETRIES}), "
                        f"waiting {wait:.1f}s...",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(wait)
                    continue
                result.error = f"HTTP {resp.status_code} after {MAX_RETRIES} attempts"
                return result

            # Non-retryable HTTP errors
            if resp.status_code >= 400:
                body = resp.text[:500]
                result.error = f"HTTP {resp.status_code}: {body}"
                return result

            # Parse response
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                result.error = "No choices in API response"
                if attempt < MAX_RETRIES - 1:
                    wait = compute_backoff(attempt)
                    print(
                        f"  No choices (attempt {attempt + 1}/{MAX_RETRIES}), "
                        f"retrying in {wait:.1f}s...",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(wait)
                    continue
                return result

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                result.error = "Empty content in API response"
                if attempt < MAX_RETRIES - 1:
                    wait = compute_backoff(attempt)
                    print(
                        f"  Empty content (attempt {attempt + 1}/{MAX_RETRIES}), "
                        f"retrying in {wait:.1f}s...",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(wait)
                    continue
                return result

            parsed = parse_llm_response(content)
            if parsed is None:
                # JSON parse failure — retry with fresh request
                if attempt < MAX_RETRIES - 1:
                    wait = compute_backoff(attempt)
                    print(
                        f"  Parse failure (attempt {attempt + 1}/{MAX_RETRIES}), "
                        f"retrying in {wait:.1f}s...",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(wait)
                    continue
                result.error = "JSON parse failure after all retries"
                return result

            result.validation = parsed
            return result

        except httpx.TimeoutException as e:
            result.elapsed += time.time() - start_time
            if attempt < MAX_RETRIES - 1:
                wait = compute_backoff(attempt)
                print(
                    f"  Timeout: {e} (attempt {attempt + 1}/{MAX_RETRIES}), waiting {wait:.1f}s...",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)
                continue
            result.error = f"Timeout after {MAX_RETRIES} attempts: {e}"
            return result

        except (httpx.ConnectError, httpx.PoolTimeout, ConnectionError) as e:
            result.elapsed += time.time() - start_time
            if attempt < MAX_RETRIES - 1:
                wait = compute_backoff(attempt)
                print(
                    f"  Connection error: {e} (attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"waiting {wait:.1f}s...",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)
                continue
            result.error = f"Connection error after {MAX_RETRIES} attempts: {e}"
            return result

        except httpx.HTTPStatusError as e:
            result.elapsed += time.time() - start_time
            status = e.response.status_code
            if is_retryable_status(status) and attempt < MAX_RETRIES - 1:
                wait = compute_backoff(attempt)
                print(
                    f"  HTTP {status} (attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"waiting {wait:.1f}s...",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)
                continue
            result.error = f"HTTP {status}: {e}"
            return result

        except Exception as e:
            result.elapsed += time.time() - start_time
            if is_retryable_exception(e) and attempt < MAX_RETRIES - 1:
                wait = compute_backoff(attempt)
                print(
                    f"  {type(e).__name__}: {e} (attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"waiting {wait:.1f}s...",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)
                continue
            result.error = f"{type(e).__name__}: {e}"
            return result

    result.error = f"Exhausted all {MAX_RETRIES} retries"
    return result


# --- Batch API with retry ---


async def submit_batch_with_retry(
    client: httpx.AsyncClient,
    batch_api_url: str,
    api_key: str,
    requests: list[dict[str, Any]],
) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "tng/eurobert-lemmatizer"}
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(
                batch_api_url,
                json={"requests": requests, "endpoint": "/v1/chat/completions"},
                headers=headers,
                timeout=BATCH_HTTP_TIMEOUT,
            )
            if resp.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES - 1:
                wait = compute_backoff(attempt)
                print(
                    f"  Batch submit HTTP {resp.status_code} "
                    f"(attempt {attempt + 1}/{MAX_RETRIES}), waiting {wait:.1f}s...",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            bid = data.get("batch_id") or data.get("id")
            if not bid:
                raise RuntimeError(f"No batch_id in response: {data}")
            return str(bid)
        except Exception as e:
            if is_retryable_exception(e) and attempt < MAX_RETRIES - 1:
                wait = compute_backoff(attempt)
                print(
                    f"  Batch submit error: {e} (attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"waiting {wait:.1f}s...",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)
                continue
            raise


async def poll_batch(
    client: httpx.AsyncClient, batch_api_url: str, api_key: str, batch_id: str
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "tng/eurobert-lemmatizer"}
    url = f"{batch_api_url}/{batch_id}"
    deadline = time.time() + BATCH_POLL_TIMEOUT
    last_status = ""
    while time.time() < deadline:
        try:
            resp = await client.get(url, headers=headers, timeout=POLL_HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")
            counts = data.get("request_counts", {})
            done = counts.get("completed", 0) + counts.get("failed", 0)
            total = counts.get("total", 0)
            if status != last_status or done != total:
                print(f"  batch {batch_id}: {status} ({done}/{total})", file=sys.stderr)
                last_status = status
            if status == "completed":
                return data
            if status == "failed":
                raise RuntimeError(f"Batch {batch_id} failed")
            await asyncio.sleep(BATCH_POLL_INTERVAL)
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            print(f"  Poll error (transient): {e}", file=sys.stderr)
            await asyncio.sleep(BATCH_POLL_INTERVAL)
            continue
    raise TimeoutError(f"Batch {batch_id} timeout after {BATCH_POLL_TIMEOUT}s")


async def validate_batch(
    endpoint: dict[str, Any],
    api_key: str,
    lang: str,
    all_batches: list[list[Sentence]],
) -> list[BatchResult]:
    results: list[BatchResult] = []
    async with httpx.AsyncClient() as client:
        for start in range(0, len(all_batches), BATCH_CHUNK_SIZE):
            chunk = all_batches[start : start + BATCH_CHUNK_SIZE]
            reqs = [build_chat_request(endpoint, lang, sents) for sents in chunk]
            print(f"  Submitting {len(reqs)} requests as batch...", file=sys.stderr)
            try:
                batch_id = await submit_batch_with_retry(
                    client, endpoint["batch_api_url"], api_key, reqs
                )
                print(f"  batch_id={batch_id}, polling...", file=sys.stderr)
                batch_data = await poll_batch(client, endpoint["batch_api_url"], api_key, batch_id)
                for raw_resp in batch_data.get("responses", []):
                    parsed = parse_batch_response(raw_resp)
                    br = BatchResult(validation=parsed)
                    if parsed is None:
                        br.error = "Parse failure in batch response"
                    results.append(br)
            except Exception as e:
                print(f"  Batch chunk failed: {e}", file=sys.stderr)
                for _ in chunk:
                    results.append(BatchResult(error=str(e)))
    return results


def parse_batch_response(raw: dict[str, Any]) -> BatchValidation | None:
    choices = raw.get("choices", [])
    if not choices:
        return None
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        return None
    return parse_llm_response(content)


# --- Utilities ---


def chunk_sentences(sentences: list[Sentence], size: int) -> list[list[Sentence]]:
    return [sentences[i : i + size] for i in range(0, len(sentences), size)]


def apply_fixes(path: Path, validations: list[TokenValidation]) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    fixes = 0
    for v in validations:
        if v.is_correct or not v.suggested_lemma:
            continue
        if v.line_number < 1 or v.line_number > len(lines):
            continue
        cols = lines[v.line_number - 1].split("\t")
        if len(cols) < 4:
            continue
        if cols[1] != v.form:
            print(
                f"  WARN: line {v.line_number} form mismatch: {cols[1]} vs {v.form}",
                file=sys.stderr,
            )
            continue
        old_lemma = cols[2]
        cols[2] = v.suggested_lemma
        lines[v.line_number - 1] = "\t".join(cols)
        print(
            f"  Fixed L{v.line_number}: {v.form}  {old_lemma} -> {v.suggested_lemma}  ({v.reason})"
        )
        fixes += 1
    if fixes > 0:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fixes


def save_partial_report(
    report_path: Path,
    model: str,
    endpoint_label: str,
    total_checked: int,
    total_errors: int,
    all_errors: list[dict[str, Any]],
    file_results: list[dict[str, Any]],
) -> None:
    """Save partial results so we don't lose progress on crash."""
    report_path.write_text(
        json.dumps(
            {
                "model": model,
                "endpoint": endpoint_label,
                "tokens_checked": total_checked,
                "errors_found": total_errors,
                "errors": all_errors,
                "file_results": file_results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# --- Main ---


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-validate CoNLL-U gold files via skainet GLM-5.2"
    )
    parser.add_argument("--model", required=True, choices=list(ENDPOINTS.keys()))
    parser.add_argument("--test", action="store_true", help="Small test: 3 sentences/lang dev")
    parser.add_argument("--full", action="store_true", help="Process all 16 gold files")
    parser.add_argument("--apply", action="store_true", help="Apply suggested fixes to gold files")
    parser.add_argument("--lang", default=None, help="Process only this language")
    args = parser.parse_args()

    if not args.test and not args.full:
        args.test = True

    api_key = resolve_api_key()
    endpoint = ENDPOINTS[args.model]
    langs = [args.lang] if args.lang else LANGUAGES
    max_sents = TEST_SENTENCES_PER_LANG if args.test else 999999
    if args.test:
        files = [(lang, GOLD_DIR / lang / "dev.conllu") for lang in langs]
    else:
        files = [(lang, GOLD_DIR / lang / "train.conllu") for lang in langs]
        if args.lang is None:
            files += [(lang, GOLD_DIR / lang / "dev.conllu") for lang in langs]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "test" if args.test else "full"
    if args.lang:
        suffix += f"_{args.lang}"
    report_path = REPORT_DIR / f"validation_{args.model}_{suffix}.json"

    all_errors: list[dict[str, Any]] = []
    file_results: list[dict[str, Any]] = []
    total_checked = 0
    total_errors = 0
    total_batches = 0
    total_failed_batches = 0
    supports_rf = True
    overall_start = time.time()

    async with httpx.AsyncClient() as client:
        for lang_idx, (lang, fpath) in enumerate(files):
            if not fpath.exists():
                print(f"SKIP {lang}/{fpath.name}: file not found", file=sys.stderr)
                continue
            sentences = parse_conllu(fpath)[:max_sents]
            if not sentences:
                continue
            batches = chunk_sentences(sentences, SENTENCES_PER_LLM_CALL)
            total_batches += len(batches)
            file_start = time.time()

            print(f"\n{'=' * 60}", file=sys.stderr)
            print(
                f"[{lang_idx + 1}/{len(files)}] {endpoint['label']} | "
                f"{lang}/{fpath.name} | {len(sentences)} sentences | "
                f"{len(batches)} batches",
                file=sys.stderr,
            )
            print(f"{'=' * 60}", file=sys.stderr)

            if endpoint["batch"]:
                batch_results = await validate_batch(endpoint, api_key, lang, batches)
                all_validations = [r.validation for r in batch_results if r.validation]
                total_failed_batches += sum(1 for r in batch_results if r.error)
            else:
                sem = asyncio.Semaphore(CONCURRENCY[args.model])
                completed = 0
                lock = asyncio.Lock()

                async def run_batch(
                    b: list[Sentence],
                    bidx: int,
                    *,
                    sem: asyncio.Semaphore = sem,
                    lang: str = lang,
                    lock: asyncio.Lock = lock,
                    batches: list[list[Sentence]] = batches,
                ) -> BatchResult:
                    nonlocal supports_rf, completed
                    async with sem:
                        result = await validate_direct_httpx(
                            client, endpoint, api_key, lang, b, supports_rf
                        )
                        if result.error and "response_format" in (result.error or ""):
                            supports_rf = False
                        async with lock:
                            completed += 1
                            elapsed = result.elapsed
                            status = "OK" if result.validation else f"FAIL: {result.error}"
                            print(
                                f"  batch {completed}/{len(batches)} ({elapsed:.1f}s) — {status}",
                                file=sys.stderr,
                            )
                        return result

                results = await asyncio.gather(*[run_batch(b, i) for i, b in enumerate(batches)])
                all_validations = [r.validation for r in results if r.validation]
                total_failed_batches += sum(1 for r in results if r.error)

            file_elapsed = time.time() - file_start
            flat: list[TokenValidation] = []
            for bv in all_validations:
                flat.extend(bv.validations)

            errors = [v for v in flat if not v.is_correct]
            total_checked += len(flat)
            total_errors += len(errors)

            print(
                f"\n  {lang}/{fpath.name}: {len(flat)} tokens checked, "
                f"{len(errors)} errors found ({file_elapsed:.1f}s)",
                file=sys.stderr,
            )
            for v in errors[:20]:
                print(
                    f"    L{v.line_number}: {v.form}  {v.current_lemma} -> "
                    f"{v.suggested_lemma}  ({v.reason})",
                    file=sys.stderr,
                )
            if len(errors) > 20:
                print(f"    ... and {len(errors) - 20} more", file=sys.stderr)

            if args.apply and errors:
                fixed = apply_fixes(fpath, errors)
                print(f"  Applied {fixed} fixes to {fpath.name}", file=sys.stderr)

            for v in errors:
                all_errors.append(
                    {
                        "lang": lang,
                        "file": fpath.name,
                        "line": v.line_number,
                        "form": v.form,
                        "old_lemma": v.current_lemma,
                        "suggested_lemma": v.suggested_lemma,
                        "upos": v.upos,
                        "reason": v.reason,
                        "model": args.model,
                    }
                )

            file_results.append(
                {
                    "lang": lang,
                    "file": fpath.name,
                    "tokens_checked": len(flat),
                    "errors_found": len(errors),
                    "batches": len(batches),
                    "failed_batches": sum(
                        1 for r in (results if not endpoint["batch"] else batch_results) if r.error
                    ),
                    "elapsed_s": round(file_elapsed, 1),
                }
            )

            # Checkpoint: save partial results after each file
            save_partial_report(
                report_path,
                args.model,
                endpoint["label"],
                total_checked,
                total_errors,
                all_errors,
                file_results,
            )

    overall_elapsed = time.time() - overall_start

    # Final report
    save_partial_report(
        report_path,
        args.model,
        endpoint["label"],
        total_checked,
        total_errors,
        all_errors,
        file_results,
    )

    print(f"\n{'=' * 60}")
    print(f"DONE: {total_checked} tokens checked, {total_errors} errors found")
    print(f"Batches: {total_batches} total, {total_failed_batches} failed")
    print(f"Elapsed: {overall_elapsed:.1f}s")
    print(f"Report: {report_path}")
    if args.apply:
        print("Fixes applied to gold files.")

    if total_failed_batches > 0:
        print(
            f"\nWARNING: {total_failed_batches} batches failed. "
            f"Re-run failed languages with --lang <code>."
        )


if __name__ == "__main__":
    asyncio.run(main())
