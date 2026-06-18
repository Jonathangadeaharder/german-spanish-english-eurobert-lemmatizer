from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class LMStudioClient:
    base_url: str = "http://127.0.0.1:1234"
    model: str = "gemma-4-e2b-it-qat"
    timeout: float = 120.0

    def chat(self, input_text: str, system_prompt: str = "") -> str:
        payload = {
            "model": self.model,
            "system_prompt": system_prompt,
            "input": input_text,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/v1/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LM Studio request failed: {exc}") from exc

        return extract_chat_text(data)


def extract_chat_text(data: object) -> str:
    if not isinstance(data, dict):
        raise ValueError("LM Studio response must be a JSON object")

    output = data.get("output")

    if isinstance(output, str):
        return output

    if isinstance(output, list):
        chunks = []
        for item in output:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                chunks.append(item["content"])
        if chunks:
            return "\n".join(chunks)

    if isinstance(data.get("content"), str):
        return data["content"]

    raise ValueError("Could not extract text from LM Studio response")


def parse_sentence_lines(text: str) -> list[str]:
    sentences = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.removeprefix("-").strip()
        if ". " in line[:4] and line[0].isdigit():
            line = line.split(". ", 1)[1].strip()
        if line:
            sentences.append(line)
    return sentences
