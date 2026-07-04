from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class LMStudioClient:
    base_url: str = "http://127.0.0.1:1234"
    model: str = os.getenv("LMSTUDIO_MODEL", "qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive-text-oq4")
    timeout: float = 120.0

    def chat(self, input_text: str, system_prompt: str = "") -> str:
        # Use OpenAI-compatible endpoint (LM Studio supports /v1/chat/completions)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": input_text})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LM Studio request failed: {exc}") from exc

        # OpenAI response format: choices[0].message.content
        choices = data.get("choices", [])
        if choices and isinstance(choices[0], dict):
            content = choices[0].get("message", {}).get("content", "")
            if content:
                return content

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
