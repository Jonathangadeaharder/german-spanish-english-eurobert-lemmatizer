"""WebVTT parser → clean sentences for training data augmentation.

Strips timing lines, joins fragmented captions, splits into
well-formed sentences. Output is plain text sentences ready for
POS/lemma annotation.
"""
from __future__ import annotations

import re
from pathlib import Path

_TIMING_RE = re.compile(r"^\d{2}:\d{2}[:.]")
_TAG_RE = re.compile(r"<[^>]+>")
_HEADER_RE = re.compile(r"^(WEBVTT|NOTE|STYLE|REGION|META)")
_SRT_TIMING_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}"
)


def parse_vtt(path: Path) -> list[str]:
    """Parse a WebVTT or SRT file into caption text blocks."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Normalize: SRT uses commas in timestamps, VTT uses periods.
    blocks: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        if _HEADER_RE.match(stripped):
            continue
        if _TIMING_RE.match(stripped):
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        if stripped.isdigit() and not current:
            continue
        clean = _TAG_RE.sub("", stripped)
        if clean:
            current.append(clean)

    if current:
        blocks.append("\n".join(current))

    return blocks


def blocks_to_sentences(blocks: list[str]) -> list[str]:
    """Join caption blocks and split into sentences.

    Joins fragmented captions within a block, then splits on
    sentence-final punctuation (., !, ?, 。 for zh).
    """
    sentences: list[str] = []

    for block in blocks:
        text = " ".join(block.splitlines())
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        # Split on sentence-final punctuation, keeping the delimiter.
        parts = re.split(r"(?<=[.!?。])\s+", text)
        for part in parts:
            part = part.strip()
            if len(part) >= 5:
                sentences.append(part)

    return sentences


def load_subtitle_sentences(path: Path) -> list[str]:
    """Parse a VTT file and return clean sentences."""
    return blocks_to_sentences(parse_vtt(path))
