"""CoNLL-U format validator for lemmatizer training data.

Validates structural correctness: 10 TAB-separated fields, valid UPOS tags,
required metadata (sent_id, text), blank line separation, text-form alignment,
unique sent_ids, UTF-8 NFC, LF line endings.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

VALID_UPOS = frozenset(
    {
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    }
)

SENT_ID_RE = re.compile(r"^#\s*sent_id\s*=\s*(.+)$")
TEXT_RE = re.compile(r"^#\s*text\s*=\s*(.*)$")


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sentence_count: int = 0
    token_count: int = 0

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


def validate_file(path: str | Path) -> ValidationResult:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return validate_text(text)


def validate_text(text: str) -> ValidationResult:
    result = ValidationResult()

    if "\r\n" in text:
        result.errors.append(
            "CRLF line endings detected (\\r\\n). CoNLL-U requires LF-only line endings."
        )

    lines = text.split("\n")

    sent_ids_seen: set[str] = set()
    in_sentence = False
    current_sent_id: str | None = None
    current_text: str | None = None
    current_forms: list[str] = []
    has_text_comment = False
    has_sent_id_comment = False

    for line_num, raw_line in enumerate(lines, 1):
        line = raw_line

        if line == "":
            if in_sentence:
                if not has_sent_id_comment:
                    result.errors.append(f"Line {line_num}: sentence missing '# sent_id' comment")
                if not has_text_comment:
                    result.errors.append(f"Line {line_num}: sentence missing '# text' comment")
                if current_sent_id is not None:
                    if current_sent_id in sent_ids_seen:
                        result.errors.append(
                            f"Line {line_num}: duplicate sent_id '{current_sent_id}'"
                        )
                    sent_ids_seen.add(current_sent_id)
                if current_text is not None and current_forms:
                    mismatch = _text_content_mismatch(current_forms, current_text)
                    if mismatch:
                        result.errors.append(f"Line {line_num}: {mismatch}")
                result.sentence_count += 1
                result.token_count += len(current_forms)
                in_sentence = False
                current_sent_id = None
                current_text = None
                current_forms = []
                has_text_comment = False
                has_sent_id_comment = False
            continue

        if line.startswith("#"):
            if not in_sentence:
                in_sentence = True
            sid_match = SENT_ID_RE.match(line)
            if sid_match:
                current_sent_id = sid_match.group(1).strip()
                has_sent_id_comment = True
            text_match = TEXT_RE.match(line)
            if text_match:
                current_text = text_match.group(1)
                has_text_comment = True
            continue

        if not in_sentence:
            in_sentence = True

        cols = line.split("\t")
        if len(cols) != 10:
            result.errors.append(
                f"Line {line_num}: expected 10 TAB-separated fields, "
                f"got {len(cols)}. Possible space-instead-of-tab issue."
            )
            current_forms.append(cols[1] if len(cols) > 1 else "")
            continue

        token_id, form, lemma, upos = cols[0], cols[1], cols[2], cols[3]

        if not form:
            result.errors.append(f"Line {line_num}: empty FORM field (token {token_id})")
        if not lemma:
            result.errors.append(
                f"Line {line_num}: empty LEMMA field (token {token_id}, form='{form}')"
            )
        if upos and upos not in VALID_UPOS:
            result.errors.append(
                f"Line {line_num}: invalid UPOS '{upos}' (token {token_id}, form='{form}')"
            )
        for i in range(4, 10):
            if cols[i] != "_":
                result.errors.append(
                    f"Line {line_num}: column {i + 1} expected '_', "
                    f"got '{cols[i]}' (token {token_id})"
                )

        if not _is_nfc(form):
            result.warnings.append(f"Line {line_num}: FORM '{form}' is not NFC-normalized")
        if not _is_nfc(lemma):
            result.warnings.append(f"Line {line_num}: LEMMA '{lemma}' is not NFC-normalized")

        current_forms.append(form)

    if in_sentence:
        result.errors.append(f"Line {len(lines)}: last sentence missing trailing blank line")
        if current_sent_id is not None:
            if current_sent_id in sent_ids_seen:
                result.errors.append(f"Line {len(lines)}: duplicate sent_id '{current_sent_id}'")
            sent_ids_seen.add(current_sent_id)
        if current_text is not None and current_forms:
            mismatch = _text_content_mismatch(current_forms, current_text)
            if mismatch:
                result.errors.append(f"Line {len(lines)}: {mismatch}")
        result.sentence_count += 1
        result.token_count += len(current_forms)

    return result


_WHITESPACE_RE = re.compile(r"\s+")


def _text_content_mismatch(forms: list[str], text: str) -> str | None:
    """Check that FORM tokens are a whitespace-agnostic re-partition of `# text`.

    CoNLL-U MISC (SpaceAfter=No, etc.) is intentionally unused in this
    dataset (columns 5-10 are always `_`), so exact inter-token spacing
    cannot be reconstructed from FORM columns alone — e.g. Stanza MWT
    expansion of "m'appelle" into "m'" + "appelle" has no space between
    them, and Chinese words have no spaces at all. Instead, we verify that
    concatenating every FORM (in order) reproduces `# text` with all
    whitespace removed: this still catches dropped/altered/reordered
    tokens without depending on a specific spacing convention.
    """
    joined_forms = "".join(forms)
    squeezed_text = _WHITESPACE_RE.sub("", text)
    if joined_forms != squeezed_text:
        return f"text mismatch — '# text'='{text}' vs tokens='{joined_forms}'"
    return None


def _is_nfc(s: str) -> bool:
    return unicodedata.normalize("NFC", s) == s
