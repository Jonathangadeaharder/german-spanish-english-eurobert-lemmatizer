"""Apply validation results to subtitle CoNLL-U files.

Reads validation result files, drops INVALID sentences, keeps VALID
and FIXED sentences. Then removes cefr-augmented-* sentences
from train.conllu and appends the validated subtitle data.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

_SENT_ID_RE = re.compile(r"# sent_id = (\S+)")


def parse_validation_results(
    results_dir: Path,
) -> set[str]:
    """Return set of sent_ids marked VALID or FIXED (keep these)."""
    keep: set[str] = set()
    for f in sorted(results_dir.glob("batch_*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not (line.startswith("VALID: ") or line.startswith("FIXED: ")):
                continue
            # Strip the " — reason" suffix if present (VALID lines may
            # not have one). The sent_id is everything after "VALID: "
            # or "FIXED: " up to the first " — ".
            prefix_and_rest = line.split(": ", 1)
            if len(prefix_and_rest) < 2:
                print(
                    f"  WARNING: skipping malformed line: {line[:60]!r}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            sent_id = prefix_and_rest[1].split(" — ", 1)[0].strip()
            if sent_id:
                keep.add(sent_id)
    return keep


def filter_conllu_by_sent_ids(
    conllu_path: Path,
    keep_ids: set[str],
) -> list[str]:
    """Read a CoNLL-U file, return only blocks whose sent_id is in keep_ids."""
    text = conllu_path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    result: list[str] = []
    found_ids: set[str] = set()
    for block in blocks:
        m = _SENT_ID_RE.search(block)
        if m is None:
            # Warn on blocks lacking sent_id so silent data loss
            # is visible to the operator.
            print(
                f"  WARNING: dropping block without sent_id (preview: {block[:40]!r})",
                file=sys.stderr,
                flush=True,
            )
            continue
        sid = m.group(1)
        if sid in keep_ids:
            result.append(block)
            found_ids.add(sid)
    missing = keep_ids - found_ids
    if missing:
        print(
            f"  NOTE: {len(missing)} keep_ids not found in {conllu_path.name} "
            f"(e.g. {sorted(missing)[:5]})",
            file=sys.stderr,
            flush=True,
        )
    return result


def remove_cefr_augmented_sentences(conllu_path: Path) -> int:
    """Remove cefr-augmented-* sentences from a CoNLL-U file.

    Writes atomically (tempfile + os.replace) so a crash mid-write
    cannot leave train.conllu truncated. Returns the number of
    sentences removed.
    """
    text = conllu_path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    kept: list[str] = []
    removed = 0
    for block in blocks:
        m = _SENT_ID_RE.search(block)
        if m is None:
            # Warn on blocks lacking sent_id so silent data loss
            # is visible to the operator.
            print(
                f"  WARNING: dropping block without sent_id (preview: {block[:40]!r})",
                file=sys.stderr,
                flush=True,
            )
            continue
        sent_id = m.group(1)
        if "cefr-augmented" in sent_id:
            removed += 1
        else:
            kept.append(block)
    new_text = "\n\n".join(kept) + "\n"
    parent = conllu_path.parent
    tmp_fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        try:
            tmp_f = os.fdopen(tmp_fd, "w", encoding="utf-8")
        except Exception:
            # os.fdopen takes ownership of tmp_fd; if it raises,
            # mkstemp already opened the FD — close it to avoid a leak.
            os.close(tmp_fd)
            raise
        with tmp_f:
            tmp_f.write(new_text)
        os.replace(tmp_path, conllu_path)
    except Exception:
        if Path(tmp_path).exists():
            os.unlink(tmp_path)
        raise
    return removed


def _has_subtitle_blocks(conllu_path: Path) -> bool:
    """Return True if the file already contains subtitle- sent_ids."""
    text = conllu_path.read_text(encoding="utf-8")
    return bool(re.search(r"# sent_id = subtitle-", text))


def _append_blocks_atomically(conllu_path: Path, blocks: list[str]) -> None:
    """Append CoNLL-U blocks to a file atomically.

    Reads the current contents, appends the blocks, then writes via a
    tempfile + os.replace so a crash mid-write cannot leave
    train.conllu truncated or half-appended.
    """
    if not blocks:
        return
    existing = conllu_path.read_text(encoding="utf-8")
    addition = "\n\n" + "\n\n".join(blocks) + "\n"
    new_text = existing + addition
    parent = conllu_path.parent
    tmp_fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        try:
            tmp_f = os.fdopen(tmp_fd, "w", encoding="utf-8")
        except Exception:
            os.close(tmp_fd)
            raise
        with tmp_f:
            tmp_f.write(new_text)
        os.replace(tmp_path, conllu_path)
    except Exception:
        if Path(tmp_path).exists():
            os.unlink(tmp_path)
        raise


def run_pipeline(langs: list[str]) -> None:
    gold_dir = Path("data/gold")
    val_dir = Path("data/validation_results")

    for lang in langs:
        print(f"\n=== {lang} ===", flush=True)

        # 1. Parse validation results.
        results_dir = val_dir / lang
        if not results_dir.exists():
            print(f"  No validation results for {lang}, skipping.")
            continue
        keep_ids = parse_validation_results(results_dir)
        print(f"  Sentences to keep: {len(keep_ids)}", flush=True)

        # 2. Filter subtitle CoNLL-U.
        sub_path = gold_dir / lang / "subtitle_sentences.conllu"
        if not sub_path.exists():
            print(f"  No subtitle file for {lang}, skipping.")
            continue
        validated_blocks = filter_conllu_by_sent_ids(sub_path, keep_ids)
        print(f"  Validated subtitle blocks: {len(validated_blocks)}")

        # Write validated subtitle file (avoid stray newlines when empty).
        validated_path = gold_dir / lang / "subtitle_validated.conllu"
        if validated_blocks:
            validated_path.write_text("\n\n".join(validated_blocks) + "\n", encoding="utf-8")
        else:
            validated_path.write_text("", encoding="utf-8")

        # 3. Remove cefr-augmented sentences from train.conllu.
        train_path = gold_dir / lang / "train.conllu"
        if train_path.exists():
            removed = remove_cefr_augmented_sentences(train_path)
            print(f"  Removed {removed} cefr-augmented sentences")

            # 4. Append validated subtitle sentences (idempotent:
            # skip if subtitle- sent_ids already present).
            if _has_subtitle_blocks(train_path):
                print("  subtitle- sent_ids already present; skipping append")
            elif not validated_blocks:
                print("  No validated blocks to append; skipping")
            else:
                _append_blocks_atomically(train_path, validated_blocks)
                print(f"  Appended {len(validated_blocks)} validated subtitle sentences")

            # 5. Snapshot train.conllu (treebank + subtitle blocks,
            # no cefr-augmented single-word sentences) into
            # train_cefr_augmented.conllu so the two stay in sync.
            # Note: the file name is historical; the snapshot no
            # longer contains CEFR-augmented sentences.
            aug_path = gold_dir / lang / "train_cefr_augmented.conllu"
            if aug_path.exists():
                aug_path.write_text(
                    train_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                print(f"  Updated {aug_path.name}")

    print("\nDone. Run `make_dataset` to rebuild HF datasets.")


if __name__ == "__main__":
    langs = sys.argv[1:] if len(sys.argv) > 1 else ["sv", "ar", "zh"]
    run_pipeline(langs)
