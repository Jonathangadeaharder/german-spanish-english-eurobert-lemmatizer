"""Tests for apply_validation fixes.

Pins:
- Empty validated_blocks → no stray newlines in output files.
- Atomic append to train.conllu via tempfile + os.replace.
- Log which keep_ids weren't found.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lemmatizer.data.apply_validation import (
    _append_blocks_atomically,
    _has_subtitle_blocks,
    filter_conllu_by_sent_ids,
    parse_validation_results,
    remove_cefr_augmented_sentences,
    run_pipeline,
)


@pytest.fixture
def conllu_with_ids(tmp_path: Path) -> Path:
    content = """# sent_id = subtitle-sv-00001
1\tHej\thej\tINTJ\t_\t_\t0\troot\t_\t_

# sent_id = subtitle-sv-00002
1\tVärlden\tvärld\tNOUN\t_\t_\t0\troot\t_\t_

# sent_id = cefr-augmented-001
1\tord\tord\tNOUN\t_\t_\t0\troot\t_\t_
"""
    p = tmp_path / "subtitle_sentences.conllu"
    p.write_text(content, encoding="utf-8")
    return p


class TestEmptyValidatedBlocksGuard:
    def test_filter_empty_result(self, conllu_with_ids: Path):
        keep = {"nonexistent-id"}
        result = filter_conllu_by_sent_ids(conllu_with_ids, keep)
        assert result == []

    def test_run_pipeline_empty_blocks_no_stray_newlines(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        gold_dir = tmp_path / "data" / "gold" / "sv"
        val_dir = tmp_path / "data" / "validation_results" / "sv"
        gold_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)

        sub_path = gold_dir / "subtitle_sentences.conllu"
        sub_path.write_text(
            "# sent_id = subtitle-sv-00001\n1\thej\thej\tNOUN\t_\t_\t0\troot\t_\t_\n",
            encoding="utf-8",
        )
        # keep_ids that don't match → empty validated_blocks.
        (val_dir / "batch_0.txt").write_text("VALID: type — subtitle-sv-99999\n", encoding="utf-8")

        train_path = gold_dir / "train.conllu"
        train_path.write_text(
            "# sent_id = treebank-1\n1\tord\tord\tNOUN\t_\t_\t0\troot\t_\t_\n",
            encoding="utf-8",
        )

        import os

        os.chdir(tmp_path)
        run_pipeline(["sv"])

        # validated file must be empty (no stray newlines).
        validated = (gold_dir / "subtitle_validated.conllu").read_text(encoding="utf-8")
        assert validated == "", f"expected empty file, got {validated!r}"

        # train.conllu must be unchanged (no stray append).
        train_text = train_path.read_text(encoding="utf-8")
        assert "subtitle-" not in train_text
        assert train_text.endswith("_\n")


class TestAtomicAppend:
    def test_append_blocks_atomically(self, tmp_path: Path):
        train = tmp_path / "train.conllu"
        original = "# sent_id = tb-1\n1\tord\tord\tNOUN\t_\t_\t0\troot\t_\t_\n"
        train.write_text(original, encoding="utf-8")

        block = "# sent_id = subtitle-sv-00001\n1\thej\thej\tNOUN\t_\t_\t0\troot\t_\t_"
        _append_blocks_atomically(train, [block])

        result = train.read_text(encoding="utf-8")
        assert result.startswith(original)
        assert "subtitle-sv-00001" in result
        assert result.endswith("_\n")

    def test_no_tmpfile_left_behind(self, tmp_path: Path):
        train = tmp_path / "train.conllu"
        train.write_text("orig\n", encoding="utf-8")
        _append_blocks_atomically(train, ["block"])
        tmps = list(tmp_path.glob("*.tmp"))
        assert tmps == []

    def test_empty_blocks_noop(self, tmp_path: Path):
        train = tmp_path / "train.conllu"
        train.write_text("orig\n", encoding="utf-8")
        _append_blocks_atomically(train, [])
        assert train.read_text(encoding="utf-8") == "orig\n"

    def test_atomic_via_replace(self, tmp_path: Path, monkeypatch):
        # Verify os.replace is used (atomic), not a direct write.
        import lemmatizer.data.apply_validation as av

        calls: list[str] = []
        real_replace = av.os.replace

        def spy_replace(src, dst):
            calls.append(str(dst))
            return real_replace(src, dst)

        monkeypatch.setattr(av.os, "replace", spy_replace)
        train = tmp_path / "train.conllu"
        train.write_text("orig\n", encoding="utf-8")
        _append_blocks_atomically(train, ["block"])
        assert calls == [str(train)]


class TestLogMissingKeepIds:
    def test_logs_missing_ids(
        self,
        conllu_with_ids: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        # One keep_id matches, one doesn't.
        keep = {"subtitle-sv-00001", "subtitle-sv-missing"}
        filter_conllu_by_sent_ids(conllu_with_ids, keep)
        err = capsys.readouterr().err
        assert "1 keep_ids not found" in err
        assert "subtitle-sv-missing" in err

    def test_no_log_when_all_found(
        self,
        conllu_with_ids: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        keep = {"subtitle-sv-00001", "subtitle-sv-00002"}
        filter_conllu_by_sent_ids(conllu_with_ids, keep)
        err = capsys.readouterr().err
        assert "not found" not in err


class TestHasSubtitleBlocks:
    def test_true_when_subtitle_present(self, tmp_path: Path):
        p = tmp_path / "train.conllu"
        p.write_text(
            "# sent_id = subtitle-x\n1\tord\tord\tNOUN\t_\t_\t0\troot\t_\t_\n",
            encoding="utf-8",
        )
        assert _has_subtitle_blocks(p) is True

    def test_false_when_absent(self, tmp_path: Path):
        p = tmp_path / "train.conllu"
        p.write_text(
            "# sent_id = tb-1\n1\tord\tord\tNOUN\t_\t_\t0\troot\t_\t_\n",
            encoding="utf-8",
        )
        assert _has_subtitle_blocks(p) is False


class TestParseValidationResults:
    def test_parses_valid_and_fixed(self, tmp_path: Path):
        f = tmp_path / "batch_0.txt"
        f.write_text(
            "VALID: subtitle-sv-00001 — ok\n"
            "FIXED: subtitle-sv-00002 — fixed lemma\n"
            "INVALID: subtitle-sv-00003 — bad upos\n"
            "garbage line\n",
            encoding="utf-8",
        )
        keep = parse_validation_results(tmp_path)
        assert keep == {"subtitle-sv-00001", "subtitle-sv-00002"}


class TestRemoveCefrAugmented:
    def test_removes_only_augmented(self, tmp_path: Path):
        p = tmp_path / "train.conllu"
        p.write_text(
            "# sent_id = tb-1\n1\tord\tord\tNOUN\t_\t_\t0\troot\t_\t_\n\n"
            "# sent_id = cefr-augmented-1\n"
            "1\tx\tx\tNOUN\t_\t_\t0\troot\t_\t_\n",
            encoding="utf-8",
        )
        removed = remove_cefr_augmented_sentences(p)
        assert removed == 1
        text = p.read_text(encoding="utf-8")
        assert "cefr-augmented" not in text
        assert "tb-1" in text
