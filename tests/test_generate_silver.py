import json
import threading

import generate_silver


class FakeEntry:
    def __init__(self, entry, level="A1", lang="de", canonical=True):
        self.entry = entry
        self.level = level
        self.lang = lang
        self.canonical = canonical


def test_generate_silver_uses_parallel_lmstudio_requests(monkeypatch, tmp_path):
    active = 0
    max_active = 0
    calls = []
    barrier = threading.Barrier(4)
    lock = threading.Lock()

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def chat(self, input_text, system_prompt=""):
            nonlocal active, max_active
            calls.append((input_text, system_prompt))
            with lock:
                active += 1
                max_active = max(max_active, active)
            barrier.wait(timeout=1)
            with lock:
                active -= 1
            return "Ein Satz."

    monkeypatch.setenv("LEMMA_LANG", "de")
    monkeypatch.setenv("SILVER_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("SILVER_LIMIT", "4")
    monkeypatch.setenv("SILVER_PARALLEL", "4")
    monkeypatch.setattr(generate_silver, "LMStudioClient", FakeClient)
    monkeypatch.setattr(
        generate_silver,
        "load_vocab_entries",
        lambda: [FakeEntry(f"wort{i}") for i in range(4)],
    )

    generate_silver.main()

    rows = [
        json.loads(line)
        for line in (tmp_path / "lemma_de_raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 4
    assert max_active > 1
    assert len(calls) == 4
