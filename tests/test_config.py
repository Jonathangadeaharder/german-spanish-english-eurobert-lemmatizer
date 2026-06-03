import os

from config import apply_env, load_profile


def test_load_profile_reads_smoke_profile():
    profile = load_profile("smoke")

    assert profile["TRAIN_MAX_STEPS"] == 1
    assert profile["MPS_BENCH_CONFIGS"] == "bf16_bs4_group"


def test_apply_env_sets_missing_values(monkeypatch):
    monkeypatch.delenv("TRAIN_MAX_STEPS", raising=False)

    apply_env({"TRAIN_MAX_STEPS": 5})

    assert os.getenv("TRAIN_MAX_STEPS") == "5"
