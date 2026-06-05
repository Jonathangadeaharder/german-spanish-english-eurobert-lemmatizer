from __future__ import annotations

import gc
import os
import resource
import sys

import torch
from transformers import TrainerCallback


def _memory_gib(value: int | float | None) -> float | None:
    if value is None:
        return None

    return round(float(value) / (1024**3), 2)


def _process_peak_rss_mib() -> float | None:
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None

    # On macOS ru_maxrss is bytes.
    return round(float(rss) / (1024**2), 2)


def mps_memory_snapshot() -> dict[str, float]:
    snapshot: dict[str, float] = {}

    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return snapshot

    for name in ["current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"]:
        fn = getattr(torch.mps, name, None)
        if callable(fn):
            try:
                snapshot[name] = _memory_gib(int(fn()))
            except Exception:
                continue

    rss = _process_peak_rss_mib()
    if rss is not None:
        snapshot["process_peak_rss_mib"] = rss

    return snapshot


def format_memory_snapshot(snapshot: dict[str, float]) -> str:
    if not snapshot:
        return "{}"

    items = [f"{key}={value}" for key, value in sorted(snapshot.items())]
    return "{" + ", ".join(items) + "}"


def cleanup_torch_mps(stage: str, emit: bool = True) -> None:
    before = mps_memory_snapshot()

    gc.collect()

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        synchronize = getattr(torch.mps, "synchronize", None)
        if callable(synchronize):
            try:
                synchronize()
            except Exception:
                pass

        empty_cache = getattr(torch.mps, "empty_cache", None)
        if callable(empty_cache):
            try:
                empty_cache()
            except Exception:
                pass

    after = mps_memory_snapshot()

    if emit:
        print(
            f"[MPS-CLEANUP] stage={stage} before={format_memory_snapshot(before)} "
            f"after={format_memory_snapshot(after)}",
            file=sys.stderr,
        )


class MPSMemoryCleanupCallback(TrainerCallback):
    def __init__(self) -> None:
        self.log_steps = int(os.getenv("TRAIN_MPS_LOG_STEPS", "0") or "0")
        self._last_log_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if self.log_steps <= 0:
            return control

        step = int(state.global_step)
        if step <= 0 or step == self._last_log_step:
            return control

        if step % self.log_steps == 0:
            self._last_log_step = step
            snapshot = mps_memory_snapshot()
            if snapshot:
                print(
                    f"[MPS-MEM] step={step} {format_memory_snapshot(snapshot)}",
                    file=sys.stderr,
                )

        return control

    def on_evaluate(self, args, state, control, **kwargs):
        cleanup_torch_mps("evaluate")
        return control

    def on_save(self, args, state, control, **kwargs):
        cleanup_torch_mps("save")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        cleanup_torch_mps("train_end")
        return control
