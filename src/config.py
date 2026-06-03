from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_profile(profile: str | None) -> dict[str, Any]:
    if not profile:
        return {}

    profile_path = Path(profile)

    if not profile_path.exists():
        profile_path = CONFIG_DIR / f"{profile}.toml"

    if not profile_path.exists():
        raise FileNotFoundError(f"Unknown profile: {profile}")

    with profile_path.open("rb") as handle:
        data = tomllib.load(handle)

    env = data.get("env")

    if isinstance(env, dict):
        return env

    return data


def env_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def apply_env(mapping: dict[str, Any], *, override: bool = False) -> None:
    for key, value in mapping.items():
        if value is None:
            continue

        if override or key not in os.environ or os.environ[key] == "":
            os.environ[key] = str(value)
