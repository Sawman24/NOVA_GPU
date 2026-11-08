"""Config loading. One YAML file (config/nova.yaml) holds settings for every machine;
each service reads only its own section."""
from __future__ import annotations

import os
import pathlib
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    p = pathlib.Path(path or os.environ.get("NOVA_CONFIG") or ROOT / "config" / "nova.yaml")
    if not p.exists():
        raise SystemExit(
            f"Config not found: {p}\n"
            "Run setup first (setup.bat on Windows, ./setup.sh on Linux), "
            "or copy config/nova.example.yaml to config/nova.yaml."
        )
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve(path: str | os.PathLike) -> pathlib.Path:
    """Resolve a path from the config relative to the NOVA folder."""
    p = pathlib.Path(path)
    return p if p.is_absolute() else ROOT / p


def secret(section: dict, key: str) -> str:
    """Read a secret either inline (`key`) or from an environment variable (`key_env`)."""
    val = section.get(key) or ""
    env_name = section.get(f"{key}_env")
    if not val and env_name:
        val = os.environ.get(env_name, "")
    return str(val).strip()
