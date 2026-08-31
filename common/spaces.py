"""Load an argument space by name from ../spaces/<name>.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["A_gc", "A_-6", "A_gc+", "A_cc"]


def load(space: str) -> dict:
    if space not in NAMES:
        raise ValueError(f"unknown space {space!r}; expected one of {NAMES}")
    return json.load(open(ROOT / "spaces" / f"{space}.json"))


def arguments(space: str) -> list[str]:
    return load(space)["arguments"]


def labels(space: str) -> list[str]:
    return [f"a{i + 1}" for i in range(load(space)["n"])]


def statement(space: str) -> str:
    return load(space)["statement"]


def data_dir(experiment: str, space: str, model: str | None = None) -> Path:
    """Where this (experiment, space, model) keeps its data."""
    p = ROOT / "data" / experiment / space
    return p / model if model else p


def figure_dir(experiment: str, space: str, model: str | None = None) -> Path:
    p = ROOT / "figures" / experiment / space
    p = p / model if model else p
    p.mkdir(parents=True, exist_ok=True)
    return p
