#!/usr/bin/env python3
"""Stable file-backed train/val/test splits for GUI-BugBench metadata."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


SPLIT_CHOICES = ("train", "val", "test", "all")
DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def split_group_key(dataframe: pd.DataFrame) -> pd.Series:
    """Return the grouping key used to prevent one UI from crossing splits."""
    if "source_interface_id" in dataframe.columns:
        return dataframe["source_interface_id"].fillna(dataframe["sample_id"]).astype(str)
    if "source_interface" in dataframe.columns:
        return dataframe["source_interface"].fillna(dataframe["sample_id"]).astype(str)
    return dataframe["sample_id"].astype(str)


def default_splits_path(metadata_path: Path) -> Path:
    return metadata_path.parent / "splits.json"


def _allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    train_count = int(round(total * ratios["train"]))
    val_count = int(round(total * ratios["val"]))
    train_count = min(max(train_count, 0), total)
    val_count = min(max(val_count, 0), total - train_count)
    test_count = total - train_count - val_count
    return {"train": train_count, "val": val_count, "test": test_count}


def create_split_payload(
    dataframe: pd.DataFrame,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    ratios = ratios or DEFAULT_RATIOS
    group_keys = sorted(split_group_key(dataframe).dropna().astype(str).unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    counts = _allocate_counts(len(group_keys), ratios)
    train_end = counts["train"]
    val_end = train_end + counts["val"]

    return {
        "train": sorted(group_keys[:train_end]),
        "val": sorted(group_keys[train_end:val_end]),
        "test": sorted(group_keys[val_end:]),
    }


def split_values_from_payload(payload: dict[str, Any], split: str) -> list[Any]:
    """Read either the current flat split format or the older wrapped format."""
    values = payload.get(split)
    if isinstance(values, list):
        return values
    wrapped = payload.get("splits")
    if isinstance(wrapped, dict):
        wrapped_values = wrapped.get(split)
        if isinstance(wrapped_values, list):
            return wrapped_values
    return []


def load_or_create_splits(
    dataframe: pd.DataFrame,
    splits_path: Path,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    if splits_path.exists():
        return json.loads(splits_path.read_text(encoding="utf-8"))

    payload = create_split_payload(dataframe=dataframe, seed=seed, ratios=ratios)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def select_split_dataframe(
    dataframe: pd.DataFrame,
    split: str,
    splits_path: Path,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, str]:
    if split not in SPLIT_CHOICES:
        raise ValueError(f"Unsupported split {split!r}. Expected one of {SPLIT_CHOICES}.")
    if split == "all":
        return dataframe.copy().reset_index(drop=True), "stable-splits:all"

    payload = load_or_create_splits(dataframe=dataframe, splits_path=splits_path, seed=seed, ratios=ratios)
    split_values = set(str(value) for value in split_values_from_payload(payload, split))
    keys = split_group_key(dataframe)
    selected = dataframe.loc[keys.isin(split_values)].copy()
    selected = selected.sort_values(["sample_id", "image_path"], kind="stable").reset_index(drop=True)
    return selected, f"stable-splits:{split}:{splits_path}"


def apply_budget(dataframe: pd.DataFrame, budget_images: int | None, seed: int = 42) -> pd.DataFrame:
    if budget_images is None or budget_images <= 0 or len(dataframe) <= budget_images:
        return dataframe.copy().reset_index(drop=True)
    return dataframe.sample(n=budget_images, random_state=seed, replace=False).sort_values(
        ["sample_id", "image_path"],
        kind="stable",
    ).reset_index(drop=True)
