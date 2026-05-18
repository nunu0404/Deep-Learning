from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analysis.hyperparam_search import simplex_grid
from evaluation.splits import create_split_payload, split_group_key


def test_coarse_simplex_grid_has_expected_count_and_unique_points() -> None:
    grid = simplex_grid(0.1)
    assert len(grid) == 66
    assert len(set(grid)) == len(grid)
    for alpha, beta, gamma in grid:
        assert math.isclose(alpha + beta + gamma, 1.0, abs_tol=1e-9)


def test_fine_simplex_grid_has_expected_count_and_unique_points() -> None:
    grid = simplex_grid(0.05)
    assert len(grid) == 231
    assert len(set(grid)) == len(grid)
    for alpha, beta, gamma in grid:
        assert math.isclose(alpha + beta + gamma, 1.0, abs_tol=1e-9)


def test_stable_splits_use_source_interface_id_and_flat_json_shape() -> None:
    dataframe = pd.DataFrame(
        {
            "sample_id": ["00001", "00001", "00002", "00002"],
            "source_interface_id": ["src-a", "src-a", "src-b", "src-b"],
            "viewport": ["375x667", "1280x800", "375x667", "1280x800"],
        }
    )

    payload = create_split_payload(dataframe, seed=42, ratios={"train": 0.5, "val": 0.0, "test": 0.5})

    assert set(payload) == {"train", "val", "test"}
    assert set(split_group_key(dataframe)) == {"src-a", "src-b"}
    assert sorted(payload["train"] + payload["val"] + payload["test"]) == ["src-a", "src-b"]
