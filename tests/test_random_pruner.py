from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.evaluate_random import RandomDropPruner


class DummyLayer:
    def __init__(self) -> None:
        self.attn = object()


class DummyModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(vision_config=SimpleNamespace(patch_size=14))
        self.visual = SimpleNamespace(
            blocks=[DummyLayer(), DummyLayer(), DummyLayer(), DummyLayer()],
            spatial_merge_size=1,
        )


def test_random_drop_is_deterministic_for_sample_rate_and_seed() -> None:
    image = Image.new("RGB", (56, 56), "white")

    first = RandomDropPruner(DummyModel(), drop_rate=0.5, seed=123)
    first.set_sample_context("00042")
    first_keep = first.get_tokens_to_keep(image, num_tokens=16)

    second = RandomDropPruner(DummyModel(), drop_rate=0.5, seed=123)
    second.set_sample_context("00042")
    second_keep = second.get_tokens_to_keep(image, num_tokens=16)

    assert first_keep == second_keep
    assert 0 in first_keep
    assert len(first_keep) == 8
