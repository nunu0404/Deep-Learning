from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.gap_pruning import GAPPruner


class DummyLayer:
    def __init__(self) -> None:
        self.attn = object()


class DummyModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(vision_config=SimpleNamespace(patch_size=14))
        self.visual = SimpleNamespace(
            blocks=[DummyLayer(), DummyLayer(), DummyLayer(), DummyLayer()],
            spatial_merge_size=2,
        )


def make_test_image() -> Image.Image:
    image = Image.new("RGB", (56, 56), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 6, 48, 16), fill="black")
    draw.rectangle((10, 28, 44, 44), outline="red", width=2)
    return image


def test_gap_pruner_drops_floor_fraction_and_keeps_cls() -> None:
    token_count = 16
    drop_rate = 0.5
    pruner = GAPPruner(DummyModel(), drop_rate=drop_rate)
    pruner.cls_attention = np.linspace(0.0, 1.0, token_count, dtype=np.float32)

    keep_indices = pruner.get_tokens_to_keep(make_test_image(), token_count)
    keep_indices = pruner._ensure_token_zero(keep_indices, token_count)

    dropped_tokens = token_count - len(keep_indices)
    assert dropped_tokens == math.floor(token_count * drop_rate)
    assert 0 in keep_indices
    assert len(set(keep_indices)) == len(keep_indices)
