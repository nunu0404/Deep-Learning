from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.evaluate_fastv import compute_fastv_image_scores, select_fastv_keep_indices


def test_fastv_topk_selection_is_deterministic_and_preserves_cls() -> None:
    attention = np.zeros((2, 6, 6), dtype=np.float32)
    text_tokens = [0, 1]
    image_tokens = [2, 3, 4, 5]

    attention[:, :, 2] = 0.10
    attention[:, :, 3] = 0.20
    attention[:, :, 4] = 0.90
    attention[:, :, 5] = 0.70

    scores = compute_fastv_image_scores(
        attention=attention,
        text_token_indices=text_tokens,
        image_token_indices=image_tokens,
    )
    np.testing.assert_allclose(scores, np.array([0.1, 0.2, 0.9, 0.7], dtype=np.float32))

    first_keep = select_fastv_keep_indices(
        attention=attention,
        text_token_indices=text_tokens,
        image_token_indices=image_tokens,
        num_tokens=4,
        drop_rate=0.5,
    )
    second_keep = select_fastv_keep_indices(
        attention=attention,
        text_token_indices=text_tokens,
        image_token_indices=image_tokens,
        num_tokens=4,
        drop_rate=0.5,
    )

    assert first_keep == second_keep
    assert first_keep == [0, 2]
