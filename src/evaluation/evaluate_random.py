#!/usr/bin/env python3
"""Evaluate uniform random vision-token pruning baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from evaluation.pruning_sweep import evaluate_pruning_sweep
    from models.gap_pruning import (
        DEFAULT_DROP_RATES,
        GAPPruner,
        Qwen2VLGapBackend,
        format_drop_rate,
        parse_drop_rates,
        resolve_model_spec,
    )
except ImportError:  # pragma: no cover - package-relative execution fallback
    from .pruning_sweep import evaluate_pruning_sweep
    from ..models.gap_pruning import (
        DEFAULT_DROP_RATES,
        GAPPruner,
        Qwen2VLGapBackend,
        format_drop_rate,
        parse_drop_rates,
        resolve_model_spec,
    )


def deterministic_random_seed(sample_id: str, drop_rate: float, seed: int) -> int:
    payload = f"{seed}|{sample_id}|{drop_rate:.4f}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


class RandomDropPruner(GAPPruner):
    """Uniformly sample kept token indices with per-sample deterministic seeds."""

    def __init__(self, model: Any, drop_rate: float = 0.5, seed: int = 42) -> None:
        super().__init__(model=model, drop_rate=drop_rate)
        self.seed = int(seed)
        self.current_sample_id = "unknown"

    def set_sample_context(self, sample_id: str) -> None:
        self.current_sample_id = str(sample_id)

    def get_tokens_to_keep(self, image: Image.Image, num_tokens: int) -> list[int]:
        if num_tokens <= 0:
            self.last_pruning_info = {
                "keep_indices": [],
                "drop_indices": [],
                "blank_dropped_pct": 0.0,
                "used_uniform_attention": False,
            }
            return []

        random_seed = deterministic_random_seed(
            sample_id=self.current_sample_id,
            drop_rate=self.drop_rate,
            seed=self.seed,
        )
        dropped_count = min(num_tokens, int(math.floor(num_tokens * self.drop_rate)))
        keep_count = max(1, num_tokens - dropped_count)
        rng = np.random.default_rng(random_seed)
        selected = np.sort(rng.choice(num_tokens, size=keep_count, replace=False).astype(int))

        drop_mask = np.ones((num_tokens,), dtype=bool)
        drop_mask[selected] = False
        drop_indices = np.nonzero(drop_mask)[0].astype(int).tolist()
        entropy = np.ones((num_tokens,), dtype=np.float32)

        self.last_pruning_info = {
            "num_tokens": int(num_tokens),
            "kept_tokens": int(len(selected)),
            "dropped_tokens": int(len(drop_indices)),
            "keep_indices": selected.astype(int).tolist(),
            "drop_indices": drop_indices,
            "blank_dropped_count": 0,
            "blank_dropped_pct": 0.0,
            "entropy": entropy,
            "edge_density": np.zeros((num_tokens,), dtype=np.float32),
            "attention": np.ones((num_tokens,), dtype=np.float32),
            "gui_scores": np.zeros((num_tokens,), dtype=np.float32),
            "used_uniform_attention": False,
            "random_seed": int(random_seed),
            "sample_id": self.current_sample_id,
        }
        return self._ensure_token_zero(selected.astype(int).tolist(), num_tokens)


class Qwen2VLRandomDropBackend(Qwen2VLGapBackend):
    """Qwen2-VL backend using uniform random token selection instead of GAP scores."""

    def __init__(
        self,
        model_name: str,
        model_id: str,
        max_new_tokens: int,
        drop_rate: float,
        seed: int,
        hf_token: str | None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            drop_rate=drop_rate,
            alpha=0.0,
            beta=0.0,
            gamma=0.0,
            hf_token=hf_token,
        )
        self.pruner.remove_hooks()
        self.pruner = RandomDropPruner(self.model, drop_rate=drop_rate, seed=seed)
        self.pruner.register_hooks()
        self.pruner.apply_pruning_hook()

    def set_sample_context(self, sample_id: str) -> None:
        self.pruner.set_sample_context(sample_id)


def build_random_backend(
    model_name: str,
    model_id: str | None,
    drop_rate: float,
    max_new_tokens: int,
    seed: int,
    hf_token: str | None,
) -> Qwen2VLRandomDropBackend:
    spec = resolve_model_spec(model_name=model_name, model_id=model_id)
    if spec["loader"] != "qwen2vl":
        raise ValueError(f"Unsupported random-drop loader type: {spec['loader']}")
    return Qwen2VLRandomDropBackend(
        model_name=model_name,
        model_id=spec["model_id"],
        max_new_tokens=max_new_tokens,
        drop_rate=drop_rate,
        seed=seed,
        hf_token=hf_token,
    )


def evaluate_random(
    model_name: str,
    drop_rates: list[float] = DEFAULT_DROP_RATES,
    metadata_csv: str | Path = "data/metadata.csv",
    output_dir: str | Path = "results/random",
    test_size: int = 750,
    seed: int = 42,
    dry_run: bool = False,
    max_new_tokens: int = 16,
    hf_token: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    resolved = resolve_model_spec(model_name=model_name, model_id=model_id)

    def factory(initial_drop_rate: float) -> Qwen2VLRandomDropBackend:
        return build_random_backend(
            model_name=model_name,
            model_id=resolved["model_id"],
            drop_rate=initial_drop_rate,
            max_new_tokens=max_new_tokens,
            seed=seed,
            hf_token=hf_token,
        )

    return evaluate_pruning_sweep(
        method_key="random",
        model_name=model_name,
        model_id=resolved["model_id"],
        drop_rates=[float(rate) for rate in drop_rates],
        metadata_csv=metadata_csv,
        output_dir=output_dir,
        test_size=test_size,
        seed=seed,
        dry_run=dry_run,
        logger_name="evaluate_random",
        backend_factory=factory,
        section_extras=lambda backend: {"seed": int(backend.pruner.seed)},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2vl")
    parser.add_argument("--model-id")
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/random"))
    parser.add_argument("--drop-rates", default=",".join(format_drop_rate(rate) for rate in DEFAULT_DROP_RATES))
    parser.add_argument("--test-size", type=int, default=750)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_random(
        model_name=args.model,
        model_id=args.model_id,
        drop_rates=parse_drop_rates(args.drop_rates),
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        dry_run=args.dry_run,
        max_new_tokens=args.max_new_tokens,
        hf_token=args.hf_token,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
