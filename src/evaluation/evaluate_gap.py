#!/usr/bin/env python3
"""CLI entrypoint for GAP evaluation sweeps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from models.gap_pruning import evaluate_gap, parse_drop_rates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2vl")
    parser.add_argument("--model-id")
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/gap"))
    parser.add_argument("--drop-rates", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--test-size", type=int, default=750)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_gap(
        model_name=args.model,
        model_id=args.model_id,
        drop_rates=parse_drop_rates(args.drop_rates),
        metadata_csv=str(args.metadata_csv),
        output_dir=str(args.output_dir),
        test_size=args.test_size,
        seed=args.seed,
        dry_run=args.dry_run,
        max_new_tokens=args.max_new_tokens,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        hf_token=args.hf_token,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
