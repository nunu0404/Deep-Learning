#!/usr/bin/env python3
"""FocusUI adapter stub for converting external outputs into GAP-style results."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from evaluation.evaluate_baseline import (
        PROMPT,
        build_test_split,
        compute_metrics,
        load_metadata,
        percentile,
        safe_mean,
    )
    from models.gap_pruning import DEFAULT_DROP_RATES, format_drop_rate, parse_drop_rates, sanitize_tag
except ImportError:  # pragma: no cover - package-relative execution fallback
    from .evaluate_baseline import (
        PROMPT,
        build_test_split,
        compute_metrics,
        load_metadata,
        percentile,
        safe_mean,
    )
    from ..models.gap_pruning import DEFAULT_DROP_RATES, format_drop_rate, parse_drop_rates, sanitize_tag


STUB_MESSAGE = (
    "FocusUI repo not provided. Skipping. To enable, clone https://github.com/showlab/FocusUI "
    "and pass --focusui-repo /path/to/FocusUI"
)
FOCUSUI_CANDIDATE_ENTRYPOINTS = (
    "run_gap_gui_bug_adapter.py",
    "scripts/run_gap_gui_bug_adapter.py",
    "tools/run_gap_gui_bug_adapter.py",
    "evaluate.py",
)


def setup_focusui_logging(results_dir: Path) -> logging.Logger:
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("evaluate_focusui_stub")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(results_dir / "errors.log", encoding="utf-8")
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def write_focusui_manifest(dataset: pd.DataFrame, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "source_interface_id", "viewport", "image_path", "prompt", "true_class"],
        )
        writer.writeheader()
        for row in dataset.itertuples(index=False):
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "source_interface_id": row.source_interface_id,
                    "viewport": row.viewport,
                    "image_path": row.image_path,
                    "prompt": PROMPT,
                    "true_class": row.true_class,
                }
            )


def find_focusui_entrypoint(focusui_repo: Path) -> Path | None:
    for relative_path in FOCUSUI_CANDIDATE_ENTRYPOINTS:
        candidate = focusui_repo / relative_path
        if candidate.exists():
            return candidate
    return None


def run_focusui_entrypoint(
    focusui_repo: Path,
    entrypoint: Path,
    manifest_path: Path,
    output_jsonl: Path,
    drop_rate: float,
    prune_layer: int,
    dry_run: bool,
    logger: logging.Logger,
) -> None:
    command = [
        os.environ.get("PYTHON", "python"),
        str(entrypoint),
        "--manifest",
        str(manifest_path),
        "--output-jsonl",
        str(output_jsonl),
        "--drop-rate",
        format_drop_rate(drop_rate),
        "--prune-layer",
        str(prune_layer),
    ]
    if dry_run:
        command.append("--dry-run")
    logger.info("Invoking FocusUI adapter: %s", " ".join(command))
    subprocess.run(command, cwd=focusui_repo, check=True)


def normalize_focusui_prediction(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return "INVALID"
    if text in {"CLEAN", "OVERLAP", "OVERFLOW", "ZINDEX", "TRUNCATION", "CONTRAST"}:
        return text
    if "OVERLAP" in text:
        return "OVERLAP"
    if "OVERFLOW" in text:
        return "OVERFLOW"
    if "ZINDEX" in text or "Z-INDEX" in text:
        return "ZINDEX"
    if "TRUNC" in text:
        return "TRUNCATION"
    if "CONTRAST" in text:
        return "CONTRAST"
    if "CLEAN" in text:
        return "CLEAN"
    return "INVALID"


def load_focusui_predictions(output_jsonl: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def convert_focusui_outputs(
    focusui_rows: list[dict[str, Any]],
    dataset: pd.DataFrame,
    model_name: str,
    model_id: str,
    metadata_path: Path,
    output_dir: Path,
    split_strategy: str,
    dry_run: bool,
    drop_rate: float,
    prune_layer: int,
) -> dict[str, Any]:
    by_key = {
        (str(row.get("sample_id")), str(Path(str(row.get("image_path", ""))).resolve())): row
        for row in focusui_rows
    }
    by_sample = {str(row.get("sample_id")): row for row in focusui_rows}

    prediction_rows: list[dict[str, Any]] = []
    true_classes: list[str] = []
    predicted_classes: list[str] = []
    latencies: list[float] = []
    peak_vram_values: list[float] = []
    error_count = 0

    for row in dataset.itertuples(index=False):
        sample_id = str(row.sample_id)
        image_path = str(row.image_path)
        focusui_row = by_key.get((sample_id, str(Path(image_path).resolve()))) or by_sample.get(sample_id) or {}
        predicted_class = normalize_focusui_prediction(
            focusui_row.get("predicted_class")
            or focusui_row.get("prediction")
            or focusui_row.get("label")
            or focusui_row.get("raw_output")
        )
        latency_ms = float(focusui_row.get("latency_ms", focusui_row.get("wall_clock_latency_ms", 0.0)) or 0.0)
        peak_vram_mb = float(focusui_row.get("peak_vram_mb", 0.0) or 0.0)
        error = focusui_row.get("error")
        if error:
            error_count += 1

        true_classes.append(row.true_class)
        predicted_classes.append(predicted_class)
        latencies.append(latency_ms)
        peak_vram_values.append(peak_vram_mb)
        prediction_rows.append(
            {
                "sample_id": row.sample_id,
                "source_interface_id": row.source_interface_id,
                "viewport": row.viewport,
                "image_path": row.image_path,
                "true_class": row.true_class,
                "true_label": int(row.true_label),
                "true_bug_type": None if row.true_class == "CLEAN" else row.true_class,
                "predicted_class": predicted_class,
                "predicted_label": 0 if predicted_class == "CLEAN" else (1 if predicted_class != "INVALID" else -1),
                "predicted_bug_type": None if predicted_class in {"CLEAN", "INVALID"} else predicted_class,
                "raw_output": str(focusui_row.get("raw_output", "")),
                "wall_clock_latency_ms": round(latency_ms, 3),
                "peak_vram_mb": round(peak_vram_mb, 3),
                "used_half_resolution_retry": bool(focusui_row.get("used_half_resolution_retry", False)),
                "drop_rate": float(drop_rate),
                "num_tokens": int(focusui_row.get("num_tokens", 0) or 0),
                "kept_tokens": int(focusui_row.get("kept_tokens", 0) or 0),
                "dropped_tokens": int(focusui_row.get("dropped_tokens", 0) or 0),
                "blank_dropped_pct": float(focusui_row.get("blank_dropped_pct", 0.0) or 0.0),
                "used_uniform_attention": False,
                "error": error,
            }
        )

    safe_model_name = sanitize_tag(model_name)
    tag = format_drop_rate(drop_rate)
    predictions_path = output_dir / f"{safe_model_name}_dr{tag}_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    payload = {
        "model_name": model_name,
        "model_id": model_id,
        "backend": "focusui-external",
        "metadata_path": str(metadata_path.resolve()),
        "results_dir": str(output_dir.resolve()),
        "split_strategy": split_strategy,
        "dry_run": bool(dry_run),
        "num_samples": int(len(dataset)),
        "num_errors": int(error_count),
        "prompt": PROMPT,
        "metrics": compute_metrics(true_classes=true_classes, predicted_classes=predicted_classes),
        "latency_ms": {
            "mean": safe_mean(latencies),
            "p95": percentile(latencies, 0.95),
        },
        "vram_mb": {
            "peak": max(peak_vram_values) if peak_vram_values else 0.0,
            "mean": safe_mean(peak_vram_values),
            "p95": percentile(peak_vram_values, 0.95),
        },
        "artifacts": {
            "predictions_csv": str(predictions_path.resolve()),
            "errors_log": str((output_dir / "errors.log").resolve()),
        },
        "library_versions": {},
        "focusui": {
            "drop_rate": float(drop_rate),
            "prune_layer": int(prune_layer),
            "adapter": "external",
            "official_repo": "https://github.com/showlab/FocusUI",
        },
    }

    result_path = output_dir / f"{safe_model_name}_dr{tag}.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def evaluate_focusui(
    focusui_repo: Path | None,
    model_name: str,
    model_id: str,
    drop_rates: list[float],
    metadata_csv: Path,
    output_dir: Path,
    test_size: int,
    seed: int,
    dry_run: bool,
    prune_layer: int,
) -> dict[str, Any]:
    if focusui_repo is None:
        print(STUB_MESSAGE)
        return {"skipped": True, "reason": STUB_MESSAGE}

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_focusui_logging(output_dir)
    focusui_repo = focusui_repo.resolve()
    if not focusui_repo.exists():
        raise FileNotFoundError(f"FocusUI repo not found: {focusui_repo}")

    metadata = load_metadata(metadata_csv)
    dataset, split_strategy = build_test_split(metadata, test_size=test_size, seed=seed)
    if dry_run:
        dataset = dataset.head(10).copy()

    manifest_path = output_dir / "focusui_manifest.csv"
    write_focusui_manifest(dataset, manifest_path)

    entrypoint = find_focusui_entrypoint(focusui_repo)
    if entrypoint is None:
        message = (
            "FocusUI repo was provided, but no supported GAP-GUI-Bug adapter entrypoint was found. "
            "Expected one of: " + ", ".join(FOCUSUI_CANDIDATE_ENTRYPOINTS)
        )
        logger.error(message)
        print(message)
        return {"skipped": True, "reason": message, "manifest": str(manifest_path.resolve())}

    results: dict[str, Any] = {}
    for drop_rate in drop_rates:
        tag = format_drop_rate(float(drop_rate))
        raw_output_path = output_dir / f"{sanitize_tag(model_name)}_dr{tag}_focusui_raw.jsonl"
        run_focusui_entrypoint(
            focusui_repo=focusui_repo,
            entrypoint=entrypoint,
            manifest_path=manifest_path,
            output_jsonl=raw_output_path,
            drop_rate=float(drop_rate),
            prune_layer=prune_layer,
            dry_run=dry_run,
            logger=logger,
        )
        focusui_rows = load_focusui_predictions(raw_output_path)
        results[tag] = convert_focusui_outputs(
            focusui_rows=focusui_rows,
            dataset=dataset,
            model_name=model_name,
            model_id=model_id,
            metadata_path=metadata_csv,
            output_dir=output_dir,
            split_strategy=split_strategy,
            dry_run=dry_run,
            drop_rate=float(drop_rate),
            prune_layer=prune_layer,
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focusui-repo", type=Path)
    parser.add_argument("--model", default="qwen2vl")
    parser.add_argument("--model-id", default="FocusUI")
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/focusui"))
    parser.add_argument("--drop-rates", default=",".join(format_drop_rate(rate) for rate in DEFAULT_DROP_RATES))
    parser.add_argument("--test-size", type=int, default=750)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prune-layer", type=int, default=2)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_focusui(
        focusui_repo=args.focusui_repo,
        model_name=args.model,
        model_id=args.model_id,
        drop_rates=parse_drop_rates(args.drop_rates),
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        dry_run=args.dry_run,
        prune_layer=args.prune_layer,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
