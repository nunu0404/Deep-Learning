#!/usr/bin/env python3
"""Shared evaluation loop for token-pruning baselines."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from tqdm import tqdm

try:
    from evaluation.evaluate_baseline import (
        PROMPT,
        PredictionResult,
        build_test_split,
        compute_metrics,
        load_metadata,
        percentile,
        run_inference_with_retry,
        safe_mean,
    )
    from models.gap_pruning import format_drop_rate, sanitize_tag, summarize_pruning_info
except ImportError:  # pragma: no cover - package-relative execution fallback
    from .evaluate_baseline import (
        PROMPT,
        PredictionResult,
        build_test_split,
        compute_metrics,
        load_metadata,
        percentile,
        run_inference_with_retry,
        safe_mean,
    )
    from ..models.gap_pruning import format_drop_rate, sanitize_tag, summarize_pruning_info


BackendFactory = Callable[[float], Any]
SectionExtras = Callable[[Any], dict[str, Any]]


def setup_pruning_logging(results_dir: Path, logger_name: str) -> logging.Logger:
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
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


def _set_sample_context(backend: Any, sample_id: str) -> None:
    setter = getattr(backend, "set_sample_context", None)
    if callable(setter):
        setter(sample_id)


def evaluate_single_pruning_drop_rate(
    backend: Any,
    method_key: str,
    model_name: str,
    model_id: str,
    drop_rate: float,
    dataset: pd.DataFrame,
    metadata_path: Path,
    output_dir: Path,
    split_strategy: str,
    dry_run: bool,
    logger: logging.Logger,
    section_extras: SectionExtras | None = None,
) -> dict[str, Any]:
    prediction_rows: list[dict[str, Any]] = []
    true_classes: list[str] = []
    predicted_classes: list[str] = []
    latencies: list[float] = []
    peak_vram_values: list[float] = []
    blank_drop_pct_values: list[float] = []
    kept_token_values: list[float] = []
    dropped_token_values: list[float] = []
    error_count = 0

    tag = format_drop_rate(drop_rate)
    desc = f"{method_key} {model_name} dr={tag}"

    with tqdm(dataset.itertuples(index=False), total=len(dataset), desc=desc, unit="image") as progress:
        for row in progress:
            image_path = Path(row.image_path)
            pruning_info: dict[str, Any] = {}
            _set_sample_context(backend, str(row.sample_id))

            if not image_path.exists():
                error_message = f"Image not found: {image_path}"
                logger.error(error_message)
                result = PredictionResult(
                    raw_output="",
                    predicted_class="INVALID",
                    predicted_label=-1,
                    predicted_bug_type=None,
                    latency_ms=0.0,
                    peak_vram_mb=0.0,
                    used_retry=False,
                    error=error_message,
                )
            else:
                result = run_inference_with_retry(
                    backend=backend,
                    image_path=image_path,
                    prompt=PROMPT,
                    logger=logger,
                )
                pruning_info = summarize_pruning_info(backend.pruner.last_pruning_info)
                if result.error:
                    logger.error("Inference failed for %s: %s", image_path, result.error)

            true_classes.append(row.true_class)
            predicted_classes.append(result.predicted_class)
            latencies.append(float(result.latency_ms))
            peak_vram_values.append(float(result.peak_vram_mb))
            blank_drop_pct_values.append(float(pruning_info.get("blank_dropped_pct", 0.0)))
            kept_token_values.append(float(pruning_info.get("kept_tokens", 0)))
            dropped_token_values.append(float(pruning_info.get("dropped_tokens", 0)))
            if result.error:
                error_count += 1

            prediction_rows.append(
                {
                    "sample_id": row.sample_id,
                    "source_interface_id": row.source_interface_id,
                    "viewport": row.viewport,
                    "image_path": row.image_path,
                    "true_class": row.true_class,
                    "true_label": int(row.true_label),
                    "true_bug_type": None if row.true_class == "CLEAN" else row.true_class,
                    "predicted_class": result.predicted_class,
                    "predicted_label": int(result.predicted_label),
                    "predicted_bug_type": result.predicted_bug_type,
                    "raw_output": result.raw_output,
                    "wall_clock_latency_ms": round(float(result.latency_ms), 3),
                    "peak_vram_mb": round(float(result.peak_vram_mb), 3),
                    "used_half_resolution_retry": result.used_retry,
                    "drop_rate": float(drop_rate),
                    "num_tokens": int(pruning_info.get("num_tokens", 0)),
                    "kept_tokens": int(pruning_info.get("kept_tokens", 0)),
                    "dropped_tokens": int(pruning_info.get("dropped_tokens", 0)),
                    "blank_dropped_pct": round(float(pruning_info.get("blank_dropped_pct", 0.0)), 3),
                    "used_uniform_attention": bool(pruning_info.get("used_uniform_attention", False)),
                    "error": result.error,
                }
            )

    metrics = compute_metrics(true_classes=true_classes, predicted_classes=predicted_classes)

    safe_model_name = sanitize_tag(model_name)
    predictions_path = output_dir / f"{safe_model_name}_dr{tag}_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    method_section = {
        "drop_rate": float(drop_rate),
        "blank_dropped_pct_mean": safe_mean(blank_drop_pct_values),
        "blank_dropped_pct_p95": percentile(blank_drop_pct_values, 0.95),
        "kept_tokens_mean": safe_mean(kept_token_values),
        "dropped_tokens_mean": safe_mean(dropped_token_values),
    }
    if section_extras is not None:
        method_section.update(section_extras(backend))

    payload = {
        "model_name": model_name,
        "model_id": model_id,
        "backend": "transformers",
        "metadata_path": str(metadata_path.resolve()),
        "results_dir": str(output_dir.resolve()),
        "split_strategy": split_strategy,
        "dry_run": bool(dry_run),
        "num_samples": int(len(dataset)),
        "num_errors": int(error_count),
        "prompt": PROMPT,
        "metrics": metrics,
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
        "library_versions": backend.get_library_versions(),
        method_key: method_section,
    }

    result_path = output_dir / f"{safe_model_name}_dr{tag}.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def evaluate_pruning_sweep(
    method_key: str,
    model_name: str,
    model_id: str,
    drop_rates: list[float],
    metadata_csv: str | Path,
    output_dir: str | Path,
    test_size: int,
    seed: int,
    dry_run: bool,
    logger_name: str,
    backend_factory: BackendFactory,
    section_extras: SectionExtras | None = None,
) -> dict[str, Any]:
    if not drop_rates:
        raise ValueError(f"{method_key} evaluation requires at least one drop rate.")

    metadata_path = Path(metadata_csv)
    results_dir = Path(output_dir)
    logger = setup_pruning_logging(results_dir, logger_name=logger_name)

    metadata = load_metadata(metadata_path)
    dataset, split_strategy = build_test_split(metadata, test_size=test_size, seed=seed)
    if dry_run:
        dataset = dataset.head(10).copy()

    backend = backend_factory(float(drop_rates[0]))
    results: dict[str, Any] = {}
    try:
        for drop_rate in drop_rates:
            backend.set_drop_rate(float(drop_rate))
            results[format_drop_rate(float(drop_rate))] = evaluate_single_pruning_drop_rate(
                backend=backend,
                method_key=method_key,
                model_name=model_name,
                model_id=model_id,
                drop_rate=float(drop_rate),
                dataset=dataset,
                metadata_path=metadata_path,
                output_dir=results_dir,
                split_strategy=split_strategy,
                dry_run=dry_run,
                logger=logger,
                section_extras=section_extras,
            )
            if backend.torch is not None and backend.torch.cuda.is_available():
                backend.torch.cuda.empty_cache()
    finally:
        backend.cleanup()

    return results
