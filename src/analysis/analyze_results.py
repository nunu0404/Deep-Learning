#!/usr/bin/env python3
"""Generate paper figures, tables, and statistical summaries from experiment outputs."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, ttest_rel

try:
    from evaluation.evaluate_baseline import load_metadata
    from models.gap_pruning import GAPPruner, align_signal, extract_grid_thw, normalize_zero_one
except ImportError:  # pragma: no cover - package-relative execution fallback
    from ..evaluation.evaluate_baseline import load_metadata
    from ..models.gap_pruning import GAPPruner, align_signal, extract_grid_thw, normalize_zero_one


COLORBLIND = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#D55E00",
    "#CC78BC",
    "#CA9161",
    "#FBAFE4",
    "#949494",
]
METHOD_COLORS = {
    "baseline": COLORBLIND[7],
    "random": COLORBLIND[1],
    "fastv": COLORBLIND[0],
    "gap": COLORBLIND[2],
}
BUG_CODE_TO_CLASS = {
    "B1": "OVERLAP",
    "B2": "OVERFLOW",
    "B3": "ZINDEX",
    "B4": "TRUNCATION",
    "B5": "CONTRAST",
}
BUG_CLASS_TO_CODE = {value: key for key, value in BUG_CODE_TO_CLASS.items()}
BUG_DISPLAY = {
    "B1": "B1 Overlap",
    "B2": "B2 Overflow",
    "B3": "B3 Z-index",
    "B4": "B4 Truncation",
    "B5": "B5 Contrast",
}
METHOD_DISPLAY = {
    "baseline": "No pruning",
    "random": "Random drop",
    "fastv": "FastV",
    "gap": "GAP (ours)",
}
LABELS = ["CLEAN", "OVERLAP", "OVERFLOW", "ZINDEX", "TRUNCATION", "CONTRAST"]
PLOT = None


@dataclass
class ResultRun:
    method: str
    label: str
    file_path: Path
    payload: dict[str, Any]
    model_name: str
    drop_rate: float
    flops_pct: float
    accuracy: float
    f1_macro: float
    latency_ms: float
    vram_mb: float
    per_bug_recall: dict[str, float]
    predictions_csv: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="qwen2vl")
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"))
    parser.add_argument("--baseline-glob", default="results/**/*baseline.json")
    parser.add_argument("--random-glob", default="results/random/**/*.json")
    parser.add_argument("--fastv-glob", default="results/fastv/**/*.json")
    parser.add_argument("--gap-glob", default="results/gap/**/*.json")
    parser.add_argument("--ablation-glob", default="results/ablation/**/*.json")
    parser.add_argument("--recommended-drop-rate", type=float)
    parser.add_argument("--ablation-drop-rate", type=float)
    parser.add_argument("--patch-viz-model-id", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--patch-viz-drop-rate", type=float, default=0.5)
    parser.add_argument("--skip-patch-viz", action="store_true")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("analyze_results")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def ensure_plotting_backend():
    global PLOT
    if PLOT is not None:
        return PLOT

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "matplotlib is required to generate figures. Install it with `pip install matplotlib`."
        ) from exc

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    PLOT = plt
    return PLOT


def expand_glob(pattern: str) -> list[Path]:
    if not pattern.strip():
        return []
    return sorted(Path(path) for path in glob.glob(pattern, recursive=True))


def sanitize_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_method(path: Path, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    lowered = str(path).lower()
    if "fastv" in lowered:
        return "fastv"
    if "random" in lowered:
        return "random"
    if "gap" in lowered:
        return "gap"
    if "baseline" in lowered:
        return "baseline"
    return "unknown"


def extract_drop_rate(payload: dict[str, Any], path: Path, default: float = 0.0) -> float:
    if "drop_rate" in payload:
        return float(payload["drop_rate"])

    for section_name in ("gap", "fastv", "random"):
        section = payload.get(section_name)
        if isinstance(section, dict) and "drop_rate" in section:
            return float(section["drop_rate"])

    match = re.search(r"_dr([0-9]+(?:\.[0-9]+)?)", path.stem)
    if match:
        return float(match.group(1))
    return float(default)


def load_result_runs(pattern: str, method: str, model_name: str, logger: logging.Logger) -> list[ResultRun]:
    runs: list[ResultRun] = []
    for path in expand_glob(pattern):
        payload = load_json(path)
        payload_model = str(payload.get("model_name", "")).strip()
        if payload_model and payload_model != model_name:
            continue

        metrics = payload.get("metrics", {})
        per_bug_type = metrics.get("per_bug_type", {})
        per_bug_recall = {
            BUG_CLASS_TO_CODE[class_name]: float(per_bug_type.get(class_name, {}).get("recall", 0.0))
            for class_name in BUG_CLASS_TO_CODE
        }

        predictions_csv = payload.get("artifacts", {}).get("predictions_csv")
        runs.append(
            ResultRun(
                method=method,
                label=METHOD_DISPLAY.get(method, method),
                file_path=path,
                payload=payload,
                model_name=payload_model or model_name,
                drop_rate=float(extract_drop_rate(payload, path, default=0.0 if method == "baseline" else math.nan)),
                flops_pct=float(extract_drop_rate(payload, path, default=0.0) ** 2 * 100.0),
                accuracy=float(metrics.get("accuracy", 0.0)),
                f1_macro=float(metrics.get("macro", {}).get("f1", 0.0)),
                latency_ms=float(payload.get("latency_ms", {}).get("mean", 0.0)),
                vram_mb=float(payload.get("vram_mb", {}).get("peak", 0.0)),
                per_bug_recall=per_bug_recall,
                predictions_csv=Path(predictions_csv) if predictions_csv else None,
            )
        )

    deduped: dict[Path, ResultRun] = {}
    for run in runs:
        deduped[run.file_path.resolve()] = run

    loaded = sorted(deduped.values(), key=lambda item: (item.drop_rate, str(item.file_path)))
    logger.info("Loaded %d result files for %s from %s", len(loaded), method, pattern)
    return loaded


def select_nearest_run(runs: list[ResultRun], target_drop_rate: float) -> ResultRun | None:
    if not runs:
        return None
    return min(runs, key=lambda run: (abs(run.drop_rate - target_drop_rate), run.drop_rate))


def compute_lossless_threshold(runs: list[ResultRun], baseline_f1: float) -> tuple[ResultRun | None, ResultRun | None]:
    if not runs:
        return None, None

    safe_floor = baseline_f1 * 0.99
    previous_run: ResultRun | None = None
    for run in sorted(runs, key=lambda item: item.drop_rate):
        if run.f1_macro < safe_floor:
            return run, previous_run or sorted(runs, key=lambda item: item.drop_rate)[0]
        previous_run = run
    ordered = sorted(runs, key=lambda item: item.drop_rate)
    return None, ordered[-1]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_figure(fig: Any, path: Path) -> None:
    ensure_parent(path)
    fig.savefig(path, bbox_inches="tight")


def plot_pareto_curve(
    baseline_run: ResultRun,
    random_runs: list[ResultRun],
    fastv_runs: list[ResultRun],
    gap_runs: list[ResultRun],
    gap_threshold: ResultRun | None,
    figures_dir: Path,
) -> None:
    plt = ensure_plotting_backend()
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(
        [0.0],
        [baseline_run.f1_macro],
        color=METHOD_COLORS["baseline"],
        s=70,
        label=METHOD_DISPLAY["baseline"],
        zorder=5,
    )

    for method_key, runs, linestyle in (
        ("random", random_runs, "--"),
        ("fastv", fastv_runs, "-."),
        ("gap", gap_runs, "-"),
    ):
        if not runs:
            continue
        x_values = [run.flops_pct for run in runs]
        y_values = [run.f1_macro for run in runs]
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.0 if method_key == "gap" else 1.6,
            markersize=4.5,
            linestyle=linestyle,
            color=METHOD_COLORS[method_key],
            label=METHOD_DISPLAY[method_key],
        )

    if gap_threshold is not None:
        ax.scatter(
            [gap_threshold.flops_pct],
            [gap_threshold.f1_macro],
            color=METHOD_COLORS["gap"],
            edgecolor="black",
            marker="*",
            s=180,
            zorder=6,
        )
        ax.annotate(
            f"Lossless threshold\nr={gap_threshold.drop_rate:.1f}",
            xy=(gap_threshold.flops_pct, gap_threshold.f1_macro),
            xytext=(8, -28),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "lw": 0.8},
        )

    ax.set_xlabel("Estimated FLOPs reduction (%)")
    ax.set_ylabel("F1-macro")
    ax.set_title("Accuracy-FLOPs Pareto")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    save_figure(fig, figures_dir / "pareto_curve.pdf")
    save_figure(fig, figures_dir / "pareto_curve.png")
    plt.close(fig)


def compute_recommended_drop_rate(gap_runs: list[ResultRun], baseline_run: ResultRun) -> tuple[float, ResultRun | None]:
    threshold_run, safe_run = compute_lossless_threshold(gap_runs, baseline_run.f1_macro)
    if safe_run is None:
        return 0.0, threshold_run
    return float(safe_run.drop_rate), threshold_run


def plot_sensitivity_curves(
    gap_runs: list[ResultRun],
    recommended_drop_rate: float,
    figures_dir: Path,
) -> str | None:
    if not gap_runs:
        return None

    plt = ensure_plotting_backend()
    ordered = sorted(gap_runs, key=lambda item: item.drop_rate)
    x_values = [run.drop_rate * 100.0 for run in ordered]

    fig, ax = plt.subplots(figsize=(6, 4))

    collapse_bug: str | None = None
    collapse_point: tuple[float, float] | None = None
    earliest_collapse = float("inf")

    for color_index, (bug_code, bug_class) in enumerate(BUG_CODE_TO_CLASS.items()):
        recalls = [run.per_bug_recall.get(bug_code, 0.0) for run in ordered]
        ax.plot(
            x_values,
            recalls,
            marker="o",
            linewidth=1.8,
            markersize=4.0,
            color=COLORBLIND[color_index],
            label=BUG_DISPLAY[bug_code],
        )

        collapse_index = next((index for index, value in enumerate(recalls) if value < 0.90), None)
        if collapse_index is not None:
            drop_pct = x_values[collapse_index]
            if drop_pct < earliest_collapse:
                earliest_collapse = drop_pct
                collapse_bug = bug_code
                collapse_point = (drop_pct, recalls[collapse_index])

    ax.axvline(recommended_drop_rate * 100.0, color="black", linestyle="--", linewidth=1.0)
    ax.annotate(
        f"Recommended CI/CD r={recommended_drop_rate:.1f}",
        xy=(recommended_drop_rate * 100.0, 0.12),
        xytext=(6, 0),
        textcoords="offset points",
        rotation=90,
        va="bottom",
    )

    if collapse_bug is not None and collapse_point is not None:
        ax.scatter([collapse_point[0]], [collapse_point[1]], color="black", s=40, zorder=6)
        ax.annotate(
            f"Collapses first: {BUG_DISPLAY[collapse_bug]}",
            xy=collapse_point,
            xytext=(10, 12),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "lw": 0.8},
        )

    ax.set_xlabel("Drop rate (%)")
    ax.set_ylabel("Recall")
    ax.set_title("Sensitivity degradation by bug type")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, frameon=True)

    save_figure(fig, figures_dir / "sensitivity_curves.pdf")
    save_figure(fig, figures_dir / "sensitivity_curves.png")
    plt.close(fig)

    return collapse_bug


def load_predictions(run: ResultRun) -> pd.DataFrame:
    if run.predictions_csv is None or not run.predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV missing for {run.label}: {run.predictions_csv}")
    dataframe = pd.read_csv(run.predictions_csv)
    dataframe["sample_key"] = dataframe["sample_id"].astype(str) + "::" + dataframe["image_path"].astype(str)
    return dataframe


def per_sample_macro_f1_scores(true_classes: pd.Series, predicted_classes: pd.Series) -> np.ndarray:
    reward = 1.0 / len(LABELS)
    return np.where(true_classes.to_numpy() == predicted_classes.to_numpy(), reward, 0.0).astype(np.float64)


def run_gap_vs_fastv_ttest(
    gap_run: ResultRun | None,
    fastv_run: ResultRun | None,
    logger: logging.Logger,
) -> float | None:
    if gap_run is None or fastv_run is None:
        logger.warning("Skipping GAP vs FastV paired t-test because one of the runs is missing.")
        return None

    gap_predictions = load_predictions(gap_run)
    fastv_predictions = load_predictions(fastv_run)

    merged = gap_predictions.merge(
        fastv_predictions,
        on="sample_key",
        suffixes=("_gap", "_fastv"),
        how="inner",
    )
    if merged.empty:
        logger.warning("Skipping GAP vs FastV paired t-test because no overlapping samples were found.")
        return None

    gap_scores = per_sample_macro_f1_scores(merged["true_class_gap"], merged["predicted_class_gap"])
    fastv_scores = per_sample_macro_f1_scores(merged["true_class_gap"], merged["predicted_class_fastv"])
    statistic = ttest_rel(gap_scores, fastv_scores, nan_policy="omit")
    p_value = float(statistic.pvalue)
    logger.info("Paired t-test GAP vs FastV at operating point: p-value = %.6g", p_value)
    return p_value


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def format_metric(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def bold_best(values: list[float], prefer: str) -> set[int]:
    if not values:
        return set()
    best_value = max(values) if prefer == "max" else min(values)
    return {index for index, value in enumerate(values) if math.isclose(value, best_value, rel_tol=1e-9, abs_tol=1e-12)}


def write_main_results_table(
    baseline_run: ResultRun | None,
    random_run: ResultRun | None,
    fastv_run: ResultRun | None,
    gap_run: ResultRun | None,
    output_path: Path,
) -> None:
    rows = [run for run in [baseline_run, random_run, fastv_run, gap_run] if run is not None]
    if not rows:
        output_path.write_text("% No rows available for main results table.\n", encoding="utf-8")
        return

    accuracy_values = [run.accuracy for run in rows]
    f1_values = [run.f1_macro for run in rows]
    latency_values = [run.latency_ms for run in rows]
    vram_values = [run.vram_mb for run in rows]
    flops_values = [run.flops_pct for run in rows]

    best_accuracy = bold_best(accuracy_values, "max")
    best_f1 = bold_best(f1_values, "max")
    best_latency = bold_best(latency_values, "min")
    best_vram = bold_best(vram_values, "min")
    best_flops = bold_best(flops_values, "max")

    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Accuracy & F1-macro & Latency (ms) & VRAM (MB) & FLOPs (\%) \\",
        r"\midrule",
    ]

    for index, run in enumerate(rows):
        values = [
            latex_escape(run.label),
            format_metric(run.accuracy, 3),
            format_metric(run.f1_macro, 3),
            format_metric(run.latency_ms, 1),
            format_metric(run.vram_mb, 1),
            format_metric(run.flops_pct, 1),
        ]
        if index in best_accuracy:
            values[1] = rf"\textbf{{{values[1]}}}"
        if index in best_f1:
            values[2] = rf"\textbf{{{values[2]}}}"
        if index in best_latency:
            values[3] = rf"\textbf{{{values[3]}}}"
        if index in best_vram:
            values[4] = rf"\textbf{{{values[4]}}}"
        if index in best_flops:
            values[5] = rf"\textbf{{{values[5]}}}"
        lines.append(" & ".join(values) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    ensure_parent(output_path)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def derive_ablation_label(run: ResultRun) -> str:
    payload_label = str(run.payload.get("ablation_name") or run.payload.get("method_label") or "").strip()
    if payload_label:
        return payload_label

    gap_section = run.payload.get("gap", {})
    alpha = float(gap_section.get("alpha", 0.0))
    beta = float(gap_section.get("beta", 0.0))
    gamma = float(gap_section.get("gamma", 0.0))
    components: list[str] = []
    if alpha > 0:
        components.append("Attention")
    if beta > 0:
        components.append("1-Entropy")
    if gamma > 0:
        components.append("Edge")
    return " + ".join(components) if components else run.file_path.stem


def write_ablation_table(runs: list[ResultRun], output_path: Path) -> None:
    if not runs:
        output_path.write_text("% No rows available for ablation table.\n", encoding="utf-8")
        return

    accuracy_values = [run.accuracy for run in runs]
    f1_values = [run.f1_macro for run in runs]
    best_accuracy = bold_best(accuracy_values, "max")
    best_f1 = bold_best(f1_values, "max")

    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Variant & $\alpha$ & $\beta$ & $\gamma$ & Accuracy & F1-macro \\",
        r"\midrule",
    ]

    for index, run in enumerate(runs):
        gap_section = run.payload.get("gap", {})
        values = [
            latex_escape(derive_ablation_label(run)),
            format_metric(float(gap_section.get("alpha", 0.0)), 2),
            format_metric(float(gap_section.get("beta", 0.0)), 2),
            format_metric(float(gap_section.get("gamma", 0.0)), 2),
            format_metric(run.accuracy, 3),
            format_metric(run.f1_macro, 3),
        ]
        if index in best_accuracy:
            values[4] = rf"\textbf{{{values[4]}}}"
        if index in best_f1:
            values[5] = rf"\textbf{{{values[5]}}}"
        lines.append(" & ".join(values) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    ensure_parent(output_path)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def metadata_with_keys(metadata_path: Path) -> pd.DataFrame:
    metadata = load_metadata(metadata_path)
    metadata["sample_key"] = metadata["sample_id"].astype(str) + "::" + metadata["image_path"].astype(str)
    return metadata


def plot_vss_correlation(
    metadata: pd.DataFrame,
    gap_runs: list[ResultRun],
    figures_dir: Path,
    logger: logging.Logger,
) -> float | None:
    if "vss_score" not in metadata.columns:
        logger.warning("Skipping VSS correlation because metadata.csv does not contain vss_score.")
        return None
    if not gap_runs:
        logger.warning("Skipping VSS correlation because GAP runs are unavailable.")
        return None

    ordered = sorted(gap_runs, key=lambda item: item.drop_rate)
    merged_frames: list[pd.DataFrame] = []

    for run in ordered:
        predictions = load_predictions(run)
        predictions["drop_rate"] = run.drop_rate
        predictions["correct"] = (predictions["predicted_class"] == predictions["true_class"]).astype(int)
        merged_frames.append(predictions[["sample_key", "drop_rate", "correct", "true_label"]])

    if not merged_frames:
        logger.warning("Skipping VSS correlation because no prediction files were found.")
        return None

    predictions_all = pd.concat(merged_frames, ignore_index=True)
    predictions_all = predictions_all.loc[predictions_all["true_label"] == 1].copy()
    if predictions_all.empty:
        logger.warning("Skipping VSS correlation because there are no buggy samples in the prediction files.")
        return None

    robust_drop = (
        predictions_all.loc[predictions_all["correct"] == 1]
        .groupby("sample_key")["drop_rate"]
        .max()
        .rename("max_safe_drop_rate")
    )
    joined = metadata.merge(robust_drop, on="sample_key", how="left")
    joined = joined.loc[(joined["label"] == 1) & joined["vss_score"].notna()].copy()
    joined["max_safe_drop_rate"] = joined["max_safe_drop_rate"].fillna(0.0)

    if len(joined) < 2:
        logger.warning("Skipping VSS correlation because fewer than two bug samples are available.")
        return None

    correlation = pearsonr(joined["vss_score"].astype(float), joined["max_safe_drop_rate"].astype(float))
    r_value = float(correlation.statistic)
    p_value = float(correlation.pvalue)
    logger.info("Pearson correlation VSS vs max safe drop rate: r = %.4f, p = %.6g", r_value, p_value)

    plt = ensure_plotting_backend()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        joined["vss_score"].astype(float),
        joined["max_safe_drop_rate"].astype(float) * 100.0,
        color=METHOD_COLORS["gap"],
        alpha=0.7,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel("VSS score")
    ax.set_ylabel("Max safe drop rate (%)")
    ax.set_title(f"VSS correlation (r={r_value:.2f})")
    ax.grid(True, alpha=0.25)

    save_figure(fig, figures_dir / "vss_correlation.pdf")
    plt.close(fig)
    return r_value


def select_representative_samples(metadata: pd.DataFrame, bug_codes: list[str]) -> dict[str, pd.Series]:
    selections: dict[str, pd.Series] = {}
    for bug_code in bug_codes:
        subset = metadata.loc[(metadata["label"] == 1) & (metadata["bug_type"] == bug_code)].copy()
        if subset.empty:
            continue
        if "vss_score" in subset.columns and subset["vss_score"].notna().any():
            subset["vss_score"] = subset["vss_score"].fillna(subset["vss_score"].median())
            subset = subset.sort_values("vss_score", kind="stable").reset_index(drop=True)
            selections[bug_code] = subset.iloc[len(subset) // 2]
        else:
            subset = subset.sort_values(["sample_id", "image_path"], kind="stable").reset_index(drop=True)
            selections[bug_code] = subset.iloc[0]
    return selections


def load_patch_viz_backend(model_id: str, hf_token: str | None) -> tuple[Any, Any, GAPPruner, Any]:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model_kwargs: dict[str, Any] = {
        "token": hf_token,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    model.eval()

    try:
        pruner = GAPPruner(model, drop_rate=0.5)
    except ValueError:
        pruner = GAPPruner(model.model, drop_rate=0.5)
    pruner.register_hooks()
    return model, processor, pruner, torch


def to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def compute_visualization_maps(
    image: Image.Image,
    model: Any,
    processor: Any,
    pruner: GAPPruner,
    torch_module: Any,
    drop_rate: float,
) -> tuple[np.ndarray, np.ndarray, list[int], tuple[int, int, int] | None]:
    batch = processor(text=["visualize gui patches"], images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    pruner.drop_rate = float(drop_rate)
    pruner.set_current_image(image, grid_thw=batch.get("image_grid_thw"))

    batch = to_device(batch, device)
    with torch_module.inference_mode():
        if hasattr(model, "get_image_features"):
            _ = model.get_image_features(pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"])
        elif hasattr(model, "model") and hasattr(model.model, "get_image_features"):
            _ = model.model.get_image_features(pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"])
        else:
            raise ValueError("Provided model does not expose get_image_features for patch visualization.")

    signals = pruner.compute_patch_signals(image)
    num_tokens = len(signals["entropy"])
    attention = pruner._get_attention_signal(num_tokens)
    gui_ss = pruner.compute_gui_ss(image)
    keep_indices = pruner.get_tokens_to_keep(image, num_tokens)
    return attention, gui_ss, keep_indices, extract_grid_thw(batch.get("image_grid_thw"))


def patch_scores_to_grid(
    scores: np.ndarray,
    grid_thw: tuple[int, int, int] | None,
    image: Image.Image,
    patch_size: int,
) -> np.ndarray:
    if grid_thw is None:
        grid_h = max(1, math.ceil(image.height / patch_size))
        grid_w = max(1, math.ceil(image.width / patch_size))
        grid_t = 1
    else:
        grid_t, grid_h, grid_w = grid_thw
    aligned = align_signal(scores, max(1, grid_t * grid_h * grid_w))
    grid = aligned.reshape(grid_t, grid_h, grid_w).mean(axis=0)
    return normalize_zero_one(grid)


def render_heatmap_overlay(image: Image.Image, heatmap_grid: np.ndarray, patch_size: int) -> np.ndarray:
    upsampled = np.kron(heatmap_grid, np.ones((patch_size, patch_size), dtype=np.float32))
    upsampled = Image.fromarray((normalize_zero_one(upsampled) * 255).astype(np.uint8)).resize(
        image.size, Image.Resampling.BILINEAR
    )
    return np.asarray(upsampled, dtype=np.float32) / 255.0


def render_pruned_visual(
    image: Image.Image,
    keep_indices: list[int],
    grid_thw: tuple[int, int, int] | None,
    patch_size: int,
) -> np.ndarray:
    if grid_thw is None:
        grid_t, grid_h, grid_w = 1, max(1, math.ceil(image.height / patch_size)), max(1, math.ceil(image.width / patch_size))
    else:
        grid_t, grid_h, grid_w = grid_thw

    mask = np.zeros((grid_t * grid_h * grid_w,), dtype=np.float32)
    mask[np.asarray(keep_indices, dtype=int)] = 1.0
    mask_grid = mask.reshape(grid_t, grid_h, grid_w).max(axis=0)
    mask_img = np.kron(mask_grid, np.ones((patch_size, patch_size), dtype=np.float32))
    mask_img = Image.fromarray((mask_img * 255).astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST)
    mask_arr = np.asarray(mask_img, dtype=np.float32) / 255.0

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    dimmed = rgb * (0.25 + 0.75 * mask_arr[..., None])
    return np.clip(dimmed, 0.0, 1.0)


def plot_patch_visualizations(
    metadata: pd.DataFrame,
    figures_dir: Path,
    model_id: str,
    hf_token: str | None,
    drop_rate: float,
    logger: logging.Logger,
) -> None:
    selections = select_representative_samples(metadata, ["B1", "B2", "B5"])
    if not selections:
        logger.warning("Skipping patch visualization because no representative images were found.")
        return

    plt = ensure_plotting_backend()
    model, processor, pruner, torch_module = load_patch_viz_backend(model_id=model_id, hf_token=hf_token)
    device = next(model.parameters()).device
    logger.info("Loaded patch-visualization model on %s", device)

    try:
        for bug_code, row in selections.items():
            image_path = Path(row["image_path"])
            if not image_path.exists():
                logger.warning("Skipping patch visualization for %s because %s does not exist.", bug_code, image_path)
                continue

            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")

            attention, gui_ss, keep_indices, grid_thw = compute_visualization_maps(
                image=image,
                model=model,
                processor=processor,
                pruner=pruner,
                torch_module=torch_module,
                drop_rate=drop_rate,
            )
            attention_grid = patch_scores_to_grid(attention, grid_thw, image, pruner.patch_size)
            gui_grid = patch_scores_to_grid(gui_ss, grid_thw, image, pruner.patch_size)
            attention_overlay = render_heatmap_overlay(image, attention_grid, pruner.patch_size)
            gui_overlay = render_heatmap_overlay(image, gui_grid, pruner.patch_size)
            pruned_visual = render_pruned_visual(image, keep_indices, grid_thw, pruner.patch_size)

            fig, axes = plt.subplots(1, 4, figsize=(10, 4))
            titles = ["Original", "Attention heatmap", "GUI-SS heatmap", f"Pruned result (r={drop_rate:.1f})"]
            panels = [
                np.asarray(image),
                np.asarray(image),
                np.asarray(image),
                (pruned_visual * 255).astype(np.uint8),
            ]
            overlays = [None, attention_overlay, gui_overlay, None]

            for axis, title, panel, overlay in zip(axes, titles, panels, overlays, strict=True):
                axis.imshow(panel)
                if overlay is not None:
                    axis.imshow(overlay, cmap="jet", alpha=0.5)
                axis.set_title(title)
                axis.axis("off")

            fig.suptitle(BUG_DISPLAY.get(bug_code, bug_code))
            save_figure(fig, figures_dir / f"patch_viz_{bug_code}.pdf")
            plt.close(fig)
    finally:
        pruner.remove_hooks()
        del model
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()


def collect_ablation_runs(pattern: str, model_name: str, target_drop_rate: float | None, logger: logging.Logger) -> list[ResultRun]:
    runs = load_result_runs(pattern, method="gap", model_name=model_name, logger=logger)
    if target_drop_rate is None:
        return runs

    grouped: dict[str, list[ResultRun]] = {}
    for run in runs:
        grouped.setdefault(derive_ablation_label(run), []).append(run)

    selected: list[ResultRun] = []
    for group_runs in grouped.values():
        chosen = select_nearest_run(group_runs, target_drop_rate)
        if chosen is not None:
            selected.append(chosen)
    return sorted(selected, key=lambda run: derive_ablation_label(run))


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    figures_dir: Path = args.figures_dir
    tables_dir: Path = args.tables_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    baseline_runs = load_result_runs(args.baseline_glob, method="baseline", model_name=args.model_name, logger=logger)
    random_runs = load_result_runs(args.random_glob, method="random", model_name=args.model_name, logger=logger)
    fastv_runs = load_result_runs(args.fastv_glob, method="fastv", model_name=args.model_name, logger=logger)
    gap_runs = load_result_runs(args.gap_glob, method="gap", model_name=args.model_name, logger=logger)

    baseline_run = baseline_runs[0] if baseline_runs else None
    if baseline_run is None:
        raise FileNotFoundError("No baseline result JSON was found. Expected a file matching --baseline-glob.")

    if args.recommended_drop_rate is None:
        recommended_drop_rate, gap_threshold = compute_recommended_drop_rate(gap_runs, baseline_run)
    else:
        recommended_drop_rate = float(args.recommended_drop_rate)
        gap_threshold, _ = compute_lossless_threshold(gap_runs, baseline_run.f1_macro)
    logger.info("Recommended CI/CD drop rate: %.3f", recommended_drop_rate)

    plot_pareto_curve(
        baseline_run=baseline_run,
        random_runs=random_runs,
        fastv_runs=fastv_runs,
        gap_runs=gap_runs,
        gap_threshold=gap_threshold,
        figures_dir=figures_dir,
    )
    collapse_bug = plot_sensitivity_curves(
        gap_runs=gap_runs,
        recommended_drop_rate=recommended_drop_rate,
        figures_dir=figures_dir,
    )
    if collapse_bug is not None:
        logger.info("First collapsing bug type: %s", BUG_DISPLAY.get(collapse_bug, collapse_bug))

    random_operating_run = select_nearest_run(random_runs, recommended_drop_rate)
    fastv_operating_run = select_nearest_run(fastv_runs, recommended_drop_rate)
    gap_operating_run = select_nearest_run(gap_runs, recommended_drop_rate)

    write_main_results_table(
        baseline_run=baseline_run,
        random_run=random_operating_run,
        fastv_run=fastv_operating_run,
        gap_run=gap_operating_run,
        output_path=tables_dir / "main_results.tex",
    )

    ablation_drop_rate = args.ablation_drop_rate if args.ablation_drop_rate is not None else recommended_drop_rate
    ablation_runs = collect_ablation_runs(
        pattern=args.ablation_glob,
        model_name=args.model_name,
        target_drop_rate=ablation_drop_rate,
        logger=logger,
    )
    write_ablation_table(ablation_runs, tables_dir / "ablation.tex")

    p_value = run_gap_vs_fastv_ttest(gap_operating_run, fastv_operating_run, logger=logger)
    metadata = metadata_with_keys(args.metadata_csv)
    correlation = plot_vss_correlation(metadata=metadata, gap_runs=gap_runs, figures_dir=figures_dir, logger=logger)

    if not args.skip_patch_viz:
        try:
            plot_patch_visualizations(
                metadata=metadata,
                figures_dir=figures_dir,
                model_id=args.patch_viz_model_id,
                hf_token=args.hf_token,
                drop_rate=args.patch_viz_drop_rate,
                logger=logger,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Patch visualization failed: %s: %s", type(exc).__name__, exc)

    summary = {
        "recommended_drop_rate": recommended_drop_rate,
        "gap_lossless_threshold": None if gap_threshold is None else gap_threshold.drop_rate,
        "collapse_bug_type": collapse_bug,
        "gap_vs_fastv_p_value": p_value,
        "vss_correlation_r": correlation,
        "figures_dir": str(figures_dir.resolve()),
        "tables_dir": str(tables_dir.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
