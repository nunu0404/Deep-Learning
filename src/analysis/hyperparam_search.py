#!/usr/bin/env python3
"""Validation-set grid search for GAP alpha/beta/gamma weights."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from evaluation.evaluate_baseline import BUG_TYPE_MAP, load_metadata
    from evaluation.splits import apply_budget, default_splits_path, select_split_dataframe
    from models.gap_pruning import evaluate_gap, format_drop_rate
except ImportError:  # pragma: no cover - package-relative execution fallback
    from ..evaluation.evaluate_baseline import BUG_TYPE_MAP, load_metadata
    from ..evaluation.splits import apply_budget, default_splits_path, select_split_dataframe
    from ..models.gap_pruning import evaluate_gap, format_drop_rate


BUG_CODE_BY_CLASS = {value: key for key, value in BUG_TYPE_MAP.items() if value != "CLEAN"}


def simplex_grid(step: float) -> list[tuple[float, float, float]]:
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1].")
    divisions = round(1.0 / step)
    if not math.isclose(divisions * step, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("step must evenly divide 1.0.")

    grid: list[tuple[float, float, float]] = []
    for alpha_index in range(divisions + 1):
        for beta_index in range(divisions - alpha_index + 1):
            gamma_index = divisions - alpha_index - beta_index
            grid.append(
                (
                    round(alpha_index / divisions, 6),
                    round(beta_index / divisions, 6),
                    round(gamma_index / divisions, 6),
                )
            )
    return grid


def combo_tag(alpha: float, beta: float, gamma: float) -> str:
    return f"a{alpha:.2f}_b{beta:.2f}_g{gamma:.2f}".replace(".", "p")


def ensure_dirs(results_dir: Path, figures_dir: Path, tables_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)


def write_filtered_metadata(dataframe: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def payload_to_row(payload: dict[str, Any], alpha: float, beta: float, gamma: float, drop_rate: float) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    row = {
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "drop_rate": float(drop_rate),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "macro_f1": float(metrics.get("macro", {}).get("f1", 0.0)),
        "macro_precision": float(metrics.get("macro", {}).get("precision", 0.0)),
        "macro_recall": float(metrics.get("macro", {}).get("recall", 0.0)),
        "latency_ms": float(payload.get("latency_ms", {}).get("mean", 0.0)),
        "vram_mb": float(payload.get("vram_mb", {}).get("peak", 0.0)),
        "result_path": "",
    }
    for bug_class, bug_code in BUG_CODE_BY_CLASS.items():
        row[f"{bug_code}_f1"] = float(metrics.get("per_bug_type", {}).get(bug_class, {}).get("f1", 0.0))
        row[f"{bug_code}_recall"] = float(metrics.get("per_bug_type", {}).get(bug_class, {}).get("recall", 0.0))
    return row


def pareto_mask(dataframe: pd.DataFrame) -> pd.Series:
    mask = []
    for row in dataframe.itertuples(index=False):
        dominated = (
            (dataframe["macro_f1"] >= row.macro_f1)
            & (dataframe["latency_ms"] <= row.latency_ms)
            & ((dataframe["macro_f1"] > row.macro_f1) | (dataframe["latency_ms"] < row.latency_ms))
        ).any()
        mask.append(not bool(dominated))
    return pd.Series(mask, index=dataframe.index)


def plot_simplex(dataframe: pd.DataFrame, output_path: Path, step: float) -> None:
    try:
        import ternary
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("python-ternary is required. Install it with `pip install python-ternary`.") from exc

    scale = round(1.0 / step)
    heatmap_data = {
        (
            int(round(row.alpha * scale)),
            int(round(row.beta * scale)),
            int(round(row.gamma * scale)),
        ): float(row.macro_f1)
        for row in dataframe.itertuples(index=False)
    }

    figure, tax = ternary.figure(scale=scale)
    tax.heatmap(heatmap_data, style="hexagonal", use_rgba=False, colorbar=True)
    tax.boundary(linewidth=1.0)
    tax.gridlines(color="black", multiple=max(1, scale // 10), linewidth=0.4, alpha=0.25)
    tax.left_axis_label("gamma: edge density", offset=0.14)
    tax.right_axis_label("beta: 1-entropy", offset=0.14)
    tax.bottom_axis_label("alpha: attention", offset=0.06)
    tax.clear_matplotlib_ticks()
    tax.get_axes().set_title("Validation macro F1 over GAP alpha/beta/gamma simplex")
    figure.savefig(output_path, bbox_inches="tight")


def plot_pareto(dataframe: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(dataframe["latency_ms"], dataframe["macro_f1"], alpha=0.65, s=28, label="Grid points")
    front = dataframe.loc[pareto_mask(dataframe)].sort_values("latency_ms")
    if not front.empty:
        axis.plot(front["latency_ms"], front["macro_f1"], color="black", linewidth=1.4, label="Pareto front")
    axis.set_xlabel("Latency (ms)")
    axis.set_ylabel("Validation macro F1")
    axis.set_title("Alpha/Beta/Gamma validation Pareto")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def write_top10_table(dataframe: pd.DataFrame, output_path: Path) -> None:
    top = dataframe.sort_values(["macro_f1", "latency_ms"], ascending=[False, True]).head(10)
    lines = [
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"Rank & $\alpha$ & $\beta$ & $\gamma$ & F1-macro & Latency (ms) \\",
        r"\midrule",
    ]
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(
            f"{rank} & {row.alpha:.2f} & {row.beta:.2f} & {row.gamma:.2f} & {row.macro_f1:.3f} & {row.latency_ms:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_per_bug_best(dataframe: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bug_code in ("B1", "B2", "B3", "B4", "B5"):
        metric_name = f"{bug_code}_f1"
        if metric_name not in dataframe.columns:
            continue
        best = dataframe.sort_values([metric_name, "macro_f1", "latency_ms"], ascending=[False, False, True]).iloc[0]
        rows.append(
            {
                "bug_type": bug_code,
                "alpha": float(best["alpha"]),
                "beta": float(best["beta"]),
                "gamma": float(best["gamma"]),
                "drop_rate": float(best["drop_rate"]),
                "bug_f1": float(best[metric_name]),
                "macro_f1": float(best["macro_f1"]),
                "latency_ms": float(best["latency_ms"]),
            }
        )
    best_df = pd.DataFrame(rows)
    best_df.to_csv(output_path, index=False)
    return best_df


def append_tuned_gap_row(main_results_path: Path, payload: dict[str, Any], alpha: float, beta: float, gamma: float) -> None:
    metrics = payload.get("metrics", {})
    line = (
        f"Validation-tuned GAP ({alpha:.2f}/{beta:.2f}/{gamma:.2f}) & "
        f"{float(metrics.get('accuracy', 0.0)):.3f} & "
        f"{float(metrics.get('macro', {}).get('f1', 0.0)):.3f} & "
        f"{float(payload.get('latency_ms', {}).get('mean', 0.0)):.1f} & "
        f"{float(payload.get('vram_mb', {}).get('peak', 0.0)):.1f} & "
        f"{float(payload.get('gap', {}).get('drop_rate', 0.0)) ** 2 * 100.0:.1f} \\\\"
    )
    if main_results_path.exists():
        text = main_results_path.read_text(encoding="utf-8")
        if "Validation-tuned GAP" in text:
            return
        marker = r"\bottomrule"
        if marker in text:
            main_results_path.write_text(text.replace(marker, line + "\n" + marker), encoding="utf-8")
            return
    main_results_path.write_text(
        "\n".join(
            [
                r"\begin{tabular}{lccccc}",
                r"\toprule",
                r"Method & Accuracy & F1-macro & Latency (ms) & VRAM (MB) & FLOPs (\%) \\",
                r"\midrule",
                line,
                r"\bottomrule",
                r"\end{tabular}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dirs(args.results_dir, args.figures_dir, args.tables_dir)
    metadata_path = args.metadata_csv
    splits_path = args.splits_path or default_splits_path(metadata_path)

    metadata = load_metadata(metadata_path)
    val_df, split_strategy = select_split_dataframe(metadata, split="val", splits_path=splits_path, seed=args.seed)
    val_df = apply_budget(val_df, budget_images=args.budget_images, seed=args.seed)
    val_metadata_path = write_filtered_metadata(val_df, args.results_dir / "val_metadata.csv")

    step = 0.05 if args.fine else 0.1
    grid = simplex_grid(step)
    rows: list[dict[str, Any]] = []

    for alpha, beta, gamma in grid:
        run_dir = args.results_dir / "runs" / combo_tag(alpha, beta, gamma)
        payload_by_rate = evaluate_gap(
            model_name=args.model,
            model_id=args.model_id,
            drop_rates=[args.drop_rate],
            metadata_csv=val_metadata_path,
            output_dir=run_dir,
            test_size=max(len(val_df), 1),
            seed=args.seed,
            split="all",
            budget_images=None,
            dry_run=args.dry_run,
            max_new_tokens=args.max_new_tokens,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            hf_token=args.hf_token,
        )
        payload = payload_by_rate[format_drop_rate(args.drop_rate)]
        row = payload_to_row(payload, alpha=alpha, beta=beta, gamma=gamma, drop_rate=args.drop_rate)
        row["result_path"] = str((run_dir / f"{args.model}_dr{format_drop_rate(args.drop_rate)}.json").resolve())
        rows.append(row)

    grid_df = pd.DataFrame(rows).sort_values(["macro_f1", "latency_ms"], ascending=[False, True]).reset_index(drop=True)
    grid_csv = args.results_dir / "grid_results.csv"
    grid_df.to_csv(grid_csv, index=False)

    plot_simplex(grid_df, args.figures_dir / "abg_simplex.pdf", step=step)
    plot_pareto(grid_df, args.figures_dir / "abg_pareto.pdf")
    write_top10_table(grid_df, args.tables_dir / "abg_top10.tex")
    per_bug_path = args.results_dir / "per_bug_best.csv"
    per_bug_best = write_per_bug_best(grid_df, per_bug_path)

    best = grid_df.iloc[0]
    final_payload: dict[str, Any] | None = None
    if not args.skip_final_test:
        final_results = evaluate_gap(
            model_name=args.model,
            model_id=args.model_id,
            drop_rates=[args.drop_rate],
            metadata_csv=metadata_path,
            output_dir=args.results_dir / "final_test",
            test_size=args.test_size,
            seed=args.seed,
            split="test",
            splits_path=splits_path,
            budget_images=None,
            dry_run=args.dry_run,
            max_new_tokens=args.max_new_tokens,
            alpha=float(best["alpha"]),
            beta=float(best["beta"]),
            gamma=float(best["gamma"]),
            hf_token=args.hf_token,
        )
        final_payload = final_results[format_drop_rate(args.drop_rate)]
        append_tuned_gap_row(args.tables_dir / "main_results.tex", final_payload, float(best["alpha"]), float(best["beta"]), float(best["gamma"]))

    summary = {
        "split_strategy": split_strategy,
        "splits_path": str(splits_path.resolve()),
        "grid_size": int(len(grid_df)),
        "best": {
            "alpha": float(best["alpha"]),
            "beta": float(best["beta"]),
            "gamma": float(best["gamma"]),
            "drop_rate": float(best["drop_rate"]),
            "macro_f1": float(best["macro_f1"]),
            "latency_ms": float(best["latency_ms"]),
        },
        "grid_results_csv": str(grid_csv.resolve()),
        "per_bug_best_csv": str(per_bug_path.resolve()),
        "per_bug_best": per_bug_best.to_dict(orient="records"),
        "final_test": None if final_payload is None else final_payload.get("artifacts", {}),
    }
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2vl")
    parser.add_argument("--model-id")
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--splits-path", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results/hyperparam_search"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"))
    parser.add_argument("--drop-rate", type=float, default=0.5)
    parser.add_argument("--budget-images", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=750)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fine", action="store_true")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    return parser.parse_args()


def main() -> None:
    summary = run_search(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
