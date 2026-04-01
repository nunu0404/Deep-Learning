#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

DEMO_DIR="${DEMO_DIR:-$ROOT_DIR/demo_artifacts}"
DATA_DIR="$DEMO_DIR/data"
RESULTS_DIR="$DEMO_DIR/results"
FIGURES_DIR="$DEMO_DIR/figures"
TABLES_DIR="$DEMO_DIR/tables"

print_section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$FIGURES_DIR" "$TABLES_DIR"

print_section "GAP GUI Bug Quick Demo"
echo "Root dir:        $ROOT_DIR"
echo "Demo dir:        $DEMO_DIR"
echo "Data dir:        $DATA_DIR"
echo "Results dir:     $RESULTS_DIR"
echo "Figures dir:     $FIGURES_DIR"
echo "Tables dir:      $TABLES_DIR"
echo "HF_XET disabled: $HF_HUB_DISABLE_XET"
echo
echo "Demo target:     total_images=10, samples_per_bug=1, seed=42"
echo "Models used:     qwen2vl only"

print_section "[1/4] Building Demo Dataset"
python -m dataset.build_dataset \
  --n_samples 10 \
  --samples-per-bug 1 \
  --seed 42 \
  --output-dir "$DATA_DIR"
echo "[done] Demo dataset build finished. Metadata: $DATA_DIR/metadata.csv"

print_section "[2/4] Baseline Dry Run"
python -m evaluation.evaluate_baseline \
  --model qwen2vl \
  --metadata-path "$DATA_DIR/metadata.csv" \
  --results-dir "$RESULTS_DIR/baseline" \
  --dry_run
echo "[done] Baseline dry run finished."

print_section "[3/4] GAP Dry Run"
python -m evaluation.evaluate_gap \
  --model qwen2vl \
  --metadata-csv "$DATA_DIR/metadata.csv" \
  --output-dir "$RESULTS_DIR/gap" \
  --drop-rates "0.0,0.5" \
  --dry-run
echo "[done] GAP dry run finished."

print_section "[4/4] Analysis"
python -m analysis.analyze_results \
  --model-name qwen2vl \
  --metadata-csv "$DATA_DIR/metadata.csv" \
  --figures-dir "$FIGURES_DIR" \
  --tables-dir "$TABLES_DIR" \
  --baseline-glob "$RESULTS_DIR/baseline/**/*baseline.json" \
  --gap-glob "$RESULTS_DIR/gap/**/*.json" \
  --random-glob "$RESULTS_DIR/random/**/*.json" \
  --fastv-glob "$RESULTS_DIR/fastv/**/*.json" \
  --ablation-glob "$RESULTS_DIR/ablation/**/*.json" \
  --skip-patch-viz
echo "[done] Figure and table generation finished."

print_section "Quick Demo Complete"
echo "Artifacts"
echo "  Data:    $DATA_DIR"
echo "  Results: $RESULTS_DIR"
echo "  Figures: $FIGURES_DIR"
echo "  Tables:  $TABLES_DIR"
