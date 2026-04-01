#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/default.yaml}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results}"
FIGURES_DIR="${FIGURES_DIR:-$ROOT_DIR/figures}"
TABLES_DIR="${TABLES_DIR:-$ROOT_DIR/tables}"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

print_section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

config_get() {
  python - "$CONFIG_PATH" "$1" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
key_path = sys.argv[2].split(".")
with open(config_path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

value = config
for key in key_path:
    value = value[key]

if isinstance(value, list):
    print(",".join(str(item) for item in value))
else:
    print(value)
PY
}

warn_gpu_memory() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[warn] nvidia-smi not found; skipping GPU memory check." >&2
    return
  fi

  local mem_mb
  mem_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  if [[ -n "$mem_mb" ]] && (( mem_mb < 8192 )); then
    echo "[warn] Detected GPU memory ${mem_mb} MB (< 8192 MB). Some evaluations may OOM." >&2
  fi
}

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$FIGURES_DIR" "$TABLES_DIR"

N_SAMPLES="$(config_get dataset.n_samples)"
SAMPLES_PER_BUG="$(config_get dataset.samples_per_bug)"
SEED="$(config_get dataset.seed)"
ALPHA="$(config_get gap.alpha)"
BETA="$(config_get gap.beta)"
GAMMA="$(config_get gap.gamma)"
DROP_RATES="$(config_get gap.drop_rates)"
MODELS="$(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
for key in config["models"]:
    print(key)
PY
)"

print_section "GAP GUI Bug Full Pipeline"
echo "Root dir:        $ROOT_DIR"
echo "Config:          $CONFIG_PATH"
echo "Data dir:        $DATA_DIR"
echo "Results dir:     $RESULTS_DIR"
echo "Figures dir:     $FIGURES_DIR"
echo "Tables dir:      $TABLES_DIR"
echo "HF_XET disabled: $HF_HUB_DISABLE_XET"
echo
echo "Dataset target:  total_images=$N_SAMPLES, samples_per_bug=$SAMPLES_PER_BUG, seed=$SEED"
echo "GAP config:      alpha=$ALPHA beta=$BETA gamma=$GAMMA drop_rates=$DROP_RATES"
echo "Baseline models: $(echo "$MODELS" | tr '\n' ' ' | sed 's/ $//')"

warn_gpu_memory

print_section "[1/5] Building Dataset"
python -m dataset.build_dataset \
  --n_samples "$N_SAMPLES" \
  --samples-per-bug "$SAMPLES_PER_BUG" \
  --seed "$SEED" \
  --output-dir "$DATA_DIR"
echo "[done] Dataset build finished. Metadata: $DATA_DIR/metadata.csv"

print_section "[2/5] Baseline Evaluations"
for model in $MODELS; do
  echo "[baseline] Dry run for model: $model"
  python -m evaluation.evaluate_baseline \
    --model "$model" \
    --metadata-path "$DATA_DIR/metadata.csv" \
    --results-dir "$RESULTS_DIR/baseline" \
    --dry_run

  echo "[baseline] Full evaluation for model: $model"
  python -m evaluation.evaluate_baseline \
    --model "$model" \
    --metadata-path "$DATA_DIR/metadata.csv" \
    --results-dir "$RESULTS_DIR/baseline"
done
echo "[done] Baseline evaluations finished."

print_section "[3/5] GAP Sweep"
python -m evaluation.evaluate_gap \
  --model qwen2vl \
  --metadata-csv "$DATA_DIR/metadata.csv" \
  --output-dir "$RESULTS_DIR/gap" \
  --drop-rates "$DROP_RATES" \
  --alpha "$ALPHA" \
  --beta "$BETA" \
  --gamma "$GAMMA"
echo "[done] GAP sweep finished."

print_section "[4/5] Analysis"
if [[ ! -d "$RESULTS_DIR/random" ]]; then
  echo "[warn] results/random is missing; Pareto plots will omit Random drop." >&2
fi
if [[ ! -d "$RESULTS_DIR/fastv" ]]; then
  echo "[warn] results/fastv is missing; Pareto plots will omit FastV." >&2
fi

python -m analysis.analyze_results \
  --model-name qwen2vl \
  --metadata-csv "$DATA_DIR/metadata.csv" \
  --figures-dir "$FIGURES_DIR" \
  --tables-dir "$TABLES_DIR" \
  --baseline-glob "$RESULTS_DIR/baseline/**/*baseline.json" \
  --random-glob "$RESULTS_DIR/random/**/*.json" \
  --fastv-glob "$RESULTS_DIR/fastv/**/*.json" \
  --gap-glob "$RESULTS_DIR/gap/**/*.json" \
  --ablation-glob "$RESULTS_DIR/ablation/**/*.json" \
  --skip-patch-viz
echo "[done] Figure and table generation finished."

print_section "[5/5] Summary"
python - "$RESULTS_DIR" <<'PY'
import json
import glob
import os

results_dir = os.path.abspath(__import__("sys").argv[1])
baseline_paths = sorted(glob.glob(os.path.join(results_dir, "baseline", "*_baseline.json")))
gap_paths = sorted(glob.glob(os.path.join(results_dir, "gap", "*.json")))

best_gap = None
for path in gap_paths:
    if path.endswith("_predictions.csv"):
        continue
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    f1 = payload.get("metrics", {}).get("macro", {}).get("f1", 0.0)
    drop_rate = payload.get("gap", {}).get("drop_rate", 0.0)
    if best_gap is None or f1 > best_gap["f1"]:
        best_gap = {"path": path, "f1": f1, "drop_rate": drop_rate}

print()
print("Summary")
print("model\taccuracy\tf1_macro\tlatency_ms\tvram_mb")
for path in baseline_paths:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    print(
        f"{payload['model_name']}\t"
        f"{payload['metrics']['accuracy']:.3f}\t"
        f"{payload['metrics']['macro']['f1']:.3f}\t"
        f"{payload['latency_ms']['mean']:.1f}\t"
        f"{payload['vram_mb']['peak']:.1f}"
    )

if best_gap is not None:
    print(
        f"GAP@r={best_gap['drop_rate']:.1f}\t-\t{best_gap['f1']:.3f}\t-\t-"
    )
PY

echo
echo "Artifacts"
echo "  Data:    $DATA_DIR"
echo "  Results: $RESULTS_DIR"
echo "  Figures: $FIGURES_DIR"
echo "  Tables:  $TABLES_DIR"
