#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"

: "${MODEL:?Set MODEL to the trained explanation model directory or HF model name.}"
: "${TEST_PT:?Set TEST_PT to the explanation test .pt file.}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/explanation_eval}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
HOOK_LAYER="${HOOK_LAYER:-1}"
STEERING_MODE="${STEERING_MODE:-add}"
STEERING_COEFFICIENT="${STEERING_COEFFICIENT:-1.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DTYPE="${DTYPE:-bfloat16}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-100352}"
THRESHOLDS="${THRESHOLDS:-0.1 0.3 0.5 0.75}"
SEED="${SEED:-3023}"

mkdir -p "$OUTPUT_DIR"
launcher=("$PYTHON")
if (( NPROC_PER_NODE > 1 )); then
  launcher=("$TORCHRUN" --standalone --nproc_per_node "$NPROC_PER_NODE")
fi

args=(
  "$ROOT_DIR/explanation/experiments/custom_pt_eval_to_csv.py"
  --model "$MODEL"
  --pt "$TEST_PT"
  --hook-layer "$HOOK_LAYER"
  --output-csv "$OUTPUT_DIR/predictions.csv"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --steering-coefficient "$STEERING_COEFFICIENT"
  --steering-mode "$STEERING_MODE"
  --dtype "$DTYPE"
  --image-max-pixels "$IMAGE_MAX_PIXELS"
  --seed "$SEED"
)

if [[ -n "${LORA_PATH:-}" ]]; then args+=(--lora-path "$LORA_PATH"); fi
if [[ -n "${MAX_EXAMPLES:-}" ]]; then args+=(--max-examples "$MAX_EXAMPLES"); fi

"${launcher[@]}" "${args[@]}"

"$PYTHON" "$ROOT_DIR/explanation/experiments/bbox_metrics_from_csv.py" \
  --input-csv "$OUTPUT_DIR/predictions.csv" \
  --output-json "$OUTPUT_DIR/bbox_metrics.json" \
  --per-sample-csv "$OUTPUT_DIR/bbox_per_sample.csv" \
  --thresholds $THRESHOLDS

