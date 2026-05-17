#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"

: "${MODEL:?Set MODEL to the trained detection model directory or HF model name.}"
: "${TEST_PT:?Set TEST_PT to the detection test .pt file.}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/detection_eval}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
HOOK_LAYER="${HOOK_LAYER:-1}"
STEERING_MODE="${STEERING_MODE:-add}"
STEERING_COEFFICIENT="${STEERING_COEFFICIENT:-1.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
DTYPE="${DTYPE:-bfloat16}"
OUTPUT_MODE="${OUTPUT_MODE:-metrics}"
SEED="${SEED:-3023}"

mkdir -p "$OUTPUT_DIR"
launcher=("$PYTHON")
if (( NPROC_PER_NODE > 1 )); then
  launcher=("$TORCHRUN" --standalone --nproc_per_node "$NPROC_PER_NODE")
fi

args=(
  "$ROOT_DIR/detection/experiments/custom_pt_eval_to_csv.py"
  --model "$MODEL"
  --pt "$TEST_PT"
  --hook-layer "$HOOK_LAYER"
  --output-csv "$OUTPUT_DIR/predictions.csv"
  --output-json "$OUTPUT_DIR/metrics.json"
  --output-mode "$OUTPUT_MODE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --steering-coefficient "$STEERING_COEFFICIENT"
  --steering-mode "$STEERING_MODE"
  --dtype "$DTYPE"
  --seed "$SEED"
)

if [[ -n "${LORA_PATH:-}" ]]; then args+=(--lora-path "$LORA_PATH"); fi
if [[ -n "${SOURCE_CSV:-}" ]]; then args+=(--source-csv "$SOURCE_CSV"); fi
if [[ -n "${MAX_EXAMPLES:-}" ]]; then args+=(--max-examples "$MAX_EXAMPLES"); fi

"${launcher[@]}" "${args[@]}"

