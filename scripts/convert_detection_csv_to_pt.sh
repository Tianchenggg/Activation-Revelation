#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

: "${INPUT_CSV:?Set INPUT_CSV to the detection split CSV.}"
: "${OUTPUT_PT:?Set OUTPUT_PT to the output .pt path.}"

ACTIVATION_MODEL="${ACTIVATION_MODEL:-${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}}"
DATASET_ROOT="${DATASET_ROOT:-/data}"
LAYER_PATH="${LAYER_PATH:-model.language_model.layers.16}"
LAYER_ID="${LAYER_ID:-1}"
RESPONSE_MODE="${RESPONSE_MODE:-no_thinking}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-3023}"
DTYPE="${DTYPE:-bfloat16}"
ACTIVATION_STORAGE_DTYPE="${ACTIVATION_STORAGE_DTYPE:-bfloat16}"
SINGLE_FILE_OUTPUT="${SINGLE_FILE_OUTPUT:-1}"

extra_args=()
if [[ -n "${MAX_ROWS:-}" ]]; then extra_args+=(--max_rows "$MAX_ROWS"); fi
if [[ -n "${MAX_ENTRIES:-}" ]]; then extra_args+=(--max_entries "$MAX_ENTRIES"); fi
if [[ -n "${MAX_LENGTH:-}" ]]; then extra_args+=(--max_length "$MAX_LENGTH"); fi
if [[ "$SINGLE_FILE_OUTPUT" == "1" ]]; then extra_args+=(--single-file-output); fi

"$PYTHON" "$ROOT_DIR/detection/pt_converters/csv_to_pt_part_span.py" \
  "$INPUT_CSV" \
  "$OUTPUT_PT" \
  --dataset-root "$DATASET_ROOT" \
  --model_path "$ACTIVATION_MODEL" \
  --layer "$LAYER_PATH" \
  --layer_id "$LAYER_ID" \
  --response-modes "$RESPONSE_MODE" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --dtype "$DTYPE" \
  --activation_storage_dtype "$ACTIVATION_STORAGE_DTYPE" \
  "${extra_args[@]}"

