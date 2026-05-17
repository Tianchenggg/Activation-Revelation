#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

: "${INPUT_CSV:?Set INPUT_CSV to the explanation split CSV.}"
: "${OUTPUT_PT:?Set OUTPUT_PT to the output .pt path.}"

ACTIVATION_MODEL="${ACTIVATION_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
LAYER_PATH="${LAYER_PATH:-model.language_model.layers.16}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-3023}"
DTYPE="${DTYPE:-bfloat16}"
ACTIVATION_STORAGE_DTYPE="${ACTIVATION_STORAGE_DTYPE:-bfloat16}"
ACTIVATION_SOURCE="${ACTIVATION_SOURCE:-span_tokens}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-200704}"
SINGLE_FILE_OUTPUT="${SINGLE_FILE_OUTPUT:-1}"

extra_args=()
if [[ -n "${DATASET_ROOT:-}" ]]; then extra_args+=(--dataset-root "$DATASET_ROOT"); fi
if [[ -n "${LAYER_ID:-}" ]]; then extra_args+=(--layer_id "$LAYER_ID"); fi
if [[ -n "${MAX_ROWS:-}" ]]; then extra_args+=(--max_rows "$MAX_ROWS"); fi
if [[ -n "${MAX_ENTRIES:-}" ]]; then extra_args+=(--max_entries "$MAX_ENTRIES"); fi
if [[ -n "${MAX_LENGTH:-}" ]]; then extra_args+=(--max_length "$MAX_LENGTH"); fi
if [[ "$SINGLE_FILE_OUTPUT" == "1" ]]; then extra_args+=(--single-file-output); fi

"$PYTHON" "$ROOT_DIR/explanation/pt_converters/csv_to_pt_part_span.py" \
  "$INPUT_CSV" \
  "$OUTPUT_PT" \
  --model_path "$ACTIVATION_MODEL" \
  --layer "$LAYER_PATH" \
  --device "$DEVICE" \
  --activation-source "$ACTIVATION_SOURCE" \
  --image-max-pixels "$IMAGE_MAX_PIXELS" \
  --seed "$SEED" \
  --dtype "$DTYPE" \
  --activation_storage_dtype "$ACTIVATION_STORAGE_DTYPE" \
  "${extra_args[@]}"

