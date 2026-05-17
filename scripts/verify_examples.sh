#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/example_verification}"

mkdir -p "$OUTPUT_DIR"

"$PYTHON" "$ROOT_DIR/detection/experiments/detection_metrics_from_csv.py" \
  --predictions "$ROOT_DIR/examples/spavl_detection_predictions.csv" \
  --source-csv "$ROOT_DIR/examples/spavl_detection.csv" \
  --output-json "$OUTPUT_DIR/detection_metrics.json"

"$PYTHON" "$ROOT_DIR/explanation/experiments/bbox_metrics_from_csv.py" \
  --input-csv "$ROOT_DIR/examples/spavl_explanation_bbox_predictions.csv" \
  --output-json "$OUTPUT_DIR/bbox_metrics.json" \
  --per-sample-csv "$OUTPUT_DIR/bbox_per_sample.csv" \
  --thresholds 0.1 0.3 0.5 0.75

echo "Example verification complete: $OUTPUT_DIR"

