"""Compute bbox set-matching metrics from custom PT prediction CSV output."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_THRESHOLDS = (0.1, 0.3, 0.5, 0.75)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _parse_json(raw: str) -> Any:
    return json.loads(str(raw).strip())


def _coerce_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        coords = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(coord) for coord in coords):
        return None

    x1, y1, x2, y2 = coords
    x1 = max(0.0, min(1000.0, x1))
    y1 = max(0.0, min(1000.0, y1))
    x2 = max(0.0, min(1000.0, x2))
    y2 = max(0.0, min(1000.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _extract_boxes(obj: Any) -> tuple[list[tuple[float, float, float, float]], int]:
    raw_boxes = obj.get("bbox") if isinstance(obj, dict) else obj
    if raw_boxes is None:
        return [], 0

    if isinstance(raw_boxes, list) and len(raw_boxes) == 4 and not any(isinstance(item, list) for item in raw_boxes):
        raw_boxes = [raw_boxes]
    if not isinstance(raw_boxes, list):
        return [], 1

    boxes: list[tuple[float, float, float, float]] = []
    invalid = 0
    for raw_box in raw_boxes:
        box = _coerce_bbox(raw_box)
        if box is None:
            invalid += 1
            continue
        boxes.append(box)
    return boxes, invalid


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area <= 0:
        return 0.0
    union_area = _area(box_a) + _area(box_b) - inter_area
    return _safe_div(inter_area, union_area)


def _match_boxes(
    gt_boxes: list[tuple[float, float, float, float]],
    pred_boxes: list[tuple[float, float, float, float]],
    threshold: float,
) -> tuple[int, int, int, list[float]]:
    pairs: list[tuple[float, int, int]] = []
    for gt_idx, gt_box in enumerate(gt_boxes):
        for pred_idx, pred_box in enumerate(pred_boxes):
            pairs.append((_iou(gt_box, pred_box), gt_idx, pred_idx))
    pairs.sort(reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matched_ious: list[float] = []
    for pair_iou, gt_idx, pred_idx in pairs:
        if pair_iou < threshold:
            break
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matched_ious.append(pair_iou)

    tp = len(matched_ious)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn, matched_ious


def _best_ious(
    source_boxes: list[tuple[float, float, float, float]],
    target_boxes: list[tuple[float, float, float, float]],
) -> list[float]:
    if not source_boxes:
        return []
    if not target_boxes:
        return [0.0] * len(source_boxes)
    return [max(_iou(source_box, target_box) for target_box in target_boxes) for source_box in source_boxes]


def _metrics_from_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _compute_metrics_from_rows(
    rows: list[dict[str, str]],
    thresholds: tuple[float, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    threshold_counts = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in thresholds}
    threshold_sample_counts = {
        threshold: {"any_match": 0, "all_gt_matched": 0, "perfect_set": 0}
        for threshold in thresholds
    }
    per_sample_rows: list[dict[str, Any]] = []
    all_best_gt_ious: list[float] = []
    all_best_pred_ious: list[float] = []
    all_gt_box_counts: list[int] = []
    all_pred_box_counts: list[int] = []
    valid_pred_json = 0
    invalid_pred_json = 0
    invalid_true_json = 0
    invalid_gt_boxes = 0
    invalid_pred_boxes = 0

    for row in rows:
        try:
            true_obj = _parse_json(row.get("true_bbox_json", ""))
        except json.JSONDecodeError:
            true_obj = {}
            invalid_true_json += 1
        try:
            pred_obj = _parse_json(row.get("pred_bbox_json", ""))
            valid_pred_json += 1
        except json.JSONDecodeError:
            pred_obj = {}
            invalid_pred_json += 1

        gt_boxes, gt_invalid = _extract_boxes(true_obj)
        pred_boxes, pred_invalid = _extract_boxes(pred_obj)
        invalid_gt_boxes += gt_invalid
        invalid_pred_boxes += pred_invalid
        all_gt_box_counts.append(len(gt_boxes))
        all_pred_box_counts.append(len(pred_boxes))

        best_gt_ious = _best_ious(gt_boxes, pred_boxes)
        best_pred_ious = _best_ious(pred_boxes, gt_boxes)
        all_best_gt_ious.extend(best_gt_ious)
        all_best_pred_ious.extend(best_pred_ious)

        sample_record: dict[str, Any] = {
            "sample_key": row.get("sample_key", ""),
            "feature_idx": row.get("feature_idx", ""),
            "row_idx": row.get("row_idx", ""),
            "row_id": row.get("row_id", ""),
            "span_idx": row.get("span_idx", ""),
            "source": row.get("source", ""),
            "source_split": row.get("source_split", ""),
            "gt_box_count": len(gt_boxes),
            "pred_box_count": len(pred_boxes),
            "mean_best_gt_iou": mean(best_gt_ious) if best_gt_ious else 0.0,
            "max_best_gt_iou": max(best_gt_ious) if best_gt_ious else 0.0,
        }

        for threshold in thresholds:
            tp, fp, fn, _ = _match_boxes(gt_boxes, pred_boxes, threshold)
            threshold_counts[threshold]["tp"] += tp
            threshold_counts[threshold]["fp"] += fp
            threshold_counts[threshold]["fn"] += fn
            if tp > 0:
                threshold_sample_counts[threshold]["any_match"] += 1
            if len(gt_boxes) > 0 and fn == 0:
                threshold_sample_counts[threshold]["all_gt_matched"] += 1
            if len(gt_boxes) > 0 and fn == 0 and fp == 0:
                threshold_sample_counts[threshold]["perfect_set"] += 1
            sample_record[f"tp@{threshold:g}"] = tp
            sample_record[f"fp@{threshold:g}"] = fp
            sample_record[f"fn@{threshold:g}"] = fn
        per_sample_rows.append(sample_record)

    num_samples = len(rows)
    threshold_metrics: dict[str, Any] = {}
    for threshold in thresholds:
        counts = threshold_counts[threshold]
        metrics = _metrics_from_counts(counts["tp"], counts["fp"], counts["fn"])
        sample_counts = threshold_sample_counts[threshold]
        metrics.update(
            {
                "sample_any_match_rate": _safe_div(sample_counts["any_match"], num_samples),
                "sample_all_gt_matched_rate": _safe_div(sample_counts["all_gt_matched"], num_samples),
                "sample_perfect_set_rate": _safe_div(sample_counts["perfect_set"], num_samples),
            }
        )
        threshold_metrics[f"{threshold:g}"] = metrics

    summary = {
        "num_samples": num_samples,
        "num_gt_boxes": sum(all_gt_box_counts),
        "num_pred_boxes": sum(all_pred_box_counts),
        "avg_gt_boxes_per_sample": mean(all_gt_box_counts) if all_gt_box_counts else 0.0,
        "avg_pred_boxes_per_sample": mean(all_pred_box_counts) if all_pred_box_counts else 0.0,
        "box_count_accuracy": _safe_div(
            sum(1 for gt_count, pred_count in zip(all_gt_box_counts, all_pred_box_counts, strict=True) if gt_count == pred_count),
            num_samples,
        ),
        "parse": {
            "valid_pred_json": valid_pred_json,
            "invalid_pred_json": invalid_pred_json,
            "invalid_true_json": invalid_true_json,
            "invalid_gt_boxes": invalid_gt_boxes,
            "invalid_pred_boxes": invalid_pred_boxes,
            "empty_pred_box_samples": sum(1 for count in all_pred_box_counts if count == 0),
        },
        "iou": {
            "mean_best_gt_iou": mean(all_best_gt_ious) if all_best_gt_ious else 0.0,
            "median_best_gt_iou": median(all_best_gt_ious) if all_best_gt_ious else 0.0,
            "mean_best_pred_iou": mean(all_best_pred_ious) if all_best_pred_ious else 0.0,
            "median_best_pred_iou": median(all_best_pred_ious) if all_best_pred_ious else 0.0,
        },
        "threshold_metrics": threshold_metrics,
    }
    return summary, per_sample_rows


def _group_rows_by_field(
    rows: list[dict[str, str]],
    field_name: str,
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = str(row.get(field_name, "")).strip()
        if not key:
            continue
        groups.setdefault(key, []).append(row)
    return groups


def compute_metrics(
    input_csv: Path,
    thresholds: tuple[float, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    overall_summary, per_sample_rows = _compute_metrics_from_rows(rows, thresholds)
    grouped_by_source = {
        group_name: _compute_metrics_from_rows(group_rows, thresholds)[0]
        for group_name, group_rows in sorted(_group_rows_by_field(rows, "source").items())
    }
    grouped_by_source_split = {
        group_name: _compute_metrics_from_rows(group_rows, thresholds)[0]
        for group_name, group_rows in sorted(_group_rows_by_field(rows, "source_split").items())
    }

    summary = {
        "input_csv": str(input_csv),
        **overall_summary,
        "grouped_by_source": grouped_by_source,
        "grouped_by_source_split": grouped_by_source_split,
    }
    return summary, per_sample_rows


def _write_per_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute bbox metrics from custom PT prediction CSV output.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--per-sample-csv", default="")
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    if not input_csv.exists():
        parser.error(f"--input-csv does not exist: {input_csv}")
    thresholds = tuple(sorted(set(float(threshold) for threshold in args.thresholds)))
    if any(threshold < 0 or threshold > 1 for threshold in thresholds):
        parser.error("--thresholds must be in [0, 1]")

    output_json = Path(args.output_json).resolve() if args.output_json else input_csv.with_suffix(".bbox_metrics.json")
    per_sample_csv = (
        Path(args.per_sample_csv).resolve()
        if args.per_sample_csv
        else input_csv.with_suffix(".bbox_per_sample.csv")
    )

    summary, per_sample_rows = compute_metrics(input_csv, thresholds)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_per_sample_csv(per_sample_csv, per_sample_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote metrics to {output_json}")
    print(f"Wrote per-sample metrics to {per_sample_csv}")


if __name__ == "__main__":
    main()
