"""Evaluate custom PT activation-oracle data and stream predictions to CSV."""

import argparse
import csv
import fcntl
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TPAMI_ROOT = PROJECT_ROOT.parents[3]

from nl_probes.dataset_classes.custom_pt_dataset import load_training_datapoints_from_pt
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import (
    TrainingDataPoint,
    get_disable_thinking_prefix_ids,
    is_qwen3_reasoning_model,
    maybe_add_disable_thinking_prefix,
)
from nl_probes.utils.eval import run_evaluation

csv.field_size_limit(sys.maxsize)

CSV_FIELDNAMES = [
    "sample_key",
    "feature_idx",
    "row_idx",
    "span_idx",
    "layer_path",
    "true_answer",
    "pred_answer",
    "status",
]
_ANSWER_END_RE = re.compile(r"</answer\s*>", re.IGNORECASE)
_THINKING_END_RE = re.compile(r"</thinking\s*>", re.IGNORECASE)
_ANSWER_BLOCK_RE = re.compile(r"<answer\s*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL)
TPAMI_ROOT = PROJECT_ROOT.parents[3]
_SAFE_LABELS = {"safe", "yes", "benign", "harmless"}
_UNSAFE_LABELS = {"unsafe", "harmful", "no", "dangerous", "toxic", "not safe"}


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_rank_zero() -> bool:
    return _local_rank() == 0


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _maybe_init_distributed() -> bool:
    world_size = _world_size()
    if world_size <= 1:
        return False
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed evaluation requires CUDA devices.")
    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    if not _dist_is_initialized():
        dist.init_process_group(backend="nccl", device_id=local_rank)
    return True


def _maybe_destroy_distributed() -> None:
    if _dist_is_initialized():
        dist.barrier(device_ids=[_local_rank()])
        dist.destroy_process_group()


def _parse_layer_selector(raw_value: str) -> int | str:
    value = raw_value.strip()
    if not value:
        raise argparse.ArgumentTypeError("Layer selector must be non-empty.")
    if value.lstrip("-").isdigit():
        return int(value)
    return value


def _dtype_from_name(name: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map[name]


def _load_eval_data(
    *,
    pt_path: str,
    model_name: str,
    max_examples: int,
) -> list[TrainingDataPoint]:
    return load_training_datapoints_from_pt(
        pt_path=pt_path,
        model_name=model_name,
        datapoint_type="custom_pt_simple_eval",
        max_records=max_examples if max_examples > 0 else 0,
    )


def _resolve_model_and_lora_paths(model_arg: str, lora_arg: str) -> tuple[str, str]:
    if lora_arg:
        return model_arg, lora_arg

    model_path = Path(model_arg).expanduser()
    adapter_config_path = model_path / "adapter_config.json"
    if not adapter_config_path.is_file():
        return model_arg, lora_arg

    with adapter_config_path.open("r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    base_model_name = str(adapter_config.get("base_model_name_or_path") or "").strip()
    if not base_model_name:
        raise RuntimeError(
            f"Adapter config exists but has no base_model_name_or_path: {adapter_config_path}"
        )

    print(
        "Detected adapter-only --model path. "
        f"Using base model '{base_model_name}' with LoRA adapter '{model_path}'."
    )
    return base_model_name, str(model_path)


def _maybe_align_qwen3_eval_inputs(
    eval_data: list[TrainingDataPoint],
    tokenizer,
    model_name: str,
) -> list[TrainingDataPoint]:
    if not is_qwen3_reasoning_model(model_name, tokenizer=tokenizer):
        return eval_data

    prefix_ids = get_disable_thinking_prefix_ids(tokenizer)
    if not prefix_ids:
        return eval_data

    updated_eval_data = [
        maybe_add_disable_thinking_prefix(data_point, tokenizer, model_name, prefix_ids)
        for data_point in eval_data
    ]
    num_modified = sum(
        1 for original, updated in zip(eval_data, updated_eval_data, strict=True) if original != updated
    )
    print(f"Applied Qwen3 disable-thinking prefix to {num_modified}/{len(eval_data)} eval examples.")
    return updated_eval_data


def _match_end_tag(
    text: str,
    *,
    output_mode: str,
    eval_strategy: str,
) -> re.Match[str] | None:
    if output_mode == "metrics":
        return _ANSWER_END_RE.search(text)

    if output_mode == "full":
        if eval_strategy == "thinking_answer_first_label_only":
            return _THINKING_END_RE.search(text)
        if eval_strategy == "thinking_answer_last_until_answer_end":
            return _ANSWER_END_RE.search(text)

    return None


def _trim_generated_answer(
    pred_answer: str,
    *,
    output_mode: str,
    eval_strategy: str,
) -> str:
    text = str(pred_answer)
    match = _match_end_tag(
        text,
        output_mode=output_mode,
        eval_strategy=eval_strategy,
    )
    if match is not None:
        return text[: match.end()].strip()
    return text.strip()


def _validate_pt_path(parser: argparse.ArgumentParser, path: str) -> None:
    if not os.path.exists(path):
        parser.error(f"--pt does not exist: {path}")
    if Path(path).suffix.lower() != ".pt":
        parser.error(f"--pt must point to a .pt file: {path}")


def _meta_str(meta_info: dict, key: str) -> str:
    value = meta_info.get(key)
    return "" if value is None else str(value)


def _sample_key(dp: TrainingDataPoint) -> str:
    meta_info = dict(dp.meta_info or {})
    row_id = meta_info.get("row_id")
    segment_idx = meta_info.get("segment_idx")
    layer_path = meta_info.get("layer_path")
    if row_id in (None, "") or segment_idx in (None, "") or layer_path in (None, ""):
        raise RuntimeError(
            "Current PT eval expects meta_info to contain non-empty "
            "'row_id', 'segment_idx', and 'layer_path'. "
            f"Got keys: {sorted(meta_info.keys())}"
        )
    return f"row_id={row_id}|span_idx={segment_idx}|layer_path={layer_path}"


def _csv_row(
    dp: TrainingDataPoint,
    pred_answer: str,
    *,
    output_mode: str,
    eval_strategy: str,
) -> dict[str, str]:
    meta_info = dict(dp.meta_info or {})
    return {
        "sample_key": _sample_key(dp),
        "feature_idx": str(int(dp.feature_idx)),
        "row_idx": _meta_str(meta_info, "row_idx"),
        "span_idx": _meta_str(meta_info, "segment_idx"),
        "layer_path": _meta_str(meta_info, "layer_path"),
        "true_answer": str(dp.target_output),
        "pred_answer": _trim_generated_answer(
            pred_answer,
            output_mode=output_mode,
            eval_strategy=eval_strategy,
        ),
        "status": "done",
    }


def _load_processed_keys(output_csv: str) -> set[str]:
    output_path = Path(output_csv)
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()

    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "sample_key" not in fieldnames:
            raise RuntimeError(
                f"Existing CSV {output_csv} is missing 'sample_key'. "
                "Delete it or use a CSV produced by the resumable writer."
            )
        return {
            str(row["sample_key"]).strip()
            for row in reader
            if row.get("sample_key") not in (None, "")
            and (
                "status" not in fieldnames
                or str(row.get("status", "")).strip() == "done"
            )
        }


def _prepare_output_csv(output_csv: str) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            first_line = f.readline()
            if not first_line:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()
                f.flush()
                os.fsync(f.fileno())
                return

            header = next(csv.reader([first_line.rstrip("\n")]))
            if header != CSV_FIELDNAMES:
                raise RuntimeError(
                    f"Existing CSV {output_csv} uses unexpected header {header}. "
                    "Delete it or write to a new output path for resumable mode."
                )
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _append_csv_rows(output_csv: str, rows: list[dict[str, str]]) -> None:
    if not rows:
        return

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _filter_unfinished(eval_data: list[TrainingDataPoint], processed_keys: set[str]) -> list[TrainingDataPoint]:
    if not processed_keys:
        return eval_data
    return [dp for dp in eval_data if _sample_key(dp) not in processed_keys]


def _validate_unique_keys(eval_data: list[TrainingDataPoint]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for dp in eval_data:
        sample_key = _sample_key(dp)
        if sample_key in seen:
            duplicates.append(sample_key)
            if len(duplicates) >= 5:
                break
        seen.add(sample_key)
    if duplicates:
        raise RuntimeError(
            "Detected duplicate sample keys in eval data, cannot resume safely. "
            f"Examples: {duplicates}"
        )




def _infer_source_csv_from_pt(pt_path: str, source_csv_arg: str = "") -> Path:
    if source_csv_arg:
        source_csv = Path(source_csv_arg).expanduser().resolve()
        if not source_csv.is_file():
            raise FileNotFoundError(f"--source-csv does not exist: {source_csv}")
        return source_csv

    split_name = Path(pt_path).stem
    candidates: list[Path] = [Path(pt_path).with_suffix(".csv")]
    for root in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
        candidates.append(root / "data" / "combine" / f"{split_name}.csv")

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    searched = "\n".join(f"  - {candidate}" for candidate in seen)
    raise FileNotFoundError(
        "Could not infer the source split CSV needed for dataset-level metrics. "
        "Pass --source-csv explicitly. Searched:\n"
        f"{searched}"
    )


def _load_source_by_row_idx(source_csv: Path) -> dict[int, str]:
    source_by_row_idx: dict[int, str] = {}
    with source_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "source" not in (reader.fieldnames or []):
            raise RuntimeError(f"Source CSV {source_csv} is missing required 'source' column.")
        for row_idx, row in enumerate(reader):
            source = str(row.get("source", "")).strip()
            if source not in {"dataset_A", "dataset_B", "dataset_C"}:
                raise RuntimeError(
                    f"Unexpected source value at row_idx={row_idx} in {source_csv}: {source!r}"
                )
            source_by_row_idx[row_idx] = source
    return source_by_row_idx


def _build_sample_metadata(
    eval_data: list[TrainingDataPoint],
    *,
    source_csv: Path,
) -> dict[str, dict[str, str | int]]:
    source_by_row_idx = _load_source_by_row_idx(source_csv)
    sample_metadata: dict[str, dict[str, str | int]] = {}
    for dp in eval_data:
        sample_key = _sample_key(dp)
        meta_info = dict(dp.meta_info or {})
        row_idx_raw = meta_info.get("row_idx")
        if row_idx_raw in (None, ""):
            raise RuntimeError(f"Missing row_idx in PT meta_info for sample_key={sample_key}")
        row_idx = int(row_idx_raw)
        if row_idx not in source_by_row_idx:
            raise RuntimeError(
                f"row_idx={row_idx} from PT was not found in inferred source CSV {source_csv}"
            )
        sample_metadata[sample_key] = {
            "row_idx": row_idx,
            "source": source_by_row_idx[row_idx],
        }
    return sample_metadata


def _load_completed_rows(output_csv: str) -> dict[str, dict[str, str]]:
    output_path = Path(output_csv)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Output CSV does not exist or is empty: {output_csv}")

    completed_rows: dict[str, dict[str, str]] = {}
    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "sample_key" not in fieldnames:
            raise RuntimeError(f"Output CSV {output_csv} is missing 'sample_key'.")
        for row in reader:
            if "status" in fieldnames and str(row.get("status", "")).strip() != "done":
                continue
            sample_key = str(row.get("sample_key", "")).strip()
            if not sample_key:
                continue
            if sample_key in completed_rows:
                raise RuntimeError(f"Duplicate sample_key found in output CSV: {sample_key}")
            completed_rows[sample_key] = row
    return completed_rows


def _normalize_label_text(text: str) -> str:
    cleaned = str(text).strip().lower()
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = cleaned.rstrip(".!?,;:")  # keep normalization minimal for current outputs
    return " ".join(cleaned.split())


def _extract_binary_label(text: str) -> str | None:
    text_str = str(text)
    match = _ANSWER_BLOCK_RE.search(text_str)
    candidate = match.group(1) if match is not None else text_str
    normalized = _normalize_label_text(candidate)
    first_label_match = re.search(r"unsafe|safe", normalized)
    if first_label_match is not None:
        return first_label_match.group(0)
    if normalized in _SAFE_LABELS:
        return "safe"
    if normalized in _UNSAFE_LABELS:
        return "unsafe"
    if normalized.startswith("safe "):
        return "safe"
    if normalized.startswith("unsafe ") or normalized.startswith("harmful "):
        return "unsafe"
    return None


def _should_use_constrained_binary_decode(eval_data: list[TrainingDataPoint]) -> bool:
    if not eval_data:
        return False
    normalized_targets = {
        _normalize_label_text(str(dp.target_output))
        for dp in eval_data
    }
    return normalized_targets.issubset({"safe", "unsafe"})


def _infer_eval_strategy(eval_data: list[TrainingDataPoint]) -> str:
    if not eval_data:
        return "empty"
    if _should_use_constrained_binary_decode(eval_data):
        return "binary_label_autoregressive"

    sample_targets = [str(dp.target_output).lower() for dp in eval_data[: min(len(eval_data), 64)]]
    answer_first = 0
    answer_last = 0
    for target in sample_targets:
        answer_idx = target.find("<answer>")
        thinking_idx = target.find("<thinking>")
        if answer_idx == -1:
            continue
        if thinking_idx == -1 or answer_idx < thinking_idx:
            answer_first += 1
        elif thinking_idx < answer_idx:
            answer_last += 1

    if answer_first and not answer_last:
        return "thinking_answer_first_label_only"
    if answer_last and not answer_first:
        return "thinking_answer_last_until_answer_end"
    if answer_first or answer_last:
        return "mixed_answer_template"
    return "free_form_generation"


def _generation_stop_strings(
    *,
    output_mode: str,
    eval_strategy: str,
) -> list[str] | None:
    if output_mode == "metrics":
        if eval_strategy in {
            "thinking_answer_first_label_only",
            "thinking_answer_last_until_answer_end",
            "mixed_answer_template",
        }:
            return ["</answer>"]
        return None

    if output_mode == "full":
        if eval_strategy == "thinking_answer_first_label_only":
            return ["</thinking>"]
        if eval_strategy == "thinking_answer_last_until_answer_end":
            return ["</answer>"]
        return None

    raise ValueError(f"Unexpected output_mode: {output_mode!r}")


def _safe_div(num: int | float, denom: int | float) -> float:
    return float(num) / float(denom) if denom else 0.0


def _round_metric(value: float) -> float:
    return round(float(value), 6)


def _compute_binary_metrics(records: list[dict[str, str]]) -> dict[str, object]:
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "unparsed_predictions": 0,
            "Acc": 0.0,
            "Mac-P": 0.0,
            "Mac-R": 0.0,
            "Mac-F1": 0.0,
            "Mic-P": 0.0,
            "Mic-R": 0.0,
            "Mic-F1": 0.0,
            "safe-P": 0.0,
            "safe-R": 0.0,
            "safe-F1": 0.0,
            "unsafe-P": 0.0,
            "unsafe-R": 0.0,
            "unsafe-F1": 0.0,
            "confusion_matrix": {
                "labels": ["safe", "unsafe"],
                "rows": "true",
                "cols": "pred",
                "matrix": [[0, 0], [0, 0]],
            },
        }

    true_safe_pred_safe = 0
    true_safe_pred_unsafe = 0
    true_unsafe_pred_safe = 0
    true_unsafe_pred_unsafe = 0
    unparsed_predictions = 0

    for record in records:
        true_label = record["true_label"]
        pred_label = record["pred_label"]
        if true_label == "safe":
            if pred_label == "safe":
                true_safe_pred_safe += 1
            elif pred_label == "unsafe":
                true_safe_pred_unsafe += 1
            else:
                unparsed_predictions += 1
        elif true_label == "unsafe":
            if pred_label == "safe":
                true_unsafe_pred_safe += 1
            elif pred_label == "unsafe":
                true_unsafe_pred_unsafe += 1
            else:
                unparsed_predictions += 1
        else:
            raise RuntimeError(f"Unexpected true label: {true_label!r}")

    safe_tp = true_safe_pred_safe
    safe_fp = true_unsafe_pred_safe
    safe_fn = true_safe_pred_unsafe + sum(
        1 for record in records if record["true_label"] == "safe" and record["pred_label"] is None
    )

    unsafe_tp = true_unsafe_pred_unsafe
    unsafe_fp = true_safe_pred_unsafe
    unsafe_fn = true_unsafe_pred_safe + sum(
        1 for record in records if record["true_label"] == "unsafe" and record["pred_label"] is None
    )

    safe_p = _safe_div(safe_tp, safe_tp + safe_fp)
    safe_r = _safe_div(safe_tp, safe_tp + safe_fn)
    safe_f1 = _safe_div(2 * safe_p * safe_r, safe_p + safe_r)

    unsafe_p = _safe_div(unsafe_tp, unsafe_tp + unsafe_fp)
    unsafe_r = _safe_div(unsafe_tp, unsafe_tp + unsafe_fn)
    unsafe_f1 = _safe_div(2 * unsafe_p * unsafe_r, unsafe_p + unsafe_r)

    correct = true_safe_pred_safe + true_unsafe_pred_unsafe
    total_tp = safe_tp + unsafe_tp
    total_fp = safe_fp + unsafe_fp
    total_fn = safe_fn + unsafe_fn

    return {
        "total": total,
        "unparsed_predictions": unparsed_predictions,
        "Acc": _round_metric(_safe_div(correct, total)),
        "Mac-P": _round_metric((safe_p + unsafe_p) / 2.0),
        "Mac-R": _round_metric((safe_r + unsafe_r) / 2.0),
        "Mac-F1": _round_metric((safe_f1 + unsafe_f1) / 2.0),
        "Mic-P": _round_metric(_safe_div(total_tp, total_tp + total_fp)),
        "Mic-R": _round_metric(_safe_div(total_tp, total_tp + total_fn)),
        "Mic-F1": _round_metric(
            _safe_div(
                2 * _safe_div(total_tp, total_tp + total_fp) * _safe_div(total_tp, total_tp + total_fn),
                _safe_div(total_tp, total_tp + total_fp) + _safe_div(total_tp, total_tp + total_fn),
            )
        ),
        "safe-P": _round_metric(safe_p),
        "safe-R": _round_metric(safe_r),
        "safe-F1": _round_metric(safe_f1),
        "unsafe-P": _round_metric(unsafe_p),
        "unsafe-R": _round_metric(unsafe_r),
        "unsafe-F1": _round_metric(unsafe_f1),
        "confusion_matrix": {
            "labels": ["safe", "unsafe"],
            "rows": "true",
            "cols": "pred",
            "matrix": [
                [true_safe_pred_safe, true_safe_pred_unsafe],
                [true_unsafe_pred_safe, true_unsafe_pred_unsafe],
            ],
        },
    }


def _write_metrics_json(
    *,
    output_json: str,
    output_csv: str,
    pt_path: str,
    eval_data: list[TrainingDataPoint],
    source_csv_arg: str = "",
) -> None:
    source_csv = _infer_source_csv_from_pt(pt_path, source_csv_arg=source_csv_arg)
    sample_metadata = _build_sample_metadata(eval_data, source_csv=source_csv)
    completed_rows = _load_completed_rows(output_csv)

    expected_keys = set(sample_metadata.keys())
    actual_keys = set(completed_rows.keys())
    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    if missing_keys or extra_keys:
        raise RuntimeError(
            "Output CSV rows do not match eval PT entries. "
            f"missing={missing_keys[:5]}, extra={extra_keys[:5]}"
        )

    grouped_records: dict[str, list[dict[str, str]]] = {
        "overall": [],
        "dataset_A": [],
        "dataset_B": [],
        "dataset_C": [],
    }
    for sample_key, meta in sample_metadata.items():
        row = completed_rows[sample_key]
        true_label = _extract_binary_label(str(row.get("true_answer", "")))
        pred_label = _extract_binary_label(str(row.get("pred_answer", "")))
        if true_label not in {"safe", "unsafe"}:
            raise RuntimeError(
                f"Could not parse true label for sample_key={sample_key}: {row.get('true_answer', '')!r}"
            )
        record = {
            "sample_key": sample_key,
            "true_label": true_label,
            "pred_label": pred_label,
            "source": str(meta["source"]),
        }
        grouped_records["overall"].append(record)
        grouped_records[str(meta["source"])].append(record)

    payload = {
        "pt_path": str(Path(pt_path).resolve()),
        "output_csv": str(Path(output_csv).resolve()),
        "source_csv": str(source_csv.resolve()),
        "overall": _compute_binary_metrics(grouped_records["overall"]),
        "dataset_A": _compute_binary_metrics(grouped_records["dataset_A"]),
        "dataset_B": _compute_binary_metrics(grouped_records["dataset_B"]),
        "dataset_C": _compute_binary_metrics(grouped_records["dataset_C"]),
    }

    output_path = Path(output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple custom-PT evaluation that writes predictions to CSV.",
    )
    parser.add_argument("--model", required=True, help="Base model name/path used for evaluation.")
    parser.add_argument(
        "--lora-path",
        default="",
        help="Optional LoRA adapter path (e.g. your training output/final).",
    )
    parser.add_argument(
        "--pt",
        dest="pt_path",
        required=True,
        help="Path to evaluation PT dataset.",
    )
    parser.add_argument(
        "--hook-layer",
        type=_parse_layer_selector,
        required=True,
        help="Hook target. Integer layer index or full module path.",
    )
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument("--output-json", default="", help="Optional JSON path for dataset_A/dataset_B/dataset_C metrics.")
    parser.add_argument(
        "--source-csv",
        default="",
        help=(
            "Optional source split CSV used to map PT row_idx values to dataset_A/dataset_B/dataset_C. "
            "Required for portable dataset-level metrics when the original data/combine path is unavailable."
        ),
    )
    parser.add_argument(
        "--output-mode",
        choices=["metrics", "full"],
        default="metrics",
        help=(
            "Prediction text retention mode. "
            "'metrics' keeps output only through the first </answer>. "
            "'full' keeps the full templated completion "
            "(answer-first stops at </thinking>; answer-last stops at </answer>)."
        ),
    )
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--steering-coefficient", type=float, default=1.0)
    parser.add_argument("--steering-mode", choices=["replace", "add"], default="replace")
    parser.add_argument("--max-examples", type=int, default=0, help="If >0, evaluate only first N examples.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--device-map",
        choices=["auto", "single"],
        default="auto",
        help="Use 'single' to keep the full model on one GPU. Under torchrun this is forced automatically.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be > 0")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be > 0")
    if args.steering_coefficient <= 0:
        parser.error("--steering-coefficient must be > 0")
    if args.max_examples < 0:
        parser.error("--max-examples must be >= 0")
    _validate_pt_path(parser, args.pt_path)
    if args.lora_path and not os.path.exists(args.lora_path):
        parser.error(f"--lora-path does not exist: {args.lora_path}")

    args.model, args.lora_path = _resolve_model_and_lora_paths(args.model, args.lora_path)
    if args.lora_path and not os.path.exists(args.lora_path):
        parser.error(f"--lora-path does not exist: {args.lora_path}")

    set_seed(args.seed)
    dtype = _dtype_from_name(args.dtype)
    use_distributed = _maybe_init_distributed()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{_local_rank()}" if use_distributed else "cuda")
    else:
        device = torch.device("cpu")
    use_lora = bool(args.lora_path)

    try:
        if _is_rank_zero():
            print(f"Loading tokenizer: {args.model}")
        tokenizer = load_tokenizer(args.model)

        model_device_map: str | dict[str, str]
        if use_distributed or args.device_map == "single":
            if device.type != "cuda":
                raise RuntimeError("--device-map single requires CUDA.")
            model_device_map = {"": str(device)}
        else:
            model_device_map = "auto"

        if _is_rank_zero():
            print(f"Loading model: {args.model} (dtype={dtype}, device_map={model_device_map})")
        model = load_model(args.model, dtype=dtype, device_map=model_device_map)

        if use_lora:
            if _is_rank_zero():
                print(f"Loading LoRA adapter: {args.lora_path}")
            model = PeftModel.from_pretrained(
                model,
                args.lora_path,
                is_trainable=False,
                autocast_adapter_dtype=True,
            )
        model.eval()

        if _is_rank_zero():
            print(f"Loading eval data: {args.pt_path}")
        eval_data = _load_eval_data(
            pt_path=args.pt_path,
            model_name=args.model,
            max_examples=args.max_examples,
        )
        eval_data = _maybe_align_qwen3_eval_inputs(eval_data, tokenizer, args.model)
        if _is_rank_zero():
            print(f"Loaded {len(eval_data)} examples")

        _validate_unique_keys(eval_data)
        if _is_rank_zero():
            _prepare_output_csv(args.output_csv)
        if _dist_is_initialized():
            dist.barrier()
        processed_keys = _load_processed_keys(args.output_csv)
        remaining_eval_data = _filter_unfinished(eval_data, processed_keys)
        if _is_rank_zero():
            print(
                f"Resume scan: existing_keys={len(processed_keys)} "
                f"remaining={len(remaining_eval_data)}"
            )
        if not remaining_eval_data:
            if _is_rank_zero():
                print(f"No remaining examples. Output CSV is up to date: {args.output_csv}")
                if args.output_json:
                    _write_metrics_json(
                        output_json=args.output_json,
                        output_csv=args.output_csv,
                        pt_path=args.pt_path,
                        eval_data=eval_data,
                        source_csv_arg=args.source_csv,
                    )
                    print(f"Metrics JSON complete: {args.output_json}")
            return

        submodule = get_hf_submodule(model, args.hook_layer, use_lora=use_lora)
        eval_strategy = _infer_eval_strategy(remaining_eval_data)
        if _is_rank_zero():
            print(f"Eval strategy: {eval_strategy}")
            print(f"Output mode: {args.output_mode}")

        generation_kwargs = {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
        }
        stop_strings = _generation_stop_strings(
            output_mode=args.output_mode,
            eval_strategy=eval_strategy,
        )
        if stop_strings:
            generation_kwargs["stop_strings"] = stop_strings
            if _is_rank_zero():
                print(f"Stop strings: {stop_strings}")
        if eval_strategy == "binary_label_autoregressive" and _is_rank_zero():
            print("Binary-label dataset detected; keeping autoregressive decoding.")

        def write_batch(feature_results, source_batch) -> None:
            rows = [
                _csv_row(
                    dp,
                    pred.api_response,
                    output_mode=args.output_mode,
                    eval_strategy=eval_strategy,
                )
                for dp, pred in zip(source_batch, feature_results, strict=True)
            ]
            _append_csv_rows(args.output_csv, rows)

        run_evaluation(
            eval_data=remaining_eval_data,
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=device,
            dtype=dtype,
            global_step=-1,
            lora_path=None,
            eval_batch_size=args.eval_batch_size,
            steering_coefficient=args.steering_coefficient,
            steering_mode=args.steering_mode,
            generation_kwargs=generation_kwargs,
            verbose=False,
            collect_prompts=False,
            batch_callback=write_batch,
            distributed=use_distributed,
        )
        if _dist_is_initialized():
            dist.barrier()
        if _is_rank_zero():
            print(f"Streaming CSV complete: {args.output_csv}")
            if args.output_json:
                _write_metrics_json(
                    output_json=args.output_json,
                    output_csv=args.output_csv,
                    pt_path=args.pt_path,
                    eval_data=eval_data,
                    source_csv_arg=args.source_csv,
                )
                print(f"Metrics JSON complete: {args.output_json}")
    finally:
        _maybe_destroy_distributed()


if __name__ == "__main__":
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
