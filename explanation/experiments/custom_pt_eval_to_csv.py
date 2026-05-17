"""Evaluate multimodal custom PT data and stream bbox JSON predictions to CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nl_probes.dataset_classes.custom_pt_dataset import load_training_datapoints_from_pt
from nl_probes.utils.activation_oracle_cli import DEFAULT_IMAGE_MAX_PIXELS, DEFAULT_VLM_MODEL
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_processor, set_seed
from nl_probes.utils.dataset_utils import TrainingDataPoint
from nl_probes.utils.eval import run_evaluation

CSV_FIELDNAMES = [
    "sample_key",
    "feature_idx",
    "row_idx",
    "row_id",
    "span_idx",
    "layer_path",
    "source",
    "source_split",
    "image_path",
    "true_bbox_json",
    "pred_bbox_json",
]


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_rank_zero() -> bool:
    return _local_rank() == 0


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _maybe_init_distributed() -> bool:
    if _world_size() <= 1:
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
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _validate_pt_path(parser: argparse.ArgumentParser, path: str) -> None:
    if not os.path.exists(path):
        parser.error(f"--pt does not exist: {path}")
    if Path(path).suffix.lower() != ".pt":
        parser.error(f"--pt must point to a .pt file: {path}")


def _load_eval_data(*, pt_path: str, max_examples: int) -> list[TrainingDataPoint]:
    return load_training_datapoints_from_pt(
        pt_path=pt_path,
        model_name="",
        datapoint_type="custom_pt_eval",
        max_records=max_examples if max_examples > 0 else 0,
    )


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


def _extract_first_json_object(text: str) -> str:
    raw = str(text).strip()
    if not raw:
        return raw

    fenced = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    if fenced.startswith("{"):
        brace_depth = 0
        for index, char in enumerate(fenced):
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    candidate = fenced[: index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))

    for start_idx, char in enumerate(raw):
        if char != "{":
            continue
        brace_depth = 0
        for end_idx in range(start_idx, len(raw)):
            current = raw[end_idx]
            if current == "{":
                brace_depth += 1
            elif current == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    candidate = raw[start_idx : end_idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    return raw


def _csv_row(dp: TrainingDataPoint, pred_answer: str) -> dict[str, str]:
    meta_info = dict(dp.meta_info or {})
    return {
        "sample_key": _sample_key(dp),
        "feature_idx": str(int(dp.feature_idx)),
        "row_idx": "" if meta_info.get("row_idx") is None else str(meta_info.get("row_idx")),
        "row_id": "" if meta_info.get("row_id") is None else str(meta_info.get("row_id")),
        "span_idx": "" if meta_info.get("segment_idx") is None else str(meta_info.get("segment_idx")),
        "layer_path": "" if meta_info.get("layer_path") is None else str(meta_info.get("layer_path")),
        "source": "" if meta_info.get("source") is None else str(meta_info.get("source")),
        "source_split": "" if meta_info.get("source_split") is None else str(meta_info.get("source_split")),
        "image_path": "" if meta_info.get("image_path") is None else str(meta_info.get("image_path")),
        "true_bbox_json": str(dp.target_output),
        "pred_bbox_json": _extract_first_json_object(pred_answer),
    }


def _write_csv_rows(output_csv: str, rows: list[dict[str, str]]) -> None:
    output_path = Path(output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate custom PT bbox data and write predictions to CSV.")
    parser.add_argument("--model", default=DEFAULT_VLM_MODEL, help="Base model name/path used for evaluation.")
    parser.add_argument("--lora-path", default="", help="Optional LoRA adapter path.")
    parser.add_argument("--pt", dest="pt_path", required=True, help="Path to evaluation PT dataset.")
    parser.add_argument("--hook-layer", type=_parse_layer_selector, required=True)
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steering-coefficient", type=float, default=1.0)
    parser.add_argument("--steering-mode", choices=["replace", "add"], default="replace")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--image-max-pixels", type=int, default=DEFAULT_IMAGE_MAX_PIXELS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be > 0")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be > 0")
    if args.max_examples < 0:
        parser.error("--max-examples must be >= 0")
    if args.steering_coefficient <= 0:
        parser.error("--steering-coefficient must be > 0")
    if args.image_max_pixels <= 0:
        parser.error("--image-max-pixels must be > 0")
    _validate_pt_path(parser, args.pt_path)
    if args.lora_path and not os.path.exists(args.lora_path):
        parser.error(f"--lora-path does not exist: {args.lora_path}")

    set_seed(args.seed)
    dtype = _dtype_from_name(args.dtype)
    use_distributed = _maybe_init_distributed()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{_local_rank()}" if use_distributed else "cuda")
    else:
        device = torch.device("cpu")

    try:
        processor = load_processor(args.model)
        model_device_map: str | dict[str, str]
        if use_distributed:
            model_device_map = {"": str(device)}
        elif device.type == "cuda":
            model_device_map = {"": str(device)}
        else:
            model_device_map = "auto"

        model = load_model(args.model, dtype=dtype, device_map=model_device_map)
        if args.lora_path:
            model = PeftModel.from_pretrained(
                model,
                args.lora_path,
                is_trainable=False,
                autocast_adapter_dtype=True,
            )
        model.eval()

        eval_data = _load_eval_data(pt_path=args.pt_path, max_examples=args.max_examples)
        if _is_rank_zero():
            print(f"Loaded {len(eval_data)} eval examples from {args.pt_path}")

        output_path = Path(args.output_csv).resolve()
        if _is_rank_zero() and output_path.exists():
            output_path.unlink()
        if _dist_is_initialized():
            dist.barrier()

        submodule = get_hf_submodule(model, args.hook_layer, use_lora=bool(args.lora_path))

        def write_batch(feature_results, source_batch) -> None:
            rows = [
                _csv_row(dp, pred.api_response)
                for dp, pred in zip(source_batch, feature_results, strict=True)
            ]
            _write_csv_rows(args.output_csv, rows)

        run_evaluation(
            eval_data=eval_data,
            model=model,
            processor=processor,
            submodule=submodule,
            device=device,
            dtype=dtype,
            eval_batch_size=args.eval_batch_size,
            steering_coefficient=args.steering_coefficient,
            steering_mode=args.steering_mode,
            generation_kwargs={
                "max_new_tokens": args.max_new_tokens,
                "image_max_pixels": args.image_max_pixels,
            },
            batch_callback=write_batch,
            distributed=use_distributed,
        )
        if _dist_is_initialized():
            dist.barrier()
        if _is_rank_zero():
            print(f"Finished writing predictions to {args.output_csv}")
    finally:
        _maybe_destroy_distributed()


if __name__ == "__main__":
    main()
