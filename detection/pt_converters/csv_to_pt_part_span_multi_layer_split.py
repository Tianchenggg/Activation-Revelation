#!/usr/bin/env python3
"""
Convert assess-only CSV files to per-layer PT files using one shared forward pass.

This is intended for layer-wise ablations where many extraction layers should be
materialized from the same source model without repeating the full multimodal
forward once per layer.
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch

csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import csv_to_pt_part_span as base
import pt_converter_common as common
import pt_converter_support as support


def _layer_output_path(output_root: Path, layer_id: int, output_filename: str) -> Path:
    return output_root / f"layer_{layer_id:02d}" / output_filename


def _all_writers_full(writers: dict[str, common.PtShardWriter], max_entries: Optional[int]) -> bool:
    if max_entries is None:
        return False
    return all(writer.total_entries >= max_entries for writer in writers.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert assess-only CSV to one PT file per extraction layer using shared multimodal forwards."
    )
    parser.add_argument("input_csv", help="Path to assess-only CSV")
    parser.add_argument("output_root", help="Root directory under which layer_<id>/<output_filename> is written")
    parser.add_argument("--output-filename", required=True, help="Per-layer PT filename, e.g. train.pt")
    parser.add_argument("--dataset-root", required=True, help="Root directory used to resolve relative image paths")
    parser.add_argument("--model_path", required=True, help="HuggingFace model path or hub ID")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--layers", nargs="+", default=None)
    parser.add_argument("--layer_id", type=int, default=None)
    parser.add_argument("--layer_ids", nargs="+", type=int, default=None)
    parser.add_argument("--response-modes", nargs="+", choices=base.ANSWER_MODES, default=list(base.ANSWER_MODES))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text_key", default="conversation")
    parser.add_argument("--detail_key", default="Detailed annotations")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--max_entries", type=int, default=0, help="Maximum PT entries to write per layer")
    parser.add_argument("--shard_size", type=int, default=4096)
    parser.add_argument("--single-file-output", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answergen1_system_prompt", default=support.ANSWERGEN1_DEFAULT_SYS_MSG)
    parser.add_argument("--answergen1_force_rgb", action="store_true")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument(
        "--activation_storage_dtype",
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="On-disk dtype used for saved activation tensors.",
    )

    args = parser.parse_args()
    if args.shard_size <= 0:
        raise ValueError("--shard_size must be > 0")

    dataset_root = str(Path(args.dataset_root).expanduser().resolve())
    if not Path(dataset_root).is_dir():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    torch_dtype = common.DTYPE_MAP[args.dtype]
    activation_storage_dtype = common.ACTIVATION_STORAGE_DTYPE_MAP[args.activation_storage_dtype]
    max_rows: Optional[int] = args.max_rows if args.max_rows and args.max_rows > 0 else None
    max_entries: Optional[int] = args.max_entries if args.max_entries and args.max_entries > 0 else None
    layer_paths, layer_ids, layer_id_by_path = common.resolve_layer_paths_and_ids(args, support.infer_layer_id_from_path)

    processor, tokenizer, processor_load_error = common.load_processor_and_tokenizer(
        model_path=args.model_path,
        allow_text_only_image_fallback=False,
        patch_mistral_fn=support.patch_mistral_common_tokenizer_utils,
    )

    print(f"Loading model from: {args.model_path} (dtype={args.dtype}, device={args.device})", file=sys.stderr)
    model = support.load_model_with_fallback(
        model_path=args.model_path,
        torch_dtype=torch_dtype,
        device=args.device,
    )
    model.eval()
    captured, handles = common.register_forward_hooks(model, layer_paths, layer_id_by_path, support.get_target_module)

    total_rows = 0
    empty_rows = 0
    controversial_segments = 0
    kept_segments = 0
    row_infos: list[Optional[dict[str, object]]] = []
    specs_by_row: dict[int, list[dict[str, object]]] = defaultdict(list)
    rng = random.Random(args.seed)

    print(f"Processing: {args.input_csv} -> {output_root}/layer_<id>/{args.output_filename}", file=sys.stderr)
    print(f"Dataset root: {dataset_root}", file=sys.stderr)
    print(f"Response modes: {args.response_modes}", file=sys.stderr)
    print(f"Layer paths: {layer_paths}", file=sys.stderr)
    print(f"Layer ids (PT): {layer_ids}", file=sys.stderr)
    print(f"Activation storage dtype: {args.activation_storage_dtype}", file=sys.stderr)

    with open(args.input_csv, "r", encoding="utf-8-sig", newline="") as in_f:
        reader = csv.DictReader(in_f)
        row_iter = enumerate(reader)
        if support.tqdm is not None:
            row_iter = support.tqdm(row_iter, total=max_rows, desc="Pass 1: index rows", unit="row")

        for row_idx, row in row_iter:
            if max_rows is not None and row_idx >= max_rows:
                break

            total_rows += 1
            detail = base._safe_json_loads(row.get(args.detail_key))
            if not isinstance(detail, dict):
                raise base.AnnotationValidationError(f"[row {row_idx}] Detailed annotations is not a JSON object.")
            segments = detail.get("segments")
            if not isinstance(segments, list):
                raise base.AnnotationValidationError(f"[row {row_idx}] Detailed annotations.segments is not a list.")
            controversial_segments += sum(
                1
                for segment in segments
                if isinstance(segment, dict) and int(segment.get("label_id", -1)) == base.CONTROVERSIAL_LABEL_ID
            )

            info = base._build_span_row_info(
                row=row,
                tokenizer=tokenizer,
                processor=processor,
                text_key=args.text_key,
                detail_key=args.detail_key,
                dataset_root=dataset_root,
                max_length=args.max_length,
                row_idx=row_idx,
                answergen1_system_prompt=args.answergen1_system_prompt,
                answergen1_force_rgb=args.answergen1_force_rgb,
                processor_error=str(processor_load_error) if processor_load_error is not None else None,
            )
            row_infos.append(info)
            if info is None:
                empty_rows += 1
                continue

            spans = info["spans"]  # type: ignore[index]
            kept_segments += len(spans)
            for span_idx, span in enumerate(spans):
                for response_mode in args.response_modes:
                    specs_by_row[row_idx].append(
                        {
                            "span_idx": span_idx,
                            "response_mode": response_mode,
                            "target_response": base._build_target_response(span, response_mode),
                        }
                    )

    writers: dict[str, common.PtShardWriter] = {}
    feature_idx_by_layer: dict[str, int] = {}
    for layer_path, layer_id in zip(layer_paths, layer_ids):
        output_pt = _layer_output_path(output_root, layer_id, args.output_filename)
        writers[layer_path] = common.PtShardWriter(
            str(output_pt),
            shard_size=args.shard_size,
            activation_dtype=activation_storage_dtype,
            single_file_output=args.single_file_output,
        )
        feature_idx_by_layer[layer_path] = 0

    row_indices = list(specs_by_row.keys())
    rng.shuffle(row_indices)
    row_iter = row_indices
    if support.tqdm is not None:
        row_iter = support.tqdm(row_indices, desc="Pass 2: collect PT datapoints", unit="row")

    for row_idx in row_iter:
        if _all_writers_full(writers, max_entries):
            break

        info = row_infos[row_idx]
        if info is None:
            continue

        hidden_by_layer = support.forward_row_hidden(
            row_info=info,
            tokenizer=tokenizer,
            processor=processor,
            model=model,
            captured=captured,
            layer_keys=layer_paths,
            max_length=args.max_length,
            row_idx=row_idx,
            answergen1_force_rgb=args.answergen1_force_rgb,
            require_image_processor=True,
            processor_error=str(processor_load_error) if processor_load_error is not None else None,
        )
        if hidden_by_layer is None:
            raise RuntimeError(f"[row {row_idx}] forward_row_hidden returned None.")

        spans = info["spans"]  # type: ignore[index]
        for spec in specs_by_row[row_idx]:
            span_idx = int(spec["span_idx"])
            if not (0 <= span_idx < len(spans)):
                raise IndexError(f"[row {row_idx}] invalid span_idx={span_idx} for {len(spans)} spans.")

            span = spans[span_idx]
            token_indices = [int(idx) for idx in span["token_indices"]]
            idx_tensor = torch.tensor(token_indices, dtype=torch.long)

            for layer_path in layer_paths:
                writer = writers[layer_path]
                if max_entries is not None and writer.total_entries >= max_entries:
                    continue

                hidden = hidden_by_layer.get(layer_path)
                if hidden is None:
                    raise RuntimeError(f"[row {row_idx}] missing captured activations for layer '{layer_path}'.")
                if any(idx < 0 or idx >= hidden.shape[1] for idx in token_indices):
                    raise IndexError(
                        f"[row {row_idx}] span {span_idx} token positions exceed activation length {hidden.shape[1]}."
                    )

                acts_BD = hidden[0][idx_tensor].detach().to(
                    device="cpu",
                    dtype=activation_storage_dtype,
                ).contiguous()
                layer_id = layer_id_by_path[layer_path]

                dp = common.create_training_datapoint(
                    datapoint_type="custom_pt_span",
                    prompt=base.PROMPTS[str(spec["response_mode"])],
                    target_response=str(spec["target_response"]),
                    layer=layer_id,
                    num_positions=acts_BD.shape[0],
                    tokenizer=tokenizer,
                    acts_BD=acts_BD,
                    feature_idx=feature_idx_by_layer[layer_path],
                    ds_label=None,
                    meta_info={
                        "row_idx": row_idx,
                        "row_id": info["row_id"],  # type: ignore[index]
                        "segment_idx": span["segment_idx"],
                        "layer_path": layer_path,
                        "prompt_text": base.PROMPTS[str(spec["response_mode"])],
                        "prompt_tokenizer_model": str(args.model_path),
                        "safety_label": span["safety_label"],
                        "label_id": span["label_id"],
                        "response_mode": spec["response_mode"],
                        "span_text": span["span_text_for_output"],
                        "reasoning_text": span["reasoning_text"],
                        "reasoning_fields": dict(span["reasoning_fields"]),
                    },
                )
                writer.append(dp)
                feature_idx_by_layer[layer_path] += 1

    common.remove_forward_handles(handles)

    per_layer_counts: dict[int, int] = {}
    for layer_path, layer_id in zip(layer_paths, layer_ids):
        writer = writers[layer_path]
        if writer.total_entries == 0:
            raise ValueError(f"No PT entries were written for layer_id={layer_id}.")
        writer.close(
            metadata={
                "source_csv": str(Path(args.input_csv).resolve()),
                "dataset_root": dataset_root,
                "model_path": args.model_path,
                "layer_paths": [layer_path],
                "layer_ids": [layer_id],
                "response_modes": list(args.response_modes),
                "seed": int(args.seed),
                "activation_storage_dtype": args.activation_storage_dtype,
                "variant": "part_span_assess_only",
                "labels": ["safe", "unsafe"],
            }
        )
        per_layer_counts[layer_id] = writer.total_entries

    counts_preview = ", ".join(f"{layer_id}:{count}" for layer_id, count in sorted(per_layer_counts.items()))
    print(
        f"\nDone. Rows processed: {total_rows}, rows without safe/unsafe spans: {empty_rows}, "
        f"controversial segments dropped: {controversial_segments}, non-controversial spans kept: {kept_segments}, "
        f"per-layer PT entries written: {counts_preview}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
