#!/usr/bin/env python3
"""
Convert merged detailed-annotation CSV rows to custom PT image-bbox-abduction data.

Input CSV keeps the original row-level schema:
  id, conversation, annotations, [model], Detailed annotations, source

"Detailed annotations" is intentionally minimal: each segment keeps only
segment_idx, label_id, label_name, and optional abduction fields. Span text and
offsets are read from the unchanged "annotations" column by segment_idx.
Only spans where every image bbox evidence item is qualified are converted into
targets; query/text evidence is ignored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import torch

csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pt_converter_common as common
import pt_converter_support as support


ACTIVATION_SOURCE_SPAN_TOKENS = "span_tokens"

PROMPT_SPAN_TOKENS = (
    "The previous tokens are internal activations extracted from one unsafe assistant response span. "
    "Infer only the single image bounding box that visually grounds that span. "
    'Return exactly one JSON object with key "bbox". '
    '"bbox" must be one [x1,y1,x2,y2] box normalized to the 0-1000 image coordinate space. '
    "Do not output text snippets, explanations, object names, markdown, safety labels, or thinking."
)

UNSAFE_LABEL_ID = 1
BBOX_FORMAT = "normalized_0_1000_xyxy"
DEFAULT_IMAGE_MAX_PIXELS = 256 * 28 * 28
DEFAULT_IMAGE_JPEG_QUALITY = 90


class AbductionValidationError(ValueError):
    """Raised when detailed annotations cannot be converted into abduction targets."""


def _safe_json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return json.loads(stripped)
    return None


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    return text if text else None


def _parse_int(value: Any, *, field_name: str, row_idx: int) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise AbductionValidationError(f"[row {row_idx}] invalid integer field {field_name}: {value!r}") from exc


def _normalize_bbox(value: Any) -> Optional[list[int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(coord))) for coord in value]
    except (TypeError, ValueError):
        return None
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    return [x1, y1, x2, y2]


def _normalize_image_evidence_item(item: Any) -> Optional[dict[str, Any]]:
    if not isinstance(item, dict):
        return None

    object_name = _normalize_text(item.get("object_name"))
    explanation = _normalize_text(item.get("explanation"))
    bbox = _normalize_bbox(item.get("bbox"))
    flags = item.get("bbox_quality_flags") if isinstance(item.get("bbox_quality_flags"), list) else []
    semantic = item.get("bbox_semantic_verification") if isinstance(item.get("bbox_semantic_verification"), dict) else None
    if object_name is None or explanation is None or bbox is None or flags:
        return None
    if semantic is not None and semantic.get("verdict") != "pass":
        return None
    return {
        "object_name": object_name,
        "bbox": bbox,
        "bbox_format": BBOX_FORMAT,
        "explanation": explanation,
    }


def _normalize_image_evidence(raw_items: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list) or not raw_items:
        return []

    output: list[dict[str, Any]] = []
    for item in raw_items:
        normalized = _normalize_image_evidence_item(item)
        if normalized is None:
            return []
        output.append(normalized)
    return output


def _normalize_single_primary_image_evidence(abduction: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(abduction, dict):
        return []
    if _normalize_text(abduction.get("warning")) is not None:
        return []
    if abduction.get("evidence_type") != "single_primary":
        return []

    primary = _normalize_image_evidence_item(abduction.get("primary_image_evidence"))
    if primary is None:
        return []
    return [primary]


def _extract_qualified_single_image_evidence(abduction: dict[str, Any]) -> list[dict[str, Any]]:
    image_evidence = _normalize_single_primary_image_evidence(abduction)
    if image_evidence:
        return image_evidence

    # Backward-compatible fallback for legacy merged CSVs that still store image_evidence.
    image_evidence = _normalize_image_evidence(abduction.get("image_evidence"))
    if len(image_evidence) != 1:
        return []
    return image_evidence


def _build_abduction_target(
    *,
    abduction: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]] | None:
    image_evidence = _extract_qualified_single_image_evidence(abduction)
    if not image_evidence:
        return None

    target = {
        "bbox": image_evidence[0]["bbox"],
    }
    return _compact_json(target), image_evidence


def _round_down_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return max(1, value)
    if value <= multiple:
        return multiple
    return max(multiple, (value // multiple) * multiple)


def _resize_dims_for_max_pixels(width: int, height: int, max_pixels: Optional[int]) -> tuple[int, int]:
    if max_pixels is None or max_pixels <= 0:
        return width, height
    if width * height <= max_pixels:
        return width, height

    scale = math.sqrt(float(max_pixels) / float(width * height))
    new_width = max(28, int(math.floor(width * scale)))
    new_height = max(28, int(math.floor(height * scale)))
    new_width = _round_down_to_multiple(new_width, 28)
    new_height = _round_down_to_multiple(new_height, 28)

    while new_width * new_height > max_pixels and (new_width > 28 or new_height > 28):
        if new_width >= new_height and new_width > 28:
            new_width = max(28, new_width - 28)
        elif new_height > 28:
            new_height = max(28, new_height - 28)
        else:
            break

    return new_width, new_height


def _resolve_image_cache_dir(output_pt: str, explicit_cache_dir: Optional[str]) -> str:
    if explicit_cache_dir:
        return str(Path(explicit_cache_dir).expanduser().resolve())
    output_path = Path(output_pt).expanduser().resolve()
    return str((output_path.parent / f"{output_path.stem}_images").resolve())


def _prepare_resized_image_path(
    *,
    image_path: Optional[str],
    image_cache_dir: str,
    image_max_pixels: Optional[int],
    jpeg_quality: int,
    force_rgb: bool,
) -> Optional[str]:
    if image_path is None:
        return None

    from PIL import Image as PILImage

    source_path = Path(image_path).expanduser().resolve()
    with PILImage.open(source_path) as opened_image:
        source_mode = str(opened_image.mode or "")
        image = opened_image.convert("RGB") if (force_rgb or source_mode != "RGB") else opened_image.copy()

    target_size = _resize_dims_for_max_pixels(image.width, image.height, image_max_pixels)
    needs_resize = target_size != image.size
    needs_reencode = force_rgb or source_mode != "RGB" or needs_resize
    if not needs_reencode:
        return str(source_path)

    if needs_resize:
        image = image.resize(target_size, PILImage.Resampling.LANCZOS)
    if image.mode != "RGB":
        image = image.convert("RGB")

    cache_root = Path(image_cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha1(
        f"{source_path}|{target_size[0]}x{target_size[1]}|q={jpeg_quality}|rgb=1".encode("utf-8")
    ).hexdigest()[:16]
    cached_path = cache_root / f"{source_path.stem}_{fingerprint}.jpg"
    if not cached_path.exists():
        image.save(cached_path, format="JPEG", quality=jpeg_quality, optimize=True)
    return str(cached_path)


def _parse_source_annotations(raw_annotations: Any, *, row_idx: int) -> dict[int, dict[str, Any]]:
    annotations = _safe_json_loads(raw_annotations)
    if not isinstance(annotations, list):
        raise AbductionValidationError(f"[row {row_idx}] annotations is not a JSON list.")

    parsed: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(annotations):
        if not isinstance(item, dict):
            continue
        segment_idx = _parse_int(item.get("segment_idx", index), field_name="annotations.segment_idx", row_idx=row_idx)
        char_start = _parse_int(item.get("char_start"), field_name=f"annotations[{index}].char_start", row_idx=row_idx)
        char_end = _parse_int(item.get("char_end"), field_name=f"annotations[{index}].char_end", row_idx=row_idx)
        span = item.get("span")
        if not isinstance(span, str) or not span:
            raise AbductionValidationError(f"[row {row_idx}] annotations[{index}] is missing span text.")
        if char_start < 0 or char_end <= char_start:
            raise AbductionValidationError(
                f"[row {row_idx}] annotations[{index}] has invalid char range [{char_start}, {char_end})."
            )
        parsed[segment_idx] = {
            "segment_idx": segment_idx,
            "index": char_start,
            "char_end": char_end,
            "span": span,
        }
    return parsed


def _parse_detailed_segments(
    raw_detail: Any,
    raw_annotations: Any,
    *,
    row_idx: int,
) -> list[dict[str, Any]]:
    detail = _safe_json_loads(raw_detail)
    if not isinstance(detail, dict):
        raise AbductionValidationError(f"[row {row_idx}] Detailed annotations is not a JSON object.")

    segments = detail.get("segments")
    if not isinstance(segments, list):
        raise AbductionValidationError(f"[row {row_idx}] Detailed annotations.segments is not a list.")

    annotations_by_idx = _parse_source_annotations(raw_annotations, row_idx=row_idx)
    parsed: list[dict[str, Any]] = []
    for raw_segment in segments:
        if not isinstance(raw_segment, dict):
            continue
        try:
            label_id = int(raw_segment.get("label_id", -1))
        except (TypeError, ValueError):
            continue
        if label_id != UNSAFE_LABEL_ID:
            continue

        abduction = raw_segment.get("abduction")
        if not isinstance(abduction, dict):
            # Backward-compatible fallback for older detailed CSVs.
            abduction = raw_segment.get("unsafe_context_linkage")
        if not isinstance(abduction, dict):
            continue
        target = _build_abduction_target(abduction=abduction)
        if target is None:
            continue
        target_response, image_evidence = target

        segment_idx = _parse_int(raw_segment.get("segment_idx"), field_name="segment_idx", row_idx=row_idx)
        source_annotation = annotations_by_idx.get(segment_idx)
        if source_annotation is None:
            raise AbductionValidationError(
                f"[row {row_idx}] Detailed annotations segment {segment_idx} has no matching annotations entry."
            )

        parsed.append(
            {
                "segment_idx": segment_idx,
                "index": source_annotation["index"],
                "char_end": source_annotation["char_end"],
                "span": source_annotation["span"],
                "target_response": target_response,
                "image_evidence": image_evidence,
            }
        )
    return parsed


def _validate_span_alignment(
    *,
    completion_text: str,
    span: dict[str, Any],
    row_idx: int,
) -> None:
    start = int(span["index"])
    end = int(span["char_end"])
    observed = completion_text[start:end]
    if observed != span["span"]:
        raise AbductionValidationError(
            f"[row {row_idx}] segment {span['segment_idx']} does not match completion text at "
            f"[{start}, {end})."
        )


def _build_span_row_info(
    *,
    row: dict[str, Any],
    tokenizer,
    processor,
    text_key: str,
    detail_key: str,
    dataset_root: Optional[str],
    max_length: Optional[int],
    image_max_pixels: Optional[int] = None,
    row_idx: int,
    answergen1_system_prompt: Optional[str],
    answergen1_force_rgb: bool,
    processor_error: Optional[str],
    image_cache_dir: str,
    image_jpeg_quality: int,
) -> Optional[dict[str, Any]]:
    text_state = support._build_text_and_token_state(
        tokenizer=tokenizer,
        raw_conv=row.get(text_key),
        max_length=max_length,
        row_idx=row_idx,
        answergen1_system_prompt=answergen1_system_prompt,
        answergen1_force_rgb=answergen1_force_rgb,
        processor=processor,
        require_image_processor=True,
        processor_error=processor_error,
        dataset_root=dataset_root,
    )
    if text_state is None:
        return None

    segments = _parse_detailed_segments(
        row.get(detail_key),
        row.get("annotations"),
        row_idx=row_idx,
    )
    if not segments:
        return None

    prepared_image_path = _prepare_resized_image_path(
        image_path=text_state["image_path"],
        image_cache_dir=image_cache_dir,
        image_max_pixels=image_max_pixels,
        jpeg_quality=image_jpeg_quality,
        force_rgb=answergen1_force_rgb,
    )

    full_messages = text_state.get("full_messages")
    if prepared_image_path and isinstance(full_messages, list):
        updated_messages: list[dict[str, Any]] = []
        for message in full_messages:
            cloned = dict(message)
            content = cloned.get("content")
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, dict) and str(item.get("type", "")).lower() == "image":
                        new_item = dict(item)
                        new_item["image"] = prepared_image_path
                        new_content.append(new_item)
                    else:
                        new_content.append(dict(item) if isinstance(item, dict) else item)
                cloned["content"] = new_content
            updated_messages.append(cloned)
        full_messages = updated_messages

    pos_map = support._build_pos_map(
        processor=processor,
        full_text=text_state["full_text"],
        image_path=prepared_image_path,
        text_input_ids=text_state["text_input_ids"],
        max_length=max_length,
        image_max_pixels=image_max_pixels,
        row_idx=row_idx,
        answergen1_force_rgb=answergen1_force_rgb,
        require_image_processor=True,
        processor_error=processor_error,
        full_messages=full_messages,
    )

    spans: list[dict[str, Any]] = []
    for segment in segments:
        _validate_span_alignment(
            completion_text=text_state["completion_text"],
            span=segment,
            row_idx=row_idx,
        )

        text_indices = support.get_span_token_indices(
            ann=segment,
            offsets=text_state["offsets"],
            input_ids=text_state["text_input_ids"],
            special_ids=text_state["special_ids"],
            completion_start_tok=text_state["completion_start_tok"],
            completion_char0=text_state["completion_char0"],
            completion_text=text_state["completion_text"],
            full_text=text_state["full_text"],
        )
        if not text_indices:
            raise AbductionValidationError(
                f"[row {row_idx}] segment {segment['segment_idx']} could not be aligned to text tokens."
            )

        token_indices = [pos_map[idx] for idx in text_indices if idx in pos_map]
        if len(token_indices) != len(text_indices):
            raise AbductionValidationError(
                f"[row {row_idx}] segment {segment['segment_idx']} could not be aligned after multimodal remapping."
            )

        span = dict(segment)
        span["span_token_indices"] = token_indices
        span["token_indices"] = token_indices
        spans.append(span)

    if not spans:
        return None

    return {
        "row_id": row.get("id"),
        "source": row.get("source"),
        "source_split": row.get("source_split"),
        "full_text": text_state["full_text"],
        "image_path": prepared_image_path,
        "original_image_path": text_state["image_path"],
        "prompt_messages": text_state["prompt_messages"],
        "full_messages": full_messages,
        "spans": spans,
    }


def _build_pt_prompt_messages(
    *,
    prompt_text: str,
    image_path: str,
    layer: int,
    num_positions: int,
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    return common.build_vlm_prompt_messages(
        prompt_text=prompt_text,
        image_path=image_path,
        layer=layer,
        num_positions=num_positions,
        system_prompt=system_prompt,
    )


def _compute_full_sequence_length(
    *,
    processor,
    prompt_messages: list[dict[str, Any]],
    target_output: str,
    image_max_pixels: Optional[int],
) -> int:
    encoded = support.vlm_compat.apply_chat_template(
        processor,
        prompt_messages + [{"role": "assistant", "content": [{"type": "text", "text": str(target_output)}]}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors=None,
        padding=False,
        max_pixels=image_max_pixels,
    )
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return len(input_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert row-level detailed annotation CSV to PT image-bbox-abduction data.")
    parser.add_argument("input_csv", help="Path to merged detailed annotation CSV.")
    parser.add_argument("output_pt", help="Path to output PT manifest or packed shard.")
    parser.add_argument("--model_path", required=True, help="HuggingFace model path or hub ID.")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--layers", nargs="+", default=None)
    parser.add_argument("--layer_id", type=int, default=None)
    parser.add_argument("--layer_ids", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text_key", default="conversation")
    parser.add_argument("--detail_key", default="Detailed annotations")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional root for relative image paths. The merged CSV stores absolute paths, so this is usually unnecessary.",
    )
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--max_entries", type=int, default=0)
    parser.add_argument(
        "--max-entries-per-target",
        type=int,
        default=0,
        help="If >0, cap the number of PT entries with the same exact target JSON. Useful for avoiding fixed-box collapse.",
    )
    parser.add_argument("--shard_size", type=int, default=4096)
    parser.add_argument("--single-file-output", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answergen1_system_prompt", default=support.ANSWERGEN1_DEFAULT_SYS_MSG)
    parser.add_argument("--answergen1_force_rgb", action="store_true")
    parser.add_argument(
        "--activation-source",
        default=ACTIVATION_SOURCE_SPAN_TOKENS,
        choices=[ACTIVATION_SOURCE_SPAN_TOKENS],
        help=(
            "Which multimodal positions to store as steering vectors. "
            "Only 'span_tokens' is supported."
        ),
    )
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument(
        "--activation_storage_dtype",
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="On-disk dtype used for saved activation tensors.",
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=DEFAULT_IMAGE_MAX_PIXELS,
        help="Maximum pixels passed to the processor for every image.",
    )
    parser.add_argument(
        "--image-cache-dir",
        default="",
        help="Directory used to store resized images referenced by the output PT.",
    )
    parser.add_argument(
        "--image-jpeg-quality",
        type=int,
        default=DEFAULT_IMAGE_JPEG_QUALITY,
        help="JPEG quality used for resized image cache files.",
    )

    args = parser.parse_args()
    if args.shard_size <= 0:
        raise ValueError("--shard_size must be > 0")
    if args.image_max_pixels <= 0:
        raise ValueError("--image-max-pixels must be > 0")
    if args.image_jpeg_quality <= 0 or args.image_jpeg_quality > 100:
        raise ValueError("--image-jpeg-quality must be in [1, 100]")

    dataset_root = str(Path(args.dataset_root).expanduser().resolve()) if args.dataset_root else None
    if dataset_root is not None and not Path(dataset_root).is_dir():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")
    image_cache_dir = _resolve_image_cache_dir(args.output_pt, args.image_cache_dir or None)

    torch_dtype = common.DTYPE_MAP[args.dtype]
    activation_storage_dtype = common.ACTIVATION_STORAGE_DTYPE_MAP[args.activation_storage_dtype]
    max_rows: Optional[int] = args.max_rows if args.max_rows and args.max_rows > 0 else None
    max_entries: Optional[int] = args.max_entries if args.max_entries and args.max_entries > 0 else None
    max_entries_per_target: Optional[int] = (
        args.max_entries_per_target if args.max_entries_per_target and args.max_entries_per_target > 0 else None
    )
    layer_paths, layer_ids, layer_id_by_path = common.resolve_layer_paths_and_ids(args, support.infer_layer_id_from_path)
    rng = random.Random(args.seed)
    prompt = PROMPT_SPAN_TOKENS

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
    kept_segments = 0
    row_infos: list[Optional[dict[str, Any]]] = []
    valid_row_indices: list[int] = []

    print(f"Processing: {args.input_csv} -> {args.output_pt}", file=sys.stderr)
    print(f"Dataset root: {dataset_root or '<absolute paths from CSV>'}", file=sys.stderr)
    print("Target mode: image bbox abduction_result JSON only", file=sys.stderr)
    print(f"Activation source: {args.activation_source}", file=sys.stderr)
    print(f"Image max pixels: {args.image_max_pixels}", file=sys.stderr)
    print(f"Image cache dir: {image_cache_dir}", file=sys.stderr)
    if max_entries_per_target is not None:
        print(f"Max entries per exact target: {max_entries_per_target}", file=sys.stderr)
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
            info = _build_span_row_info(
                row=row,
                tokenizer=tokenizer,
                processor=processor,
                text_key=args.text_key,
                detail_key=args.detail_key,
                dataset_root=dataset_root,
                max_length=args.max_length,
                image_max_pixels=args.image_max_pixels,
                row_idx=row_idx,
                answergen1_system_prompt=args.answergen1_system_prompt,
                answergen1_force_rgb=args.answergen1_force_rgb,
                processor_error=str(processor_load_error) if processor_load_error is not None else None,
                image_cache_dir=image_cache_dir,
                image_jpeg_quality=args.image_jpeg_quality,
            )
            row_infos.append(info)
            if info is None:
                empty_rows += 1
                continue
            valid_row_indices.append(row_idx)
            kept_segments += len(info["spans"])

    writer = common.PtShardWriter(
        args.output_pt,
        shard_size=args.shard_size,
        activation_dtype=activation_storage_dtype,
        single_file_output=args.single_file_output,
    )
    feature_idx_counter = 0
    target_entry_counts: Counter[str] = Counter()

    rng.shuffle(valid_row_indices)
    row_iter = valid_row_indices
    if support.tqdm is not None:
        row_iter = support.tqdm(valid_row_indices, desc="Pass 2: collect PT datapoints", unit="row")

    try:
        for row_idx in row_iter:
            if max_entries is not None and writer.total_entries >= max_entries:
                break

            info = row_infos[row_idx]
            if info is None:
                continue
            if max_entries_per_target is not None and all(
                target_entry_counts[str(span["target_response"])] >= max_entries_per_target for span in info["spans"]
            ):
                continue

            hidden_by_layer = support.forward_row_hidden(
                row_info=info,
                tokenizer=tokenizer,
                processor=processor,
                model=model,
                captured=captured,
                layer_keys=layer_paths,
                max_length=args.max_length,
                image_max_pixels=args.image_max_pixels,
                row_idx=row_idx,
                answergen1_force_rgb=args.answergen1_force_rgb,
                require_image_processor=True,
                processor_error=str(processor_load_error) if processor_load_error is not None else None,
            )
            if hidden_by_layer is None:
                raise RuntimeError(f"[row {row_idx}] forward_row_hidden returned None.")

            for layer_path in layer_paths:
                if max_entries is not None and writer.total_entries >= max_entries:
                    break

                hidden = hidden_by_layer.get(layer_path)
                if hidden is None:
                    raise RuntimeError(f"[row {row_idx}] missing captured activations for layer '{layer_path}'.")

                layer_id = layer_id_by_path[layer_path]
                for span in info["spans"]:
                    if max_entries is not None and writer.total_entries >= max_entries:
                        break
                    target_key = str(span["target_response"])
                    if (
                        max_entries_per_target is not None
                        and target_entry_counts[target_key] >= max_entries_per_target
                    ):
                        continue

                    token_indices = [int(idx) for idx in span["token_indices"]]
                    if any(idx < 0 or idx >= hidden.shape[1] for idx in token_indices):
                        raise IndexError(
                            f"[row {row_idx}] span {span['segment_idx']} token positions exceed activation length "
                            f"{hidden.shape[1]}."
                        )

                    idx_tensor = torch.tensor(token_indices, dtype=torch.long)
                    acts_BD = hidden[0][idx_tensor].detach().to(
                        device="cpu",
                        dtype=activation_storage_dtype,
                    ).contiguous()
                    prompt_messages = _build_pt_prompt_messages(
                        prompt_text=prompt,
                        image_path=str(info["image_path"]),
                        layer=layer_id,
                        num_positions=int(acts_BD.shape[0]),
                        system_prompt=args.answergen1_system_prompt,
                    )
                    full_sequence_length = _compute_full_sequence_length(
                        processor=processor,
                        prompt_messages=prompt_messages,
                        target_output=str(span["target_response"]),
                        image_max_pixels=args.image_max_pixels,
                    )

                    dp = common.create_vlm_training_datapoint(
                        datapoint_type="custom_pt_span",
                        prompt_messages=prompt_messages,
                        target_response=str(span["target_response"]),
                        layer=layer_id,
                        num_positions=acts_BD.shape[0],
                        acts_BD=acts_BD,
                        feature_idx=feature_idx_counter,
                        ds_label=None,
                        meta_info={
                            "row_idx": row_idx,
                            "row_id": info["row_id"],
                            "source": info["source"],
                            "source_split": info.get("source_split"),
                            "segment_idx": span["segment_idx"],
                            "layer_path": layer_path,
                            "activation_source": args.activation_source,
                            "span_text": span["span"],
                            "token_indices": token_indices,
                            "span_token_indices": list(span.get("span_token_indices", [])),
                            "image_evidence": list(span["image_evidence"]),
                            "image_path": str(info["image_path"]),
                            "original_image_path": str(info["original_image_path"])
                            if info.get("original_image_path") is not None
                            else None,
                            "image_max_pixels": int(args.image_max_pixels),
                            "full_sequence_length": int(full_sequence_length),
                            "target_response": span["target_response"],
                        },
                    )
                    writer.append(dp)
                    target_entry_counts[target_key] += 1
                    feature_idx_counter += 1
    finally:
        common.remove_forward_handles(handles)

    if writer.total_entries == 0:
        raise ValueError("No PT entries were written.")

    writer.close(
        metadata={
            "source_csv": str(Path(args.input_csv).resolve()),
            "dataset_root": dataset_root,
            "model_path": args.model_path,
            "layer_paths": layer_paths,
            "layer_ids": layer_ids,
            "seed": int(args.seed),
            "activation_storage_dtype": args.activation_storage_dtype,
            "variant": "part_span_bbox_abduction",
            "activation_source": args.activation_source,
            "max_entries_per_target": int(max_entries_per_target or 0),
            "prompt": prompt,
            "image_max_pixels": int(args.image_max_pixels),
            "image_cache_dir": image_cache_dir,
            "image_jpeg_quality": int(args.image_jpeg_quality),
        }
    )
    print(
        f"\nDone. Rows processed: {total_rows}, rows without qualified spans: {empty_rows}, "
        f"qualified spans kept: {kept_segments}, PT entries written: {writer.total_entries}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
