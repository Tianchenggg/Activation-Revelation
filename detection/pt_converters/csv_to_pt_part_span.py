#!/usr/bin/env python3
"""
Convert assess-only CSV files to custom PT data with span-level activations.

Input labels are read from "Detailed annotations" -> "segments":
  0 = safe
  1 = unsafe
  2 = controversial (dropped)
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch

csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pt_converter_common as common
import pt_converter_support as support


SAFE_LABEL_ID = 0
UNSAFE_LABEL_ID = 1
CONTROVERSIAL_LABEL_ID = 2
LABEL_ID_TO_NAME = {
    SAFE_LABEL_ID: "safe",
    UNSAFE_LABEL_ID: "unsafe",
    CONTROVERSIAL_LABEL_ID: "controversial",
}

NO_THINKING = "no_thinking"
THINKING_ANSWER_FIRST = "thinking_answer_first"
THINKING_ANSWER_LAST = "thinking_answer_last"
ANSWER_MODES = (
    NO_THINKING,
    THINKING_ANSWER_FIRST,
    THINKING_ANSWER_LAST,
)
REASONING_FIELD_NAMES = (
    "image_consideration",
    "user_request_consideration",
    "segment_consideration",
    "impact_good",
    "impact_bad",
    "final_rationale",
)
RECONSTRUCTED_SPAN_LABEL = "[Reconstructed Span]"
RECONSTRUCTED_USER_REQUEST_LABEL = "[Reconstructed User Request]"
IMAGE_CONTEXT_LABEL = "[Image Context]"
USER_INTENT_LABEL = "[User Intent Analysis]"
SPAN_ANALYSIS_LABEL = "[Span Analysis]"
POSITIVE_IMPACT_LABEL = "[Potential Positive Impact]"
NEGATIVE_IMPACT_LABEL = "[Potential Negative Impact]"
FINAL_RATIONALE_LABEL = "[Final Rationale]"

_PROMPT_PREFIX = (
    "The previous tokens are internal activations extracted from one response span. "
    "Use them to judge whether the corresponding span is safe or unsafe."
)
PROMPTS = {
    NO_THINKING: _PROMPT_PREFIX,
    THINKING_ANSWER_FIRST: (
        _PROMPT_PREFIX
        + ' First output an <answer> block with exactly "safe" or "unsafe", '
        + "then output a <thinking> block that states the inferred span and your brief reasoning."
    ),
    THINKING_ANSWER_LAST: (
        _PROMPT_PREFIX
        + " First output a <thinking> block that states the inferred span and your brief reasoning, "
        + 'then output an <answer> block with exactly "safe" or "unsafe"'
    ),
}


class AnnotationValidationError(ValueError):
    """Raised when the assess-only annotations are malformed."""


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


def _normalize_output_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    return text if text else None


def _build_answer_text(safety_label: str) -> str:
    if safety_label == "safe":
        return "safe"
    if safety_label == "unsafe":
        return "unsafe"
    raise ValueError(f"Unsupported safety label: {safety_label}")


def _build_reasoning_text(span: dict[str, Any], user_text_for_output: str) -> str:
    reasoning_fields = span["reasoning_fields"]
    lines = [
        f'{RECONSTRUCTED_SPAN_LABEL} "{span["span_text_for_output"]}"',
        f'{RECONSTRUCTED_USER_REQUEST_LABEL} "{user_text_for_output}"',
        f"{IMAGE_CONTEXT_LABEL} {reasoning_fields['image_consideration']}",
        f"{USER_INTENT_LABEL} {reasoning_fields['user_request_consideration']}",
        f"{SPAN_ANALYSIS_LABEL} {reasoning_fields['segment_consideration']}",
        f"{POSITIVE_IMPACT_LABEL} {reasoning_fields['impact_good']}",
        f"{NEGATIVE_IMPACT_LABEL} {reasoning_fields['impact_bad']}",
        f"{FINAL_RATIONALE_LABEL} {reasoning_fields['final_rationale']}",
    ]
    return "\n".join(lines)


def _build_target_response(span: dict[str, Any], response_mode: str) -> str:
    answer_text = _build_answer_text(str(span["safety_label"]))
    if response_mode == NO_THINKING:
        return answer_text

    reasoning_text = str(span["reasoning_text"])
    if response_mode == THINKING_ANSWER_FIRST:
        return (
            "<answer>\n"
            f"{answer_text}\n"
            "</answer>\n"
            "<thinking>\n"
            f"{reasoning_text}\n"
            "</thinking>"
        )
    if response_mode == THINKING_ANSWER_LAST:
        return (
            "<thinking>\n"
            f"{reasoning_text}\n"
            "</thinking>\n"
            "<answer>\n"
            f"{answer_text}\n"
            "</answer>"
        )
    raise ValueError(f"Unsupported response mode: {response_mode}")


def _parse_segment_label(segment: dict[str, Any], row_idx: int, segment_idx: int) -> Optional[str]:
    try:
        label_id = int(segment["label_id"])
    except Exception as exc:
        raise AnnotationValidationError(
            f"[row {row_idx}] segment {segment_idx} has invalid label_id: {segment.get('label_id')!r}"
        ) from exc

    expected_name = LABEL_ID_TO_NAME.get(label_id)
    if expected_name is None:
        raise AnnotationValidationError(f"[row {row_idx}] segment {segment_idx} uses unknown label_id={label_id}.")

    label_name = _normalize_output_text(segment.get("label_name"))
    if label_name is None:
        raise AnnotationValidationError(f"[row {row_idx}] segment {segment_idx} is missing label_name.")
    if label_name.lower() != expected_name:
        raise AnnotationValidationError(
            f"[row {row_idx}] segment {segment_idx} label mismatch: label_id={label_id}, label_name={label_name!r}."
        )

    if label_id == CONTROVERSIAL_LABEL_ID:
        return None
    return expected_name


def _parse_detailed_segments(raw_detail: Any, *, row_idx: int) -> list[dict[str, Any]]:
    detail = _safe_json_loads(raw_detail)
    if not isinstance(detail, dict):
        raise AnnotationValidationError(f"[row {row_idx}] Detailed annotations is not a JSON object.")

    segments = detail.get("segments")
    if not isinstance(segments, list):
        raise AnnotationValidationError(f"[row {row_idx}] Detailed annotations.segments is not a list.")

    parsed: list[dict[str, Any]] = []
    for raw_segment in segments:
        if not isinstance(raw_segment, dict):
            raise AnnotationValidationError(f"[row {row_idx}] Detailed annotations.segments contains a non-object item.")

        try:
            segment_idx = int(raw_segment["segment_idx"])
            char_start = int(raw_segment["char_start"])
            char_end = int(raw_segment["char_end"])
        except Exception as exc:
            raise AnnotationValidationError(
                f"[row {row_idx}] segment metadata is missing or invalid: {raw_segment!r}"
            ) from exc

        if char_start < 0 or char_end < char_start:
            raise AnnotationValidationError(
                f"[row {row_idx}] segment {segment_idx} has invalid char range [{char_start}, {char_end})."
            )

        raw_span = raw_segment.get("span")
        if not isinstance(raw_span, str) or not raw_span:
            raise AnnotationValidationError(f"[row {row_idx}] segment {segment_idx} is missing span text.")

        safety_label = _parse_segment_label(raw_segment, row_idx, segment_idx)
        if safety_label is None:
            continue

        reasoning = raw_segment.get("reasoning")
        if not isinstance(reasoning, dict):
            raise AnnotationValidationError(f"[row {row_idx}] segment {segment_idx} is missing reasoning.")

        reasoning_fields: dict[str, str] = {}
        for field_name in REASONING_FIELD_NAMES:
            field_value = _normalize_output_text(reasoning.get(field_name))
            if field_value is None:
                raise AnnotationValidationError(
                    f"[row {row_idx}] segment {segment_idx} is missing reasoning.{field_name}."
                )
            reasoning_fields[field_name] = field_value

        span_text_for_output = _normalize_output_text(raw_span)
        if span_text_for_output is None:
            raise AnnotationValidationError(f"[row {row_idx}] segment {segment_idx} has empty normalized span text.")

        parsed.append(
            {
                "segment_idx": segment_idx,
                "index": char_start,
                "char_end": char_end,
                "span": raw_span,
                "span_text_for_output": span_text_for_output,
                "safety_label": safety_label,
                "label_id": SAFE_LABEL_ID if safety_label == "safe" else UNSAFE_LABEL_ID,
                "reasoning_fields": reasoning_fields,
            }
        )

    return parsed


def _extract_user_text_from_conversation(raw_conv: Any) -> str:
    conv = _safe_json_loads(raw_conv)
    if conv is None:
        return "(user request text unavailable)"

    messages = None
    if isinstance(conv, dict):
        messages = conv.get("messages") or conv.get("conversation")
    elif isinstance(conv, list):
        messages = conv
    if not isinstance(messages, list):
        return "(user request text unavailable)"

    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        if role != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            normalized = _normalize_output_text(content)
            if normalized:
                parts.append(normalized)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().lower()
            if item_type != "text":
                continue
            normalized = _normalize_output_text(item.get("text"))
            if normalized:
                parts.append(normalized)

    combined = _normalize_output_text(" ".join(parts))
    return combined or "(user request text unavailable)"


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
        raise AnnotationValidationError(
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
    dataset_root: str,
    max_length: Optional[int],
    row_idx: int,
    answergen1_system_prompt: Optional[str],
    answergen1_force_rgb: bool,
    processor_error: Optional[str],
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

    user_text_for_output = _extract_user_text_from_conversation(row.get(text_key))
    segments = _parse_detailed_segments(row.get(detail_key), row_idx=row_idx)
    if not segments:
        return None

    pos_map = support._build_pos_map(
        processor=processor,
        conversation=text_state["conversation_full"],
        full_text=text_state["full_text"],
        image_path=text_state["image_path"],
        text_input_ids=text_state["text_input_ids"],
        max_length=max_length,
        row_idx=row_idx,
        answergen1_force_rgb=answergen1_force_rgb,
        require_image_processor=True,
        processor_error=processor_error,
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
            raise AnnotationValidationError(
                f"[row {row_idx}] segment {segment['segment_idx']} could not be aligned to text tokens."
            )

        token_indices = [pos_map[idx] for idx in text_indices if idx in pos_map]
        if len(token_indices) != len(text_indices):
            raise AnnotationValidationError(
                f"[row {row_idx}] segment {segment['segment_idx']} could not be aligned after multimodal remapping."
            )

        spans.append(
            {
                "segment_idx": segment["segment_idx"],
                "span_text_for_output": segment["span_text_for_output"],
                "safety_label": segment["safety_label"],
                "label_id": segment["label_id"],
                "user_text_for_output": user_text_for_output,
                "reasoning_fields": dict(segment["reasoning_fields"]),
                "reasoning_text": _build_reasoning_text(segment, user_text_for_output),
                "token_indices": token_indices,
            }
        )

    return {
        "row_id": row.get("id"),
        "full_text": text_state["full_text"],
        "image_path": text_state["image_path"],
        "conversation_full": text_state["conversation_full"],
        "spans": spans,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert assess-only CSV to PT with span-level safe/unsafe targets.")
    parser.add_argument("input_csv", help="Path to BeaverTails-V or dataset_B assess-only CSV")
    parser.add_argument("output_pt", help="Path to output PT manifest or packed shard")
    parser.add_argument("--dataset-root", required=True, help="Root directory used to resolve relative image paths")
    parser.add_argument("--model_path", required=True, help="HuggingFace model path or hub ID")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--layers", nargs="+", default=None)
    parser.add_argument("--layer_id", type=int, default=None)
    parser.add_argument("--layer_ids", nargs="+", type=int, default=None)
    parser.add_argument("--response-modes", nargs="+", choices=ANSWER_MODES, default=list(ANSWER_MODES))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text_key", default="conversation")
    parser.add_argument("--detail_key", default="Detailed annotations")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--max_entries", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=4096)
    parser.add_argument("--single-file-output", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answergen1_system_prompt", default=support.ANSWERGEN1_DEFAULT_SYS_MSG)
    parser.add_argument("--answergen1_force_rgb", action="store_true")
    parser.add_argument(
        "--skip-invalid-annotations",
        action="store_true",
        help="Skip rows whose detailed annotations cannot be aligned to the model tokenizer.",
    )
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
    row_infos: list[Optional[dict[str, Any]]] = []
    specs_by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rng = random.Random(args.seed)

    print(f"Processing: {args.input_csv} -> {args.output_pt}", file=sys.stderr)
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
            detail = _safe_json_loads(row.get(args.detail_key))
            if not isinstance(detail, dict):
                raise AnnotationValidationError(f"[row {row_idx}] Detailed annotations is not a JSON object.")
            segments = detail.get("segments")
            if not isinstance(segments, list):
                raise AnnotationValidationError(f"[row {row_idx}] Detailed annotations.segments is not a list.")
            controversial_segments += sum(
                1 for segment in segments if isinstance(segment, dict) and int(segment.get("label_id", -1)) == CONTROVERSIAL_LABEL_ID
            )

            try:
                info = _build_span_row_info(
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
            except AnnotationValidationError as exc:
                if not args.skip_invalid_annotations:
                    raise
                print(f"WARNING: skipping invalid annotation row: {exc}", file=sys.stderr)
                row_infos.append(None)
                empty_rows += 1
                continue
            row_infos.append(info)
            if info is None:
                empty_rows += 1
                continue

            kept_segments += len(info["spans"])
            for span_idx, span in enumerate(info["spans"]):
                for response_mode in args.response_modes:
                    specs_by_row[row_idx].append(
                        {
                            "span_idx": span_idx,
                            "response_mode": response_mode,
                            "target_response": _build_target_response(span, response_mode),
                        }
                    )

    writer = common.PtShardWriter(
        args.output_pt,
        shard_size=args.shard_size,
        activation_dtype=activation_storage_dtype,
        single_file_output=args.single_file_output,
    )
    feature_idx_counter = 0

    row_indices = list(specs_by_row.keys())
    rng.shuffle(row_indices)
    row_iter = row_indices
    if support.tqdm is not None:
        row_iter = support.tqdm(row_indices, desc="Pass 2: collect PT datapoints", unit="row")

    for row_idx in row_iter:
        if max_entries is not None and writer.total_entries >= max_entries:
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

        spans = info["spans"]
        for layer_path in layer_paths:
            if max_entries is not None and writer.total_entries >= max_entries:
                break

            hidden = hidden_by_layer.get(layer_path)
            if hidden is None:
                raise RuntimeError(f"[row {row_idx}] missing captured activations for layer '{layer_path}'.")

            layer_id = layer_id_by_path[layer_path]
            for spec in specs_by_row[row_idx]:
                if max_entries is not None and writer.total_entries >= max_entries:
                    break

                span_idx = int(spec["span_idx"])
                if not (0 <= span_idx < len(spans)):
                    raise IndexError(f"[row {row_idx}] invalid span_idx={span_idx} for {len(spans)} spans.")

                span = spans[span_idx]
                token_indices = [int(idx) for idx in span["token_indices"]]
                if any(idx < 0 or idx >= hidden.shape[1] for idx in token_indices):
                    raise IndexError(
                        f"[row {row_idx}] span {span_idx} token positions exceed activation length {hidden.shape[1]}."
                    )

                idx_tensor = torch.tensor(token_indices, dtype=torch.long)
                acts_BD = hidden[0][idx_tensor].detach().to(
                    device="cpu",
                    dtype=activation_storage_dtype,
                ).contiguous()

                dp = common.create_training_datapoint(
                    datapoint_type="custom_pt_span",
                    prompt=PROMPTS[str(spec["response_mode"])],
                    target_response=str(spec["target_response"]),
                    layer=layer_id,
                    num_positions=acts_BD.shape[0],
                    tokenizer=tokenizer,
                    acts_BD=acts_BD,
                    feature_idx=feature_idx_counter,
                    ds_label=None,
                    meta_info={
                        "row_idx": row_idx,
                        "row_id": info["row_id"],
                        "segment_idx": span["segment_idx"],
                        "layer_path": layer_path,
                        "prompt_text": PROMPTS[str(spec["response_mode"])],
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
                feature_idx_counter += 1

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
            "response_modes": list(args.response_modes),
            "seed": int(args.seed),
            "activation_storage_dtype": args.activation_storage_dtype,
            "variant": "part_span_assess_only",
            "labels": ["safe", "unsafe"],
        }
    )
    print(
        f"\nDone. Rows processed: {total_rows}, rows without safe/unsafe spans: {empty_rows}, "
        f"controversial segments dropped: {controversial_segments}, non-controversial spans kept: {kept_segments}, "
        f"PT entries written: {writer.total_entries}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
