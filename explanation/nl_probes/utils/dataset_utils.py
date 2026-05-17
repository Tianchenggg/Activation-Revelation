from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nl_probes.utils.vlm_compat import apply_chat_template

SPECIAL_TOKEN = " ?"
REMOVED_ACTIVATION_SOURCES = {"image_bbox_tokens"}


def validate_supported_activation_source(meta_info: Mapping[str, Any] | None, *, context: str = "") -> None:
    if not meta_info:
        return
    activation_source = str(meta_info.get("activation_source") or "").strip()
    if activation_source in REMOVED_ACTIVATION_SOURCES:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}activation_source={activation_source!r} has been removed because it selects activations "
            "using the target bbox and can leak the label. Regenerate the PT dataset with span_tokens."
        )


def normalize_chat_template_token_ids(token_ids: Any, field_name: str) -> list[int]:
    original = token_ids

    if isinstance(token_ids, Mapping):
        if "input_ids" not in token_ids:
            raise TypeError(
                f"Expected {field_name} to contain 'input_ids', got keys: {list(token_ids.keys())}"
            )
        token_ids = token_ids["input_ids"]

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise TypeError(
                f"Expected single sequence for {field_name}, got batched output with length {len(token_ids)}"
            )
        token_ids = token_ids[0]

    if not isinstance(token_ids, list) or any(not isinstance(token_id, int) for token_id in token_ids):
        preview = repr(original)
        if len(preview) > 220:
            preview = preview[:220] + "..."
        raise TypeError(
            f"Expected list[int] token ids for {field_name}, got {type(original).__name__}. "
            f"Value preview: {preview}"
        )
    return token_ids


def get_introspection_prefix(sae_layer: int, num_positions: int) -> str:
    return f"Layer: {sae_layer}\n" + (SPECIAL_TOKEN * num_positions) + " \n"


def build_vlm_prompt_messages(
    *,
    prompt_text: str,
    image_path: str,
    layer: int,
    num_positions: int,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    prompt_with_prefix = get_introspection_prefix(layer, num_positions) + str(prompt_text)
    messages: list[dict[str, Any]] = []
    if system_prompt is not None and str(system_prompt).strip():
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": str(system_prompt)}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt_with_prefix},
            ],
        }
    )
    return messages


def clone_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for message in messages:
        new_message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            new_message["content"] = [dict(item) if isinstance(item, dict) else item for item in content]
        cloned.append(new_message)
    return cloned


def build_full_messages(prompt_messages: list[dict[str, Any]], target_output: str) -> list[dict[str, Any]]:
    messages = clone_messages(prompt_messages)
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(target_output)}],
        }
    )
    return messages


def build_empty_assistant_messages(prompt_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return build_full_messages(prompt_messages, "")


class FeatureResult(BaseModel):
    feature_idx: int
    api_response: str
    prompt: str
    meta_info: Mapping[str, Any] = {}


class EvalStepResult(BaseModel):
    step: int
    results: list[FeatureResult]


class TrainingDataPoint(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    datapoint_type: str
    prompt_messages: list[dict[str, Any]]
    layer: int
    steering_vectors: torch.Tensor
    num_positions: int
    feature_idx: int
    target_output: str
    prompt_only: bool = False
    ds_label: str | None = None
    meta_info: Mapping[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_multimodal_fields(cls, values):
        if not values.prompt_messages:
            raise ValueError("prompt_messages must be non-empty")
        if values.num_positions <= 0:
            raise ValueError("num_positions must be > 0")
        acts_BD = values.steering_vectors
        if not isinstance(acts_BD, torch.Tensor):
            raise ValueError("steering_vectors must be a torch.Tensor")
        if acts_BD.ndim != 2:
            raise ValueError(f"steering_vectors must be 2-D, got shape={list(acts_BD.shape)}")
        if int(acts_BD.shape[0]) != int(values.num_positions):
            raise ValueError(
                "num_positions and steering_vectors.shape[0] must match: "
                f"{values.num_positions} vs {int(acts_BD.shape[0])}"
            )
        validate_supported_activation_source(values.meta_info, context=f"feature_idx={values.feature_idx}")
        return values


class BatchData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    steering_vectors: list[torch.Tensor]
    positions: list[list[int]]
    feature_indices: list[int]
    position_ids: torch.Tensor | None = None
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    mm_token_type_ids: torch.Tensor | None = None
    images: torch.Tensor | None = None
    aspect_ratio_ids: torch.Tensor | None = None
    aspect_ratio_mask: torch.Tensor | None = None
    cross_attention_mask: torch.Tensor | None = None


def get_prompt_tokens_only(training_data_point: TrainingDataPoint) -> TrainingDataPoint:
    updated = training_data_point.model_copy(deep=False)
    updated.prompt_only = True
    return updated


def find_pattern_in_tokens(
    token_ids: list[int],
    special_token_str: str,
    num_positions: int,
    tokenizer,
) -> list[int]:
    special_token_id = tokenizer.encode(special_token_str, add_special_tokens=False)
    if len(special_token_id) != 1:
        raise ValueError(f"Expected '{special_token_str}' to map to a single token, got {special_token_id}")
    special_token_id = special_token_id[0]

    positions: list[int] = []
    for index, token_id in enumerate(token_ids):
        if len(positions) == num_positions:
            break
        if token_id == special_token_id:
            positions.append(index)

    if len(positions) != num_positions:
        raise ValueError(f"Expected {num_positions} positions, got {len(positions)}")
    if positions[-1] - positions[0] != num_positions - 1:
        raise ValueError(f"Positions are not consecutive: {positions}")

    final_pos = positions[-1] + 1
    final_tokens = token_ids[final_pos : final_pos + 2]
    final_str = tokenizer.decode(final_tokens, skip_special_tokens=False)
    if "\n" not in final_str:
        raise ValueError(f"Expected newline after introspection prefix, got {final_str!r}")
    return positions


def create_vlm_training_datapoint(
    *,
    datapoint_type: str,
    prompt_messages: list[dict[str, Any]],
    target_response: str,
    layer: int,
    num_positions: int,
    acts_BD: torch.Tensor,
    feature_idx: int,
    ds_label: str | None = None,
    meta_info: Mapping[str, Any] | None = None,
) -> TrainingDataPoint:
    if meta_info is None:
        meta_info = {}

    acts_BD = acts_BD.detach().to(device="cpu")
    if acts_BD.ndim != 2:
        raise ValueError(f"Expected acts_BD to be 2-D, got shape={list(acts_BD.shape)}")
    if int(acts_BD.shape[0]) != int(num_positions):
        raise ValueError(
            f"num_positions ({num_positions}) must equal acts_BD.shape[0] ({int(acts_BD.shape[0])})"
        )

    return TrainingDataPoint(
        datapoint_type=datapoint_type,
        prompt_messages=clone_messages(prompt_messages),
        layer=int(layer),
        steering_vectors=acts_BD.contiguous(),
        num_positions=int(num_positions),
        feature_idx=int(feature_idx),
        target_output=str(target_response),
        prompt_only=False,
        ds_label=ds_label,
        meta_info=dict(meta_info),
    )


def estimate_sequence_length(
    *,
    data_point: TrainingDataPoint,
    processor,
    image_max_pixels: int | None,
) -> int:
    if data_point.prompt_only:
        messages = clone_messages(data_point.prompt_messages)
        encoded = apply_chat_template(
            processor,
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors=None,
            padding=False,
            max_pixels=image_max_pixels,
        )
    else:
        messages = build_full_messages(data_point.prompt_messages, data_point.target_output)
        encoded = apply_chat_template(
            processor,
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors=None,
            padding=False,
            max_pixels=image_max_pixels,
        )
    return len(normalize_chat_template_token_ids(encoded, "estimate_sequence_length"))


def create_training_datapoint(*args, **kwargs):
    raise RuntimeError(
        "The text-only create_training_datapoint interface has been removed for this VLM training path. "
        "Use create_vlm_training_datapoint instead."
    )
