from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

CUSTOM_PT_MANIFEST_FORMAT = "ao_custom_pt_manifest_v3"
CUSTOM_PT_SHARD_FORMAT = "ao_custom_pt_shard_v3"

_INT_STORAGE_DTYPE = torch.int32
_OFFSET_DTYPE = torch.int64


def _record_get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _as_2d_activation_tensor(values: Any, activation_dtype: torch.dtype) -> torch.Tensor:
    if values is None:
        raise ValueError("steering_vectors cannot be None in packed PT shards.")

    if isinstance(values, torch.Tensor):
        tensor = values.detach().to(device="cpu")
    else:
        tensor = torch.tensor(values)

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"steering_vectors must be 2-D, got shape={list(tensor.shape)}")
    return tensor.to(dtype=activation_dtype).contiguous()


def _pack_required_ragged_activations(records: list[Any], activation_dtype: torch.dtype) -> dict[str, Any]:
    values: list[torch.Tensor] = []
    offsets = [0]
    hidden_dim: int | None = None

    for record in records:
        tensor = _as_2d_activation_tensor(_record_get(record, "steering_vectors"), activation_dtype)
        if hidden_dim is None:
            hidden_dim = int(tensor.shape[1])
        elif int(tensor.shape[1]) != hidden_dim:
            raise ValueError(
                "All steering_vectors in a packed shard must share the same hidden size. "
                f"Expected {hidden_dim}, got {int(tensor.shape[1])}."
            )
        values.append(tensor)
        offsets.append(offsets[-1] + int(tensor.shape[0]))

    packed_values = torch.cat(values, dim=0) if values else torch.empty((0, 0), dtype=activation_dtype)
    return {
        "values": packed_values,
        "offsets": torch.tensor(offsets, dtype=_OFFSET_DTYPE),
        "hidden_dim": hidden_dim or 0,
    }


def pack_training_datapoint_records(records: list[Any], *, activation_dtype: torch.dtype) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot pack an empty PT shard.")

    datapoint_types = [str(_record_get(record, "datapoint_type", "")) for record in records]
    prompt_messages = [list(_record_get(record, "prompt_messages") or []) for record in records]
    target_outputs = [str(_record_get(record, "target_output", "")) for record in records]
    ds_labels = [_record_get(record, "ds_label") for record in records]
    meta_infos = [dict(_record_get(record, "meta_info", {}) or {}) for record in records]
    layers = torch.tensor([int(_record_get(record, "layer")) for record in records], dtype=_INT_STORAGE_DTYPE)
    num_positions = torch.tensor(
        [int(_record_get(record, "num_positions")) for record in records],
        dtype=_INT_STORAGE_DTYPE,
    )
    feature_indices = torch.tensor(
        [int(_record_get(record, "feature_idx")) for record in records],
        dtype=_INT_STORAGE_DTYPE,
    )

    return {
        "format": CUSTOM_PT_SHARD_FORMAT,
        "num_entries": len(records),
        "activation_dtype": str(activation_dtype).replace("torch.", ""),
        "datapoint_type": datapoint_types,
        "prompt_messages": prompt_messages,
        "target_output": target_outputs,
        "ds_label": ds_labels,
        "meta_info": meta_infos,
        "layer": layers,
        "num_positions": num_positions,
        "feature_idx": feature_indices,
        "steering_vectors": _pack_required_ragged_activations(records, activation_dtype),
    }


def is_packed_custom_pt_shard(pt_obj: Any) -> bool:
    return isinstance(pt_obj, dict) and pt_obj.get("format") == CUSTOM_PT_SHARD_FORMAT


def is_custom_pt_manifest(pt_obj: Any) -> bool:
    return isinstance(pt_obj, dict) and pt_obj.get("format") == CUSTOM_PT_MANIFEST_FORMAT


def get_training_datapoint_record_from_packed_shard(pt_obj: dict[str, Any], index: int) -> dict[str, Any]:
    if not is_packed_custom_pt_shard(pt_obj):
        raise ValueError("PT object is not a packed custom PT shard.")

    num_entries = int(pt_obj["num_entries"])
    if index < 0 or index >= num_entries:
        raise IndexError(f"Packed shard index out of range: {index} (num_entries={num_entries})")

    steering_values = pt_obj["steering_vectors"]["values"]
    steering_offsets = pt_obj["steering_vectors"]["offsets"]
    sv_start = int(steering_offsets[index].item())
    sv_end = int(steering_offsets[index + 1].item())

    datapoint_types = pt_obj["datapoint_type"]
    prompt_messages = pt_obj["prompt_messages"]
    target_outputs = pt_obj["target_output"]
    ds_labels = pt_obj["ds_label"]
    meta_infos = pt_obj["meta_info"]
    layers = pt_obj["layer"]
    num_positions = pt_obj["num_positions"]
    feature_indices = pt_obj["feature_idx"]

    return {
        "datapoint_type": str(datapoint_types[index]),
        "prompt_messages": list(prompt_messages[index]),
        "layer": int(layers[index].item()),
        "steering_vectors": steering_values[sv_start:sv_end],
        "num_positions": int(num_positions[index].item()),
        "feature_idx": int(feature_indices[index].item()),
        "target_output": str(target_outputs[index]),
        "prompt_only": False,
        "ds_label": ds_labels[index],
        "meta_info": dict(meta_infos[index] or {}),
    }


def iter_training_datapoint_records_from_packed_shard(pt_obj: dict[str, Any]):
    if not is_packed_custom_pt_shard(pt_obj):
        raise ValueError("PT object is not a packed custom PT shard.")

    num_entries = int(pt_obj["num_entries"])
    for index in range(num_entries):
        yield get_training_datapoint_record_from_packed_shard(pt_obj, index)
