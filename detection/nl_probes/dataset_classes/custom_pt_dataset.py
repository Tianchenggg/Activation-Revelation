import os
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    BaseDatasetConfig,
    DatasetLoaderConfig,
)
from nl_probes.dataset_classes.custom_pt_serialization import (
    get_training_datapoint_record_from_packed_shard,
    is_custom_pt_manifest,
    iter_training_datapoint_records_from_packed_shard,
    is_packed_custom_pt_shard,
)
from nl_probes.utils.common import load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint
from nl_probes.utils.dataset_utils import (
    get_introspection_prefix,
    normalize_chat_template_token_ids,
)

CUSTOM_PT_DATASET_BUILD_VERSION = 1
_TOKENIZER_CACHE: dict[str, Any] = {}


def _normalize_meta_info(raw_meta: Any) -> dict[str, Any]:
    return raw_meta if isinstance(raw_meta, dict) else {}


def _normalize_model_name_for_compare(model_name: str) -> str:
    raw = str(model_name or "").strip()
    if not raw:
        return raw
    model_path = Path(raw).expanduser()
    if model_path.exists():
        return str(model_path.resolve())
    return raw


def _should_retokenize_record_for_model(record: Any, model_name: str) -> bool:
    meta_info = _normalize_meta_info(_record_get(record, "meta_info"))
    prompt_text = meta_info.get("prompt_text")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return False

    prompt_model_name = str(
        meta_info.get("prompt_tokenizer_model")
        or meta_info.get("prompt_source_model")
        or meta_info.get("source_model_name")
        or ""
    ).strip()
    if not prompt_model_name:
        return True

    return _normalize_model_name_for_compare(prompt_model_name) != _normalize_model_name_for_compare(model_name)


def _retokenize_materialized_record(
    record: Any,
    *,
    model_name: str,
    datapoint_type: str,
    record_index: int,
) -> TrainingDataPoint:
    meta_info = dict(_normalize_meta_info(_record_get(record, "meta_info")))
    prompt_text = str(meta_info["prompt_text"])
    acts_BD = _activation_to_tensor(_record_get(record, "steering_vectors"))
    tokenizer = _get_cached_tokenizer(model_name)

    meta_info["retokenized_for_model"] = model_name
    return create_training_datapoint(
        datapoint_type=str(_record_get(record, "datapoint_type", datapoint_type) or datapoint_type),
        prompt=prompt_text,
        target_response=str(_record_get(record, "target_output", "")),
        layer=int(_record_get(record, "layer")),
        num_positions=acts_BD.shape[0],
        tokenizer=tokenizer,
        acts_BD=acts_BD,
        feature_idx=int(_record_get(record, "feature_idx", record_index)),
        ds_label=_record_get(record, "ds_label"),
        meta_info=meta_info,
    )


@dataclass
class CustomPtDatasetConfig(BaseDatasetConfig):
    """Config for loading custom activation-oracle datasets from PT files."""

    pt_train_path: str = ""
    pt_test_path: str = ""
    pt_train_signature: str = ""
    pt_test_signature: str = ""
    build_version: int = CUSTOM_PT_DATASET_BUILD_VERSION


def _file_stat_signature(path: str) -> str:
    stat = os.stat(path)
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _is_materialized_datapoint_dict(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    required = {"input_ids", "labels", "positions", "layer", "feature_idx", "target_output"}
    return required.issubset(record.keys())


def _activation_to_tensor(activation: Any) -> torch.Tensor:
    if isinstance(activation, torch.Tensor):
        acts_BD = activation.detach().to(dtype=torch.float32, device="cpu")
    else:
        acts_BD = torch.tensor(activation, dtype=torch.float32)
    if acts_BD.ndim == 1:
        acts_BD = acts_BD.unsqueeze(0)
    if acts_BD.ndim != 2:
        raise ValueError(f"'activation' must be 2-D, got shape={list(acts_BD.shape)}")
    return acts_BD


def _get_cached_tokenizer(model_name: str):
    tokenizer = _TOKENIZER_CACHE.get(model_name)
    if tokenizer is None:
        tokenizer = load_tokenizer(model_name)
        _TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer


def _torch_load_cpu(path: Path, *, mmap: bool = False) -> Any:
    load_kwargs = {"map_location": "cpu"}
    if mmap:
        load_kwargs["mmap"] = True

    try:
        return torch.load(path, weights_only=False, **load_kwargs)
    except TypeError:
        load_kwargs.pop("mmap", None)
        try:
            return torch.load(path, weights_only=False, **load_kwargs)
        except TypeError:
            return torch.load(path, **load_kwargs)


def _record_get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _maybe_item(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _sequence_value(values: Any, index: int, default: Any = None) -> Any:
    if values is None:
        return default
    try:
        return values[index]
    except (IndexError, KeyError, TypeError):
        return default


def _ragged_sequence_length(packed: Any, index: int) -> int:
    if not isinstance(packed, dict) or "offsets" not in packed:
        raise ValueError("Expected packed ragged field with offsets.")
    offsets = packed["offsets"]
    start = int(_maybe_item(offsets[index]))
    end = int(_maybe_item(offsets[index + 1]))
    return end - start


def _packed_activation_length(pt_obj: dict[str, Any], index: int) -> int:
    steering_vectors = pt_obj.get("steering_vectors")
    if not isinstance(steering_vectors, dict) or "offsets" not in steering_vectors:
        raise ValueError("Packed shard is missing steering vector offsets.")
    return _ragged_sequence_length(steering_vectors, index)


def _retokenized_input_length_from_packed_record(
    pt_obj: dict[str, Any],
    index: int,
    *,
    tokenizer,
    length_cache: dict[tuple[str, int, int, str], int] | None = None,
) -> int:
    meta_info = _normalize_meta_info(_sequence_value(pt_obj.get("meta_info"), index, {}))
    prompt_text = meta_info.get("prompt_text")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("Cannot retokenize packed record length without meta_info['prompt_text'].")

    layer = int(_maybe_item(_sequence_value(pt_obj.get("layer"), index)))
    target_output = str(_sequence_value(pt_obj.get("target_output"), index, ""))
    try:
        num_positions = _ragged_sequence_length(pt_obj.get("positions"), index)
    except ValueError:
        num_positions = _packed_activation_length(pt_obj, index)

    cache_key = (prompt_text, layer, num_positions, target_output)
    if length_cache is not None and cache_key in length_cache:
        return length_cache[cache_key]

    prompt = get_introspection_prefix(layer, num_positions) + prompt_text
    full_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target_output},
    ]
    full_prompt_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    input_length = len(normalize_chat_template_token_ids(full_prompt_ids, "full_prompt_ids"))
    if length_cache is not None:
        length_cache[cache_key] = input_length
    return input_length


def _packed_record_input_length_for_model(
    pt_obj: dict[str, Any],
    index: int,
    *,
    model_name: str,
    tokenizer,
    length_cache: dict[tuple[str, int, int, str], int] | None = None,
) -> int:
    meta_info = _normalize_meta_info(_sequence_value(pt_obj.get("meta_info"), index, {}))
    if _should_retokenize_record_for_model({"meta_info": meta_info}, model_name):
        return _retokenized_input_length_from_packed_record(
            pt_obj,
            index,
            tokenizer=tokenizer,
            length_cache=length_cache,
        )
    return _ragged_sequence_length(pt_obj.get("input_ids"), index)


def _materialize_training_datapoint(
    record: Any,
    *,
    model_name: str,
    datapoint_type: str,
    record_index: int,
) -> TrainingDataPoint:
    if _should_retokenize_record_for_model(record, model_name):
        return _retokenize_materialized_record(
            record,
            model_name=model_name,
            datapoint_type=datapoint_type,
            record_index=record_index,
        )

    if isinstance(record, TrainingDataPoint):
        return record

    if _is_materialized_datapoint_dict(record):
        return TrainingDataPoint(**record)

    if not isinstance(record, dict):
        raise TypeError(
            f"Unsupported PT record type at index {record_index}: {type(record).__name__}"
        )

    for required_key in ("question", "answer", "activation", "layer"):
        if required_key not in record:
            raise KeyError(
                f"Missing key '{required_key}' in PT record #{record_index}"
            )

    tokenizer = _get_cached_tokenizer(model_name)
    acts_BD = _activation_to_tensor(record["activation"])

    raw_meta = record.get("meta_info")
    meta_info = _normalize_meta_info(raw_meta)

    return create_training_datapoint(
        datapoint_type=datapoint_type,
        prompt=str(record["question"]),
        target_response=str(record["answer"]),
        layer=int(record["layer"]),
        num_positions=acts_BD.shape[0],
        tokenizer=tokenizer,
        acts_BD=acts_BD,
        feature_idx=int(record.get("feature_idx", record_index)),
        ds_label=record.get("ds_label"),
        meta_info=meta_info,
    )


def _iter_records_from_pt_object(pt_obj: Any, source_path: Path):
    if is_packed_custom_pt_shard(pt_obj):
        yield from iter_training_datapoint_records_from_packed_shard(pt_obj)
        return

    if isinstance(pt_obj, dict):
        if "shards" in pt_obj:
            for shard in pt_obj["shards"]:
                if isinstance(shard, dict):
                    shard_rel = shard.get("path")
                else:
                    shard_rel = shard
                if not isinstance(shard_rel, str) or not shard_rel.strip():
                    raise ValueError(f"Invalid shard entry in manifest: {shard!r}")
                shard_path = (source_path.parent / shard_rel).resolve()
                shard_obj = _torch_load_cpu(shard_path)
                yield from _iter_records_from_pt_object(shard_obj, shard_path)
            return

        if "data" in pt_obj:
            data = pt_obj["data"]
            if not isinstance(data, list):
                raise ValueError(f"'data' must be a list in {source_path}")
            for record in data:
                yield record
            return

    if isinstance(pt_obj, list):
        for record in pt_obj:
            yield record
        return

    raise ValueError(
        f"Unsupported PT dataset format in {source_path}. "
        "Expected a packed shard, a dict with 'data' or 'shards', or a top-level list."
    )


@dataclass(frozen=True)
class _LazyShardSpec:
    path: Path
    num_entries: int


class LazyCustomPtDataset(Dataset[TrainingDataPoint]):
    """Map-style dataset that loads packed PT shards on demand."""

    def __init__(
        self,
        *,
        pt_path: str,
        model_name: str,
        datapoint_type: str,
        max_records: int = 0,
        max_cached_shards: int = 8,
    ) -> None:
        self.source_path = Path(pt_path).resolve()
        if not self.source_path.exists():
            raise FileNotFoundError(f"PT dataset file not found: {self.source_path}")

        self.model_name = model_name
        self.datapoint_type = datapoint_type
        self.max_records = max(0, int(max_records))
        self.max_cached_shards = max(1, int(max_cached_shards))
        self._tokenizer = None
        self._shard_specs: list[_LazyShardSpec] = []
        self._shard_start_indices: list[int] = []
        self._shard_cache: OrderedDict[int, Any] = OrderedDict()
        self._input_length_cache: dict[tuple[str, int, int, str], int] = {}

        self._initialize_shards()
        self._total_records = sum(spec.num_entries for spec in self._shard_specs)
        if self._total_records <= 0:
            raise ValueError(f"No usable datapoints found in {self.source_path}")

    def _initialize_shards(self) -> None:
        root_obj = _torch_load_cpu(self.source_path, mmap=True)

        if is_packed_custom_pt_shard(root_obj):
            num_entries = int(root_obj["num_entries"])
            if self.max_records > 0:
                num_entries = min(num_entries, self.max_records)
            if num_entries > 0:
                self._shard_specs = [_LazyShardSpec(path=self.source_path, num_entries=num_entries)]
                self._shard_start_indices = [0]
                self._shard_cache[0] = root_obj
            return

        if not is_custom_pt_manifest(root_obj):
            raise ValueError(
                "Lazy PT loading requires a packed shard or manifest produced by the PT converters. "
                f"Received unsupported format in {self.source_path}."
            )

        shard_stats = root_obj.get("shard_stats")
        shard_entries = root_obj.get("shards")
        if not isinstance(shard_entries, list) or not shard_entries:
            raise ValueError(f"Manifest has no shards: {self.source_path}")

        remaining = self.max_records if self.max_records > 0 else None
        specs: list[_LazyShardSpec] = []

        for shard_index, shard_entry in enumerate(shard_entries):
            if remaining is not None and remaining <= 0:
                break

            if isinstance(shard_entry, dict):
                shard_rel = shard_entry.get("path")
                num_entries = shard_entry.get("num_entries")
            else:
                shard_rel = shard_entry
                num_entries = None

            if not isinstance(shard_rel, str) or not shard_rel.strip():
                raise ValueError(f"Invalid shard entry in manifest: {shard_entry!r}")

            if num_entries is None and isinstance(shard_stats, list) and shard_index < len(shard_stats):
                shard_stat = shard_stats[shard_index]
                if isinstance(shard_stat, dict):
                    num_entries = shard_stat.get("num_entries")

            shard_path = (self.source_path.parent / shard_rel).resolve()
            if num_entries is None:
                shard_obj = _torch_load_cpu(shard_path, mmap=True)
                if not is_packed_custom_pt_shard(shard_obj):
                    raise ValueError(f"Shard is not in packed custom PT format: {shard_path}")
                num_entries = int(shard_obj["num_entries"])

            shard_num_entries = int(num_entries)
            if remaining is not None:
                shard_num_entries = min(shard_num_entries, remaining)

            if shard_num_entries > 0:
                specs.append(_LazyShardSpec(path=shard_path, num_entries=shard_num_entries))
                if remaining is not None:
                    remaining -= shard_num_entries

        self._shard_specs = specs

        start_index = 0
        for spec in self._shard_specs:
            self._shard_start_indices.append(start_index)
            start_index += spec.num_entries

    def __len__(self) -> int:
        return self._total_records

    def __iter__(self) -> Iterator[TrainingDataPoint]:
        for index in range(len(self)):
            yield self[index]

    def _load_shard(self, shard_index: int) -> Any:
        cached = self._shard_cache.get(shard_index)
        if cached is not None:
            self._shard_cache.move_to_end(shard_index)
            return cached

        shard_spec = self._shard_specs[shard_index]
        shard_obj = _torch_load_cpu(shard_spec.path, mmap=True)
        if not is_packed_custom_pt_shard(shard_obj):
            raise ValueError(f"Shard is not in packed custom PT format: {shard_spec.path}")

        self._shard_cache[shard_index] = shard_obj
        self._shard_cache.move_to_end(shard_index)
        while len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)
        return shard_obj

    def _resolve_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _get_cached_tokenizer(self.model_name)
        return self._tokenizer

    def _getitem_int(self, index: int) -> TrainingDataPoint:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}")

        shard_index = bisect_right(self._shard_start_indices, index) - 1
        shard_start = self._shard_start_indices[shard_index]
        local_index = index - shard_start
        shard_obj = self._load_shard(shard_index)
        record = get_training_datapoint_record_from_packed_shard(shard_obj, local_index)

        if isinstance(record, dict) and "question" in record and "answer" in record and "activation" in record:
            tokenizer = self._resolve_tokenizer()
            acts_BD = _activation_to_tensor(record["activation"])
            raw_meta = record.get("meta_info")
            meta_info = _normalize_meta_info(raw_meta)
            return create_training_datapoint(
                datapoint_type=self.datapoint_type,
                prompt=str(record["question"]),
                target_response=str(record["answer"]),
                layer=int(record["layer"]),
                num_positions=acts_BD.shape[0],
                tokenizer=tokenizer,
                acts_BD=acts_BD,
                feature_idx=int(record.get("feature_idx", index)),
                ds_label=record.get("ds_label"),
                meta_info=meta_info,
            )

        return _materialize_training_datapoint(
            record,
            model_name=self.model_name,
            datapoint_type=self.datapoint_type,
            record_index=index,
        )

    def get_input_length(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}")

        shard_index = bisect_right(self._shard_start_indices, index) - 1
        shard_start = self._shard_start_indices[shard_index]
        local_index = index - shard_start
        shard_obj = self._load_shard(shard_index)
        tokenizer = self._resolve_tokenizer()
        return _packed_record_input_length_for_model(
            shard_obj,
            local_index,
            model_name=self.model_name,
            tokenizer=tokenizer,
            length_cache=self._input_length_cache,
        )

    def __getitem__(self, index: int | slice) -> TrainingDataPoint | list[TrainingDataPoint]:
        if isinstance(index, slice):
            return [self._getitem_int(i) for i in range(*index.indices(len(self)))]
        return self._getitem_int(index)


class TransformedTrainingDataPointDataset(Dataset[TrainingDataPoint]):
    """Dataset wrapper that applies a transform per item without materializing the full dataset."""

    def __init__(self, base_dataset: Any, transform) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __iter__(self) -> Iterator[TrainingDataPoint]:
        for index in range(len(self)):
            yield self[index]

    def _normalize_item(self, item: Any) -> TrainingDataPoint:
        if isinstance(item, TrainingDataPoint):
            return item
        return TrainingDataPoint(**item)

    def __getitem__(self, index: int | slice) -> TrainingDataPoint | list[TrainingDataPoint]:
        item = self.base_dataset[index]
        if isinstance(index, slice):
            return [self.transform(self._normalize_item(entry)) for entry in item]
        return self.transform(self._normalize_item(item))


def load_training_dataset_from_pt(
    *,
    pt_path: str,
    model_name: str,
    datapoint_type: str,
    max_records: int = 0,
    prefer_lazy: bool = True,
    max_cached_shards: int = 8,
) -> Dataset[TrainingDataPoint] | list[TrainingDataPoint]:
    if prefer_lazy:
        try:
            return LazyCustomPtDataset(
                pt_path=pt_path,
                model_name=model_name,
                datapoint_type=datapoint_type,
                max_records=max_records,
                max_cached_shards=max_cached_shards,
            )
        except ValueError as exc:
            print(f"WARNING: {exc} Falling back to eager PT loading for {pt_path}.")

    return load_training_datapoints_from_pt(
        pt_path=pt_path,
        model_name=model_name,
        datapoint_type=datapoint_type,
        max_records=max_records,
    )


def load_training_datapoints_from_pt(
    *,
    pt_path: str,
    model_name: str,
    datapoint_type: str,
    max_records: int = 0,
) -> list[TrainingDataPoint]:
    source_path = Path(pt_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"PT dataset file not found: {source_path}")

    pt_obj = _torch_load_cpu(source_path)
    data: list[TrainingDataPoint] = []

    for idx, record in enumerate(_iter_records_from_pt_object(pt_obj, source_path)):
        data.append(
            _materialize_training_datapoint(
                record,
                model_name=model_name,
                datapoint_type=datapoint_type,
                record_index=idx,
            )
        )

        if max_records > 0 and len(data) >= max_records:
            break

    if not data:
        raise ValueError(f"No usable datapoints found in {source_path}")
    return data


class CustomPtDatasetLoader(ActDatasetLoader):
    """Dataset loader that consumes prebuilt PT datasets directly."""

    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        self.dataset_config.dataset_name = "custom_pt"
        self._custom_cfg: CustomPtDatasetConfig = dataset_config.custom_dataset_params
        if self._custom_cfg.pt_train_path:
            self._custom_cfg.pt_train_signature = _file_stat_signature(self._custom_cfg.pt_train_path)
        if self._custom_cfg.pt_test_path:
            self._custom_cfg.pt_test_signature = _file_stat_signature(self._custom_cfg.pt_test_path)

    def create_dataset(self) -> None:
        raise RuntimeError(
            "CustomPtDatasetLoader reads existing PT files directly and does not support create_dataset()."
        )

    def _split_path(self, split: str) -> str:
        if split == "train":
            return self._custom_cfg.pt_train_path
        return self._custom_cfg.pt_test_path

    def _split_limit(self, split: str) -> int:
        if split == "train":
            return max(0, int(self.dataset_config.num_train))
        return max(0, int(self.dataset_config.num_test))

    def load_dataset(self, split: str) -> list[TrainingDataPoint]:
        path = self._split_path(split)
        if not path:
            raise ValueError(
                f"CustomPtDatasetConfig.pt_{split}_path is empty but split '{split}' was requested."
            )

        data = load_training_datapoints_from_pt(
            pt_path=path,
            model_name=self.dataset_config.model_name,
            datapoint_type=self.dataset_config.dataset_name,
            max_records=self._split_limit(split),
        )
        print(f"Loaded {len(data)} datapoints from {path}")
        return data
