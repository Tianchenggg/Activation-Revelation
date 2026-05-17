from __future__ import annotations

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
    is_packed_custom_pt_shard,
)
from nl_probes.utils.dataset_utils import TrainingDataPoint, validate_supported_activation_source

CUSTOM_PT_DATASET_BUILD_VERSION = 2


@dataclass
class CustomPtDatasetConfig(BaseDatasetConfig):
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
    required = {"prompt_messages", "num_positions", "layer", "feature_idx", "target_output", "steering_vectors"}
    return required.issubset(record.keys())


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


def _validate_pt_object_activation_source(pt_obj: Any, source_path: Path) -> None:
    if not isinstance(pt_obj, dict):
        return
    try:
        validate_supported_activation_source(pt_obj, context=str(source_path))
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _materialize_training_datapoint(record: Any, *, record_index: int) -> TrainingDataPoint:
    if isinstance(record, TrainingDataPoint):
        validate_supported_activation_source(record.meta_info, context=f"record_index={record_index}")
        return record
    if _is_materialized_datapoint_dict(record):
        data_point = TrainingDataPoint(**record)
        validate_supported_activation_source(data_point.meta_info, context=f"record_index={record_index}")
        return data_point
    raise TypeError(f"Unsupported PT record type at index {record_index}: {type(record).__name__}")


def _iter_records_from_pt_object(pt_obj: Any, source_path: Path):
    if is_packed_custom_pt_shard(pt_obj):
        num_entries = int(pt_obj["num_entries"])
        for index in range(num_entries):
            yield get_training_datapoint_record_from_packed_shard(pt_obj, index)
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
    def __init__(
        self,
        *,
        pt_path: str,
        datapoint_type: str,
        max_records: int = 0,
        max_cached_shards: int = 8,
    ) -> None:
        self.source_path = Path(pt_path).resolve()
        if not self.source_path.exists():
            raise FileNotFoundError(f"PT dataset file not found: {self.source_path}")

        self.datapoint_type = datapoint_type
        self.max_records = max(0, int(max_records))
        self.max_cached_shards = max(1, int(max_cached_shards))
        self._shard_specs: list[_LazyShardSpec] = []
        self._shard_start_indices: list[int] = []
        self._shard_cache: OrderedDict[int, Any] = OrderedDict()

        self._initialize_shards()
        self._total_records = sum(spec.num_entries for spec in self._shard_specs)
        if self._total_records <= 0:
            raise ValueError(f"No usable datapoints found in {self.source_path}")

    def _initialize_shards(self) -> None:
        root_obj = _torch_load_cpu(self.source_path, mmap=True)
        _validate_pt_object_activation_source(root_obj, self.source_path)

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

        shard_entries = root_obj.get("shards")
        shard_stats = root_obj.get("shard_stats")
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
        return _materialize_training_datapoint(record, record_index=index)

    def __getitem__(self, index: int | slice) -> TrainingDataPoint | list[TrainingDataPoint]:
        if isinstance(index, slice):
            return [self._getitem_int(i) for i in range(*index.indices(len(self)))]
        return self._getitem_int(index)


class TransformedTrainingDataPointDataset(Dataset[TrainingDataPoint]):
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
    del model_name
    if prefer_lazy:
        try:
            return LazyCustomPtDataset(
                pt_path=pt_path,
                datapoint_type=datapoint_type,
                max_records=max_records,
                max_cached_shards=max_cached_shards,
            )
        except ValueError as exc:
            print(f"WARNING: {exc} Falling back to eager PT loading for {pt_path}.")

    return load_training_datapoints_from_pt(
        pt_path=pt_path,
        model_name="",
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
    del model_name, datapoint_type
    source_path = Path(pt_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"PT dataset file not found: {source_path}")

    pt_obj = _torch_load_cpu(source_path)
    _validate_pt_object_activation_source(pt_obj, source_path)
    data: list[TrainingDataPoint] = []
    for idx, record in enumerate(_iter_records_from_pt_object(pt_obj, source_path)):
        data.append(_materialize_training_datapoint(record, record_index=idx))
        if max_records > 0 and len(data) >= max_records:
            break

    if not data:
        raise ValueError(f"No usable datapoints found in {source_path}")
    return data


class CustomPtDatasetLoader(ActDatasetLoader):
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
            model_name="",
            datapoint_type=self.dataset_config.dataset_name,
            max_records=self._split_limit(split),
        )
        print(f"Loaded {len(data)} datapoints from {path}")
        return data
