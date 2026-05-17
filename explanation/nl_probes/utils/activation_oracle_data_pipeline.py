from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nl_probes.dataset_classes.custom_pt_dataset import load_training_dataset_from_pt
from nl_probes.utils.activation_oracle_runtime import IndexedTrainingDataPointDataset
from nl_probes.utils.dataset_utils import TrainingDataPoint, estimate_sequence_length


@dataclass
class DatasetFilterStats:
    original_count: int
    kept_count: int
    dropped_count: int
    max_kept_length: int
    max_dropped_length: int


@dataclass
class ActivationOracleDatasetBundle:
    training_data: Any
    eval_datasets: dict[str, Any]
    train_filter_stats: DatasetFilterStats | None = None
    eval_filter_stats: dict[str, DatasetFilterStats] | None = None


def load_datasets(
    *,
    model_name: str,
    train_pt_path: str,
    test_pt_path: str,
    max_train_records: int = 0,
    max_eval_records: int = 0,
) -> tuple[Any, dict[str, Any]]:
    training_data = load_training_dataset_from_pt(
        pt_path=train_pt_path,
        model_name=model_name,
        datapoint_type="custom_pt_train",
        max_records=max_train_records,
    )
    eval_datasets: dict[str, Any] = {}
    if test_pt_path:
        eval_name = Path(test_pt_path).stem
        eval_datasets[eval_name] = load_training_dataset_from_pt(
            pt_path=test_pt_path,
            model_name=model_name,
            datapoint_type=f"custom_pt_{eval_name}",
            max_records=max_eval_records,
        )
    return training_data, eval_datasets


def _sequence_length_from_meta(data_point: TrainingDataPoint) -> int | None:
    raw_value = dict(data_point.meta_info or {}).get("full_sequence_length")
    if raw_value in (None, ""):
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _use_cached_sequence_length() -> bool:
    return os.environ.get("ACTIVATION_ORACLE_USE_CACHED_SEQ_LENGTH", "0") == "1"


def filter_dataset_by_max_seq_length(
    dataset,
    *,
    processor,
    max_seq_length: int,
    image_max_pixels: int | None,
) -> tuple[Any, DatasetFilterStats]:
    if max_seq_length <= 0:
        return dataset, DatasetFilterStats(
            original_count=len(dataset),
            kept_count=len(dataset),
            dropped_count=0,
            max_kept_length=0,
            max_dropped_length=0,
        )

    kept_indices: list[int] = []
    dropped_count = 0
    max_kept_length = 0
    max_dropped_length = 0

    for index in range(len(dataset)):
        data_point = dataset[index]
        input_length = None
        if _use_cached_sequence_length():
            input_length = _sequence_length_from_meta(data_point)
        if input_length is None:
            input_length = estimate_sequence_length(
                data_point=data_point,
                processor=processor,
                image_max_pixels=image_max_pixels,
            )
        if input_length <= max_seq_length:
            kept_indices.append(index)
            max_kept_length = max(max_kept_length, input_length)
        else:
            dropped_count += 1
            max_dropped_length = max(max_dropped_length, input_length)

    filtered_dataset = IndexedTrainingDataPointDataset(dataset, kept_indices)
    if len(filtered_dataset) == 0:
        raise ValueError(
            f"--max-seq-length={max_seq_length} filtered out every example. "
            f"Longest dropped sequence length was {max_dropped_length}."
        )

    return filtered_dataset, DatasetFilterStats(
        original_count=len(dataset),
        kept_count=len(filtered_dataset),
        dropped_count=dropped_count,
        max_kept_length=max_kept_length,
        max_dropped_length=max_dropped_length,
    )


def prepare_datasets(
    *,
    model_name: str,
    processor,
    train_pt_path: str,
    test_pt_path: str,
    max_train_records: int = 0,
    max_eval_records: int = 0,
    max_seq_length: int = 0,
    image_max_pixels: int | None = None,
) -> ActivationOracleDatasetBundle:
    training_data, eval_datasets = load_datasets(
        model_name=model_name,
        train_pt_path=train_pt_path,
        test_pt_path=test_pt_path,
        max_train_records=max_train_records,
        max_eval_records=max_eval_records,
    )

    train_filter_stats = None
    eval_filter_stats = None
    if max_seq_length > 0:
        training_data, train_filter_stats = filter_dataset_by_max_seq_length(
            training_data,
            processor=processor,
            max_seq_length=max_seq_length,
            image_max_pixels=image_max_pixels,
        )
        eval_filtered: dict[str, Any] = {}
        eval_filter_stats = {}
        for dataset_name, dataset_points in eval_datasets.items():
            filtered_dataset, filter_stats = filter_dataset_by_max_seq_length(
                dataset_points,
                processor=processor,
                max_seq_length=max_seq_length,
                image_max_pixels=image_max_pixels,
            )
            eval_filtered[dataset_name] = filtered_dataset
            eval_filter_stats[dataset_name] = filter_stats
        eval_datasets = eval_filtered

    return ActivationOracleDatasetBundle(
        training_data=training_data,
        eval_datasets=eval_datasets,
        train_filter_stats=train_filter_stats,
        eval_filter_stats=eval_filter_stats,
    )
