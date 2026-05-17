from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nl_probes.dataset_classes.custom_pt_dataset import (
    TransformedTrainingDataPointDataset,
    load_training_dataset_from_pt,
)
from nl_probes.utils.activation_oracle_runtime import IndexedTrainingDataPointDataset
from nl_probes.utils.dataset_utils import (
    get_disable_thinking_prefix_ids,
    is_qwen3_reasoning_model,
    maybe_add_disable_thinking_prefix,
)


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


def maybe_wrap_dataset_with_disable_thinking_prefix(dataset, tokenizer, model_name: str):
    if not is_qwen3_reasoning_model(model_name, tokenizer=tokenizer):
        return dataset

    prefix_ids = get_disable_thinking_prefix_ids(tokenizer)
    if not prefix_ids:
        return dataset

    return TransformedTrainingDataPointDataset(
        dataset,
        lambda data_point: maybe_add_disable_thinking_prefix(data_point, tokenizer, model_name, prefix_ids),
    )


def filter_dataset_by_max_seq_length(dataset, max_seq_length: int) -> tuple[Any, DatasetFilterStats]:
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
    get_input_length = getattr(dataset, "get_input_length", None)

    for index in range(len(dataset)):
        if callable(get_input_length):
            input_length = int(get_input_length(index))
        else:
            data_point = dataset[index]
            input_length = len(data_point.input_ids)
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
    tokenizer,
    train_pt_path: str,
    test_pt_path: str,
    max_train_records: int = 0,
    max_eval_records: int = 0,
    max_seq_length: int = 0,
) -> ActivationOracleDatasetBundle:
    training_data, eval_datasets = load_datasets(
        model_name=model_name,
        train_pt_path=train_pt_path,
        test_pt_path=test_pt_path,
        max_train_records=max_train_records,
        max_eval_records=max_eval_records,
    )

    training_data = maybe_wrap_dataset_with_disable_thinking_prefix(training_data, tokenizer, model_name)
    eval_datasets = {
        dataset_name: maybe_wrap_dataset_with_disable_thinking_prefix(dataset_points, tokenizer, model_name)
        for dataset_name, dataset_points in eval_datasets.items()
    }

    train_filter_stats = None
    eval_filter_stats = None
    if max_seq_length > 0:
        training_data, train_filter_stats = filter_dataset_by_max_seq_length(training_data, max_seq_length)
        eval_filtered: dict[str, Any] = {}
        eval_filter_stats = {}
        for dataset_name, dataset_points in eval_datasets.items():
            filtered_dataset, filter_stats = filter_dataset_by_max_seq_length(dataset_points, max_seq_length)
            eval_filtered[dataset_name] = filtered_dataset
            eval_filter_stats[dataset_name] = filter_stats
        eval_datasets = eval_filtered

    return ActivationOracleDatasetBundle(
        training_data=training_data,
        eval_datasets=eval_datasets,
        train_filter_stats=train_filter_stats,
        eval_filter_stats=eval_filter_stats,
    )
