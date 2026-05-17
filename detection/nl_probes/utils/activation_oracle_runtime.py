from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from nl_probes.utils.dataset_utils import TrainingDataPoint, construct_batch
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.common import (
    maybe_save_dynamic_model_code,
    temporarily_disable_generation_parameter_validation,
)


def _normalize_feature(feature: TrainingDataPoint | dict[str, Any]) -> TrainingDataPoint:
    if isinstance(feature, TrainingDataPoint):
        return feature
    return TrainingDataPoint(**feature)


class ActivationOracleBatchCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.cpu_device = torch.device("cpu")

    def __call__(self, features: list[TrainingDataPoint | dict[str, Any]]) -> dict[str, Any]:
        datapoints = [_normalize_feature(feature) for feature in features]
        batch = construct_batch(datapoints, self.tokenizer, self.cpu_device)
        return {
            "input_ids": batch.input_ids,
            "labels": batch.labels,
            "attention_mask": batch.attention_mask,
            "steering_vectors": batch.steering_vectors,
            "positions": batch.positions,
        }


class IndexedTrainingDataPointDataset(torch.utils.data.Dataset):
    """Dataset view that keeps only a fixed list of indices from a base dataset."""

    def __init__(self, base_dataset, indices: list[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            selected_indices = self.indices[index]
            return [self.base_dataset[item_index] for item_index in selected_indices]
        return self.base_dataset[self.indices[index]]


class ActivationOracleWrapper(torch.nn.Module):
    accepts_loss_kwargs = False

    def __init__(
        self,
        base_model: torch.nn.Module,
        hook_submodule: torch.nn.Module,
        steering_coefficient: float,
        steering_mode: str,
        hook_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.hook_submodule = hook_submodule
        self.steering_coefficient = steering_coefficient
        self.steering_mode = steering_mode
        self.hook_dtype = hook_dtype
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)
        self.main_input_name = getattr(base_model, "main_input_name", "input_ids")

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        steering_vectors: list[torch.Tensor] | None = None,
        positions: list[list[int]] | None = None,
        **kwargs,
    ):
        if steering_vectors is None or positions is None:
            raise ValueError("ActivationOracleWrapper requires steering_vectors and positions for every batch.")

        hook_fn = get_hf_activation_steering_hook(
            vectors=steering_vectors,
            positions=positions,
            steering_coefficient=self.steering_coefficient,
            steering_mode=self.steering_mode,
            device=input_ids.device,
            dtype=self.hook_dtype,
        )

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "use_cache": False,
        }
        model_inputs.update(kwargs)

        with add_hook(self.hook_submodule, hook_fn):
            return self.base_model(**model_inputs)

    def save_pretrained(self, output_dir: str | Path, safe_serialization: bool = True) -> None:
        with temporarily_disable_generation_parameter_validation(getattr(self.base_model, "config", None)):
            self.base_model.save_pretrained(str(output_dir), safe_serialization=safe_serialization)
        maybe_save_dynamic_model_code(self.base_model, output_dir)
