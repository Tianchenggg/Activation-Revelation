from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from nl_probes.utils.dataset_utils import (
    BatchData,
    SPECIAL_TOKEN,
    TrainingDataPoint,
    build_empty_assistant_messages,
    build_full_messages,
    clone_messages,
    find_pattern_in_tokens,
    normalize_chat_template_token_ids,
    validate_supported_activation_source,
)
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.vlm_compat import adjust_positions_for_model, apply_chat_template, get_processor_tokenizer


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_feature(feature: TrainingDataPoint | dict[str, Any]) -> TrainingDataPoint:
    if isinstance(feature, TrainingDataPoint):
        validate_supported_activation_source(feature.meta_info, context=f"feature_idx={feature.feature_idx}")
        return feature
    data_point = TrainingDataPoint(**feature)
    validate_supported_activation_source(data_point.meta_info, context=f"feature_idx={data_point.feature_idx}")
    return data_point


class ActivationOracleBatchCollator:
    def __init__(self, processor, image_max_pixels: int | None) -> None:
        self.processor = processor
        self.tokenizer = get_processor_tokenizer(processor)
        self.image_max_pixels = image_max_pixels

    def _encode_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> dict[str, Any]:
        return apply_chat_template(
            self.processor,
            clone_messages(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
            padding=False,
            max_pixels=self.image_max_pixels,
            return_mm_token_type_ids=True,
        )

    def _encode_message_ids(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> list[int]:
        encoded = apply_chat_template(
            self.processor,
            clone_messages(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors=None,
            padding=False,
            max_pixels=self.image_max_pixels,
        )
        return normalize_chat_template_token_ids(encoded, "message_ids")

    @staticmethod
    def _format_debug_info(
        data_point: TrainingDataPoint,
        *,
        input_length: int,
        positions: list[int],
    ) -> str:
        meta = dict(data_point.meta_info or {})
        fields = [
            f"feature_idx={int(data_point.feature_idx)}",
            f"input_length={input_length}",
            f"positions={positions}",
        ]
        for key in ("row_idx", "row_id", "source", "segment_idx", "full_sequence_length"):
            value = meta.get(key)
            if value not in (None, ""):
                fields.append(f"{key}={value}")
        image_path = meta.get("image_path")
        if image_path:
            fields.append(f"image_path={image_path}")
        return ", ".join(fields)

    def __call__(self, features: list[TrainingDataPoint | dict[str, Any]]) -> dict[str, Any]:
        datapoints = [_normalize_feature(feature) for feature in features]
        encoded_samples: list[dict[str, Any]] = []
        unpadded_lengths: list[int] = []
        labels_per_sample: list[list[int]] = []
        positions_per_sample: list[list[int]] = []

        for data_point in datapoints:
            prompt_messages = clone_messages(data_point.prompt_messages)
            if data_point.prompt_only:
                encoded_full = self._encode_messages(
                    prompt_messages,
                    add_generation_prompt=True,
                )
                full_ids = normalize_chat_template_token_ids(encoded_full, "prompt_only_input_ids")
                labels = [-100] * len(full_ids)
            else:
                full_messages = build_full_messages(prompt_messages, data_point.target_output)
                empty_messages = build_empty_assistant_messages(prompt_messages)
                encoded_full = self._encode_messages(full_messages, add_generation_prompt=False)
                full_ids = normalize_chat_template_token_ids(encoded_full, "full_input_ids")
                prompt_ids = self._encode_message_ids(prompt_messages, add_generation_prompt=True)
                empty_ids = self._encode_message_ids(empty_messages, add_generation_prompt=False)

                assistant_start_idx = len(prompt_ids)
                full_suffix = full_ids[assistant_start_idx:]
                empty_suffix = empty_ids[assistant_start_idx:]
                common_suffix_len = 0
                max_common = min(len(full_suffix), len(empty_suffix))
                while (
                    common_suffix_len < max_common
                    and full_suffix[-1 - common_suffix_len] == empty_suffix[-1 - common_suffix_len]
                ):
                    common_suffix_len += 1

                response_end_idx = len(full_ids) - common_suffix_len
                if response_end_idx <= assistant_start_idx:
                    raise ValueError(
                        "Failed to isolate assistant response tokens from chat-template suffix. "
                        f"assistant_start_idx={assistant_start_idx}, response_end_idx={response_end_idx}"
                    )

                labels = [-100] * len(full_ids)
                for token_idx in range(assistant_start_idx, response_end_idx):
                    labels[token_idx] = full_ids[token_idx]

            positions = find_pattern_in_tokens(
                full_ids,
                SPECIAL_TOKEN,
                data_point.num_positions,
                self.tokenizer,
            )
            encoded_samples.append(encoded_full)
            unpadded_lengths.append(len(full_ids))
            labels_per_sample.append(labels)
            positions_per_sample.append(positions)

        max_length = max(unpadded_lengths)
        pad_token_id = self.tokenizer.pad_token_id
        batch_input_ids: list[torch.Tensor] = []
        batch_attention_mask: list[torch.Tensor] = []
        batch_labels: list[torch.Tensor] = []
        batch_positions: list[list[int]] = []
        batch_steering_vectors: list[torch.Tensor] = []
        batch_feature_indices: list[int] = []
        batch_debug_infos: list[str] = []
        sequence_extra_tensors: dict[str, list[torch.Tensor]] = {
            "position_ids": [],
            "mm_token_type_ids": [],
            "cross_attention_mask": [],
        }
        non_sequence_extra_tensors: dict[str, list[torch.Tensor]] = {
            "pixel_values": [],
            "image_grid_thw": [],
            "images": [],
            "aspect_ratio_ids": [],
            "aspect_ratio_mask": [],
        }

        for data_point, encoded_sample, input_length, labels, positions in zip(
            datapoints,
            encoded_samples,
            unpadded_lengths,
            labels_per_sample,
            positions_per_sample,
            strict=True,
        ):
            input_ids = encoded_sample["input_ids"][0].to(dtype=torch.long, device="cpu")
            attention_mask = encoded_sample["attention_mask"][0].to(dtype=torch.bool, device="cpu")
            padding_length = max_length - input_length

            if padding_length > 0:
                pad_ids = torch.full((padding_length,), pad_token_id, dtype=torch.long)
                pad_mask = torch.zeros((padding_length,), dtype=torch.bool)
                padded_input_ids = torch.cat([pad_ids, input_ids], dim=0)
                padded_attention_mask = torch.cat([pad_mask, attention_mask], dim=0)
                padded_labels = torch.tensor(([-100] * padding_length) + labels, dtype=torch.long)
            else:
                padded_input_ids = input_ids
                padded_attention_mask = attention_mask
                padded_labels = torch.tensor(labels, dtype=torch.long)

            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attention_mask)
            batch_labels.append(padded_labels)
            batch_positions.append([position + padding_length for position in positions])
            batch_steering_vectors.append(data_point.steering_vectors.to(device="cpu"))
            batch_feature_indices.append(int(data_point.feature_idx))
            batch_debug_infos.append(
                self._format_debug_info(
                    data_point,
                    input_length=input_length,
                    positions=[position + padding_length for position in positions],
                )
            )
            for key in sequence_extra_tensors:
                value = encoded_sample.get(key)
                if value is None:
                    continue
                tensor = value.to(device="cpu")
                if tensor.ndim == 0:
                    continue
                if tensor.shape[0] != 1:
                    raise ValueError(f"Expected {key} to have batch size 1, got shape={list(tensor.shape)}")
                tensor = tensor[0]
                if tensor.shape[0] != input_length:
                    raise ValueError(
                        f"Expected {key} sequence length {input_length}, got shape={list(tensor.shape)}"
                    )
                if padding_length > 0:
                    pad_shape = (padding_length, *tensor.shape[1:])
                    pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype)
                    tensor = torch.cat([pad_tensor, tensor], dim=0)
                sequence_extra_tensors[key].append(tensor)

            for key in non_sequence_extra_tensors:
                value = encoded_sample.get(key)
                if value is not None:
                    non_sequence_extra_tensors[key].append(value.to(device="cpu"))

        batch = BatchData(
            input_ids=torch.stack(batch_input_ids, dim=0),
            labels=torch.stack(batch_labels, dim=0),
            attention_mask=torch.stack(batch_attention_mask, dim=0),
            steering_vectors=batch_steering_vectors,
            positions=batch_positions,
            feature_indices=batch_feature_indices,
            position_ids=torch.stack(sequence_extra_tensors["position_ids"], dim=0)
            if sequence_extra_tensors["position_ids"]
            else None,
            pixel_values=torch.cat(non_sequence_extra_tensors["pixel_values"], dim=0)
            if non_sequence_extra_tensors["pixel_values"]
            else None,
            image_grid_thw=torch.cat(non_sequence_extra_tensors["image_grid_thw"], dim=0)
            if non_sequence_extra_tensors["image_grid_thw"]
            else None,
            mm_token_type_ids=torch.stack(sequence_extra_tensors["mm_token_type_ids"], dim=0)
            if sequence_extra_tensors["mm_token_type_ids"]
            else None,
            images=torch.cat(non_sequence_extra_tensors["images"], dim=0)
            if non_sequence_extra_tensors["images"]
            else None,
            aspect_ratio_ids=torch.cat(non_sequence_extra_tensors["aspect_ratio_ids"], dim=0)
            if non_sequence_extra_tensors["aspect_ratio_ids"]
            else None,
            aspect_ratio_mask=torch.cat(non_sequence_extra_tensors["aspect_ratio_mask"], dim=0)
            if non_sequence_extra_tensors["aspect_ratio_mask"]
            else None,
            cross_attention_mask=torch.stack(sequence_extra_tensors["cross_attention_mask"], dim=0)
            if sequence_extra_tensors["cross_attention_mask"]
            else None,
        )
        output = {
            "input_ids": batch.input_ids,
            "labels": batch.labels,
            "attention_mask": batch.attention_mask,
            "steering_vectors": batch.steering_vectors,
            "positions": batch.positions,
            "debug_infos": batch_debug_infos,
        }
        if batch.pixel_values is not None:
            output["pixel_values"] = batch.pixel_values
        if batch.image_grid_thw is not None:
            output["image_grid_thw"] = batch.image_grid_thw
        if batch.mm_token_type_ids is not None:
            output["mm_token_type_ids"] = batch.mm_token_type_ids
        if batch.position_ids is not None:
            output["position_ids"] = batch.position_ids
        if batch.images is not None:
            output["images"] = batch.images
        if batch.aspect_ratio_ids is not None:
            output["aspect_ratio_ids"] = batch.aspect_ratio_ids
        if batch.aspect_ratio_mask is not None:
            output["aspect_ratio_mask"] = batch.aspect_ratio_mask
        if batch.cross_attention_mask is not None:
            output["cross_attention_mask"] = batch.cross_attention_mask
        return output


class IndexedTrainingDataPointDataset(torch.utils.data.Dataset):
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
        self._warned_nonfinite_loss = False

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def _zero_loss(self, device: torch.device) -> torch.Tensor:
        for param in self.base_model.parameters():
            if param.requires_grad and param.numel() > 0:
                return param.float().reshape(-1)[0] * 0.0
        return torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)

    @staticmethod
    def _replace_loss(outputs, loss: torch.Tensor):
        if isinstance(outputs, tuple):
            return (loss, *outputs[1:])
        try:
            outputs.loss = loss
        except Exception:
            pass
        try:
            outputs["loss"] = loss
        except Exception:
            pass
        return outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        steering_vectors: list[torch.Tensor] | None = None,
        positions: list[list[int]] | None = None,
        debug_infos: list[str] | None = None,
        **kwargs,
    ):
        if steering_vectors is None or positions is None:
            raise ValueError("ActivationOracleWrapper requires steering_vectors and positions for every batch.")

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "use_cache": False,
        }
        model_inputs.update(kwargs)
        adjusted_positions = adjust_positions_for_model(self.base_model, model_inputs, positions)
        hook_fn = get_hf_activation_steering_hook(
            vectors=steering_vectors,
            positions=adjusted_positions,
            steering_coefficient=self.steering_coefficient,
            steering_mode=self.steering_mode,
            device=input_ids.device,
            dtype=self.hook_dtype,
            debug_infos=debug_infos,
        )

        with add_hook(self.hook_submodule, hook_fn):
            outputs = self.base_model(**model_inputs)
        loss = getattr(outputs, "loss", None)
        if loss is None and isinstance(outputs, tuple) and outputs:
            loss = outputs[0]
        if torch.is_tensor(loss) and not torch.isfinite(loss).all():
            sample_info = debug_infos[0] if debug_infos else "no sample debug info"
            if _env_flag("AO_SKIP_NONFINITE_LOSS"):
                if not self._warned_nonfinite_loss:
                    print(
                        "WARNING: Non-finite model loss after activation steering; "
                        "replacing this micro-batch loss with 0 because AO_SKIP_NONFINITE_LOSS=1. "
                        f"{sample_info}"
                    )
                    self._warned_nonfinite_loss = True
                return self._replace_loss(outputs, self._zero_loss(input_ids.device))
            raise FloatingPointError(f"Non-finite model loss after activation steering. {sample_info}")
        return outputs

    def save_pretrained(self, output_dir: str | Path, safe_serialization: bool = True) -> None:
        self.base_model.save_pretrained(str(output_dir), safe_serialization=safe_serialization)
