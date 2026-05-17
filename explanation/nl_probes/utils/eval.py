from __future__ import annotations

from collections.abc import Callable
import os

import torch
from tqdm import tqdm

from nl_probes.utils.activation_oracle_runtime import ActivationOracleBatchCollator
from nl_probes.utils.dataset_utils import FeatureResult, TrainingDataPoint, get_prompt_tokens_only
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.vlm_compat import adjust_positions_for_model, get_processor_tokenizer


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_nonfinite_generation_error(exc: Exception) -> bool:
    if isinstance(exc, FloatingPointError):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(pattern in text for pattern in ("non-finite", "nonfinite", "nan", "inf"))


def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _dist_rank() -> int:
    if not _dist_is_initialized():
        return 0
    return torch.distributed.get_rank()


def _dist_world_size() -> int:
    if not _dist_is_initialized():
        return 1
    return torch.distributed.get_world_size()


def _get_contiguous_rank_range(total_size: int, rank: int, world_size: int) -> tuple[int, int]:
    base = total_size // world_size
    remainder = total_size % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def _gather_ordered_feature_results(
    local_results: list[FeatureResult],
    *,
    global_start_index: int,
    total_size: int,
) -> list[FeatureResult]:
    if not _dist_is_initialized():
        return local_results

    payload = [
        (global_start_index + local_offset, feature_result.model_dump())
        for local_offset, feature_result in enumerate(local_results)
    ]
    gathered_payloads = [None] * _dist_world_size()
    torch.distributed.all_gather_object(gathered_payloads, payload)

    if _dist_rank() != 0:
        return []

    merged_results: list[FeatureResult | None] = [None] * total_size
    for rank_payload in gathered_payloads:
        if rank_payload is None:
            continue
        for global_index, result_payload in rank_payload:
            merged_results[global_index] = FeatureResult(**result_payload)

    missing_indices = [idx for idx, result in enumerate(merged_results) if result is None]
    if missing_indices:
        preview = missing_indices[:10]
        raise RuntimeError(
            "Distributed generation eval did not gather all results. "
            f"Missing indices: {preview}{'...' if len(missing_indices) > 10 else ''}"
        )
    return [result for result in merged_results if result is not None]


@torch.no_grad()
def run_evaluation(
    *,
    eval_data: list[TrainingDataPoint],
    model,
    processor,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    eval_batch_size: int,
    steering_coefficient: float,
    generation_kwargs: dict,
    steering_mode: str = "replace",
    batch_callback: Callable[[list[FeatureResult], list[TrainingDataPoint]], None] | None = None,
    distributed: bool = False,
) -> list[FeatureResult]:
    del dtype
    rank = _dist_rank()
    world_size = _dist_world_size()
    use_distributed = distributed and world_size > 1

    if use_distributed:
        local_start, local_end = _get_contiguous_rank_range(len(eval_data), rank, world_size)
        local_eval_data = eval_data[local_start:local_end]
    else:
        local_start, local_end = 0, len(eval_data)
        local_eval_data = eval_data

    tokenizer = get_processor_tokenizer(processor)
    image_max_pixels = generation_kwargs.get("image_max_pixels")
    collator = ActivationOracleBatchCollator(processor, image_max_pixels=image_max_pixels)
    all_feature_results: list[FeatureResult] = []

    for batch_start in tqdm(
        range(0, len(local_eval_data), eval_batch_size),
        desc=f"Evaluating model (rank {rank})" if use_distributed else "Evaluating model",
        disable=use_distributed and rank != 0,
    ):
        source_batch = local_eval_data[batch_start : batch_start + eval_batch_size]
        prompt_only_batch = [get_prompt_tokens_only(item) for item in source_batch]
        eval_batch = collator(prompt_only_batch)
        model_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in eval_batch.items()
            if key not in {"labels", "steering_vectors", "positions", "debug_infos"}
        }
        prompt_length = int(model_inputs["input_ids"].shape[1])
        adjusted_positions = adjust_positions_for_model(model, model_inputs, eval_batch["positions"])

        hook_fn = get_hf_activation_steering_hook(
            vectors=[tensor.to(device) for tensor in eval_batch["steering_vectors"]],
            positions=adjusted_positions,
            steering_coefficient=steering_coefficient,
            steering_mode=steering_mode,
            device=device,
            dtype=next(model.parameters()).dtype,
            debug_infos=eval_batch.get("debug_infos"),
        )

        try:
            with add_hook(submodule, hook_fn):
                output_ids = model.generate(
                    **model_inputs,
                    do_sample=False,
                    max_new_tokens=int(generation_kwargs["max_new_tokens"]),
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        except Exception as exc:
            if not _env_flag("AO_SKIP_NONFINITE_EVAL_BATCH") or not _is_nonfinite_generation_error(exc):
                raise
            print(
                "WARNING: Generation failed for one eval batch with a non-finite error; "
                "recording empty predictions because AO_SKIP_NONFINITE_EVAL_BATCH=1. "
                f"rank={rank}, local_batch_start={batch_start}, error={type(exc).__name__}: {exc}"
            )
            decoded_prompts = tokenizer.batch_decode(
                model_inputs["input_ids"],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            feature_results = []
            for prompt_text, eval_data_point in zip(decoded_prompts, source_batch, strict=True):
                meta_info = dict(eval_data_point.meta_info or {})
                meta_info["eval_error"] = f"{type(exc).__name__}: {exc}"
                feature_results.append(
                    FeatureResult(
                        feature_idx=int(eval_data_point.feature_idx),
                        api_response="",
                        prompt=str(prompt_text),
                        meta_info=meta_info,
                    )
                )
            if batch_callback is not None and not use_distributed:
                batch_callback(feature_results, source_batch)
            all_feature_results.extend(feature_results)
            continue

        generated_tokens = output_ids[:, prompt_length:]
        decoded_output = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decoded_prompts = tokenizer.batch_decode(
            model_inputs["input_ids"],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        feature_results = []
        for output_text, prompt_text, eval_data_point in zip(
            decoded_output,
            decoded_prompts,
            source_batch,
            strict=True,
        ):
            feature_results.append(
                FeatureResult(
                    feature_idx=int(eval_data_point.feature_idx),
                    api_response=str(output_text),
                    prompt=str(prompt_text),
                    meta_info=dict(eval_data_point.meta_info or {}),
                )
            )

        if batch_callback is not None and not use_distributed:
            batch_callback(feature_results, source_batch)
        all_feature_results.extend(feature_results)

    final_results = (
        _gather_ordered_feature_results(
            all_feature_results,
            global_start_index=local_start,
            total_size=len(eval_data),
        )
        if use_distributed
        else all_feature_results
    )
    if batch_callback is not None and use_distributed and rank == 0:
        batch_callback(final_results, eval_data)
    return final_results
