import json
import math
from collections.abc import Callable
from collections.abc import Mapping

import torch
import torch._dynamo as dynamo
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.dataset_utils import (
    BatchData,
    EvalStepResult,
    FeatureResult,
    TrainingDataPoint,
    construct_batch,
    get_prompt_tokens_only,
)


SAFE_LABEL_ALIASES = {
    "safe",
    "yes",
    "benign",
    "harmless",
    "non harmful",
    "not harmful",
}
HARMFUL_LABEL_ALIASES = {
    "harmful",
    "unsafe",
    "no",
    "dangerous",
    "toxic",
    "not safe",
}
SAFETY_PROMPT_KEYWORDS = {
    "safe",
    "unsafe",
    "harmful",
    "harm",
    "toxic",
    "toxicity",
    "benign",
    "harmless",
}
_LOGGED_BINARY_SAFETY_STEPS: set[int] = set()
_WARNED_CACHELESS_GENERATION_FALLBACK = False


def _as_model_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.dtype == torch.bool:
        return attention_mask.to(dtype=torch.long)
    return attention_mask


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
    if total_size < 0:
        raise ValueError(f"total_size must be >= 0, got {total_size}")
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

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
            if global_index < 0 or global_index >= total_size:
                raise IndexError(
                    f"Gathered generation-eval result index out of range: {global_index} (total_size={total_size})"
                )
            merged_results[global_index] = FeatureResult(**result_payload)

    missing_indices = [idx for idx, result in enumerate(merged_results) if result is None]
    if missing_indices:
        preview = missing_indices[:10]
        raise RuntimeError(
            "Distributed generation eval did not gather all results. "
            f"Missing indices: {preview}{'...' if len(missing_indices) > 10 else ''}"
        )

    return [result for result in merged_results if result is not None]


def _active_adapter_set(model: PreTrainedModel) -> set[str]:
    active = getattr(model, "active_adapters", None)
    if callable(active):
        try:
            active = active()
        except Exception:
            active = None

    if active is None:
        return set()
    if isinstance(active, str):
        return {active}
    if isinstance(active, (list, tuple, set)):
        return {str(x) for x in active}
    return {str(active)}


def _ensure_peft_state_for_load_adapter(model: PreTrainedModel) -> None:
    """
    Keep transformers PEFT mixin state consistent before calling `load_adapter`.

    Some stacks can end up with `_hf_peft_config_loaded=True` while `peft_config`
    is absent/empty after a failed adapter init. `load_adapter` then crashes when
    it probes `adapter_name in self.peft_config`.
    """
    peft_cfg_obj = getattr(model, "peft_config", None)
    if peft_cfg_obj is None:
        model.peft_config = {}
        peft_cfg_obj = model.peft_config

    if not isinstance(peft_cfg_obj, Mapping):
        try:
            peft_cfg_obj = dict(peft_cfg_obj)
            model.peft_config = peft_cfg_obj
        except Exception:
            model.peft_config = {}
            peft_cfg_obj = model.peft_config

    loaded_flag = bool(getattr(model, "_hf_peft_config_loaded", False))
    if loaded_flag and len(peft_cfg_obj) == 0:
        # Reset inconsistent half-initialized state.
        model._hf_peft_config_loaded = False


def _sync_adapter_to_base_layer_devices(model: PreTrainedModel, adapter_name: str) -> None:
    """
    Ensure all modules for `adapter_name` live on the same device as their base layers.

    This avoids CUDA/CPU mixed-device errors when loading adapters into a model that was
    already dispatched across devices.
    """
    for module in model.modules():
        move_adapter_fn = getattr(module, "_move_adapter_to_device_of_base_layer", None)
        if callable(move_adapter_fn):
            move_adapter_fn(adapter_name)


@dynamo.disable
@torch.no_grad()
def _prepare_stop_token_sequences(
    tokenizer: PreTrainedTokenizer,
    *,
    device: torch.device,
    stop_strings: list[str] | None,
) -> list[torch.Tensor]:
    stop_token_sequences: list[torch.Tensor] = []
    seen_sequences: set[tuple[int, ...]] = set()
    for stop_string in stop_strings or []:
        token_ids = tokenizer.encode(str(stop_string), add_special_tokens=False)
        if not token_ids:
            continue
        token_key = tuple(int(token_id) for token_id in token_ids)
        if token_key in seen_sequences:
            continue
        seen_sequences.add(token_key)
        stop_token_sequences.append(
            torch.tensor(token_key, dtype=torch.long, device=device)
        )
    return stop_token_sequences


def _update_finished_mask(
    *,
    generated_ids: torch.Tensor,
    current_length: int,
    next_token: torch.Tensor,
    finished_mask: torch.Tensor,
    eos_token_id: int | None,
    stop_token_sequences: list[torch.Tensor],
) -> torch.Tensor:
    updated_mask = finished_mask
    if eos_token_id is not None:
        updated_mask = updated_mask | (next_token.squeeze(-1) == eos_token_id)

    if stop_token_sequences:
        current_output = generated_ids[:, :current_length]
        for stop_token_sequence in stop_token_sequences:
            stop_length = int(stop_token_sequence.numel())
            if current_length < stop_length:
                continue
            updated_mask = updated_mask | torch.all(
                current_output[:, current_length - stop_length : current_length]
                == stop_token_sequence.view(1, -1),
                dim=1,
            )

    return updated_mask


@dynamo.disable
@torch.no_grad()
def _greedy_generate_without_cache(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    stop_token_sequences: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    attention_mask = _as_model_attention_mask(attention_mask)
    batch_size, prompt_length = input_ids.shape
    total_length = prompt_length + max_new_tokens
    generated_ids = input_ids.new_empty((batch_size, total_length))
    generated_ids[:, :prompt_length] = input_ids
    generated_mask = attention_mask.new_zeros((batch_size, total_length))
    generated_mask[:, :prompt_length] = attention_mask
    current_length = prompt_length
    finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    stop_token_sequences = stop_token_sequences or []

    for _ in range(max_new_tokens):
        logits = model(
            input_ids=generated_ids[:, :current_length],
            attention_mask=generated_mask[:, :current_length],
            use_cache=False,
        ).logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if eos_token_id is not None and bool(torch.any(finished_mask)):
            next_token = torch.where(
                finished_mask.unsqueeze(-1),
                torch.full_like(next_token, eos_token_id),
                next_token,
            )

        generated_ids[:, current_length] = next_token.squeeze(-1)
        generated_mask[:, current_length] = True
        current_length += 1
        finished_mask = _update_finished_mask(
            generated_ids=generated_ids,
            current_length=current_length,
            next_token=next_token,
            finished_mask=finished_mask,
            eos_token_id=eos_token_id,
            stop_token_sequences=stop_token_sequences,
        )
        if bool(torch.all(finished_mask)):
            break

    return generated_ids[:, :current_length]


@dynamo.disable
@torch.no_grad()
def _greedy_generate_with_cache(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    stop_token_sequences: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    attention_mask = _as_model_attention_mask(attention_mask)
    batch_size, prompt_length = input_ids.shape
    total_length = prompt_length + max_new_tokens
    generated_ids = input_ids.new_empty((batch_size, total_length))
    generated_ids[:, :prompt_length] = input_ids
    generated_mask = attention_mask.new_zeros((batch_size, total_length))
    generated_mask[:, :prompt_length] = attention_mask
    current_length = prompt_length
    finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    stop_token_sequences = stop_token_sequences or []

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = getattr(outputs, "past_key_values", None)
    if past_key_values is None:
        raise RuntimeError("Model did not return past_key_values during cached generation.")

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated_ids[:, current_length] = next_token.squeeze(-1)
    generated_mask[:, current_length] = True
    current_length += 1
    finished_mask = _update_finished_mask(
        generated_ids=generated_ids,
        current_length=current_length,
        next_token=next_token,
        finished_mask=finished_mask,
        eos_token_id=eos_token_id,
        stop_token_sequences=stop_token_sequences,
    )
    if bool(torch.all(finished_mask)):
        return generated_ids[:, :current_length]

    current_input_ids = next_token
    for _ in range(max_new_tokens - 1):
        outputs = model(
            input_ids=current_input_ids,
            attention_mask=generated_mask[:, :current_length],
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError("Model stopped returning past_key_values during cached generation.")

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        if eos_token_id is not None and bool(torch.any(finished_mask)):
            next_token = torch.where(
                finished_mask.unsqueeze(-1),
                torch.full_like(next_token, eos_token_id),
                next_token,
            )
        generated_ids[:, current_length] = next_token.squeeze(-1)
        generated_mask[:, current_length] = True
        current_length += 1
        finished_mask = _update_finished_mask(
            generated_ids=generated_ids,
            current_length=current_length,
            next_token=next_token,
            finished_mask=finished_mask,
            eos_token_id=eos_token_id,
            stop_token_sequences=stop_token_sequences,
        )
        current_input_ids = next_token

        if bool(torch.all(finished_mask)):
            break

    return generated_ids[:, :current_length]


@dynamo.disable
@torch.no_grad()
def eval_features_batch(
    eval_batch: BatchData,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    steering_coefficient: float,
    generation_kwargs: dict,
    steering_mode: str = "replace",
    collect_prompts: bool = True,
) -> list[FeatureResult]:
    if generation_kwargs["do_sample"]:
        raise ValueError("Evaluation currently supports only greedy decoding (do_sample=False).")

    max_new_tokens = int(generation_kwargs["max_new_tokens"])
    stop_token_sequences = _prepare_stop_token_sequences(
        tokenizer,
        device=device,
        stop_strings=generation_kwargs.get("stop_strings"),
    )
    batch_steering_vectors = eval_batch.steering_vectors
    batch_positions = eval_batch.positions

    # 3. Create and apply the activation steering hook
    hook_fn = get_hf_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=steering_coefficient,
        steering_mode=steering_mode,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {
        "input_ids": eval_batch.input_ids,
        "attention_mask": eval_batch.attention_mask,
    }

    if collect_prompts:
        prompt_tokens = eval_batch.input_ids[:, : eval_batch.input_ids.shape[1]]
        decoded_prompts = tokenizer.batch_decode(prompt_tokens, skip_special_tokens=False)
    else:
        decoded_prompts = [""] * len(eval_batch.feature_indices)

    feature_results = []

    with add_hook(submodule, hook_fn):
        global _WARNED_CACHELESS_GENERATION_FALLBACK
        try:
            output_ids = _greedy_generate_with_cache(
                model=model,
                input_ids=tokenized_input["input_ids"],
                attention_mask=tokenized_input["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                stop_token_sequences=stop_token_sequences,
            )
        except Exception as err:
            if not _WARNED_CACHELESS_GENERATION_FALLBACK:
                print(f"WARNING: cached generation eval unavailable, falling back to cacheless path: {err}")
                _WARNED_CACHELESS_GENERATION_FALLBACK = True
            output_ids = _greedy_generate_without_cache(
                model=model,
                input_ids=tokenized_input["input_ids"],
                attention_mask=tokenized_input["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                stop_token_sequences=stop_token_sequences,
            )

    # Decode only the newly generated tokens
    generated_tokens = output_ids[:, eval_batch.input_ids.shape[1] :]
    decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # Now display and process both samples for each feature consecutively
    for i in range(len(eval_batch.feature_indices)):
        feature_idx = eval_batch.feature_indices[i]

        output = decoded_output[i]

        feature_result = FeatureResult(
            feature_idx=feature_idx,
            api_response=output,
            prompt=decoded_prompts[i],
        )
        feature_results.append(feature_result)

    return feature_results


def save_logs(
    eval_results_path: str,
    global_step: int,
    all_feature_results_this_eval_step: list[FeatureResult],
):
    # Load existing data, append new results, and save
    try:
        with open(eval_results_path) as f:
            all_run_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_run_results = []

    # Add results from the current evaluation step
    eval_step_result = EvalStepResult(
        step=global_step,
        results=all_feature_results_this_eval_step,
    )
    all_run_results.append(eval_step_result.model_dump())

    with open(eval_results_path, "w") as f:
        json.dump(all_run_results, f, indent=2)


def run_evaluation(
    eval_data: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
    lora_path: str | None,
    eval_batch_size: int,
    steering_coefficient: float,
    generation_kwargs: dict,
    steering_mode: str = "replace",
    verbose: bool = False,
    prompt_only_inputs: bool = False,
    collect_prompts: bool = True,
    batch_callback: Callable[[list[FeatureResult], list[TrainingDataPoint]], None] | None = None,
    distributed: bool = False,
    auto_log_binary_safety_metrics: bool = True,
) -> list[FeatureResult]:
    """Run evaluation and save results."""
    rank = _dist_rank()
    world_size = _dist_world_size()
    use_distributed = distributed and world_size > 1

    if use_distributed:
        local_start, local_end = _get_contiguous_rank_range(len(eval_data), rank, world_size)
        local_eval_data = eval_data[local_start:local_end]
    else:
        local_start, local_end = 0, len(eval_data)
        local_eval_data = eval_data

    if lora_path is not None:
        if not hasattr(model, "load_adapter"):
            raise RuntimeError(
                "Model does not expose `load_adapter`, so --lora_paths cannot be applied in evaluation."
            )

        _ensure_peft_state_for_load_adapter(model)

        adapter_name = lora_path
        loaded_adapters = getattr(model, "peft_config", {})
        if adapter_name not in loaded_adapters:
            model.load_adapter(
                lora_path,
                adapter_name=adapter_name,
                is_trainable=False,
                low_cpu_mem_usage=False,
                hotswap=False,
            )

        _ensure_peft_state_for_load_adapter(model)
        loaded_adapters_after = getattr(model, "peft_config", {})
        if adapter_name not in loaded_adapters_after:
            loaded_names = sorted(str(k) for k in loaded_adapters_after.keys())
            raise RuntimeError(
                "LoRA adapter load did not register the requested adapter name. "
                f"requested={adapter_name}, loaded={loaded_names}"
            )

        _sync_adapter_to_base_layer_devices(model, adapter_name)
        model.set_adapter(adapter_name)
        active_adapters = _active_adapter_set(model)
        print(
            "LoRA adapter ready: "
            f"requested={adapter_name}, "
            f"active={sorted(active_adapters) if active_adapters else []}"
        )
        if active_adapters and adapter_name not in active_adapters:
            print(
                "WARNING: active adapter does not match requested adapter. "
                f"requested={adapter_name}, active={sorted(active_adapters)}"
            )
    with torch.no_grad():
        all_feature_results: list[FeatureResult] = []
        for i in tqdm(
            range(0, len(local_eval_data), eval_batch_size),
            desc=f"Evaluating model (rank {rank})" if use_distributed else "Evaluating model",
            disable=use_distributed and rank != 0,
        ):
            source_batch = local_eval_data[i : i + eval_batch_size]
            e_batch = source_batch

            if not prompt_only_inputs:
                e_batch = [get_prompt_tokens_only(item) for item in e_batch]

            e_batch = construct_batch(e_batch, tokenizer, device)

            feature_results = eval_features_batch(
                eval_batch=e_batch,
                model=model,
                submodule=submodule,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                steering_coefficient=steering_coefficient,
                steering_mode=steering_mode,
                generation_kwargs=generation_kwargs,
                collect_prompts=collect_prompts,
            )
            for feature_result, eval_data_point in zip(feature_results, source_batch, strict=True):
                feature_result.meta_info = eval_data_point.meta_info
            if batch_callback is not None:
                batch_callback(feature_results, source_batch)
            if verbose:
                for feature_result in feature_results:
                    print(f"\n=== Feature {feature_result.feature_idx} : {feature_result.api_response} ===\n")
            all_feature_results.extend(feature_results)

        # save_logs(
        #     eval_results_path="eval_logs.json",
        #     global_step=global_step,
        #     all_feature_results_this_eval_step=all_feature_results,
        # )
    # now add the meta info to the local feature results before any distributed gather
    assert len(all_feature_results) == len(local_eval_data), (
        "Number of local feature results and local evaluation data points must match"
    )
    all_feature_results = _gather_ordered_feature_results(
        all_feature_results,
        global_start_index=local_start,
        total_size=len(eval_data),
    ) if use_distributed else all_feature_results

    if auto_log_binary_safety_metrics and (not use_distributed or rank == 0):
        _log_binary_safety_metrics_to_wandb(
            eval_responses=all_feature_results,
            eval_dataset=eval_data,
            global_step=global_step,
        )
    return all_feature_results


def parse_answer(answer: str) -> str:
    return answer.rstrip(".!?,;:").strip().lower()


def _strip_leading_thinking(text: str) -> str:
    cleaned = text.lstrip()
    if not cleaned.startswith("<think>"):
        return cleaned
    end_idx = cleaned.find("</think>")
    if end_idx == -1:
        return cleaned
    return cleaned[end_idx + len("</think>") :].lstrip()


def _normalize_label_text(text: str) -> str:
    text = _strip_leading_thinking(text)
    cleaned = parse_answer(text)
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    return " ".join(cleaned.split())


def _starts_with_label(cleaned_text: str, label: str) -> bool:
    if cleaned_text == label:
        return True
    if not cleaned_text.startswith(label):
        return False
    if len(cleaned_text) == len(label):
        return True
    return cleaned_text[len(label)] in {" ", "\n", "\t", "/", ":", "(", "[", "{", ".", "!", "?", ",", ";", "<"}


def _extract_matching_answer(answer: str, valid_answers: list[str]) -> str | None:
    cleaned_answer = _normalize_label_text(answer)
    for valid_answer in sorted(valid_answers, key=len, reverse=True):
        if _starts_with_label(cleaned_answer, valid_answer):
            return valid_answer
    return None


def _to_binary_safety_label(text: str) -> str | None:
    if text is None:
        return None
    cleaned = _normalize_label_text(text)
    if not cleaned:
        return None

    if cleaned in SAFE_LABEL_ALIASES:
        return "safe"
    if cleaned in HARMFUL_LABEL_ALIASES:
        return "harmful"

    for alias in SAFE_LABEL_ALIASES:
        if _starts_with_label(cleaned, alias):
            return "safe"
    for alias in HARMFUL_LABEL_ALIASES:
        if _starts_with_label(cleaned, alias):
            return "harmful"
    return None


def _looks_like_safety_eval(eval_responses: list[FeatureResult]) -> bool:
    for response in eval_responses:
        prompt_text = _normalize_label_text(response.prompt)
        if any(keyword in prompt_text for keyword in SAFETY_PROMPT_KEYWORDS):
            return True
    return False


def _safe_div(num: int, denom: int) -> float:
    return num / denom if denom > 0 else 0.0


def _compute_binary_safety_confusion(
    eval_responses: list[FeatureResult],
    eval_dataset: list[TrainingDataPoint],
) -> dict[str, int] | None:
    if len(eval_responses) != len(eval_dataset):
        raise ValueError("eval_responses and eval_dataset must have the same length")
    if not eval_responses:
        return None

    valid_answers = sorted({_normalize_label_text(dp.target_output) for dp in eval_dataset})
    cleaned_targets = [_normalize_label_text(dp.target_output) for dp in eval_dataset]
    if any(_to_binary_safety_label(target) is None for target in cleaned_targets):
        return None

    # Avoid polluting generic Yes/No tasks; treat them as Safe/Harmful only
    # when prompts look like a safety evaluation.
    if set(cleaned_targets).issubset({"yes", "no"}) and not _looks_like_safety_eval(eval_responses):
        return None

    tp_harmful = 0
    fp_harmful = 0
    fn_harmful = 0
    tp_safe = 0
    fp_safe = 0
    fn_safe = 0
    pred_harmful = 0
    pred_safe = 0
    pred_unparsed = 0
    target_harmful = 0
    target_safe = 0
    canonical_correct = 0

    for eval_response, eval_data_point, target_answer in zip(eval_responses, eval_dataset, cleaned_targets, strict=True):
        pred_answer = _extract_matching_answer(eval_response.api_response, valid_answers)
        target = _to_binary_safety_label(target_answer)
        pred = _to_binary_safety_label(pred_answer)
        if target is None:
            return None

        if pred_answer == target_answer:
            canonical_correct += 1

        if pred == "harmful":
            pred_harmful += 1
        elif pred == "safe":
            pred_safe += 1
        else:
            pred_unparsed += 1

        if target == "harmful":
            target_harmful += 1
            if pred == "harmful":
                tp_harmful += 1
            else:
                fn_harmful += 1
            if pred == "safe":
                fp_safe += 1
        elif target == "safe":
            target_safe += 1
            if pred == "safe":
                tp_safe += 1
            else:
                fn_safe += 1
            if pred == "harmful":
                fp_harmful += 1
        else:
            return None

    return {
        "tp_harmful": tp_harmful,
        "fp_harmful": fp_harmful,
        "fn_harmful": fn_harmful,
        "tp_safe": tp_safe,
        "fp_safe": fp_safe,
        "fn_safe": fn_safe,
        "pred_harmful": pred_harmful,
        "pred_safe": pred_safe,
        "pred_unparsed": pred_unparsed,
        "target_harmful": target_harmful,
        "target_safe": target_safe,
        "canonical_correct": canonical_correct,
        "total": len(eval_dataset),
    }


def compute_binary_safety_confusion(
    eval_responses: list[FeatureResult],
    eval_dataset: list[TrainingDataPoint],
) -> dict[str, int] | None:
    return _compute_binary_safety_confusion(eval_responses=eval_responses, eval_dataset=eval_dataset)


def _resolve_wandb_log_step(global_step: int) -> int:
    if wandb is None:
        return global_step
    run = getattr(wandb, "run", None)
    if run is None:
        return global_step

    run_step = getattr(run, "step", None)
    if isinstance(run_step, int) and global_step < run_step:
        # W&B drops logs with decreasing step. Fall back to the current run step
        # so eval metrics are still recorded.
        return run_step
    return global_step


def log_metrics_to_wandb(
    metrics: Mapping[str, float],
    global_step: int,
    source_step_key: str | None = None,
    only_on_step_adjustment: bool = False,
) -> bool:
    if wandb is None or getattr(wandb, "run", None) is None:
        return False

    target_step = _resolve_wandb_log_step(global_step)
    if only_on_step_adjustment and target_step == global_step:
        return False

    payload = dict(metrics)
    if source_step_key is not None and target_step != global_step:
        payload[source_step_key] = float(global_step)

    wandb.log(payload, step=target_step)
    return True


def compute_binary_safety_metrics(
    eval_responses: list[FeatureResult],
    eval_dataset: list[TrainingDataPoint],
) -> dict[str, float] | None:
    confusion = _compute_binary_safety_confusion(eval_responses=eval_responses, eval_dataset=eval_dataset)
    if confusion is None:
        return None

    tp_harmful = confusion["tp_harmful"]
    fp_harmful = confusion["fp_harmful"]
    fn_harmful = confusion["fn_harmful"]
    tp_safe = confusion["tp_safe"]
    fp_safe = confusion["fp_safe"]
    fn_safe = confusion["fn_safe"]

    harmful_precision = _safe_div(tp_harmful, tp_harmful + fp_harmful)
    harmful_recall = _safe_div(tp_harmful, tp_harmful + fn_harmful)
    harmful_f1 = _safe_div(2 * harmful_precision * harmful_recall, harmful_precision + harmful_recall)

    safe_precision = _safe_div(tp_safe, tp_safe + fp_safe)
    safe_recall = _safe_div(tp_safe, tp_safe + fn_safe)
    safe_f1 = _safe_div(2 * safe_precision * safe_recall, safe_precision + safe_recall)

    macro_precision = 0.5 * (harmful_precision + safe_precision)
    macro_recall = 0.5 * (harmful_recall + safe_recall)
    macro_f1 = 0.5 * (harmful_f1 + safe_f1)

    if harmful_precision == 0.0 or safe_precision == 0.0 or confusion["pred_unparsed"] > 0:
        print(
            "Binary safety debug: "
            f"target_safe={confusion['target_safe']} "
            f"target_harmful={confusion['target_harmful']} "
            f"pred_safe={confusion['pred_safe']} "
            f"pred_harmful={confusion['pred_harmful']} "
            f"pred_unparsed={confusion['pred_unparsed']} "
            f"tp_harmful={tp_harmful} fp_harmful={fp_harmful} fn_harmful={fn_harmful} "
            f"tp_safe={tp_safe} fp_safe={fp_safe} fn_safe={fn_safe} "
            f"canonical_answer_accuracy={_safe_div(confusion['canonical_correct'], confusion['total']):.4f}"
        )

    return {
        "eval/harmful_precision": harmful_precision,
        "eval/harmful_recall": harmful_recall,
        "eval/harmful_f1": harmful_f1,
        "eval/safe_precision": safe_precision,
        "eval/safe_recall": safe_recall,
        "eval/safe_f1": safe_f1,
        "eval/macro_precision": macro_precision,
        "eval/macro_recall": macro_recall,
        "eval/macro_f1": macro_f1,
        "eval/binary_pred_unparsed_rate": _safe_div(confusion["pred_unparsed"], confusion["total"]),
        "eval/binary_pred_safe_rate": _safe_div(confusion["pred_safe"], confusion["total"]),
        "eval/binary_pred_harmful_rate": _safe_div(confusion["pred_harmful"], confusion["total"]),
        "eval/binary_target_safe_rate": _safe_div(confusion["target_safe"], confusion["total"]),
        "eval/binary_target_harmful_rate": _safe_div(confusion["target_harmful"], confusion["total"]),
        "eval/binary_canonical_answer_accuracy": _safe_div(confusion["canonical_correct"], confusion["total"]),
    }


def _log_binary_safety_metrics_to_wandb(
    eval_responses: list[FeatureResult],
    eval_dataset: list[TrainingDataPoint],
    global_step: int,
) -> None:
    if wandb is None or getattr(wandb, "run", None) is None:
        return
    if global_step in _LOGGED_BINARY_SAFETY_STEPS:
        return

    metrics = compute_binary_safety_metrics(eval_responses=eval_responses, eval_dataset=eval_dataset)
    if metrics is None:
        return

    try:
        log_metrics_to_wandb(
            metrics=metrics,
            global_step=global_step,
            source_step_key="eval/safety_source_step",
        )
        _LOGGED_BINARY_SAFETY_STEPS.add(global_step)
    except Exception as err:
        print(f"WARNING: failed to log binary safety metrics to wandb at step {global_step}: {err}")


def score_eval_responses(
    eval_responses: list[FeatureResult],
    eval_dataset: list[TrainingDataPoint],
    valid_answers: list[str] | None = None,
) -> tuple[float, float]:
    if valid_answers is None:
        valid_answers = sorted({_normalize_label_text(dp.target_output) for dp in eval_dataset})

    format_correct_list = []
    ans_correct_list = []
    for eval_response, eval_data_point in zip(eval_responses, eval_dataset, strict=True):
        matched_response = _extract_matching_answer(eval_response.api_response, valid_answers)
        target_response = _normalize_label_text(eval_data_point.target_output)
        format_correct = matched_response is not None
        ans_correct = matched_response == target_response
        format_correct_list.append(format_correct)
        ans_correct_list.append(ans_correct)

    percent_format_correct = sum(format_correct_list) / len(format_correct_list)
    percent_ans_correct = sum(ans_correct_list) / len(ans_correct_list)
    return percent_format_correct, percent_ans_correct


def proportion_confidence(correct: int, total: int, z: float = 1.96) -> tuple[float, float, float, float]:
    """
    Compute proportion statistics.

    Returns (p, se, lower, upper)
    - p: proportion correct (in [0,1])
    - se: standard error of the proportion (sqrt(p*(1-p)/n))
    - lower, upper: normal-approximation confidence interval (clamped to [0,1])

    Uses normal approx: CI = p +/- z * se. Default z=1.96 gives ~95% CI.
    """
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    p = correct / total
    se = math.sqrt(p * (1.0 - p) / total)
    lower = max(0.0, p - z * se)
    upper = min(1.0, p + z * se)
    return p, se, lower, upper


def analyze_results(results: list[dict]) -> dict[str, float]:
    clean_responses = []

    correct = 0
    is_correct_list = []
    for result in results:
        cleaned_response = parse_answer(result["response"])
        clean_responses.append(cleaned_response)
        target_response = result["target_response"].lower()
        is_correct = target_response == cleaned_response
        is_correct_list.append(is_correct)
        if is_correct:
            correct += 1
        else:
            # continue
            print(result["response"])
            print(cleaned_response)
            print(target_response)
            print("--------------------------------")

    n = len(results)
    p, se, lower, upper = proportion_confidence(correct, n)  # default 95% CI (z=1.96)

    print(f"{correct=}")
    print(f"{n=}")
    print(f"percent_correct = {p:.4f} ({p * 100:.2f}%)")
    print(f"standard_error = {se:.6f}")
    print(f"95% CI (normal approx) = [{lower:.4f}, {upper:.4f}] ({lower * 100:.2f}%, {upper * 100:.2f}%)")
    print(f"len(set(clean_responses))={len(set(clean_responses))}")

    # return values in case you want to plot programmatically
    return {
        "correct": correct,
        "n": n,
        "p": p,
        "se": se,
        "ci_lower": lower,
        "ci_upper": upper,
        "is_correct_list": is_correct_list,
    }
