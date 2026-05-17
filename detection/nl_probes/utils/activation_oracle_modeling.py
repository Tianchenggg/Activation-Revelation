from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, TrainingArguments

from nl_probes.utils.activation_oracle_runtime import ActivationOracleWrapper
from nl_probes.utils.activation_utils import get_hf_submodule, get_text_only_lora_targets
from nl_probes.utils.common import load_model


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def print_trainable_parameter_stats(model: torch.nn.Module) -> None:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    pct = (100.0 * trainable / total) if total > 0 else 0.0
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")


def get_quantized_model_kwargs(model_name: str, dtype: torch.dtype) -> dict[str, Any]:
    quantized_models = {
        "Qwen/Qwen3-32B",
        "meta-llama/Llama-3.3-70B-Instruct",
    }
    if model_name not in quantized_models:
        return {}

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=dtype,
    )
    return {"quantization_config": bnb_config}


def build_default_deepspeed_zero2_config(
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    world_size_value: int,
    max_grad_norm: float,
    dtype: torch.dtype,
) -> dict[str, Any]:
    return {
        "train_micro_batch_size_per_gpu": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "train_batch_size": per_device_train_batch_size * gradient_accumulation_steps * world_size_value,
        "gradient_clipping": max_grad_norm,
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "zero_allow_untested_optimizer": True,
        "bf16": {"enabled": dtype == torch.bfloat16},
        "fp16": {"enabled": dtype == torch.float16},
        "wall_clock_breakdown": False,
    }


def build_activation_oracle_model(
    *,
    model_name: str,
    hook_layer: int | str,
    dtype: torch.dtype,
    gradient_checkpointing: bool,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: str,
    load_lora_path: str | None,
    steering_coefficient: float,
    steering_mode: str,
) -> ActivationOracleWrapper:
    quantized_model_kwargs = get_quantized_model_kwargs(model_name, dtype)
    if quantized_model_kwargs and not (use_lora or load_lora_path is not None):
        raise ValueError(
            f"{model_name} is configured for 8-bit loading in this script. "
            "Full fine-tuning without LoRA is not supported in that mode."
        )

    model_kwargs = dict(quantized_model_kwargs)
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": f"cuda:{local_rank()}"}

    model = load_model(model_name, dtype, **model_kwargs)

    if quantized_model_kwargs:
        model = prepare_model_for_kbit_training(model)

    model.enable_input_require_grads()

    if gradient_checkpointing:
        model.use_cache = False
        model.config.use_cache = False
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            model.model.language_model.config.use_cache = False
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.use_cache = False
        model.gradient_checkpointing_enable()

    hook_submodule = get_hf_submodule(model, hook_layer)

    if use_lora and load_lora_path is None:
        target_modules = lora_target_modules
        vlm_targets = get_text_only_lora_targets(model_name)
        if vlm_targets and target_modules == "all-linear":
            print(f"VLM detected ({model_name}): excluding vision tower from LoRA")
            target_modules = vlm_targets

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)
    elif load_lora_path is not None:
        load_lora_path = str(Path(load_lora_path).resolve())
        model = PeftModel.from_pretrained(model, load_lora_path, is_trainable=True, autocast_adapter_dtype=True)

    if isinstance(model, PeftModel):
        hook_submodule = get_hf_submodule(model, hook_layer, use_lora=True)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        print_trainable_parameter_stats(model)

    return ActivationOracleWrapper(
        base_model=model,
        hook_submodule=hook_submodule,
        steering_coefficient=steering_coefficient,
        steering_mode=steering_mode,
        hook_dtype=dtype,
    )


def build_training_args(
    *,
    save_dir: str,
    run_name: str,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    lr: float,
    lr_scheduler_type: str,
    warmup_ratio: float,
    warmup_steps: int,
    num_epochs: int,
    max_steps: int,
    max_grad_norm: float,
    eval_steps: int,
    eval_on_start: bool,
    save_checkpoints: bool,
    use_deepspeed: bool,
    deepspeed_config_path: str,
    dtype: torch.dtype,
    use_lora: bool,
    load_lora_path: str | None,
    model_name: str,
    report_to_wandb: bool,
    wandb_project: str,
    has_eval_dataset: bool,
    save_steps: int,
    save_best_model_at_end: bool,
    metric_for_best_model: str | None,
    greater_is_better: bool | None,
    save_total_limit: int | None,
    seed: int,
    gradient_checkpointing: bool,
) -> TrainingArguments:
    world_size_value = world_size()
    deepspeed_config = None
    if use_deepspeed:
        if deepspeed_config_path:
            deepspeed_config = deepspeed_config_path
        else:
            deepspeed_config = build_default_deepspeed_zero2_config(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                world_size_value=world_size_value,
                max_grad_norm=max_grad_norm,
                dtype=dtype,
            )

    ddp_find_unused_parameters = False
    if not use_lora and load_lora_path is None and get_text_only_lora_targets(model_name) is not None:
        ddp_find_unused_parameters = True

    report_to = ["wandb"] if report_to_wandb else []
    if report_to_wandb:
        os.environ.setdefault("WANDB_PROJECT", wandb_project)

    return TrainingArguments(
        output_dir=save_dir,
        run_name=run_name,
        report_to=report_to,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        max_grad_norm=max_grad_norm,
        eval_strategy="steps" if has_eval_dataset else "no",
        eval_steps=eval_steps if has_eval_dataset else None,
        eval_on_start=eval_on_start and has_eval_dataset,
        save_strategy="steps" if save_checkpoints else "no",
        save_steps=save_steps if save_checkpoints else None,
        # Best/final are saved as model-only directories by ActivationOracleTrainer.
        # Keep the in-memory model at the final training step so final/ is not
        # overwritten by Trainer's native best-checkpoint reload behavior.
        load_best_model_at_end=False,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=save_total_limit,
        logging_strategy="steps",
        logging_steps=1,
        bf16=dtype == torch.bfloat16,
        bf16_full_eval=dtype == torch.bfloat16,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        gradient_checkpointing=gradient_checkpointing,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        deepspeed=deepspeed_config,
        seed=seed,
    )
