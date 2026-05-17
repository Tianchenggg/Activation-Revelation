import os
import shutil
import sys
import math
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nl_probes.utils.activation_oracle_cli import (  # noqa: E402
    parse_cli_args,
    resolve_best_model_selection,
)
from nl_probes.utils.activation_oracle_data_pipeline import prepare_datasets  # noqa: E402
from nl_probes.utils.activation_oracle_modeling import (  # noqa: E402
    build_activation_oracle_model,
    build_training_args,
)
from nl_probes.utils.activation_oracle_runtime import (  # noqa: E402
    ActivationOracleBatchCollator,
    ActivationOracleWrapper,
)
from nl_probes.utils.common import load_processor, set_seed  # noqa: E402


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


class ActivationOracleTrainer(Trainer):
    save_best_model_only: bool = False
    best_model_output_dir: Path | None = None
    best_model_metric: float | None = None

    def _barrier(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _base_model_state_dict(state_dict: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
        if state_dict is None:
            return None
        prefix = "base_model."
        base_state_dict = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        return base_state_dict or state_dict

    def save_model_only(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        if self.args.should_save and output_path.exists():
            shutil.rmtree(output_path)
        self._barrier()
        self.save_model(str(output_path))
        self._barrier()

    def _save(self, output_dir: str | None = None, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.accelerator.unwrap_model(self.model)

        if not isinstance(model_to_save, ActivationOracleWrapper):
            super()._save(output_dir=output_dir)
            return

        model_to_save.base_model.save_pretrained(
            output_dir,
            safe_serialization=True,
            state_dict=self._base_model_state_dict(state_dict),
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _best_metric_key(self) -> str:
        metric = self.args.metric_for_best_model or "eval_loss"
        return metric if metric.startswith("eval_") else f"eval_{metric}"

    def _metric_is_better(self, metric_value: float) -> bool:
        if self.best_model_metric is None:
            return True
        if self.args.greater_is_better:
            return metric_value > self.best_model_metric
        return metric_value < self.best_model_metric

    def _maybe_save_best_model_only(self, metrics: dict[str, float]) -> None:
        if not self.save_best_model_only:
            return
        if self.best_model_output_dir is None:
            raise RuntimeError("best_model_output_dir must be configured when save_best_model_only=True.")

        metric_key = self._best_metric_key()
        if metric_key not in metrics:
            return
        metric_value = float(metrics[metric_key])
        if not self._metric_is_better(metric_value):
            return

        self.best_model_metric = metric_value
        self.state.best_metric = metric_value
        self.state.best_global_step = self.state.global_step
        self.state.best_model_checkpoint = str(self.best_model_output_dir)
        if self.args.should_save:
            print(
                f"New best {metric_key}={metric_value:.6g} at step {self.state.global_step}; "
                f"saving model-only best to {self.best_model_output_dir}"
            )
        self.save_model_only(self.best_model_output_dir)

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        self._maybe_save_best_model_only(metrics)
        return metrics


def _run_single_model() -> None:
    dtype = torch.bfloat16
    set_seed(cli_args.seed)
    effective_gradient_checkpointing = bool(cli_args.gradient_checkpointing)
    if effective_gradient_checkpointing:
        if _local_rank() == 0:
            print(
                "Disabling --gradient-checkpointing for activation-injection training: "
                "batch-scoped steering hooks are not active during PyTorch checkpoint recomputation."
            )
        effective_gradient_checkpointing = False

    world_size = _world_size()
    if cli_args.global_train_batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size {cli_args.global_train_batch_size} must be divisible by world_size {world_size}"
        )
    per_device_train_batch_size = cli_args.global_train_batch_size // world_size
    if cli_args.global_eval_batch_size > 0:
        if cli_args.global_eval_batch_size % world_size != 0:
            raise ValueError(
                f"Global eval batch size {cli_args.global_eval_batch_size} must be divisible by world_size {world_size}"
            )
        per_device_eval_batch_size = cli_args.global_eval_batch_size // world_size
    else:
        per_device_eval_batch_size = per_device_train_batch_size

    processor = load_processor(cli_args.model)
    dataset_bundle = prepare_datasets(
        model_name=cli_args.model,
        processor=processor,
        train_pt_path=cli_args.custom_train_pt_path,
        test_pt_path=cli_args.custom_test_pt_path,
        max_train_records=cli_args.max_train_examples,
        max_eval_records=cli_args.max_eval_examples,
        max_seq_length=cli_args.max_seq_length,
        image_max_pixels=cli_args.image_max_pixels,
    )
    training_data = dataset_bundle.training_data
    eval_datasets = dataset_bundle.eval_datasets

    if dataset_bundle.train_filter_stats is not None and _local_rank() == 0:
        train_filter_stats = dataset_bundle.train_filter_stats
        print(
            f"Applied --max-seq-length={cli_args.max_seq_length}: "
            f"train kept {train_filter_stats.kept_count}/{train_filter_stats.original_count} "
            f"(dropped {train_filter_stats.dropped_count}), "
            f"max_kept_length={train_filter_stats.max_kept_length}, "
            f"max_dropped_length={train_filter_stats.max_dropped_length}"
        )
        for dataset_name, filter_stats in (dataset_bundle.eval_filter_stats or {}).items():
            print(
                f"  eval[{dataset_name}] kept {filter_stats.kept_count}/{filter_stats.original_count} "
                f"(dropped {filter_stats.dropped_count}), "
                f"max_kept_length={filter_stats.max_kept_length}, "
                f"max_dropped_length={filter_stats.max_dropped_length}"
            )

    if _local_rank() == 0:
        print(f"Model: {cli_args.model}")
        print(f"Image max pixels: {cli_args.image_max_pixels}")
        print(f"Training data length: {len(training_data)}")
        print(f"Eval datasets: { {name: len(ds) for name, ds in eval_datasets.items()} }")

    if cli_args.max_steps > 0:
        estimated_total_steps = cli_args.max_steps
    else:
        micro_batches_per_epoch = math.ceil(len(training_data) / cli_args.global_train_batch_size)
        estimated_total_steps = math.ceil(micro_batches_per_epoch / cli_args.gradient_accumulation_steps)
        estimated_total_steps *= cli_args.num_epochs
    effective_warmup_steps = int(cli_args.warmup_steps)
    effective_warmup_ratio = float(cli_args.warmup_ratio)
    if effective_warmup_steps > 0:
        effective_warmup_ratio = None
    elif effective_warmup_ratio > 0.0:
        effective_warmup_steps = max(1, math.ceil(estimated_total_steps * effective_warmup_ratio))
        effective_warmup_ratio = None
    else:
        effective_warmup_ratio = None
    if _local_rank() == 0:
        print(
            "LR warmup resolved: "
            f"estimated_total_steps={estimated_total_steps}, "
            f"warmup_steps={effective_warmup_steps}, warmup_ratio={effective_warmup_ratio}"
        )

    model = build_activation_oracle_model(
        model_name=cli_args.model,
        hook_layer=cli_args.hook_layer,
        dtype=dtype,
        gradient_checkpointing=effective_gradient_checkpointing,
        use_lora=cli_args.use_lora,
        lora_r=cli_args.lora_r,
        lora_alpha=cli_args.lora_alpha,
        lora_dropout=cli_args.lora_dropout,
        lora_target_modules=cli_args.lora_target_modules,
        load_lora_path=cli_args.load_lora_path,
        train_llm_only=cli_args.train_llm_only,
        steering_coefficient=cli_args.steering_coefficient,
        steering_mode=cli_args.steering_mode,
    )

    metric_for_best_model = None
    greater_is_better = None
    if cli_args.save_best_model_at_end or cli_args.metric_for_best_model.strip():
        metric_for_best_model, greater_is_better = resolve_best_model_selection(
            cli_args.metric_for_best_model,
            cli_args.greater_is_better,
        )

    run_name = cli_args.wandb_run_name or "trainer_custom_pt_qwen3_vl_8b"
    training_args = build_training_args(
        save_dir=cli_args.save_dir,
        run_name=run_name,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=cli_args.gradient_accumulation_steps,
        lr=cli_args.lr,
        lr_scheduler_type=cli_args.lr_scheduler_type,
        warmup_ratio=effective_warmup_ratio,
        warmup_steps=effective_warmup_steps,
        num_epochs=cli_args.num_epochs,
        max_steps=cli_args.max_steps,
        max_grad_norm=cli_args.max_grad_norm,
        eval_steps=cli_args.eval_steps,
        eval_on_start=cli_args.eval_on_start,
        save_checkpoints=cli_args.save_checkpoints,
        use_deepspeed=cli_args.use_deepspeed,
        deepspeed_config_path=cli_args.deepspeed_config_path,
        dtype=dtype,
        use_lora=cli_args.use_lora,
        load_lora_path=cli_args.load_lora_path,
        model_name=cli_args.model,
        report_to_wandb=cli_args.report_to_wandb,
        wandb_project=cli_args.wandb_project,
        has_eval_dataset=bool(eval_datasets),
        save_steps=cli_args.save_steps,
        save_best_model_at_end=cli_args.save_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=cli_args.save_total_limit or None,
        seed=cli_args.seed,
        gradient_checkpointing=effective_gradient_checkpointing,
    )

    trainer = ActivationOracleTrainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=next(iter(eval_datasets.values())) if eval_datasets else None,
        data_collator=ActivationOracleBatchCollator(processor, image_max_pixels=cli_args.image_max_pixels),
        processing_class=processor,
    )
    trainer.save_best_model_only = bool(cli_args.save_best_model_at_end)
    trainer.best_model_output_dir = Path(cli_args.save_dir) / "best"
    trainer.train()

    post_train_metrics = None
    if cli_args.save_best_model_at_end and eval_datasets:
        post_train_metrics = trainer.evaluate(metric_key_prefix="eval")
        if _local_rank() == 0:
            print(post_train_metrics)

    if cli_args.save_final_model:
        final_dir = Path(cli_args.save_dir) / "final"
        trainer.save_model_only(final_dir)

    if eval_datasets:
        final_metrics = post_train_metrics
        if final_metrics is None:
            final_metrics = trainer.evaluate(metric_key_prefix="final")
        if _local_rank() == 0:
            print(final_metrics)


if __name__ == "__main__":
    cli_args = parse_cli_args()
    _run_single_model()
