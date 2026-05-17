from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_model_names(csv_value: str) -> list[str]:
    values = [item.strip() for item in csv_value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one model name.")
    return values


def parse_layer_selector(raw_value: str) -> int | str:
    value = raw_value.strip()
    if not value:
        raise argparse.ArgumentTypeError("Layer selector must be non-empty.")
    if value.lstrip("-").isdigit():
        return int(value)
    return value


def _validate_pt_path(parser: argparse.ArgumentParser, flag_name: str, path: str, *, required: bool) -> None:
    if not path:
        if required:
            parser.error(f"{flag_name} is required.")
        return
    if not os.path.exists(path):
        parser.error(f"{flag_name} does not exist: {path}")
    if Path(path).suffix.lower() != ".pt":
        parser.error(f"{flag_name} must point to a .pt file: {path}")


def resolve_best_model_selection(
    metric_for_best_model: str,
    greater_is_better: bool | None,
) -> tuple[str, bool]:
    resolved_metric = metric_for_best_model.strip() or "eval_loss"
    if greater_is_better is not None:
        return resolved_metric, greater_is_better
    return resolved_metric, not resolved_metric.endswith("loss")


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trainer-based SFT for activation oracle custom PT data.")
    parser.add_argument(
        "--custom-train-pt-path",
        dest="custom_train_pt_path",
        type=str,
        required=True,
        help="Path to training PT dataset.",
    )
    parser.add_argument(
        "--custom-test-pt-path",
        dest="custom_test_pt_path",
        type=str,
        default="",
        help="Optional path to test PT dataset used for evaluation.",
    )
    parser.add_argument(
        "--models",
        type=parse_model_names,
        default=["Qwen/Qwen3-8B"],
        help="Comma-separated Hugging Face model names.",
    )
    parser.add_argument(
        "--hook-layer",
        type=parse_layer_selector,
        required=True,
        help="Hook target. Integer layer index or full module path.",
    )
    parser.add_argument(
        "--global-train-batch-size",
        type=int,
        default=8,
        help="Global train batch size across all ranks before gradient accumulation.",
    )
    parser.add_argument(
        "--global-eval-batch-size",
        type=int,
        default=0,
        help="Global teacher-forced eval batch size. Default 0 means 8x per-device train batch size.",
    )
    parser.add_argument(
        "--generation-eval-batch-size",
        type=int,
        default=0,
        help="Batch size for rank-0 free-generation eval. Default 0 means per-device eval batch size.",
    )
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="If > 0, stop after this many optimizer steps. Useful for quick debug runs.",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="linear",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
        ],
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio used when --warmup-steps is 0.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Absolute warmup steps. Overrides --warmup-ratio when > 0.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--eval-steps", type=int, default=100, help="Number of optimizer steps between evals.")
    parser.add_argument(
        "--eval-on-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run evaluation before the first optimizer step.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable gradient checkpointing.",
    )
    parser.add_argument(
        "--steering-coefficient",
        type=float,
        default=1.0,
        help="Activation steering coefficient.",
    )
    parser.add_argument(
        "--steering-mode",
        type=str,
        choices=["replace", "add"],
        default="replace",
        help="How steering vectors are applied at hook positions.",
    )
    parser.add_argument("--save-steps", type=int, default=5000, help="Checkpoint save interval in optimizer steps.")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory for checkpoints.")
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable saving Trainer checkpoint-* directories during training.",
    )
    parser.add_argument(
        "--save-final-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable saving --save-dir/final after training finishes.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--use-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable LoRA fine-tuning.",
    )
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="all-linear",
        help="LoRA target modules or regex pattern.",
    )
    parser.add_argument(
        "--load-lora-path",
        type=str,
        default=None,
        help="Optional path to an existing LoRA adapter to continue training.",
    )
    parser.add_argument(
        "--use-deepspeed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable DeepSpeed ZeRO training under Trainer.",
    )
    parser.add_argument(
        "--deepspeed-config-path",
        type=str,
        default="",
        help="Optional path to a DeepSpeed JSON config. If empty, uses built-in ZeRO-2 defaults.",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=0,
        help="Optional cap on training examples loaded from the PT file. Default 0 uses the full dataset.",
    )
    parser.add_argument(
        "--max-eval-examples",
        type=int,
        default=0,
        help="Optional cap on evaluation examples loaded from the PT file. Default 0 uses the full dataset.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=0,
        help="If > 0, drop train/eval examples whose tokenized sequence length exceeds this limit.",
    )
    parser.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens for the free-generation eval path.",
    )
    parser.add_argument(
        "--generation-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable rank-0 free-generation evaluation in addition to teacher-forced eval.",
    )
    parser.add_argument(
        "--generation-debug-max-examples",
        type=int,
        default=0,
        help="How many generation examples to print during each eval. Default 0 disables sample-level printing.",
    )
    parser.add_argument(
        "--auto-match-hook-layer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-correct integer --hook-layer when dataset layer is uniquely different. Disabled by default.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="sae_introspection",
        help="WandB project name used by Trainer.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="",
        help="Optional WandB run name. Defaults to a generated value per model.",
    )
    parser.add_argument(
        "--report-to-wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable WandB reporting.",
    )
    parser.add_argument(
        "--save-best-model-at-end",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the best evaluated model to --save-dir/best without requiring Trainer checkpoint directories.",
    )
    parser.add_argument(
        "--metric-for-best-model",
        type=str,
        default="",
        help=(
            "Metric name used to select the best checkpoint. "
            "Examples: eval_loss, eval_val_safety_macro_f1, eval_val_parent_macro_f1, eval_val_subcategory_macro_f1."
        ),
    )
    parser.add_argument(
        "--greater-is-better",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether higher values are better for --metric-for-best-model.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=0,
        help="Maximum number of Trainer checkpoints to keep. Default 0 keeps all checkpoints.",
    )

    args = parser.parse_args()
    if args.global_train_batch_size <= 0:
        parser.error("--global-train-batch-size must be > 0")
    if args.global_eval_batch_size < 0:
        parser.error("--global-eval-batch-size must be >= 0")
    if args.generation_eval_batch_size < 0:
        parser.error("--generation-eval-batch-size must be >= 0")
    if args.generation_debug_max_examples < 0:
        parser.error("--generation-debug-max-examples must be >= 0")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be > 0")
    if args.num_epochs <= 0:
        parser.error("--num-epochs must be > 0")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or a positive integer")
    if args.lr <= 0:
        parser.error("--lr must be > 0")
    if args.warmup_ratio < 0.0 or args.warmup_ratio >= 1.0:
        parser.error("--warmup-ratio must be in [0.0, 1.0)")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be > 0")
    if args.eval_steps <= 0:
        parser.error("--eval-steps must be > 0")
    if args.save_steps <= 0:
        parser.error("--save-steps must be > 0")
    if args.save_total_limit < 0:
        parser.error("--save-total-limit must be >= 0")
    if args.max_train_examples < 0:
        parser.error("--max-train-examples must be >= 0")
    if args.max_eval_examples < 0:
        parser.error("--max-eval-examples must be >= 0")
    if args.max_seq_length < 0:
        parser.error("--max-seq-length must be >= 0")
    if args.steering_coefficient <= 0:
        parser.error("--steering-coefficient must be > 0")
    if args.generation_max_new_tokens <= 0:
        parser.error("--generation-max-new-tokens must be > 0")
    if args.lora_r <= 0:
        parser.error("--lora-r must be > 0")
    if args.lora_alpha <= 0:
        parser.error("--lora-alpha must be > 0")
    if args.lora_dropout < 0.0 or args.lora_dropout > 1.0:
        parser.error("--lora-dropout must be in [0.0, 1.0]")

    _validate_pt_path(parser, "--custom-train-pt-path", args.custom_train_pt_path, required=True)
    _validate_pt_path(parser, "--custom-test-pt-path", args.custom_test_pt_path, required=False)
    if args.save_best_model_at_end:
        if not args.custom_test_pt_path:
            parser.error("--save-best-model-at-end requires --custom-test-pt-path for evaluation.")
        if args.save_checkpoints and args.save_steps % args.eval_steps != 0:
            parser.error("--save-best-model-at-end requires --save-steps to be a multiple of --eval-steps.")
    if args.load_lora_path and not os.path.exists(args.load_lora_path):
        parser.error(f"--load-lora-path does not exist: {args.load_lora_path}")
    if args.deepspeed_config_path and not os.path.exists(args.deepspeed_config_path):
        parser.error(f"--deepspeed-config-path does not exist: {args.deepspeed_config_path}")
    return args
