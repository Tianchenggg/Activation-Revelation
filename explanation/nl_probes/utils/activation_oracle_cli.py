from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_IMAGE_MAX_PIXELS = 256 * 28 * 28


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
    parser = argparse.ArgumentParser(description="Trainer-based VLM SFT for activation-oracle custom PT data.")
    parser.add_argument("--custom-train-pt-path", type=str, required=True, help="Path to training PT dataset.")
    parser.add_argument(
        "--custom-test-pt-path",
        type=str,
        default="",
        help="Optional path to evaluation PT dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_VLM_MODEL,
        help="VLM model name or path. Default uses Qwen/Qwen3-VL-8B-Instruct.",
    )
    parser.add_argument(
        "--hook-layer",
        type=parse_layer_selector,
        required=True,
        help="Hook target. Integer layer index or full module path.",
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=DEFAULT_IMAGE_MAX_PIXELS,
        help="Maximum pixels passed to the VLM processor for each image.",
    )
    parser.add_argument("--global-train-batch-size", type=int, default=8)
    parser.add_argument("--global-eval-batch-size", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-5)
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
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--eval-on-start", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--steering-coefficient", type=float, default=1.0)
    parser.add_argument("--steering-mode", type=str, choices=["replace", "add"], default="replace")
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-final-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--train-llm-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For VLMs, freeze vision/non-LLM modules and train only the language model "
            "and lm_head. Also narrows default LoRA targets to language-model layers."
        ),
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default="all-linear")
    parser.add_argument("--load-lora-path", type=str, default=None)
    parser.add_argument("--use-deepspeed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deepspeed-config-path", type=str, default="")
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="sae_introspection")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--report-to-wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-best-model-at-end", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--metric-for-best-model", type=str, default="")
    parser.add_argument("--greater-is-better", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-total-limit", type=int, default=0)

    args = parser.parse_args()
    if args.global_train_batch_size <= 0:
        parser.error("--global-train-batch-size must be > 0")
    if args.global_eval_batch_size < 0:
        parser.error("--global-eval-batch-size must be >= 0")
    if args.image_max_pixels <= 0:
        parser.error("--image-max-pixels must be > 0")
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
