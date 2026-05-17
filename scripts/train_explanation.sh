#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"

: "${BASE_MODEL:?Set BASE_MODEL, for example Qwen/Qwen3-VL-8B-Instruct.}"
: "${TRAIN_PT:?Set TRAIN_PT to the explanation train .pt file.}"
: "${VAL_PT:?Set VAL_PT to the explanation validation .pt file.}"

SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/outputs/explanation_model}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
HOOK_LAYER="${HOOK_LAYER:-1}"
STEERING_MODE="${STEERING_MODE:-add}"
STEERING_COEFFICIENT="${STEERING_COEFFICIENT:-1.0}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-100352}"
GLOBAL_TRAIN_BATCH_SIZE="${GLOBAL_TRAIN_BATCH_SIZE:-4}"
GLOBAL_EVAL_BATCH_SIZE="${GLOBAL_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-1e-5}"
EVAL_STEPS="${EVAL_STEPS:-300}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-5}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
USE_DEEPSPEED="${USE_DEEPSPEED:-1}"
USE_LORA="${USE_LORA:-0}"
REPORT_TO_WANDB="${REPORT_TO_WANDB:-0}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-explanation-activation-oracle}"

launcher=("$PYTHON")
if (( NPROC_PER_NODE > 1 )); then
  launcher=("$TORCHRUN" --standalone --nproc_per_node "$NPROC_PER_NODE")
fi

args=(
  "$ROOT_DIR/explanation/nl_probes/sft_trainer.py"
  --model "$BASE_MODEL"
  --custom-train-pt-path "$TRAIN_PT"
  --custom-test-pt-path "$VAL_PT"
  --hook-layer "$HOOK_LAYER"
  --steering-mode "$STEERING_MODE"
  --steering-coefficient "$STEERING_COEFFICIENT"
  --train-llm-only
  --global-train-batch-size "$GLOBAL_TRAIN_BATCH_SIZE"
  --global-eval-batch-size "$GLOBAL_EVAL_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --num-epochs "$NUM_EPOCHS"
  --lr "$LR"
  --lr-scheduler-type "$LR_SCHEDULER_TYPE"
  --eval-steps "$EVAL_STEPS"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --image-max-pixels "$IMAGE_MAX_PIXELS"
  --save-dir "$SAVE_DIR"
  --max-grad-norm "$MAX_GRAD_NORM"
  --no-save-checkpoints
  --save-final-model
  --save-best-model-at-end
  --metric-for-best-model eval_loss
  --no-greater-is-better
  --wandb-run-name "$WANDB_RUN_NAME"
)

if [[ -n "${MAX_STEPS:-}" ]]; then args+=(--max-steps "$MAX_STEPS"); fi
if [[ -n "${MAX_TRAIN_EXAMPLES:-}" ]]; then args+=(--max-train-examples "$MAX_TRAIN_EXAMPLES"); fi
if [[ -n "${MAX_EVAL_EXAMPLES:-}" ]]; then args+=(--max-eval-examples "$MAX_EVAL_EXAMPLES"); fi
if [[ "$USE_DEEPSPEED" == "1" ]]; then args+=(--use-deepspeed); else args+=(--no-use-deepspeed); fi
if [[ "$USE_LORA" == "1" ]]; then args+=(--use-lora); else args+=(--no-use-lora); fi
if [[ "$REPORT_TO_WANDB" == "1" ]]; then args+=(--report-to-wandb); else args+=(--no-report-to-wandb); fi

"${launcher[@]}" "${args[@]}"
