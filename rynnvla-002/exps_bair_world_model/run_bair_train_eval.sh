#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

mode=${MODE:-baseline_last_layers}
exp_name=${EXP_NAME:-}
output_dir=${OUTPUT_DIR:-../outputs/bair_robot_pushing}
gpu_ids=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}
num_gpus=${NUM_GPUS:-1}
wait_for_gpus=${WAIT_FOR_GPUS:-false}
min_free_mb=${MIN_FREE_MB:-42000}
poll_seconds=${POLL_SECONDS:-60}
run_teacher_eval=${RUN_TEACHER_EVAL:-true}
run_rollout_eval=${RUN_ROLLOUT_EVAL:-true}
rollout_samples=${ROLLOUT_SAMPLES:-32}
rollout_future_frames=${ROLLOUT_FUTURE_FRAMES:-15}
rollout_num_gpus=${ROLLOUT_NUM_GPUS:-1}
teacher_eval_max_batches=${TEACHER_EVAL_MAX_BATCHES:-32}

case "$mode" in
  baseline_last_layers)
    train_script=./train_bair_world_model_baseline_last_layers_nopretokenize.sh
    exp_name=${exp_name:-bair_world_model_baseline_last4_1gpu}
    export WITH_TRANSITION_TOKENS=${WITH_TRANSITION_TOKENS:-false}
    export TRAINABLE_SCOPE=${TRAINABLE_SCOPE:-last_layers}
    export TRAIN_LAST_N_LAYERS=${TRAIN_LAST_N_LAYERS:-4}
    export ONLY_SAVE_TRAINABLE=${ONLY_SAVE_TRAINABLE:-false}
    export PROMOTE_PARAMS_TO_FP32=${PROMOTE_PARAMS_TO_FP32:-true}
    export LOAD_MODEL_ON_ALL_RANKS=${LOAD_MODEL_ON_ALL_RANKS:-true}
    ;;
  baseline_full)
    train_script=./train_bair_world_model_baseline_nopretokenize.sh
    exp_name=${exp_name:-bair_world_model_baseline_fullfinetune}
    export WITH_TRANSITION_TOKENS=${WITH_TRANSITION_TOKENS:-false}
    export TRAINABLE_SCOPE=${TRAINABLE_SCOPE:-all}
    export ONLY_SAVE_TRAINABLE=${ONLY_SAVE_TRAINABLE:-false}
    export PROMOTE_PARAMS_TO_FP32=${PROMOTE_PARAMS_TO_FP32:-false}
    export LOAD_MODEL_ON_ALL_RANKS=${LOAD_MODEL_ON_ALL_RANKS:-false}
    export IGNORE_MISMATCHED_SIZES=${IGNORE_MISMATCHED_SIZES:-true}
    ;;
  baseline_official_awm)
    train_script=./train_bair_world_model_from_official_awm_nopretokenize.sh
    exp_name=${exp_name:-bair_world_model_from_official_awm}
    export WITH_TRANSITION_TOKENS=${WITH_TRANSITION_TOKENS:-false}
    export TRAINABLE_SCOPE=${TRAINABLE_SCOPE:-all}
    export ONLY_SAVE_TRAINABLE=${ONLY_SAVE_TRAINABLE:-false}
    export PROMOTE_PARAMS_TO_FP32=${PROMOTE_PARAMS_TO_FP32:-false}
    export LOAD_MODEL_ON_ALL_RANKS=${LOAD_MODEL_ON_ALL_RANKS:-false}
    ;;
  transition_phase1)
    train_script=./train_bair_transition_tokens_nopretokenize.sh
    exp_name=${exp_name:-bair_world_model_transition_tokens_phase1}
    export WITH_TRANSITION_TOKENS=${WITH_TRANSITION_TOKENS:-true}
    export TRAINABLE_SCOPE=${TRAINABLE_SCOPE:-transition_only}
    export ONLY_SAVE_TRAINABLE=${ONLY_SAVE_TRAINABLE:-true}
    export PROMOTE_PARAMS_TO_FP32=${PROMOTE_PARAMS_TO_FP32:-true}
    export LOAD_MODEL_ON_ALL_RANKS=${LOAD_MODEL_ON_ALL_RANKS:-false}
    ;;
  *)
    echo "Unknown MODE: $mode" >&2
    echo "Expected one of: baseline_last_layers, baseline_full, baseline_official_awm, transition_phase1" >&2
    exit 2
    ;;
esac

export EXP_NAME=$exp_name
mkdir -p "$output_dir/$exp_name"

gpu_count() {
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l
}

gpu_free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

select_free_gpus() {
  local selected=()
  local total
  total=$(gpu_count)
  for ((idx=0; idx<total; idx++)); do
    local free_mb
    free_mb=$(gpu_free_mb "$idx")
    if [[ "$free_mb" -ge "$min_free_mb" ]]; then
      selected+=("$idx")
    fi
    if [[ "${#selected[@]}" -ge "$num_gpus" ]]; then
      local joined
      joined=$(IFS=,; echo "${selected[*]}")
      echo "$joined"
      return 0
    fi
  done
  return 1
}

verify_requested_gpus() {
  local ids=$1
  IFS=',' read -r -a requested <<< "$ids"
  if [[ "${#requested[@]}" -lt "$num_gpus" ]]; then
    echo "GPU_IDS has ${#requested[@]} GPU(s), but NUM_GPUS=$num_gpus" >&2
    return 1
  fi
  for idx in "${requested[@]}"; do
    local free_mb
    free_mb=$(gpu_free_mb "$idx")
    if [[ "$free_mb" -lt "$min_free_mb" ]]; then
      echo "GPU $idx free ${free_mb}MiB < MIN_FREE_MB=${min_free_mb}" >&2
      return 1
    fi
  done
  echo "$ids"
}

resolve_gpus() {
  if [[ -n "$gpu_ids" ]]; then
    if verify_requested_gpus "$gpu_ids"; then
      return 0
    fi
    return 1
  fi
  select_free_gpus
}

if [[ "$wait_for_gpus" == "true" ]]; then
  while true; do
    if selected_gpus=$(resolve_gpus); then
      break
    fi
    date "+waiting for ${num_gpus} GPU(s) with >=${min_free_mb}MiB free at %F %T"
    sleep "$poll_seconds"
  done
else
  if ! selected_gpus=$(resolve_gpus); then
    echo "No suitable GPU set is currently available. Set WAIT_FOR_GPUS=true to queue the run." >&2
    exit 4
  fi
fi

export CUDA_VISIBLE_DEVICES=$selected_gpus
export NPROC_PER_NODE=$num_gpus

echo "mode: $mode"
echo "exp_name: $exp_name"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "train_script: $train_script"

"$train_script" 1 "$num_gpus"

latest_checkpoint=$(find "$output_dir/$exp_name" -mindepth 1 -maxdepth 1 -type d -name 'epoch*' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
if [[ -z "$latest_checkpoint" ]]; then
  echo "No checkpoint directory found under $output_dir/$exp_name" >&2
  exit 3
fi

echo "latest_checkpoint: $latest_checkpoint"

if [[ "$run_teacher_eval" == "true" ]]; then
  teacher_eval_name=${TEACHER_EVAL_EXP_NAME:-${exp_name}_teacher_eval}
  EXP_NAME=$teacher_eval_name \
  CHECKPOINT_PATH=$latest_checkpoint \
  WITH_TRANSITION_TOKENS=$WITH_TRANSITION_TOKENS \
  TRAINABLE_SCOPE=$TRAINABLE_SCOPE \
  EVAL_MAX_BATCHES=$teacher_eval_max_batches \
  NPROC_PER_NODE=1 \
  MASTER_PORT=${TEACHER_EVAL_MASTER_PORT:-16691} \
    bash ./eval_bair_transition_tokens.sh "$latest_checkpoint"
fi

if [[ "$run_rollout_eval" == "true" ]]; then
  rollout_eval_name=${ROLLOUT_EVAL_EXP_NAME:-${exp_name}_rollout_eval}
  rollout_script=./eval_bair_rollout.sh
  rollout_gpu_ids=${ROLLOUT_GPU_IDS:-${selected_gpus}}
  if [[ "$rollout_num_gpus" -gt 1 ]]; then
    rollout_script=./eval_bair_rollout_multi.sh
  fi
  EXP_NAME=$rollout_eval_name \
  RUN_LABEL=$rollout_eval_name \
  CHECKPOINT_PATH=$latest_checkpoint \
  WITH_TRANSITION_TOKENS=$WITH_TRANSITION_TOKENS \
  SAMPLES=$rollout_samples \
  FUTURE_FRAMES=$rollout_future_frames \
  GPU_IDS=$rollout_gpu_ids \
  SHARD_COUNT=$rollout_num_gpus \
  DEVICE=${ROLLOUT_DEVICE:-auto} \
  MASTER_PORT=${ROLLOUT_MASTER_PORT:-16693} \
    bash "$rollout_script"
fi

echo "done: $exp_name"
