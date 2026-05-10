#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)
rynnvla_dir=$(cd "$script_dir/.." && pwd)
cd "$script_dir"
export PYTHONPATH="$repo_dir:$rynnvla_dir:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"

checkpoint_path=${CHECKPOINT_PATH:-${1:-../ckpts/starting_point}}
base_checkpoint_path=${BASE_CHECKPOINT_PATH:-../ckpts/starting_point}
exp_name=${EXP_NAME:-bair_rollout_eval_multi}
output_dir=${OUTPUT_DIR:-../outputs/bair_robot_pushing}
out_dir=${OUT_DIR:-"$output_dir"/"$exp_name"/rollout_eval}
data_dir=${DATA_DIR:-../../../../data/bair_robot_pushing_npz/test}
gpu_ids=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}
seed=${SEED:-7}
samples=${SAMPLES:-32}
context_frames=${CONTEXT_FRAMES:-1}
future_frames=${FUTURE_FRAMES:-15}
resolution=${RESOLUTION:-256}
preview_samples=${PREVIEW_SAMPLES:-8}
video_fps=${VIDEO_FPS:-4}
with_transition_tokens=${WITH_TRANSITION_TOKENS:-false}
transition_token_count=${TRANSITION_TOKEN_COUNT:-4}
transition_token_id=${TRANSITION_TOKEN_ID:-16001}
transition_token_hidden_mult=${TRANSITION_TOKEN_HIDDEN_MULT:-1}
max_new_tokens=${MAX_NEW_TOKENS:-0}
rollout_state=${ROLLOUT_STATE:-token}
run_label=${RUN_LABEL:-$exp_name}
max_shards=${MAX_SHARDS:-0}
force_image_prefix=${FORCE_IMAGE_PREFIX:-true}
action_mode=${ACTION_MODE:-gt}
action_scale=${ACTION_SCALE:-1.0}
keep_shards=${KEEP_SHARDS:-false}

IFS=',' read -r -a gpus <<< "$gpu_ids"
shard_count=${SHARD_COUNT:-${#gpus[@]}}
if [[ "$shard_count" -lt 1 ]]; then
  echo "SHARD_COUNT must be >= 1" >&2
  exit 2
fi
if [[ "${#gpus[@]}" -lt "$shard_count" ]]; then
  echo "GPU_IDS has ${#gpus[@]} GPU(s), but SHARD_COUNT=${shard_count}" >&2
  exit 2
fi
if [[ "$samples" -lt "$shard_count" ]]; then
  echo "SAMPLES=${samples} is smaller than SHARD_COUNT=${shard_count}; using SAMPLES=${shard_count}" >&2
  samples=${shard_count}
fi

transition_token_args=()
if [[ "${with_transition_tokens}" == "true" ]]; then
  transition_token_args+=(--with-transition-tokens)
fi

optional_args=()
if [[ "${LPIPS:-false}" == "true" ]]; then
  optional_args+=(--lpips --lpips-net "${LPIPS_NET:-alex}")
fi
if [[ -n "${LPIPS_DEVICE:-}" ]]; then
  optional_args+=(--lpips-device "${LPIPS_DEVICE}")
fi
if [[ -n "${FVD_MODEL_PATH:-}" ]]; then
  echo "FVD is not supported in sharded rollout. Run FVD as a separate single-process eval." >&2
  exit 2
fi
if [[ "${SAVE_GIF:-true}" != "true" ]]; then
  optional_args+=(--no-save-gif)
fi
if [[ "${DO_SAMPLE:-false}" == "true" ]]; then
  optional_args+=(--do-sample)
fi
if [[ "${force_image_prefix}" != "true" ]]; then
  optional_args+=(--no-force-image-prefix)
fi
if [[ "${ALLOW_NONFORMAL_BASELINE:-false}" == "true" ]]; then
  optional_args+=(--allow-nonformal-baseline)
fi
aggregate_args=()
if [[ "${keep_shards}" == "true" ]]; then
  aggregate_args+=(--keep-shards)
fi

mkdir -p "${out_dir}"
echo "rollout multi eval dir: ${out_dir}" | tee -a "${out_dir}/output.log"
echo "checkpoint: ${checkpoint_path}" | tee -a "${out_dir}/output.log"
echo "GPU_IDS: ${gpu_ids}" | tee -a "${out_dir}/output.log"
echo "SHARD_COUNT: ${shard_count}" | tee -a "${out_dir}/output.log"
echo "ACTION_MODE: ${action_mode}" | tee -a "${out_dir}/output.log"

pids=()
shard_dirs=()
for ((shard_idx=0; shard_idx<shard_count; shard_idx++)); do
  gpu=${gpus[$shard_idx]}
  shard_dir="${out_dir}/shard_${shard_idx}"
  shard_dirs+=("${shard_dir}")
  mkdir -p "${shard_dir}"
  echo "[multi] start shard ${shard_idx}/${shard_count} on GPU ${gpu}" | tee -a "${out_dir}/output.log"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    ../../.venv/bin/python eval_bair_rollout.py \
      --checkpoint-path "${checkpoint_path}" \
      --base-checkpoint-path "${base_checkpoint_path}" \
      --out-dir "${shard_dir}" \
      --data-dir "${data_dir}" \
      --device cuda:0 \
      --seed "${seed}" \
      --samples "${samples}" \
      --shard-index "${shard_idx}" \
      --shard-count "${shard_count}" \
      --context-frames "${context_frames}" \
      --future-frames "${future_frames}" \
      --resolution "${resolution}" \
      --preview-samples "${preview_samples}" \
      --video-fps "${video_fps}" \
      --max-shards "${max_shards}" \
      --transition-token-id "${transition_token_id}" \
      --transition-token-count "${transition_token_count}" \
      --transition-token-hidden-mult "${transition_token_hidden_mult}" \
      --max-new-tokens "${max_new_tokens}" \
      --rollout-state "${rollout_state}" \
      --action-mode "${action_mode}" \
      --action-scale "${action_scale}" \
      --run-label "${run_label}" \
      "${transition_token_args[@]}" \
      "${optional_args[@]}" \
      2>&1 | tee -a "${shard_dir}/output.log"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "[multi] one or more rollout shards failed" | tee -a "${out_dir}/output.log" >&2
  exit 1
fi

../../.venv/bin/python aggregate_bair_rollout_shards.py \
  --out-dir "${out_dir}" \
  --shard-dirs "${shard_dirs[@]}" \
  "${aggregate_args[@]}" \
  2>&1 | tee -a "${out_dir}/output.log"

echo "rollout eval dir: ${out_dir}"
echo "checkpoint: ${checkpoint_path}"
