#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)

remote_host=${REMOTE_HOST:-casia23}
remote_root=${REMOTE_ROOT:-/data2/qixing.zhou/beta}
local_root=${LOCAL_ROOT:-$repo_root}
exp_name=${EXP_NAME:-bair_awm_from_official_pretoken_bs6_noacthead_skip_fix_8gpu_0510}
output_rel=${OUTPUT_REL:-external/RynnVLA-002/rynnvla-002/outputs/bair_robot_pushing/${exp_name}}
remote_stage_rel=${REMOTE_STAGE_REL:-external/RynnVLA-002/rynnvla-002/outputs/bair_robot_pushing/_rollout_backfill/${exp_name}}

gpu_ids=${GPU_IDS:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a gpu_array <<< "$gpu_ids"
candidate_gpu_count=${#gpu_array[@]}
max_rollout_gpus=${MAX_ROLLOUT_GPUS:-${SHARD_COUNT:-0}}
samples=${SAMPLES:-$candidate_gpu_count}
future_frames=${FUTURE_FRAMES:-15}
preview_samples=${PREVIEW_SAMPLES:-4}
rollout_state=${ROLLOUT_STATE:-token}
action_mode=${ACTION_MODE:-gt}
action_scale=${ACTION_SCALE:-1.0}
remote_min_free_mb=${REMOTE_MIN_FREE_MB:-30000}
remote_gpu_poll_seconds=${REMOTE_GPU_POLL_SECONDS:-30}
remote_gpu_wait_timeout_seconds=${REMOTE_GPU_WAIT_TIMEOUT_SECONDS:-0}
sync_code=${SYNC_CODE:-true}
sync_back_to_local=${SYNC_BACK_TO_LOCAL:-true}
cleanup_remote=${CLEANUP_REMOTE:-true}
dry_run=${DRY_RUN:-false}

local_exp="${local_root}/${output_rel}"
remote_stage_exp="${remote_root}/${remote_stage_rel}"
remote_workdir="${remote_root}/external/RynnVLA-002/rynnvla-002/exps_bair_world_model"
local_external_root="${local_root}/external/RynnVLA-002"
remote_external_root="${remote_root}/external/RynnVLA-002"

if [[ ! -d "$local_exp" ]]; then
  echo "local experiment dir does not exist: $local_exp" >&2
  exit 2
fi
if [[ "$sync_code" == "true" && ! -d "$local_external_root" ]]; then
  echo "local RynnVLA external dir does not exist: $local_external_root" >&2
  exit 2
fi
if [[ "$candidate_gpu_count" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU id" >&2
  exit 2
fi
if [[ "$max_rollout_gpus" -lt 0 ]]; then
  echo "MAX_ROLLOUT_GPUS must be >= 0" >&2
  exit 2
fi
if [[ "$max_rollout_gpus" -gt "$candidate_gpu_count" ]]; then
  echo "MAX_ROLLOUT_GPUS=${max_rollout_gpus} is larger than candidate GPU count ${candidate_gpu_count}; using ${candidate_gpu_count}" >&2
  max_rollout_gpus=$candidate_gpu_count
fi

run() {
  echo "+ $*"
  if [[ "$dry_run" != "true" ]]; then
    "$@"
  fi
}

has_rollout_gif() {
  local rollout_dir=$1
  if compgen -G "$rollout_dir/rollout_videos/*.gif" >/dev/null; then
    return 0
  fi
  if compgen -G "$rollout_dir/shard_*/rollout_videos/*.gif" >/dev/null; then
    return 0
  fi
  return 1
}

needs_rollout() {
  local rollout_dir=$1
  if ! has_rollout_gif "$rollout_dir"; then
    return 0
  fi
  return 1
}

shopt -s nullglob
missing_ckpts=()
missing_rollouts=()
for ckpt_dir in "$local_exp"/epoch*-iter*; do
  [[ -d "$ckpt_dir" ]] || continue
  ckpt_name=$(basename "$ckpt_dir")
  iter="${ckpt_name##*-iter}"
  if [[ ! "$iter" =~ ^[0-9]+$ ]]; then
    echo "[local] skip checkpoint with unrecognized iteration: $ckpt_name" >&2
    continue
  fi
  step=$((10#$iter + 1))
  rollout_name="rollout_iter${iter}_step${step}"
  rollout_dir="$local_exp/periodic_rollout_eval/$rollout_name"
  if needs_rollout "$rollout_dir"; then
    missing_ckpts+=("$ckpt_name")
    missing_rollouts+=("$rollout_name")
  fi
done

echo "remote host:       $remote_host"
echo "local exp:         $local_exp"
echo "remote stage exp:  ${remote_host}:${remote_stage_exp}"
echo "candidate GPUs:    GPU_IDS=${gpu_ids} MAX_ROLLOUT_GPUS=${max_rollout_gpus} SAMPLES=${samples}"
echo "remote GPU check:  use GPUs with free_mb>=${remote_min_free_mb}; poll=${remote_gpu_poll_seconds}s timeout=${remote_gpu_wait_timeout_seconds}s"
echo "sync code:         ${sync_code}"
echo "cleanup remote:    ${cleanup_remote}"

if [[ "${#missing_ckpts[@]}" -eq 0 ]]; then
  echo "[local] all checkpoint rollout videos are present; nothing to backfill"
  exit 0
fi

echo "[local] checkpoints missing rollout videos:"
for idx in "${!missing_ckpts[@]}"; do
  echo "  - ${missing_ckpts[$idx]} -> ${missing_rollouts[$idx]}"
done

if [[ "$sync_code" == "true" ]]; then
  run ssh "$remote_host" "mkdir -p '$remote_external_root'"
  run rsync -aH --delete --partial --info=stats2 \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude 'rynnvla-002/ckpts/' \
    --exclude 'rynnvla-002/outputs/' \
    --exclude 'rynnvla-002/processed_data/' \
    --exclude 'rynnvla-002/__pycache__/' \
    --exclude 'rynnvla-002/**/__pycache__/' \
    "${local_external_root}/" \
    "${remote_host}:${remote_external_root}/"
fi

run ssh "$remote_host" "test -d '$remote_workdir' && rm -rf '$remote_stage_exp' && mkdir -p '$remote_stage_exp/periodic_rollout_eval'"

for ckpt_name in "${missing_ckpts[@]}"; do
  run rsync -aH --partial --info=stats2 \
    "${local_exp}/${ckpt_name}/" \
    "${remote_host}:${remote_stage_exp}/${ckpt_name}/"
done

if [[ "$dry_run" == "true" ]]; then
  echo "DRY_RUN=true: skip remote rollout backfill, sync-back, and cleanup"
  exit 0
fi

ssh "$remote_host" \
  "REMOTE_STAGE_EXP='$remote_stage_exp' REMOTE_WORKDIR='$remote_workdir' GPU_IDS='$gpu_ids' MAX_ROLLOUT_GPUS='$max_rollout_gpus' SAMPLES='$samples' FUTURE_FRAMES='$future_frames' PREVIEW_SAMPLES='$preview_samples' ROLLOUT_STATE='$rollout_state' ACTION_MODE='$action_mode' ACTION_SCALE='$action_scale' REMOTE_MIN_FREE_MB='$remote_min_free_mb' REMOTE_GPU_POLL_SECONDS='$remote_gpu_poll_seconds' REMOTE_GPU_WAIT_TIMEOUT_SECONDS='$remote_gpu_wait_timeout_seconds' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail

cd "$REMOTE_WORKDIR"
shopt -s nullglob

echo "[remote] workdir: $REMOTE_WORKDIR"
echo "[remote] stage exp: $REMOTE_STAGE_EXP"
echo "[remote] candidate GPU_IDS=${GPU_IDS} MAX_ROLLOUT_GPUS=${MAX_ROLLOUT_GPUS} SAMPLES=${SAMPLES}"

select_ready_gpus() {
  local start now free_mb gpu ready_samples
  local -a gpu_array ready_gpus
  start=$(date +%s)
  IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
  while true; do
    ready_gpus=()
    echo "[remote] GPU snapshot before rollout:"
    nvidia-smi || true
    for gpu in "${gpu_array[@]}"; do
      free_mb=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
      if [[ "$free_mb" =~ ^[0-9]+$ && "$free_mb" -ge "$REMOTE_MIN_FREE_MB" ]]; then
        echo "[remote] GPU ${gpu} free ${free_mb}MB OK"
        ready_gpus+=("$gpu")
      else
        echo "[remote] GPU ${gpu} free ${free_mb:-unknown}MB < ${REMOTE_MIN_FREE_MB}MB"
      fi
    done

    if [[ "${#ready_gpus[@]}" -gt 0 ]]; then
      if [[ "$MAX_ROLLOUT_GPUS" -gt 0 && "${#ready_gpus[@]}" -gt "$MAX_ROLLOUT_GPUS" ]]; then
        ready_gpus=("${ready_gpus[@]:0:$MAX_ROLLOUT_GPUS}")
      fi
      READY_GPU_IDS=$(IFS=,; echo "${ready_gpus[*]}")
      READY_SHARD_COUNT=${#ready_gpus[@]}
      ready_samples=$SAMPLES
      if [[ "$ready_samples" -lt "$READY_SHARD_COUNT" ]]; then
        ready_samples=$READY_SHARD_COUNT
      fi
      READY_SAMPLES=$ready_samples
      echo "[remote] selected rollout GPUs: ${READY_GPU_IDS} (shards=${READY_SHARD_COUNT}, samples=${READY_SAMPLES})"
      return 0
    fi

    if [[ "$REMOTE_GPU_WAIT_TIMEOUT_SECONDS" -gt 0 ]]; then
      now=$(date +%s)
      if (( now - start >= REMOTE_GPU_WAIT_TIMEOUT_SECONDS )); then
        echo "[remote] timeout waiting for at least one eligible GPU" >&2
        return 1
      fi
    fi
    echo "[remote] no eligible GPU yet; sleep ${REMOTE_GPU_POLL_SECONDS}s"
    sleep "$REMOTE_GPU_POLL_SECONDS"
  done
}

backfilled_count=0
for ckpt_dir in "$REMOTE_STAGE_EXP"/epoch*-iter*; do
  [[ -d "$ckpt_dir" ]] || continue
  ckpt_name=$(basename "$ckpt_dir")
  iter="${ckpt_name##*-iter}"
  if [[ ! "$iter" =~ ^[0-9]+$ ]]; then
    echo "[remote] skip checkpoint with unrecognized iteration: $ckpt_name"
    continue
  fi
  step=$((10#$iter + 1))
  rollout_name="rollout_iter${iter}_step${step}"
  rollout_dir="$REMOTE_STAGE_EXP/periodic_rollout_eval/$rollout_name"

  echo "[remote] backfill: $ckpt_name -> $rollout_name"
  mkdir -p "$rollout_dir"
  rm -f "$rollout_dir/rollout_eval_status.json"

  select_ready_gpus

  CHECKPOINT_PATH="$ckpt_dir" \
  OUT_DIR="$rollout_dir" \
  EXP_NAME="$rollout_name" \
  RUN_LABEL="$rollout_name" \
  GPU_IDS="$READY_GPU_IDS" \
  SHARD_COUNT="$READY_SHARD_COUNT" \
  SAMPLES="$READY_SAMPLES" \
  FUTURE_FRAMES="$FUTURE_FRAMES" \
  PREVIEW_SAMPLES="$PREVIEW_SAMPLES" \
  SAVE_GIF=true \
  ROLLOUT_STATE="$ROLLOUT_STATE" \
  ACTION_MODE="$ACTION_MODE" \
  ACTION_SCALE="$ACTION_SCALE" \
    bash ./eval_bair_rollout_multi.sh

  if ! compgen -G "$rollout_dir/rollout_videos/*.gif" >/dev/null \
    && ! compgen -G "$rollout_dir/shard_*/rollout_videos/*.gif" >/dev/null; then
    echo "[remote] rollout finished but no GIF was produced: $rollout_dir" >&2
    exit 1
  fi
  backfilled_count=$((backfilled_count + 1))
done

echo "[remote] backfilled rollout checkpoints: $backfilled_count"
REMOTE_SCRIPT

if [[ "$sync_back_to_local" == "true" ]]; then
  mkdir -p "${local_exp}/periodic_rollout_eval"
  run rsync -aH --partial --info=stats2 \
    "${remote_host}:${remote_stage_exp}/periodic_rollout_eval/" \
    "${local_exp}/periodic_rollout_eval/"
fi

if [[ "$cleanup_remote" == "true" ]]; then
  run ssh "$remote_host" "rm -rf '$remote_stage_exp'"
fi

echo "done"
