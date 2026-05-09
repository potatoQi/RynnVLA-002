#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)
rynnvla_dir=$(cd "$script_dir/.." && pwd)
cd "$script_dir"
export PYTHONPATH="$repo_dir:$rynnvla_dir:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

tokenizer_path=${TOKENIZER_PATH:-../ckpts/base_model}
target_size=${TARGET_SIZE:-256}
gpu_ids=${GPU_IDS:-0,1,2,3}
splits=${SPLITS:-}
bair_splits=${BAIR_SPLITS:-train,test}
with_transition_tokens=${WITH_TRANSITION_TOKENS:-false}
transition_token_id=${TRANSITION_TOKEN_ID:-16001}
transition_token_count=${TRANSITION_TOKEN_COUNT:-4}
storage_format=${STORAGE_FORMAT:-npy}
shard_size=${SHARD_SIZE:-4096}
image_batch_size=${IMAGE_BATCH_SIZE:-32}
image_cache_size=${IMAGE_CACHE_SIZE:-8192}

IFS=',' read -ra gpus <<< "$gpu_ids"
if [[ -z "$splits" ]]; then
  splits=${#gpus[@]}
fi
if [[ "$splits" -lt 1 ]]; then
  echo "SPLITS must be >= 1" >&2
  exit 2
fi
if [[ "${#gpus[@]}" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU id" >&2
  exit 2
fi

transition_args=()
if [[ "$with_transition_tokens" == "true" ]]; then
  transition_args+=(--with-transition-tokens)
fi

IFS=',' read -ra split_names <<< "$bair_splits"
for split_name in "${split_names[@]}"; do
  split_name=$(echo "$split_name" | xargs)
  if [[ "$split_name" == "train" ]]; then
    source_config=${DATA_CONFIG_TRAIN:-../configs/bair_robot_pushing/his_1_world_model_nopretokenize_train.yaml}
    out_dir=${OUT_DIR_TRAIN:-../processed_data/bair_robot_pushing_tokens/his_1_world_model_train}
  elif [[ "$split_name" == "test" ]]; then
    source_config=${DATA_CONFIG_TEST:-../configs/bair_robot_pushing/his_1_world_model_nopretokenize_test.yaml}
    out_dir=${OUT_DIR_TEST:-../processed_data/bair_robot_pushing_tokens/his_1_world_model_test}
  else
    echo "Unknown BAIR split: $split_name" >&2
    exit 2
  fi

  mkdir -p "$out_dir"
  echo "Pretokenizing BAIR $split_name -> $out_dir with $splits workers"
  pids=()
  for ((rank=0; rank<splits; rank++)); do
    gpu=${gpus[$((rank % ${#gpus[@]}))]}
    CUDA_VISIBLE_DEVICES=$gpu ../../.venv/bin/python ../data/pretoken_bair_world_model.py \
      --config "$source_config" \
      --out-dir "$out_dir" \
      --tokenizer "$tokenizer_path" \
      --target-size "$target_size" \
      --splits "$splits" \
      --rank "$rank" \
      --storage-format "$storage_format" \
      --shard-size "$shard_size" \
      --image-batch-size "$image_batch_size" \
      --image-cache-size "$image_cache_size" \
      --transition-token-id "$transition_token_id" \
      --transition-token-count "$transition_token_count" \
      "${transition_args[@]}" &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  ../../.venv/bin/python ../data/pretoken_bair_world_model.py \
    --out-dir "$out_dir" \
    --aggregate-only
done
