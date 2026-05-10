#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

# BAIR AWM baseline: mixed action-prediction + world-model records.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export MASTER_PORT=${MASTER_PORT:-16931}

export EXP_NAME=${EXP_NAME:-bair_awm_from_official_pretoken_bs6_noactckpt_8gpu_0509}
export BATCH_SIZE=${BATCH_SIZE:-6}
export ACCUM_ITER=${ACCUM_ITER:-1}
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-10000}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-0}
export NUM_WORKERS=${NUM_WORKERS:-0}
export LR=${LR:-5e-6}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
export LOSS_CT_WEIGHTS=${LOSS_CT_WEIGHTS:-10}
export LOG_METRICS_INTERVAL=${LOG_METRICS_INTERVAL:-50}

export TRAIN_RECORD=${TRAIN_RECORD:-../processed_data/bair_robot_pushing_tokens/his_1_awm_train/record.json}
export TEST_RECORD=${TEST_RECORD:-../processed_data/bair_robot_pushing_tokens/his_1_awm_test/record.json}
export TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_bair_awm_from_official_awm_pretokenize.sh}

export DATA_CONFIG_TRAIN=${DATA_CONFIG_TRAIN:-../configs/bair_robot_pushing/his_1_awm_pretokenize_train.yaml}
export DATA_CONFIG_VAL_IND=${DATA_CONFIG_VAL_IND:-../configs/bair_robot_pushing/his_1_awm_pretokenize_test.yaml}
export DATA_CONFIG_VAL_OOD=${DATA_CONFIG_VAL_OOD:-../configs/bair_robot_pushing/his_1_awm_pretokenize_test.yaml}

export SAVE_AT_EPOCH_END=${SAVE_AT_EPOCH_END:-false}
export SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-2000}
export EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-2000}
export EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-32}
export AUTO_RESUME=${AUTO_RESUME:-false}
export CKPT_MAX_KEEP=${CKPT_MAX_KEEP:-1}
export BEST_CHECKPOINT_METRIC=${BEST_CHECKPOINT_METRIC:-closs}
export BEST_CHECKPOINT_SPLIT=${BEST_CHECKPOINT_SPLIT:-ind}
export BEST_CHECKPOINT_MODE=${BEST_CHECKPOINT_MODE:-min}
export BEST_CHECKPOINT_DIR=${BEST_CHECKPOINT_DIR:-best}

export CHECKPOINTING=${CHECKPOINTING:-false}
export LOAD_MODEL_ON_ALL_RANKS=${LOAD_MODEL_ON_ALL_RANKS:-true}
export PROMOTE_PARAMS_TO_FP32=${PROMOTE_PARAMS_TO_FP32:-false}
export IGNORE_MISMATCHED_SIZES=${IGNORE_MISMATCHED_SIZES:-true}

export ROLLOUT_EVAL_EVERY_STEPS=${ROLLOUT_EVAL_EVERY_STEPS:-2000}
export ROLLOUT_EVAL_GPU_IDS=${ROLLOUT_EVAL_GPU_IDS:-0,1,2,3,4,5,6,7}
export ROLLOUT_EVAL_NUM_GPUS=${ROLLOUT_EVAL_NUM_GPUS:-8}
export ROLLOUT_EVAL_SAMPLES=${ROLLOUT_EVAL_SAMPLES:-8}
export ROLLOUT_EVAL_FUTURE_FRAMES=${ROLLOUT_EVAL_FUTURE_FRAMES:-15}
export ROLLOUT_EVAL_PREVIEW_SAMPLES=${ROLLOUT_EVAL_PREVIEW_SAMPLES:-4}
export ROLLOUT_EVAL_MIN_FREE_MB=${ROLLOUT_EVAL_MIN_FREE_MB:-16000}
export ROLLOUT_EVAL_TIMEOUT_SECONDS=${ROLLOUT_EVAL_TIMEOUT_SECONDS:-1}
export ROLLOUT_EVAL_STATE=${ROLLOUT_EVAL_STATE:-token}

exec bash "$script_dir/wait_and_train_bair_pretokenize.sh" 1 "$NPROC_PER_NODE"
