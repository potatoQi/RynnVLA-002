#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

official_awm_checkpoint=${OFFICIAL_AWM_CHECKPOINT:-../ckpts/Action_World_model_512/libero_spatial}
if [[ ! -f "$script_dir/$official_awm_checkpoint/config.json" && ! -f "$official_awm_checkpoint/config.json" ]]; then
  echo "Official AWM checkpoint not found: $official_awm_checkpoint" >&2
  echo "Run download_rynnvla_official_awm_ckpt.sh first, or set OFFICIAL_AWM_CHECKPOINT." >&2
  exit 2
fi

export PREPROCESS=${PREPROCESS:-true}
export DATA_CONFIG_TRAIN=${DATA_CONFIG_TRAIN:-../configs/bair_robot_pushing/his_1_awm_pretokenize_train.yaml}
export DATA_CONFIG_VAL_IND=${DATA_CONFIG_VAL_IND:-../configs/bair_robot_pushing/his_1_awm_pretokenize_test.yaml}
export DATA_CONFIG_VAL_OOD=${DATA_CONFIG_VAL_OOD:-../configs/bair_robot_pushing/his_1_awm_pretokenize_test.yaml}
export INIT_FROM=${INIT_FROM:-$official_awm_checkpoint}
export WITH_TRANSITION_TOKENS=${WITH_TRANSITION_TOKENS:-false}
export TRAINABLE_SCOPE=${TRAINABLE_SCOPE:-all}
export ONLY_SAVE_TRAINABLE=${ONLY_SAVE_TRAINABLE:-false}
export EXP_NAME=${EXP_NAME:-bair_awm_from_official_awm_pretokenize}
export PRECISION=${PRECISION:-bf16}
export PROMOTE_PARAMS_TO_FP32=${PROMOTE_PARAMS_TO_FP32:-false}
export LOAD_MODEL_ON_ALL_RANKS=${LOAD_MODEL_ON_ALL_RANKS:-true}
export IGNORE_MISMATCHED_SIZES=${IGNORE_MISMATCHED_SIZES:-true}
export CHECKPOINTING=${CHECKPOINTING:-false}
export LOSS_CT_WEIGHTS=${LOSS_CT_WEIGHTS:-10}
export BEST_CHECKPOINT_METRIC=${BEST_CHECKPOINT_METRIC:-closs}

exec bash "$script_dir/train_bair_transition_tokens_nopretokenize.sh" "$@"
