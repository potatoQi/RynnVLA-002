#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Formal BAIR next-frame world-model baseline:
# image_t + action_t -> image_{t+1}, without transition soft tokens.
#
# This is intentionally not adapter-only. The raw starting_point checkpoint is
# only a Chameleon/RynnVLA initialization and cannot be used as the BAIR
# baseline for rollout comparisons.
export WITH_TRANSITION_TOKENS=${WITH_TRANSITION_TOKENS:-false}
export TRAINABLE_SCOPE=${TRAINABLE_SCOPE:-all}
export ONLY_SAVE_TRAINABLE=${ONLY_SAVE_TRAINABLE:-false}
export EXP_NAME=${EXP_NAME:-bair_world_model_baseline_fullfinetune}
export PRECISION=${PRECISION:-bf16}
export PROMOTE_PARAMS_TO_FP32=${PROMOTE_PARAMS_TO_FP32:-false}

exec bash "$script_dir/train_bair_transition_tokens_nopretokenize.sh" "$@"
