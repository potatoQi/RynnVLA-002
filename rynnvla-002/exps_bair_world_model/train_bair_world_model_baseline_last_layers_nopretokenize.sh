#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 2-GPU-friendly BAIR next-frame baseline.
# Trains action/image token embeddings, the final N Chameleon layers, final
# norm, lm_head, and action_head; freezes earlier decoder layers.
export WITH_TRANSITION_TOKENS=${WITH_TRANSITION_TOKENS:-false}
export TRAINABLE_SCOPE=${TRAINABLE_SCOPE:-last_layers}
export TRAIN_LAST_N_LAYERS=${TRAIN_LAST_N_LAYERS:-4}
export ONLY_SAVE_TRAINABLE=${ONLY_SAVE_TRAINABLE:-false}
export EXP_NAME=${EXP_NAME:-bair_world_model_baseline_last4}
export PRECISION=${PRECISION:-bf16}
export PROMOTE_PARAMS_TO_FP32=${PROMOTE_PARAMS_TO_FP32:-false}
export LOAD_MODEL_ON_ALL_RANKS=${LOAD_MODEL_ON_ALL_RANKS:-true}

exec bash "$script_dir/train_bair_transition_tokens_nopretokenize.sh" "$@"
