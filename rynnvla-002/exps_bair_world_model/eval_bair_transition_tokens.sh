#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)
rynnvla_dir=$(cd "$script_dir/.." && pwd)
cd "$script_dir"
export PYTHONPATH="$repo_dir:$rynnvla_dir:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

CHECKPOINT_PATH=${1:-${CHECKPOINT_PATH:-../outputs/bair_robot_pushing/bair_his_1_world_model_transition_tokens/epoch0}}

ARG_WORLD_SIZE=${WORLD_SIZE:-1}
ARG_NPROC_PER_NODE=${NPROC_PER_NODE:-1}
ARG_MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
ARG_MASTER_PORT=${MASTER_PORT:-16681}
ARG_RANK=${RANK:-0}

WORLD_SIZE=${WORLD_SIZE:-$ARG_WORLD_SIZE}
NPROC_PER_NODE=${NPROC_PER_NODE:-$ARG_NPROC_PER_NODE}
MASTER_ADDR=$ARG_MASTER_ADDR
MASTER_PORT=$ARG_MASTER_PORT
RANK=$ARG_RANK

lr=${LR:-5e-6}
wd=${WEIGHT_DECAY:-0.1}
dropout=${DROPOUT:-0.05}
z_loss_weight=${Z_LOSS_WEIGHT:-1e-5}
precision=${PRECISION:-tf32}
batch_size=${BATCH_SIZE:-1}
accum_iter=${ACCUM_ITER:-1}
eval_max_batches=${EVAL_MAX_BATCHES:-0}
num_workers=${NUM_WORKERS:-4}
trainable_scope=${TRAINABLE_SCOPE:-transition_only}
with_transition_tokens=${WITH_TRANSITION_TOKENS:-true}
transition_token_hidden_mult=${TRANSITION_TOKEN_HIDDEN_MULT:-1}
custom_att_mask=${CUSTOM_ATT_MASK:-false}
custom_att_mask_args=()
if [[ "${custom_att_mask}" != "true" ]]; then
  custom_att_mask_args+=(--disable_custom_att_mask)
fi
transition_token_args=()
if [[ "${with_transition_tokens}" == "true" ]]; then
  transition_token_args+=(--with-transition-tokens)
fi

data_config_train=${DATA_CONFIG_TRAIN:-../configs/bair_robot_pushing/his_1_world_model_nopretokenize_train.yaml}
data_config_val_ind=${DATA_CONFIG_VAL_IND:-../configs/bair_robot_pushing/his_1_world_model_nopretokenize_test.yaml}
data_config_val_ood=${DATA_CONFIG_VAL_OOD:-../configs/bair_robot_pushing/his_1_world_model_nopretokenize_test.yaml}

exp_name=${EXP_NAME:-bair_his_1_world_model_transition_tokens_eval}
output_dir=${OUTPUT_DIR:-../outputs/bair_robot_pushing}
mkdir -p "$output_dir"/"$exp_name"

../../.venv/bin/python -m torch.distributed.run \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=$WORLD_SIZE \
  --node_rank=$RANK \
  ../pretrain_solver_awm_w_ck_action_head.py \
  --disable_length_clustering \
  --eval_only true \
  --train_only false \
  --preprocess false \
  --dataset-kind bair_npz \
  --with_world_model \
  "${transition_token_args[@]}" \
  --resolution 256 \
  --init_from ../ckpts/starting_point \
  --resume_path "$CHECKPOINT_PATH" \
  --tokenizer_path ../ckpts/base_model \
  --ablation 0 \
  --model_size 7B \
  --trainable-scope "${trainable_scope}" \
  --batch_size ${batch_size} \
  --accum_iter ${accum_iter} \
  --epochs 1 \
  --warmup_epochs 0.01 \
  --lr ${lr} \
  --min_lr ${lr} \
  --wd ${wd} \
  --clip_grad 4 \
  --action_dim 4 \
  --time_horizon 1 \
  --transition-token-id 16001 \
  --transition-token-count 4 \
  --transition-token-hidden-mult ${transition_token_hidden_mult} \
  --data_config_train $data_config_train \
  --data_config_val_ind $data_config_val_ind \
  --data_config_val_ood $data_config_val_ood \
  --num_workers ${num_workers} \
  --precision ${precision} \
  --output_dir "$output_dir"/"$exp_name" \
  --metrics_dir "$output_dir"/"$exp_name"/metrics \
  --checkpointing \
  "${custom_att_mask_args[@]}" \
  --eval_max_batches ${eval_max_batches} \
  --max_seq_len 8192 \
  --unmask_image_logits \
  --dropout ${dropout} \
  --z_loss_weight ${z_loss_weight} \
  --ckpt_max_keep 1 \
  2>&1 | tee -a "$output_dir"/"$exp_name"/output.log

echo "eval exp name: $exp_name"
echo "checkpoint: $CHECKPOINT_PATH"
