#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-1}
ARG_MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
ARG_MASTER_PORT=${MASTER_PORT:-16679}
ARG_RANK=${RANK:-0}

WORLD_SIZE=${WORLD_SIZE:-$ARG_WORLD_SIZE}
NPROC_PER_NODE=${NPROC_PER_NODE:-$ARG_NPROC_PER_NODE}
MASTER_ADDR=$ARG_MASTER_ADDR
MASTER_PORT=$ARG_MASTER_PORT
RANK=$ARG_RANK

lr=5e-6
wd=0.1
dropout=0.05
z_loss_weight=1e-5

data_config=../configs/bair_robot_pushing/his_1_world_model_nopretokenize_smoke.yaml

exp_name=bair_his_1_world_model_transition_tokens_smoke
output_dir=../outputs/bair_robot_pushing
mkdir -p "$output_dir"/"$exp_name"

../../.venv/bin/python -m torch.distributed.run \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=$WORLD_SIZE \
  --node_rank=$RANK \
  ../pretrain_solver_awm_w_ck_action_head.py \
  --disable_length_clustering \
  --train_only True \
  --preprocess false \
  --dataset-kind bair_npz \
  --with_world_model \
  --with-transition-tokens \
  --resolution 256 \
  --init_from ../ckpts/starting_point \
  --tokenizer_path ../ckpts/base_model \
  --ablation 0 \
  --model_size 7B \
  --trainable-scope transition_only \
  --batch_size 1 \
  --accum_iter 1 \
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
  --data_config_train $data_config \
  --data_config_val_ind $data_config \
  --data_config_val_ood $data_config \
  --num_workers 0 \
  --output_dir "$output_dir"/"$exp_name" \
  --checkpointing \
  --max_seq_len 8192 \
  --unmask_image_logits \
  --dropout ${dropout} \
  --z_loss_weight ${z_loss_weight} \
  --ckpt_max_keep 1 \
  2>&1 | tee -a "$output_dir"/"$exp_name"/output.log

echo "exp name: $exp_name"
