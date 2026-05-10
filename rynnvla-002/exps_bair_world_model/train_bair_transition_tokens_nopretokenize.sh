#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)
rynnvla_dir=$(cd "$script_dir/.." && pwd)
cd "$script_dir"
export PYTHONPATH="$repo_dir:$rynnvla_dir:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
ARG_MASTER_PORT=${MASTER_PORT:-16677}
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
loss_ct_weights=${LOSS_CT_WEIGHTS:-10}
precision=${PRECISION:-tf32}
data_parallel=${DATA_PARALLEL:-fsdp}
preprocess=${PREPROCESS:-false}
dataset_kind=${DATASET_KIND:-bair_npz}
batch_size=${BATCH_SIZE:-1}
accum_iter=${ACCUM_ITER:-1}
epochs=${EPOCHS:-1}
warmup_epochs=${WARMUP_EPOCHS:-0.01}
num_workers=${NUM_WORKERS:-8}
train_only=${TRAIN_ONLY:-false}
save_at_epoch_end=${SAVE_AT_EPOCH_END:-true}
save_every_steps=${SAVE_EVERY_STEPS:-5000}
eval_every_steps=${EVAL_EVERY_STEPS:-5000}
eval_max_batches=${EVAL_MAX_BATCHES:-32}
rollout_eval_every_steps=${ROLLOUT_EVAL_EVERY_STEPS:-0}
rollout_eval_gpu_ids=${ROLLOUT_EVAL_GPU_IDS:-}
rollout_eval_num_gpus=${ROLLOUT_EVAL_NUM_GPUS:-1}
rollout_eval_samples=${ROLLOUT_EVAL_SAMPLES:-8}
rollout_eval_future_frames=${ROLLOUT_EVAL_FUTURE_FRAMES:-15}
rollout_eval_preview_samples=${ROLLOUT_EVAL_PREVIEW_SAMPLES:-4}
rollout_eval_min_free_mb=${ROLLOUT_EVAL_MIN_FREE_MB:-0}
rollout_eval_timeout_seconds=${ROLLOUT_EVAL_TIMEOUT_SECONDS:-0}
rollout_eval_state=${ROLLOUT_EVAL_STATE:-token}
max_train_steps=${MAX_TRAIN_STEPS:-0}
ckpt_max_keep=${CKPT_MAX_KEEP:-1}
best_checkpoint_metric=${BEST_CHECKPOINT_METRIC:-closs}
best_checkpoint_split=${BEST_CHECKPOINT_SPLIT:-ind}
best_checkpoint_mode=${BEST_CHECKPOINT_MODE:-min}
best_checkpoint_dir=${BEST_CHECKPOINT_DIR:-best}
log_metrics_interval=${LOG_METRICS_INTERVAL:-50}
trainable_scope=${TRAINABLE_SCOPE:-transition_only}
train_last_n_layers=${TRAIN_LAST_N_LAYERS:-4}
with_transition_tokens=${WITH_TRANSITION_TOKENS:-true}
transition_token_hidden_mult=${TRANSITION_TOKEN_HIDDEN_MULT:-1}
init_from=${INIT_FROM:-../ckpts/starting_point}
tokenizer_path=${TOKENIZER_PATH:-../ckpts/base_model}
checkpointing=${CHECKPOINTING:-true}
only_save_trainable=${ONLY_SAVE_TRAINABLE:-true}
auto_resume=${AUTO_RESUME:-true}
custom_att_mask=${CUSTOM_ATT_MASK:-false}
promote_params_to_fp32=${PROMOTE_PARAMS_TO_FP32:-true}
load_model_on_all_ranks=${LOAD_MODEL_ON_ALL_RANKS:-false}
ignore_mismatched_sizes=${IGNORE_MISMATCHED_SIZES:-false}
checkpointing_args=()
if [[ "${checkpointing}" == "true" ]]; then
  checkpointing_args+=(--checkpointing)
fi
only_save_trainable_args=()
if [[ "${only_save_trainable}" == "true" ]]; then
  only_save_trainable_args+=(--only_save_trainable)
fi
auto_resume_args=()
if [[ "${auto_resume}" != "true" ]]; then
  auto_resume_args+=(--no_auto_resume)
fi
custom_att_mask_args=()
if [[ "${custom_att_mask}" != "true" ]]; then
  custom_att_mask_args+=(--disable_custom_att_mask)
fi
save_at_epoch_end_args=()
if [[ "${save_at_epoch_end}" != "true" ]]; then
  save_at_epoch_end_args+=(--no_save_at_epoch_end)
fi
promote_params_args=()
if [[ "${promote_params_to_fp32}" != "true" ]]; then
  promote_params_args+=(--no_promote_params_to_fp32)
fi
load_model_args=()
if [[ "${load_model_on_all_ranks}" == "true" ]]; then
  load_model_args+=(--load_model_on_all_ranks)
fi
ignore_mismatch_args=()
if [[ "${ignore_mismatched_sizes}" == "true" ]]; then
  ignore_mismatch_args+=(--ignore_mismatched_checkpoint_sizes)
fi
transition_token_args=()
if [[ "${with_transition_tokens}" == "true" ]]; then
  transition_token_args+=(--with-transition-tokens)
fi
if [[ "${with_transition_tokens}" != "true" && "${trainable_scope}" == "transition_only" ]]; then
  echo "WITH_TRANSITION_TOKENS=false requires TRAINABLE_SCOPE=all or another trainable scope." >&2
  exit 2
fi

if [[ "${preprocess}" == "true" ]]; then
  default_data_config_train=../configs/bair_robot_pushing/his_1_world_model_pretokenize_train.yaml
  default_data_config_val=../configs/bair_robot_pushing/his_1_world_model_pretokenize_test.yaml
else
  default_data_config_train=../configs/bair_robot_pushing/his_1_world_model_nopretokenize_train.yaml
  default_data_config_val=../configs/bair_robot_pushing/his_1_world_model_nopretokenize_test.yaml
fi
data_config_train=${DATA_CONFIG_TRAIN:-$default_data_config_train}
data_config_val_ind=${DATA_CONFIG_VAL_IND:-$default_data_config_val}
data_config_val_ood=${DATA_CONFIG_VAL_OOD:-$default_data_config_val}

exp_name=${EXP_NAME:-bair_his_1_world_model_transition_tokens}
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
  --train_only "${train_only}" \
  --preprocess "${preprocess}" \
  --dataset-kind "${dataset_kind}" \
  --with_world_model \
  "${transition_token_args[@]}" \
  --resolution 256 \
  --init_from "${init_from}" \
  --tokenizer_path "${tokenizer_path}" \
  --ablation 0 \
  --model_size 7B \
  --trainable-scope "${trainable_scope}" \
  --train-last-n-layers "${train_last_n_layers}" \
  --batch_size ${batch_size} \
  --accum_iter ${accum_iter} \
  --epochs ${epochs} \
  --warmup_epochs ${warmup_epochs} \
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
  --data_parallel ${data_parallel} \
  --output_dir "$output_dir"/"$exp_name" \
  --metrics_dir "$output_dir"/"$exp_name"/metrics \
  "${checkpointing_args[@]}" \
  "${only_save_trainable_args[@]}" \
  "${auto_resume_args[@]}" \
  "${custom_att_mask_args[@]}" \
  "${save_at_epoch_end_args[@]}" \
  "${promote_params_args[@]}" \
  "${load_model_args[@]}" \
  "${ignore_mismatch_args[@]}" \
  --save_iteration_interval ${save_every_steps} \
  --eval_iteration_interval ${eval_every_steps} \
  --eval_max_batches ${eval_max_batches} \
  --rollout_eval_iteration_interval ${rollout_eval_every_steps} \
  --rollout_eval_workdir "$script_dir" \
  --rollout_eval_gpu_ids "${rollout_eval_gpu_ids}" \
  --rollout_eval_num_gpus ${rollout_eval_num_gpus} \
  --rollout_eval_samples ${rollout_eval_samples} \
  --rollout_eval_future_frames ${rollout_eval_future_frames} \
  --rollout_eval_preview_samples ${rollout_eval_preview_samples} \
  --rollout_eval_min_free_mb ${rollout_eval_min_free_mb} \
  --rollout_eval_timeout_seconds ${rollout_eval_timeout_seconds} \
  --rollout_eval_state ${rollout_eval_state} \
  --max_train_steps ${max_train_steps} \
  --log_metrics_interval ${log_metrics_interval} \
  --max_seq_len 8192 \
  --unmask_image_logits \
  --dropout ${dropout} \
  --z_loss_weight ${z_loss_weight} \
  --loss_ct_weights ${loss_ct_weights} \
  --ckpt_max_keep ${ckpt_max_keep} \
  --best_checkpoint_metric "${best_checkpoint_metric}" \
  --best_checkpoint_split "${best_checkpoint_split}" \
  --best_checkpoint_mode "${best_checkpoint_mode}" \
  --best_checkpoint_dir "${best_checkpoint_dir}" \
  2>&1 | tee -a "$output_dir"/"$exp_name"/output.log

echo "exp name: $exp_name"
