# BAIR 训练基建

## BAIR 数据入口

新增 BAIR no-pretokenize world-model 入口：

```text
--dataset-kind bair_npz
--action_dim 4
--time_horizon 1
```

新增 BAIR pretokenize 入口：

```bash
cd rynnvla-002/exps_bair_world_model
GPU_IDS=0,1,2,3 BAIR_SPLITS=train,test bash prepare_bair_pretokenize.sh
```

它会把 BAIR 原始 `.npz` 样本离线转换为 RynnVLA 训练时直接消费的 token/label，并写到：

```text
../processed_data/bair_robot_pushing_tokens/his_1_world_model_train/record.json
../processed_data/bair_robot_pushing_tokens/his_1_world_model_test/record.json
```

对应训练配置：

```text
../configs/bair_robot_pushing/his_1_world_model_pretokenize_train.yaml
../configs/bair_robot_pushing/his_1_world_model_pretokenize_test.yaml
```

pretokenize 的原理是把当前训练循环里的 deterministic 数据处理提前做完：

```text
BAIR frames/actions
-> conversation + [image_i, image_j] + action
-> Chameleon VQ image tokens + action tokens + text tokens
-> input_ids / labels
-> pkl token files + record.json manifest
```

训练时 `PREPROCESS=true` 会直接读取 token/label，不再在每个 batch 里在线跑 VQ image tokenizer 和 action discretization。

BAIR 默认不使用官方的每样本一个 pkl 形式，而是写 `.npy` 分片：

```text
record.json:
  每条样本记录 token_file / label_file / index / len

shards/*_tokens.npy:
  int32 token id array

shards/*_labels.npy:
  int32 label array
```

训练侧用 `mmap_mode="r"` 读取分片，避免 125 万个小 pkl 文件带来的 inode 和随机小文件开销。

副作用和注意事项：

```text
1. 会占额外磁盘；当前 BAIR 默认 `.npy` 分片以减少 inode 压力。
2. tokenizer、resolution、prompt、action normalization、transition token 设置一旦变化，cache 必须重建。
3. 随机数据增强如果放进 pretokenize，会被固定下来；当前 BAIR 256 resize/crop 路径基本是确定性的，所以可以接受。
4. pretokenize 只优化数据流水线，不改变模型能力，也不会修复 baseline 没学会运动这类建模问题。
```

BAIR shard 读取格式：

```text
frames:  [N, T, H, W, C]
actions: [N, T-1, 4]
```

当前支持两类 sample mode：

```text
one_step:
  image_t + action_t -> image_{t+1}

direct_endpoint:
  image_i + sum(action_i:j) -> image_j
```

## 训练脚本

第一版正式训练脚本：

```bash
cd rynnvla-002/exps_bair_world_model
bash train_bair_transition_tokens_nopretokenize.sh
```

正式 BAIR next-frame baseline 入口：

```bash
cd rynnvla-002/exps_bair_world_model
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash train_bair_world_model_baseline_nopretokenize.sh 1 4
```

这个入口从 `../ckpts/starting_point` 初始化，主要用于“从 Chameleon/WorldVLA 底座开始训练 BAIR baseline”。当前更推荐先使用 RynnVLA-002 官方 action-world-model checkpoint 初始化：

```bash
cd rynnvla-002/exps_bair_world_model
HF_ENDPOINT=https://hf-mirror.com \
MODEL_SUBDIR=Action_World_model_512/libero_spatial \
  bash download_rynnvla_official_awm_ckpt.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
OFFICIAL_AWM_CHECKPOINT=../ckpts/Action_World_model_512/libero_spatial \
EXP_NAME=bair_world_model_from_official_awm \
  bash train_bair_world_model_from_official_awm_nopretokenize.sh 1 4
```

官方 AWM 初始化入口显式设置：

```text
WITH_TRANSITION_TOKENS=false
TRAINABLE_SCOPE=all
ONLY_SAVE_TRAINABLE=false
PRECISION=bf16
PROMOTE_PARAMS_TO_FP32=false
IGNORE_MISMATCHED_SIZES=true
```

也就是说，它会从 RynnVLA-002 已训练好的 action-world-model checkpoint 出发微调整个 BAIR world model。`starting_point` 仍可用于 smoke 或从底座重训，但不是当前主线。

官方 `Action_World_model_512/libero_spatial` 使用 LIBERO action 规格：

```text
action_dim=7
time_horizon=10
```

BAIR 当前入口使用：

```text
action_dim=4
time_horizon=1
```

因此官方 AWM -> BAIR 初始化默认设置 `IGNORE_MISMATCHED_SIZES=true`：主体 Transformer / image-token world-model 权重会继承，尺寸不匹配的 `action_head` 会在 BAIR 配置下重新初始化。

可以用轻量脚本确认官方 checkpoint 是否完整，以及 action head 为什么需要按 BAIR 配置重建：

```bash
cd rynnvla-002/exps_bair_world_model
../../.venv/bin/python inspect_rynnvla_awm_checkpoint.py \
  ../ckpts/Action_World_model_512/libero_spatial
```

当前 `libero_spatial` 的检查结果应显示：

```text
num_tensors=464
total_size_bytes_from_index=14104606734
checkpoint_action_dim=7
checkpoint_time_horizon=10
target_action_dim=4
target_time_horizon=1
requires_ignore_mismatched_sizes=true
missing_shards=[]
aria2_files=[]
```

实现细节上，当 `IGNORE_MISMATCHED_SIZES=true` 时，训练代码不会使用 `device_map="cpu"` 加载 checkpoint。否则缺失/重建的 `transition_token_adapter` 和 BAIR 形状的 `action_head` 可能残留在 meta tensor 状态，导致初始化失败。

2 卡资源不足以支撑 7B 全参 AdamW baseline。2 卡先使用参数高效 baseline：

```bash
cd rynnvla-002/exps_bair_world_model
CUDA_VISIBLE_DEVICES=0,2 \
  bash train_bair_world_model_baseline_last_layers_nopretokenize.sh 1 2
```

这个版本训练：

```text
model.embed_tokens
最后 TRAIN_LAST_N_LAYERS 个 decoder layer，默认 4
model.norm
lm_head
action_head
```

它不是全参 baseline；后续如果拿到 4/8 张空卡，仍应优先补全参 baseline。

默认是单机多卡 FSDP。单机 8 卡正式跑法：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash train_bair_transition_tokens_nopretokenize.sh 1 8
```

小规模 smoke 训练脚本：

```bash
cd rynnvla-002/exps_bair_world_model
bash train_bair_transition_tokens_smoke.sh
```

smoke 配置只读取 1 个 BAIR shard、16 个样本，并设置：

```text
--trainable-scope transition_only
--batch_size 1
--num_workers 0
```

正式 BAIR one-step 入口同样默认先用 `transition_only`，也就是冻结 Chameleon backbone，只训练 `transition_token_adapter`。这是 Phase 1 的默认设置；如果后面要对 backbone 做 LoRA 或全参微调，再单独新增训练范围。

它只是 Phase 1 的 one-step transition-token conditioning，不是最终 triangle loss。

注意：`transition_only` 只能验证 transition soft token 能否稳定接入和被 checkpoint；如果底座还没有完成 BAIR next-frame 训练，adapter-only 结果不能代表视频生成质量，也不能和正式 baseline 做结论性对比。

## CLI 参数

脚本里的关键参数都可以用环境变量覆盖：

```text
BATCH_SIZE=1
ACCUM_ITER=1
EPOCHS=1
WARMUP_EPOCHS=0.01
TRAIN_ONLY=false
WITH_TRANSITION_TOKENS=true
TRAINABLE_SCOPE=transition_only
TRAIN_LAST_N_LAYERS=4
PRECISION=tf32
DATA_PARALLEL=fsdp
CHECKPOINTING=true
ONLY_SAVE_TRAINABLE=true
CUSTOM_ATT_MASK=false
AUTO_RESUME=true
TRANSITION_TOKEN_HIDDEN_MULT=1
SAVE_EVERY_STEPS=5000
SAVE_AT_EPOCH_END=true
EVAL_EVERY_STEPS=5000
EVAL_MAX_BATCHES=32
MAX_TRAIN_STEPS=0
CKPT_MAX_KEEP=3
LOG_METRICS_INTERVAL=50
EXP_NAME=bair_his_1_world_model_transition_tokens
OUTPUT_DIR=../outputs/bair_robot_pushing
INIT_FROM=../ckpts/starting_point
TOKENIZER_PATH=../ckpts/base_model
PROMOTE_PARAMS_TO_FP32=true
LOAD_MODEL_ON_ALL_RANKS=false
```

`WARMUP_EPOCHS` 是按完整 dataloader 的 epoch 长度换算的。完整 BAIR one-step 目前约 418k updates/epoch；因此 `WARMUP_EPOCHS=0.01` 约等于 4k warmup steps。短跑验证如果只设 `MAX_TRAIN_STEPS=200`，应显式设 `WARMUP_EPOCHS=0`，否则前 200 step 的 lr 基本还没有起来。

当前默认 `PRECISION=tf32`。实测 `bf16` 在 transition-token 反传路径上仍可能产生非有限梯度；脚本已经在 optimizer step 前加了梯度 finite 检查，避免写出被 NaN 污染的 adapter checkpoint。后续如果要重新启用 `bf16`，必须先让 smoke 同时满足 finite loss、finite grad norm、checkpoint 权重全 finite。

无 transition token 的全参 BAIR baseline wrapper 默认使用 `PRECISION=bf16`，这更接近 RynnVLA 原训练路径，也能显著降低全参 FSDP 的显存压力。2 卡 full baseline 上，fp32 参数/AdamW 状态会在 optimizer step OOM；baseline wrapper 因此默认 `PROMOTE_PARAMS_TO_FP32=false`。8 卡正式跑如果显存充足，可以显式设回 `PROMOTE_PARAMS_TO_FP32=true`。transition-token adapter 实验仍先用 `tf32` 和 fp32 adapter 参数。

`LOAD_MODEL_ON_ALL_RANKS=true` 会让每个 data-parallel rank 都从 CPU 侧加载同一份 checkpoint，再交给 FSDP 分片。它会增加 CPU 内存和 checkpoint 读取开销，但能绕开 rank0-only meta init + `sync_module_states` 在部分微调时出现的显存不均和 NCCL collective 卡死。当前 `last_layers` baseline 默认启用这个选项；全参 baseline 仍默认关闭，除非多卡 smoke 证明需要开启。

当前默认 `TRANSITION_TOKEN_HIDDEN_MULT=1`。`hidden_mult=4` 在 smoke 中暴露过非有限梯度风险；`hidden_mult=1` 已通过 1-step adapter-only smoke，并写出了 finite checkpoint。正式实验先以这个配置作为基线，后续再单独做 adapter 宽度消融。

当前 BAIR 脚本默认 `CUSTOM_ATT_MASK=false`。RynnVLA 原始 `generate_att_mask_3` 面向原动作/图像块格式；在 BAIR 的 `image + action + transition token -> next image` prompt 上，开启它会触发首个 batch 的 NaN loss。后续如果要重新启用，必须先重写并单测 BAIR 专用 mask。

`transition_token_adapter` 现在采用稳定冷启动：placeholder 位置的普通 reserved token embedding 会被连续 soft token 替换；初始 soft token 为 0，adapter 最后一层也从 0 初始化，训练开始后再逐步学习非零 transition token。per-slot position 只是非训练 buffer，避免单独的位置参数在冷启动阶段数值失控。

`CHECKPOINTING=true` 指 activation/gradient checkpointing：forward 不保存部分中间 activation，backward 时重算，以计算换显存。显存足够时可以设 `CHECKPOINTING=false`，通常会更快。pretokenized 官方 AWM baseline wrapper 默认关掉它；正式长跑前仍要先用相同 batch size 做 20-50 step smoke，确认不会 OOM。

## Checkpoint / Resume

```text
--save_iteration_interval:
  每多少个 optimizer update 保存一次 epochX-iterY。

--save_interval:
  每多少个 epoch 额外保存一次 epochX。

SAVE_AT_EPOCH_END=false:
  关闭 epoch 末尾保存；只用于短 memory smoke，正式训练不要关闭。

--ckpt_max_keep:
  只保留最近 N 个 checkpoint；<=0 表示全部保留。

BEST_CHECKPOINT_METRIC / BEST_CHECKPOINT_SPLIT / BEST_CHECKPOINT_MODE:
  每次 step eval 后，用指定 split 上的指标更新 best checkpoint。

--auto_resume:
  默认开启，会从 output_dir 下最新 epoch* checkpoint 恢复。

--resume_path:
  手动指定 checkpoint，优先级高于 auto_resume。
```

当前正式训练默认 `CKPT_MAX_KEEP=1`：磁盘上只长期保留一个最新 step checkpoint，并维护：

```text
latest -> epochX-iterY
best/
best_checkpoint.json
```

`latest` 用于断点恢复；`best/` 保存历史 eval 中表现最好的 checkpoint。这样等价于只保留“最新”和“最优”两份有效 checkpoint，避免每 2000 step 都堆完整 7B 权重。

`transition_only` 默认开启 `--only_save_trainable`，checkpoint 只保存：

```text
trainable_params.pt
optimizer.*.pth
args.json
checkpoint_meta.json
```

resume 时先从 `--init_from` 加载 base Chameleon checkpoint，再叠加 `trainable_params.pt`，因此不会把 7B 冻结 backbone 重复写进每个实验 checkpoint。

当 `TRAINABLE_SCOPE=all` 时必须设置 `ONLY_SAVE_TRAINABLE=false`。这种 checkpoint 会保存完整模型权重，磁盘和保存时间都会明显变大，但它才是正式 BAIR baseline 能用于 rollout 的格式。

正式 transition-token candidate 应从已经训练好的 BAIR baseline 初始化，而不是从 `starting_point` 初始化。推荐链路是：

```bash
cd rynnvla-002/exps_bair_world_model
CUDA_VISIBLE_DEVICES=0,1,2,3 \
INIT_FROM=../outputs/bair_robot_pushing/bair_world_model_from_official_awm/epoch0 \
EXP_NAME=bair_world_model_transition_tokens_on_baseline \
  bash train_bair_transition_tokens_nopretokenize.sh 1 4
```

adapter-only checkpoint 的 `checkpoint_meta.json` 会记录这个 `INIT_FROM`；离线 rollout eval 加载 adapter 时会优先使用这条 meta 里的 base checkpoint。需要手动覆盖时再传 `BASE_CHECKPOINT_PATH=...`。

## 训练中 Eval

```text
TRAIN_ONLY=false:
  打开训练过程中的 eval。

--eval_iteration_interval:
  每多少个 optimizer update 跑一次 val_ind / val_ood。

--eval_max_batches:
  每个 split 最多评估多少个 batch；smoke 用 1，正式短评估可用 32，
  论文级完整评估设为 0。

--eval_at_start:
  训练前先评估一次初始模型。

--no_eval_at_epoch_end:
  关闭 epoch 末尾 eval。
```

eval-only 脚本：

```bash
cd rynnvla-002/exps_bair_world_model
CHECKPOINT_PATH=../outputs/bair_robot_pushing/bair_his_1_world_model_transition_tokens/epoch0 \
  bash eval_bair_transition_tokens.sh
```

这个脚本是 teacher-forced validation，只看模型在给定 GT prompt / GT history 时的 token loss 和 accuracy。

## 离线 Rollout Eval

最终和 baseline 对比使用独立的 open-loop rollout 脚本：

```bash
cd rynnvla-002/exps_bair_world_model
CHECKPOINT_PATH=../outputs/bair_robot_pushing/bair_world_model_baseline/epoch0 \
WITH_TRANSITION_TOKENS=false \
EXP_NAME=bair_baseline_rollout \
bash eval_bair_rollout.sh
```

微调 checkpoint：

```bash
CHECKPOINT_PATH=../outputs/bair_robot_pushing/bair_his_1_world_model_transition_tokens/epoch0 \
WITH_TRANSITION_TOKENS=true \
EXP_NAME=bair_transition_tokens_rollout \
bash eval_bair_rollout.sh
```

常用环境变量：

```text
SAMPLES=32
CONTEXT_FRAMES=1
FUTURE_FRAMES=15
SEED=7
DEVICE=auto
SAVE_GIF=true
PREVIEW_SAMPLES=8
LPIPS=false
FVD_MODEL_PATH=
FVD_SAMPLES=32
FORCE_IMAGE_PREFIX=true
ROLLOUT_STATE=token
```

这个脚本的生成方式是：

```text
frame_t + action_t -> pred_frame_{t+1}
pred_frame_{t+1} + action_{t+1} -> pred_frame_{t+2}
...
```

它评估的是长程 open-loop drift，而不是训练中的 teacher-forced CE。

`FORCE_IMAGE_PREFIX=true` 只固定生成图像块的 start 和 256x256 grid token，不给目标图像内容；实际被评估的仍然是模型生成的 image latent token。

`ROLLOUT_STATE=token` 是默认 open-loop 递推方式：第 0 帧先 VQ 编码成 image token，后续每一步直接把模型生成出的 image token 作为下一步输入状态，只在保存 GIF / metric 时解码成 RGB。旧的 `ROLLOUT_STATE=image` 会把每一步生成图先解码成 RGB，再重新 VQ 编码进下一步 prompt，速度更慢，也会额外引入一次 encode/decode 量化噪声；它只作为排查用回退路径保留。

`../ckpts/starting_point` 可以用来做脚本链路 smoke test，但它没有 BAIR fine-tune，不是正式 baseline。
离线 rollout metrics 里会写入 `diagnostics.checkpoint_kind` 和 `diagnostics.formal_rollout_candidate`。如果 run label 含 `baseline` 但 checkpoint 不是正式 BAIR-finetuned 模型，脚本会拒绝运行，除非 label 明确包含 `smoke` 或显式传 `ALLOW_NONFORMAL_BASELINE=true`。

rollout 生成是逐 clip 自回归生成，默认脚本是单进程单卡：

```bash
CHECKPOINT_PATH=../outputs/bair_robot_pushing/bair_world_model_baseline/epoch0 \
GPU_IDS=3,5,6,7 \
SAMPLES=32 \
EXP_NAME=bair_baseline_rollout_4gpu \
bash eval_bair_rollout_multi.sh
```

多卡 rollout 不使用 DDP/FSDP，而是把同一组 `SAMPLES` 按 `sample_idx % SHARD_COUNT` 切成多个 shard，每张 GPU 跑一个独立进程，最后聚合：

```text
rollout_eval/shard_0/metrics.json
rollout_eval/shard_1/metrics.json
rollout_eval/metrics.json
rollout_eval/rollout_artifacts.json
```

训练 runner 里也可以直接启用：

```bash
ROLLOUT_NUM_GPUS=4 ROLLOUT_GPU_IDS=3,5,6,7 bash ./run_bair_train_eval.sh
```

如果有 5 张空卡，推荐 4 张用于训练、1 张用于 rollout/eval；如果正在单独跑 rollout，则推荐 4 张训练、剩余卡跑评估。5 卡 FSDP 不是不可以，但会改变 global batch 和通信形态，正式可比实验里最好固定卡数。

当前训练脚本默认 `BATCH_SIZE=1` 是保守启动配置，不是行业规定。7B 多模态模型在全参 FSDP + AdamW + image token 序列下显存波动较大，先用 micro-batch 1 能稳定排除 checkpoint、shape、mask、loss、保存恢复这些工程问题。显存足够时应该测试 `BATCH_SIZE=2` 和 `BATCH_SIZE=4`，尤其是 adapter-only 或 last-layers 微调；但要记住 global batch 会变成：

```text
global_batch = BATCH_SIZE * NUM_GPUS * ACCUM_ITER
```

因此把 4 卡 `BATCH_SIZE=1, ACCUM_ITER=1` 改成 `BATCH_SIZE=2` 会把 global batch 从 4 改成 8，不只是速度变化，也会改变优化噪声和可能的 lr 最优点。正式可比实验需要记录 global batch，必要时单独做 batch-size 消融。

rollout eval 会边跑边写 partial 产物，避免长时间看不到结果：

```text
partial_metrics.json
partial_status.json
partial_sample_manifest.json
partial_prediction_grid.png
partial_rollout_videos/
```

训练过程中可以把 open-loop rollout eval 正式接到训练 step 里。它由 rank0 在保存 checkpoint 后启动，其他 FSDP rank 等待 eval 结束，避免多个 rank 重复生成。`SAVE_EVERY_STEPS=2000` 时，训练代码通常会写出 `epoch0-iter1999`、`epoch0-iter3999`，分别表示第 2000 / 4000 个 update 后的 checkpoint：

```bash
cd rynnvla-002/exps_bair_world_model
CUDA_VISIBLE_DEVICES=4,5,6,7 \
EXP_NAME=bair_world_model_from_official_awm_full_bs4_10k_4gpu_0508 \
MAX_TRAIN_STEPS=10000 \
BATCH_SIZE=4 \
SAVE_EVERY_STEPS=2000 \
ROLLOUT_EVAL_EVERY_STEPS=2000 \
ROLLOUT_EVAL_GPU_IDS=4 \
ROLLOUT_EVAL_SAMPLES=8 \
ROLLOUT_EVAL_FUTURE_FRAMES=15 \
ROLLOUT_EVAL_PREVIEW_SAMPLES=4 \
bash train_bair_world_model_from_official_awm_nopretokenize.sh 1 4
```

周期 rollout 输出在：

```text
<TRAIN_OUTPUT_DIR>/periodic_rollout_eval/rollout_iter1999_step2000/
<TRAIN_OUTPUT_DIR>/periodic_rollout_eval/rollout_iter3999_step4000/
```

每个目录都会包含 `metrics.json`、`prediction_grid.png`、`rollout_videos/*.gif`。rollout eval 会重新加载一个 7B checkpoint，所以需要额外空闲显存；如果训练已经占满全部 GPU，就只能在训练过程中做 teacher-forced eval，open-loop rollout/GIF 会因为 `ROLLOUT_EVAL_MIN_FREE_MB` 不满足而等待或跳过。

如果目标是每 2000 step 都稳定产出 GIF，推荐固定预留至少 1 张 GPU 给 rollout eval，例如 7 卡训练 + 1 卡 rollout，或者训练结束/暂停后再用 `latest` / `best` 做离线 rollout。8 卡全量训练适合追求吞吐，但不要预期它在同一时间还能额外加载一份 7B 模型生成 GIF。

## 输出产物

每次训练 / eval 输出：

```text
output.log:
  shell tee 出来的总日志。

common.log / rank-*.log:
  solver 日志。

log_train.txt:
  flat JSONL train metrics。

log_eval_ind.txt / log_eval_ood.txt:
  flat JSONL validation metrics。

tensorboard/:
  标量曲线。

metrics/metrics.jsonl:
  grouped JSONL，按 losses / accuracies / action / optimization 分组。

metrics/latest*.json:
  最近一次 train/eval 的 grouped metrics 快照；train 记录会额外带 progress，包括 elapsed_sec、sec_per_step、remaining_steps、eta_sec、estimated_finish_time。

rollout_eval/metrics.json:
  grouped rollout 指标，按 run / config / quality / per_horizon / fvd / failures 分组。

rollout_eval/rollout_videos/*.gif:
  GT 与 autoregressive rollout 的对齐 GIF。

rollout_eval/prediction_grid.png:
  固定样本的静态预测网格。

rollout_eval/horizon_metrics.png:
  horizon drift 曲线。

rollout_comparison/comparison.md:
  baseline 与 candidate 的 grouped metric delta 表。
```

当前训练中直接计算的指标：

```text
losses:
  closs              image/action token causal LM CE
  loss_ct            continuous action head L1
  z_loss             optional logit z-loss

accuracies:
  acc_image_*        image token accuracy by block position
  acc_action_*       action token accuracy by block position

action:
  l1_loss_action_*   decoded continuous action L1 by block position

optimization:
  lr
  grad_norm          train only
```

这些是 teacher-forced 指标，用来判断训练是否正常、transition adapter 是否能接入。正式论文对比还需要离线 rollout 评估，不应该混在每 N step 的轻量训练 eval 里。
