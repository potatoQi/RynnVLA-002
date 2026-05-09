# 评估与对比协议

## 成功标准

这个方法成功的标准不是“画质单点更好”，而是：

```text
视频生成质量基本不下降；
长程稳定性显著提高；
transition token 学到了可组合、可解释的动态语义。
```

## 固定对照

第一组必须固定的对照：

```text
baseline:
  WITH_TRANSITION_TOKENS=false
  TRAINABLE_SCOPE=all
  ONLY_SAVE_TRAINABLE=false
  EXP_NAME=bair_world_model_baseline

ours phase1:
  WITH_TRANSITION_TOKENS=true
  TRAINABLE_SCOPE=transition_only
  INIT_FROM=<same BAIR baseline checkpoint>
  EXP_NAME=bair_world_model_transition_tokens
```

两组应使用同一份 BAIR train/test config、同样的 batch/global batch、训练步数、checkpoint 选择规则和完整离线 rollout 评估脚本。

Phase 1 的最低目标不是短程 teacher-forced CE 明显降低，而是画质不下降，同时长程 rollout 稳定性和 transition token 语义指标更好。

`../ckpts/starting_point` 和 `transition_adapter_on_starting_point` 都只能算链路 smoke。它们还没有学会 BAIR next-frame generation，不能进入正式 baseline/candidate 表格。

## Teacher-Forced 指标

训练中直接计算的轻量指标：

```text
closs
loss_ct
z_loss
acc_image_*
acc_action_*
l1_loss_action_*
lr
grad_norm
```

这些指标用于检查训练是否正常、transition adapter 是否接入成功、loss 是否爆炸或退化。它们不能替代最终 rollout 评估。

## 质量保持指标

```text
FVD
LPIPS
PSNR / SSIM
image-token CE / accuracy
qualitative rollout videos
```

## 长程稳定性指标

```text
open-loop rollout 的 PSNR/SSIM/LPIPS 随 horizon 的曲线
rollout FVD
5/10/15/30 step drift curve
failure rate：物体消失、画面冻结、不可控变糊、轨迹明显跑偏
```

## Transition Token 语义指标

```text
direct prediction vs composed prediction agreement
multi-hop rollout vs direct endpoint consistency
transition token -> action/dt 的 linear probe
transition token 按 action type / horizon 的 nearest-neighbor clustering
same-action different-state sensitivity
same-state different-action sensitivity
```

## 完整评估策略

训练中每 N step 的 eval 只跑少量 batch，主要用于监控。论文级对比应选择固定 checkpoint，单独跑完整离线评估：

```text
1. teacher-forced val loss / token accuracy
2. fixed-seed rollout generation
3. rollout videos / grids
4. FVD / LPIPS / PSNR / SSIM
5. horizon drift curves
6. transition token semantic probes
```

## BAIR 离线 Rollout

新增离线 rollout 入口：

```bash
cd rynnvla-002/exps_bair_world_model
CHECKPOINT_PATH=../outputs/bair_robot_pushing/bair_world_model_baseline/epoch0 \
WITH_TRANSITION_TOKENS=false \
EXP_NAME=bair_baseline_rollout \
bash eval_bair_rollout.sh
```

微调模型使用同一脚本、同一 seed、同一数据，只替换 checkpoint 和 transition-token 开关：

```bash
CHECKPOINT_PATH=../outputs/bair_robot_pushing/bair_his_1_world_model_transition_tokens/epoch0 \
WITH_TRANSITION_TOKENS=true \
EXP_NAME=bair_transition_tokens_rollout \
bash eval_bair_rollout.sh
```

这一步是 open-loop autoregressive 评估：给定第 0 帧和真实 action，生成第 1 帧；再把生成帧作为下一步输入继续生成。默认评估 BAIR test split 的 32 条 16-frame clip：

```text
context_frames = 1
future_frames  = 15
samples        = 32
resolution     = 256
```

脚本默认设置 `FORCE_IMAGE_PREFIX=true`，即在生成端固定目标图像块的 `<image_start>, h_grid, w_grid` 前缀，然后只让模型生成 VQ image content / newline / image_end。这个前缀只定义输出格式和分辨率，不包含目标图像内容；它能避免裸生成先吐文本 token，导致无法解码成图像。

主要输出：

```text
metrics.json:
  grouped JSON，只保留 run / config / quality / per_horizon / fvd / failures。

rollout_videos/*.gif:
  gt vs rollout 的逐帧可视化。

prediction_grid.png:
  固定样本的 GT / rollout 静态网格。

horizon_metrics.png:
  MSE / PSNR / SSIM / LPIPS 随 future horizon 的变化。

sample_manifest.json:
  每条 clip 来自哪个 shard、seq_idx，以及 decode failure 统计。
```

需要完整指标时开启：

```bash
LPIPS=true \
FVD_MODEL_PATH=../../../../models/fvd/i3d_torchscript.pt \
FVD_SAMPLES=32 \
bash eval_bair_rollout.sh
```

baseline 和 ours 必须使用同一份 `metrics.json` schema 做对比，避免训练中 teacher-forced 指标和最终 rollout 指标混在一起。

`../ckpts/starting_point` 只适合做链路 smoke test；它不是 BAIR fine-tuned baseline，不能作为正式对照。
`metrics.json` 的 `diagnostics` 组会标记 checkpoint 类型；正式对比只接受 `diagnostics.formal_rollout_candidate=true` 的结果。

两个 rollout 跑完后可以生成对比表：

```bash
cd rynnvla-002/exps_bair_world_model
../../.venv/bin/python compare_bair_rollout_metrics.py \
  --baseline ../outputs/bair_robot_pushing/bair_baseline_rollout/rollout_eval/metrics.json \
  --candidate ../outputs/bair_robot_pushing/bair_transition_tokens_rollout/rollout_eval/metrics.json \
  --out-dir ../outputs/bair_robot_pushing/rollout_comparison \
  --baseline-name baseline \
  --candidate-name transition_tokens
```

输出 `comparison.json` 和 `comparison.md`，重点看 `quality.*`、`per_horizon.*`、`fvd.*`、`failures.*` 的 delta。
对比脚本默认同样拒绝 `diagnostics.formal_rollout_candidate=false` 的 metrics；只做 smoke/debug 对比时才使用 `--allow-nonformal`。
