# Transition Token 接入 RynnVLA-002 计划

这份文档维护我们把 transition-field / transition-token idea 接入
RynnVLA-002 autoregressive action world model 的方案。

## 为什么选 RynnVLA-002

RynnVLA-002 当前最适合我们作为第一版 backbone，核心原因是它的 world
model 任务本身已经是 action-conditioned autoregressive 形式：

```text
历史图像 tokens + action tokens -> 未来图像 tokens
```

我们要优先改的是 Action World Model / world-model training 链路，而不是只
预测机器人动作的 VLA policy 链路。

当前最相关的文件：

```text
rynnvla-002/pretrain_solver_awm_w_ck_action_head.py
xllmx/solvers/pretrain/pretrain_ck_action_head.py
rynnvla-002/model/modeling_xllmx_chameleon_ck_action_head.py
rynnvla-002/data/dataset.py
rynnvla-002/data/item_processor.py
rynnvla-002/eval_solver_libero_g_video_512_third_wrist.py
rynnvla-002/exps_libero_world_model/calculate_world_model_performance.py
```

## 环境约定

RynnVLA-002 子模块使用独立 uv 环境，避免和 beta 主项目环境互相污染：

```text
external/RynnVLA-002/.venv
```

创建和安装命令：

```bash
cd external/RynnVLA-002
uv venv --python /usr/bin/python3.10 .venv
rg -v '^nvidia-' requirements.txt > /tmp/rynnvla-requirements-nonvidia.txt
uv pip install --python .venv/bin/python -r /tmp/rynnvla-requirements-nonvidia.txt
```

这里不是裁掉 CUDA 依赖，而是避免 requirements 中显式写死的 `nvidia-*`
版本和 `torch==2.2.0` 自身依赖互相冲突；最终 CUDA wheel 由 torch 解析得到。
当前环境验证到：

```text
torch==2.2.0+cu121
torch.cuda.is_available() == True
```

必要 checkpoint 放在：

```text
rynnvla-002/ckpts/chameleon/tokenizer
rynnvla-002/ckpts/chameleon/starting_point
rynnvla-002/ckpts/starting_point -> chameleon/starting_point
rynnvla-002/ckpts/base_model
```

当前下载优先级是：先拿到 tokenizer、VQGAN tokenizer、`starting_point`
权重和 base model 的 HF tokenizer 文件；完整 `base_model` 权重不是 Phase 0/1
训练 smoke 的必需项。

大文件下载可用 `hf-mirror + aria2c` 断点续传：

```bash
aria2c -c -x 16 -s 16 -k 1M -j 3 \
  --retry-wait=5 --max-tries=0 \
  --timeout=60 --connect-timeout=20 \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  -i /tmp/rynnvla-worldvla-aria2.txt
```

## 当前 Backbone 接口

no-pretokenize 的 LIBERO world-model dataset 当前返回：

```text
conversations, images, actions, states
```

world-model 任务对应的 conversation 大致是：

```text
human: Generate the next image ... <|image|><|image|><|action|>
gpt:   <|image|><|image|>
```

`item_processor` 会把图像和 action 转成 Chameleon 离散 tokens。训练目标是
标准 causal LM cross entropy，主要监督未来图像 tokens。评测时做 open-loop
rollout：把模型生成的帧继续喂回历史窗口。

## 我们要接入什么

transition-token 版本的目标接口是：

```text
历史图像 tokens + action tokens + transition tokens -> 未来图像 tokens
```

其中：

```text
transition_token = Pred_e(z_context, action/state, dt)
```

这里的 transition token 应该是插入 token sequence 的连续 soft tokens。它们
不作为 CE target，本身 label 保持 `-100`，只作为未来图像 token 生成的动态
条件。

## 最小工程设计

在 Chameleon wrapper 里增加一个 transition 模块：

```text
TransitionPredictor:
  输入：context image/action/state embeddings 的 pooled 表征 + dt
  输出：M 个 Chameleon hidden size 的 soft transition tokens

TransitionCompose:
  输入：e_ik, e_kj
  输出：e_ij_composed
```

在 conversation 里，在 action block 后插入 transition placeholder：

```text
<|image|><|action|><|transition|> -> <|image|>
```

forward 时做：

```text
1. 正常从 input_ids 查 embedding；
2. 找到 transition placeholder 的位置；
3. 用 Pred_e(...) 生成的 soft token embedding 替换这些位置；
4. 用 inputs_embeds 调 Chameleon LM，labels 仍然按原本方式算 CE。
```

这样 backbone 的 AR 结构保持不变，但 transition token 会成为 future image
token prediction 的真实条件。

## 当前已落地的 Scaffold

当前分支先落地了最小 soft-token scaffold：

```text
transition placeholder token id: 16001
transition token count:          4
CLI 参数：
  --with-transition-tokens
  --transition-token-id
  --transition-token-count
  --transition-token-hidden-mult
```

代码位置：

```text
rynnvla-002/model/configuration_xllmx_chameleon.py
rynnvla-002/model/modeling_xllmx_chameleon_ck_action_head.py
rynnvla-002/pretrain_solver_awm_w_ck_action_head.py
rynnvla-002/data/dataset.py
rynnvla-002/data/item_processor.py
xllmx/solvers/pretrain/pretrain_ck_action_head.py
rynnvla-002/configs/bair_robot_pushing/
rynnvla-002/exps_bair_world_model/
```

新增 BAIR no-pretokenize world-model 入口：

```text
--dataset-kind bair_npz
--action_dim 4
--time_horizon 1
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

第一版训练脚本：

```text
cd rynnvla-002/exps_bair_world_model
bash train_bair_transition_tokens_nopretokenize.sh
```

小规模 smoke 训练脚本：

```text
cd rynnvla-002/exps_bair_world_model
bash train_bair_transition_tokens_smoke.sh
```

smoke 配置只读取 1 个 BAIR shard、16 个样本，并设置：

```text
--trainable-scope transition_only
--batch_size 1
--num_workers 0
```

正式 BAIR one-step 入口同样默认先用 `transition_only`，也就是冻结 Chameleon
backbone，只训练 `transition_token_adapter`。这是 Phase 1 的默认设置；如果后面
要对 backbone 做 LoRA 或全参微调，再单独新增训练范围。

它只是 Phase 1 的 one-step transition-token conditioning，不是最终 triangle loss。

当前实现逻辑：

```text
1. 数据里如果没有 token id 16001，模型行为保持原样；
2. no-pretokenize 数据路径在加 `--with-transition-tokens` 后，会把
   `<reserved16001>` 插到 world-model prompt 的 action block 后；
3. 如果 input_ids 中出现 token id 16001，则认为这些位置是 transition placeholder；
4. 模型用 placeholder 之前的上下文 embedding 做 mean pooling；
5. TransitionTokenAdapter 把 pooled context 映射成连续 soft tokens；
6. 用这些 soft tokens 替换 placeholder 的普通 embedding；
7. placeholder 对应 labels 强制设为 -100，不参与 CE；
8. future image tokens 继续按原始 causal LM CE 训练。
```

这个版本还没有接入 triangle loss，也还没有显式区分 `e_ik/e_kj/e_ij`。它的作用是
先打通“AR token backbone 能接收连续 transition soft token 条件”的工程入口。

## Loss 如何映射

我们保留当前 transition-field baseline loss 的逻辑，但视觉目标从 latent MSE
改成 image-token CE：

```text
edge_3way:
  CE(AR(z_i, e_ik), image_k)
  CE(AR(z_k, e_kj), image_j)
  CE(AR(z_i, e_ij), image_j)

identity:
  CE(AR(z_i, e_ii), image_i)
  额外约束 zero-dt transition 接近 identity 表示

composed_target:
  CE(AR(z_i, Comp(e_ik, e_kj)), image_j)

pixel / perceptual auxiliary:
  第一阶段先作为离线评测指标；
  若 CE 指标正常但画质/轨迹仍差，再考虑加入 decode 后的辅助 loss。
```

注意：不要把 `e_ij = Comp(e_ik, e_kj)` 写成出发假设。composition 在这里是
transition representation 的正则和语义约束，核心视觉目标仍然是未来图像
tokens 是否预测正确。

## 数据需要怎么改

RynnVLA 当前 world-model sample 基本是一跳：

```text
image_t, action_t -> image_{t+1}
```

我们的 loss 需要 triangle sample：

```text
i < k < j
image_i, image_k, image_j
action_i:k, action_k:j, action_i:j
dt_ik, dt_kj, dt_ij
```

需要新增一个 dataset path，能从 BAIR / Bridge / LIBERO 中采样上述 triangle。
第一版建议先接 BAIR 或 Bridge，因为 beta 仓库已经有 BAIR shards 和 rollout
评测基建。

## 怎么证明目的达到了

这个方法成功的标准不是“画质单点更好”，而是：

```text
视频生成质量基本不下降；
长程稳定性显著提高；
transition token 学到了可组合、可解释的动态语义。
```

质量保持指标：

```text
FVD
LPIPS
PSNR / SSIM
image-token CE / accuracy
qualitative rollout videos
```

长程稳定性指标：

```text
open-loop rollout 的 PSNR/SSIM/LPIPS 随 horizon 的曲线
rollout FVD
5/10/15/30 step drift curve
failure rate：物体消失、画面冻结、不可控变糊、轨迹明显跑偏
```

transition token 语义指标：

```text
direct prediction vs composed prediction agreement
multi-hop rollout vs direct endpoint consistency
transition token -> action/dt 的 linear probe
transition token 按 action type / horizon 的 nearest-neighbor clustering
same-action different-state sensitivity
same-state different-action sensitivity
```

## 训练计划

Phase 0：复现 backbone。

```text
跑通 RynnVLA-002 world-model inference / training smoke。
确认 tokenizer、Chameleon checkpoint、FSDP、generation 和原始 metrics 都能工作。
```

Phase 1：最小 transition-token conditioning。

```text
只用 one-step samples。
冻结大部分 Chameleon backbone。
只训练 TransitionPredictor，必要时加少量 adapter / LoRA。
目标：future image token CE 和 rollout 质量不劣于原始 baseline。
```

Phase 2：加入 triangle transition-field loss。

```text
使用 i<k<j samples。
训练 edge_3way + identity + composed_target。
和同一个 RynnVLA one-step baseline 做对照。
```

Phase 3：提升 rollout 鲁棒性。

```text
加入 generated-state rollout training / scheduled sampling。
把长程 open-loop rollout 作为主评测指标。
```

## 当前最近任务

1. 准备小规模 smoke 环境，下载 RynnVLA / Chameleon 必要 checkpoint。
2. 如果已有 LIBERO 数据和 checkpoint，先跑 tiny subset 的 world-model eval。
3. 增加 BAIR / Bridge triangle dataset adapter。
4. 实现 transition placeholder 和 soft transition-token forward path。
5. 先验证 one-step 质量不下降，再加入 composition / multi-hop loss。
