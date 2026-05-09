# RynnVLA Backbone 与环境

## 为什么选 RynnVLA-002

RynnVLA-002 当前最适合作为第一版 backbone，因为它的 world model 任务本身已经是 action-conditioned autoregressive 形式：

```text
历史图像 tokens + action tokens -> 未来图像 tokens
```

我们优先改的是 Action World Model / world-model training 链路，不是只预测机器人动作的 VLA policy 链路。

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

这里不是裁掉 CUDA 依赖，而是避免 requirements 中显式写死的 `nvidia-*` 版本和 `torch==2.2.0` 自身依赖互相冲突；最终 CUDA wheel 由 torch 解析得到。

当前环境验证到：

```text
torch==2.2.0+cu121
torch.cuda.is_available() == True
```

## Checkpoint 目录

必要 checkpoint 放在：

```text
rynnvla-002/ckpts/chameleon/tokenizer
rynnvla-002/ckpts/chameleon/starting_point
rynnvla-002/ckpts/starting_point -> chameleon/starting_point
rynnvla-002/ckpts/base_model
rynnvla-002/ckpts/Action_World_model_512/libero_spatial
```

当前下载优先级是：

```text
1. tokenizer、VQGAN tokenizer、base_model 的 HF tokenizer 文件；
2. RynnVLA-002 官方 Action_World_model_512/libero_spatial；
3. starting_point 只作为 smoke / fallback / 从底座重训初始化。
```

`starting_point` 来自 WorldVLA/Chameleon minimal checkpoint，不是官方已经训练好的 RynnVLA-002 action-world-model。当前主线应优先使用 `Action_World_model_512/libero_spatial` 作为 BAIR baseline 的初始化点。

官方 AWM checkpoint 下载完成后，可以快速检查 shard 完整性和 action head 形状：

```bash
cd rynnvla-002/exps_bair_world_model
../../.venv/bin/python inspect_rynnvla_awm_checkpoint.py \
  ../ckpts/Action_World_model_512/libero_spatial
```

当前已验证 `libero_spatial` 包含 3 个 safetensors shard，总大小约 14.10GB，checkpoint 自身是 `action_dim=7,time_horizon=10`。BAIR 入口是 `action_dim=4,time_horizon=1`，所以从该 checkpoint 初始化 BAIR baseline 时需要跳过并重建尺寸不匹配的 `action_head`。

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

`item_processor` 会把图像和 action 转成 Chameleon 离散 tokens。训练目标是标准 causal LM cross entropy，主要监督未来图像 tokens。评测时做 open-loop rollout：把模型生成的帧继续喂回历史窗口。
