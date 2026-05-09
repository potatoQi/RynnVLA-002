# Transition Token 接入 RynnVLA-002

这份文档只作为入口索引，避免把环境、算法、训练和评估全部堆在同一个文件里。

## 文档结构

- [RynnVLA Backbone 与环境](rynnvla_backbone_environment.md)
  - 为什么选 RynnVLA-002
  - 子模块 uv 环境
  - checkpoint / tokenizer 目录约定
  - 当前 backbone 的 world-model 接口

- [Transition Token 设计](transition_token_design.md)
  - 接入目标
  - soft transition token 的最小工程设计
  - 当前 scaffold
  - loss 与 triangle data 后续映射

- [BAIR 训练基建](bair_training_infra.md)
  - BAIR npz 数据格式
  - smoke / 正式训练命令
  - 多卡 FSDP 启动方式
  - checkpoint / resume / step eval
  - grouped metrics 输出

- [评估与对比协议](evaluation_protocol.md)
  - baseline vs ours 的固定对照
  - teacher-forced 指标
  - rollout / FVD / LPIPS / PSNR / SSIM
  - transition token 语义指标

- [Roadmap](transition_token_roadmap.md)
  - Phase 0 到 Phase 3
  - 当前最近任务

## 当前状态

当前已经完成 Phase 1 的最小工程入口和 smoke 验证：

```text
历史图像 tokens + action tokens + transition soft tokens -> 未来图像 tokens
```

RynnVLA / Chameleon 7B checkpoint 可以用于链路 smoke，但它还不是 BAIR world-model baseline。正式 baseline 需要先训练无 transition token 的 BAIR next-frame model：

```bash
cd rynnvla-002/exps_bair_world_model
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash train_bair_world_model_baseline_nopretokenize.sh 1 4
```

当前训练脚本默认 `tf32`，因为 `bf16` 在 adapter 反传上还没有通过 finite-gradient smoke。adapter-only transition-token 训练只说明接入和保存链路可用；在底座没有 BAIR fine-tune 前，rollout 图像失败是预期现象，不能用于方法结论。
