# Transition Token Roadmap

## Phase 0：复现 Backbone

```text
跑通 RynnVLA-002 world-model inference / training smoke。
确认 tokenizer、Chameleon checkpoint、FSDP、generation 和原始 metrics 都能工作。
```

## Phase 1：最小 Transition-Token Conditioning

```text
只用 one-step samples。
冻结大部分 Chameleon backbone。
只训练 TransitionPredictor，必要时加少量 adapter / LoRA。
目标：future image token CE 和 rollout 质量不劣于原始 baseline。
```

## Phase 2：Triangle Transition-Field Loss

```text
使用 i<k<j samples。
训练 edge_3way + identity + composed_target。
和同一个 RynnVLA one-step baseline 做对照。
```

## Phase 3：提升 Rollout 鲁棒性

```text
加入 generated-state rollout training / scheduled sampling。
把长程 open-loop rollout 作为主评测指标。
```

## 当前最近任务

1. 已完成：RynnVLA / Chameleon 7B starting point、tokenizer、BAIR npz 数据入口。
2. 已完成：BAIR transition-token smoke training，确认 FSDP、adapter checkpoint、step eval 和 grouped metrics 能跑通。
3. 已完成：离线 rollout 评估脚本、GIF/grid/metrics 输出、checkpoint 诊断。
4. 下一步：启动正式 BAIR one-step baseline，即 `WITH_TRANSITION_TOKENS=false, TRAINABLE_SCOPE=all`。
5. 下一步：在一个已经能生成 BAIR future frame 的底座上，再做 transition-token candidate。
6. 后续：增加 BAIR / Bridge triangle dataset adapter，先验证 one-step 质量不下降，再加入 composition / multi-hop loss。
