# Transition Token 设计

## 接入目标

transition-token 版本的目标接口是：

```text
历史图像 tokens + action tokens + transition tokens -> 未来图像 tokens
```

其中：

```text
transition_token = Pred_e(z_context, action/state, dt)
```

transition token 插入 token sequence，作为连续 soft tokens 使用。它们不作为 CE target，本身 label 保持 `-100`，只作为未来图像 token 生成的动态条件。

## 最小工程设计

在 Chameleon wrapper 里增加 transition 模块：

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

这样 backbone 的 AR 结构保持不变，但 transition token 会成为 future image token prediction 的真实条件。

## 当前 Scaffold

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

当前实现逻辑：

```text
1. 数据里如果没有 token id 16001，模型行为保持原样；
2. no-pretokenize 数据路径在加 --with-transition-tokens 后，会把 tokenizer 中对应 id 16001 的 reserved token 插到 world-model prompt 的 action block 后；
3. 当前 Chameleon tokenizer 中这个文本是 <reserved15997>；
4. 如果 input_ids 中出现 token id 16001，则认为这些位置是 transition placeholder；
5. 模型用 placeholder 之前的上下文 embedding 做 mean pooling；
6. TransitionTokenAdapter 把 pooled context 映射成连续 soft tokens；
7. 用这些 soft tokens 替换 placeholder 的普通 embedding；
8. placeholder 对应 labels 强制设为 -100，不参与 CE；
9. future image tokens 继续按原始 causal LM CE 训练。
```

这个版本还没有接入 triangle loss，也还没有显式区分 `e_ik/e_kj/e_ij`。它的作用是先打通“AR token backbone 能接收连续 transition soft token 条件”的工程入口。

## Loss 映射

我们保留当前 transition-field baseline loss 的逻辑，但视觉目标从 latent MSE 改成 image-token CE：

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

注意：不要把 `e_ij = Comp(e_ik, e_kj)` 写成出发假设。composition 在这里是 transition representation 的正则和语义约束，核心视觉目标仍然是未来图像 tokens 是否预测正确。

## Triangle Data

RynnVLA 当前 world-model sample 基本是一跳：

```text
image_t, action_t -> image_{t+1}
```

我们的 loss 后续需要 triangle sample：

```text
i < k < j
image_i, image_k, image_j
action_i:k, action_k:j, action_i:j
dt_ik, dt_kj, dt_ij
```

需要新增一个 dataset path，能从 BAIR / Bridge / LIBERO 中采样上述 triangle。第一版建议先接 BAIR 或 Bridge，因为 beta 仓库已经有 BAIR shards 和 rollout 评测基建。
