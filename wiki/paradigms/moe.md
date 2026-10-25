---
{
  "id": "paradigm-moe",
  "title": "Sparse Mixture of Experts",
  "type": "paradigm",
  "tags": [
    "moe",
    "routing",
    "mlp",
    "quantization"
  ],
  "related": [
    "component-mlp",
    "component-router-expert",
    "paradigm-latent-moe",
    "check-quantization-preflight"
  ],
  "sources": [
    "paper-switch-transformer",
    "paper-mixtral",
    "paper-deepseek-v2",
    "paper-deepseek-v3",
    "repo-deepseek-v3"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "mixture of experts",
    "mixture-of-experts",
    "混合专家"
  ],
  "updated": "2026-10-24"
}
---

# Sparse Mixture of Experts（MoE）

## 一句话理解

稀疏 MoE 用 router 给每个 token 选少量专家：总参数量随专家数增长，而每个 token 的计算量只随被激活的专家数增长。它与 dense FFN 的差别就在这条数据流——router → dispatch → 专家 FFN → combine——以及由此引入的负载均衡与容量约束。

## 计算过程与状态

通用路由形式（以 Switch Transformer 的记法为主）：

```text
h(x)       = W_r · x                       # router logits
p_i(x)     = softmax(h(x))_i               # gate 值（可微的来源）
y          = Σ_{i∈TopK(p(x))} p_i(x) · E_i(x)
```

一次前向的过程：

1. router 产生每个 token 在 `N` 个专家上的 logits / 概率，取 top-K 得到专家集合；Mixtral 写成 `G(x)=Softmax(TopK(x·W_g))`，把未选中的 logits 置 `−∞` 后可以整个跳过那些专家；
2. 生成 token→expert 映射并 dispatch 到对应专家；分布式下这一步是 all-to-all（Expert Parallelism 的负载均衡难点就在这里）；
3. 专家各自处理自己的 token 子集（一般是独立的 gated FFN，Mixtral 用 SwiGLU），输出按 routing weight 加权；
4. combine 回原 token 顺序，再与 residual 相加。

**容量与负载**：Switch 定义为 `expert capacity = (tokens per batch / number of experts) × capacity factor`，路由溢出（dropped token）跳过该层、经残差直传；论文设定下丢弃通常 <1%，容量因子越大越浪费算力与显存。均衡靠辅助损失 `loss = α·N·Σ_i f_i·P_i`（`f_i` 不可微、`P_i` 可微），论文取 `α=1e-2`。DeepSeek-V3 换了一条路：给每个专家一个只参与 top-K 选择的 bias，按 batch 负载动态调整，序列级辅助损失的系数压到极小。

## 变体对照

| 变体 | 路由 | 专家布局 | 要点 |
| --- | --- | --- | --- |
| Switch Transformer | top-1 | 隔层替换 FFN（该论文的实验配置；Switch-C 则是每层都有专家） | 单 token 只走 1 个专家；靠容量因子 + 丢弃控制负载 |
| Mixtral 8x7B | top-2 | 每层 FFN 都换成 MoE | 8 个 SwiGLU 专家；总参 47B / 激活 13B；32k 上下文 |
| DeepSeekMoE（V2） | top-K + 共享专家 | 细粒度路由专家 + 共享专家隔离 | softmax affinity、device-limited routing，训练期会丢 token |
| DeepSeekMoE（V3） | top-K + 共享专家 + 路由 bias | 同上 | 671B 配置：每层 256 路由 + 1 共享、每 token 激活 8；sigmoid affinity、node-limited routing、不丢 token |

## 实现差异

不同实现之间可能不同的地方：top-k 取值与专家粒度、是否有 shared expert、affinity 用 softmax 还是 sigmoid、容量与丢弃策略、负载均衡用辅助损失还是路由 bias、专家权重的布局与并行方式（专家并行 / 张量并行 / grouped GEMM）、以及 dispatch-combine 的通信顺序。DeepSeekMoE 的共享专家与细粒度切分不能直接套到 Mixtral 配置上。DeepSeek-V3 官方推理实现里 `Gate`、`Expert`、`MoE` 是分开的模块，配置从 `configs/*.json` 读取——代码里的 `ModelArgs` 默认值（64 路由专家、2 共享专家）不是模型配置。

## 量化关联

**结构事实**

- 每个专家有自己的权重、激活分布与访问频率；router 的输出是离散选择（top-K），专家是否参与计算由它决定。
- DeepSeek-V3 官方推理实现里存在 fp8 路径（`act_quant` 激活量化 + `fp8_gemm`），属于官方推理契约，不等于训练期契约。

**量化推断**

- 专家访问频率通常不均衡，全局校准可能忽略低频但关键的专家。
- router logits 的小误差可以改变 top-K，误差沿离散路由路径被放大；router 应与专家 MLP 分开评估。
- 各专家的激活、权重与负载分布不同，per-expert 或分组 scale 需要后端支持。
- 量化后要同时检查路由一致率、专家输出误差、通信与 packing，以及端到端质量。

## 结果边界

- 本页的公式与容量机制来自 Switch Transformer；`softmax over top-K` 只是它采用的一种门控实现（MoE 门控存在多种候选，该说法见 `paper-mixtral` §2.1 对既有 MoE 工作的引用），不能当作所有 MoE 的通用定义。
- 参数量口径要分清：Mixtral 的 "8x7B" 是命名，总参 47B、激活 13B；激活参数不等于等量稠密模型的算力或显存。
- 各来源论文都不含低比特量化精度的完整对比；DeepSeek 侧的部署量化只到 FP8 / KV cache 6 bit 的陈述，例外是 DeepSeek-V3 报告用 FP8 vs BF16 相对误差 <0.25% 验证其 FP8 训练框架。本页量化内容均为结构推断。
- 容量公式假定张量形状在编译期静态确定（论文的 TPU 语境），不能当作与框架无关的通用做法。

## Agent 如何使用

首轮读本页即可判断目标模型是否属于稀疏 MoE、路由怎么选专家、负载与容量怎么控制。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| top-1 路由、容量因子、丢弃与辅助损失的定义 | [`paper-switch-transformer`](../../sources/papers/paper-switch-transformer.md) |
| 每层全替换、top-2、总参/激活参口径 | [`paper-mixtral`](../../sources/papers/paper-mixtral.md) |
| DeepSeekMoE 的细粒度专家、共享专家与均衡策略 | [`paper-deepseek-v2`](../../sources/papers/paper-deepseek-v2.md)、[`paper-deepseek-v3`](../../sources/papers/paper-deepseek-v3.md) |
| 官方实现里的 Gate / Expert / MoE 与 671B 配置 | [`repo-deepseek-v3`](../../sources/repos/repo-deepseek-v3.md) |
| 目标模型的专家数 / 路由配置与后端 kernel | 该 checkpoint 的 `config.json` / 官方源码 / 运行时（本库不提供） |

回答 MoE 量化问题时必须说明：路由方式与 top-k、专家总数与每 token 激活数、是否含共享专家、容量与丢弃策略、并行方式和目标后端。

## 来源与证据

- [Switch Transformer](../../sources/papers/paper-switch-transformer.md)：top-1 路由、容量因子与丢弃、辅助损失、selective precision 与初始化。
- [Mixtral of Experts](../../sources/papers/paper-mixtral.md)：每层全替换、top-2 门控、47B/13B 参数量口径与路由分析。
- [DeepSeek-V2](../../sources/papers/paper-deepseek-v2.md)、[DeepSeek-V3](../../sources/papers/paper-deepseek-v3.md)：细粒度专家 + 共享专家、路由 bias 动态调整（论文称 auxiliary-loss-free）、node-limited routing。
- [DeepSeek-V3 官方实现](../../sources/repos/repo-deepseek-v3.md)：固定 commit 下的模块划分、fp8 推理路径与 671B 配置。

本页的通用机制以 Switch 与 Mixtral 两份来源为限；DeepSeek 的配置属于具体 checkpoint，量化结论均为推断。
