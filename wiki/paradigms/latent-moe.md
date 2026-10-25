---
{
  "id": "paradigm-latent-moe",
  "title": "LatentMoE Terminology and Variants",
  "type": "paradigm",
  "tags": [
    "latent-moe",
    "moe",
    "routing",
    "quantization"
  ],
  "related": [
    "paradigm-moe",
    "paradigm-latent-moe-nvidia",
    "paradigm-multihead-latent-moe",
    "component-router-expert",
    "component-mlp"
  ],
  "sources": [
    "paper-latentmoe",
    "paper-multihead-latentmoe"
  ],
  "confidence": "inferred",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-24"
}
---

# LatentMoE 术语与变体

## 一句话理解

"LatentMoE" 不是唯一确定的架构名：它至少指向两条不同工作——NVIDIA 技术报告的 **LatentMoE** 与在其之上发展的 **Multi-Head LatentMoE**。两者都把专家计算放进低维空间，但"latent"指的张量、路由所在空间和并行契约都不同；命中该术语必须先确定论文/仓库与版本，再落到具体变体页。

## 机制与数据流

两条变体的数据流**顺序相反**，这是最容易混淆的点：

```text
NVIDIA LatentMoE:  x(d) → router(d) → W_↓ → latent 专家(ℓ) → W_↑ → 合并回 d
Multi-Head:        x(d) → W_in 投影并切分 → N_h 个子 token(d_h) → 各自 router + 专家 → 拼接 → W_out
```

NVIDIA 是"**先路由、后降维**"（路由权重 `W'_r ∈ R^{N'×d}` 作用在原始 `d` 维）；Multi-Head 是"**先投影切分、后路由**"（每个子 token 有自己的 router，作用在 `d_h` 维）。所以"latent"一词在两者里指向不同的张量，形状、参数与并行方式不能互相套用。

## 变体对照

| 变体 | 来源 | "latent"指什么 | 路由空间 | 并行方式 |
| --- | --- | --- | --- | --- |
| LatentMoE（NVIDIA） | `paper-latentmoe` | 专家输入维降到 `ℓ = d/α` | 原始 `d` 维（降维**之前**） | 不改 EP；`LMoE-eff` 通信降 `α` 倍，`LMoE-acc` 通信不降（换精度） |
| Multi-Head LatentMoE | `paper-multihead-latentmoe` | 把 token 切成 `N_h` 个 `d_h` 维子 token（总维不变） | `d_h` 维（投影切分**之后**，逐子 token） | Head Parallel：通信在路由前，`O(1)` |

Multi-Head LatentMoE 建立在 NVIDIA LatentMoE 之上（把通信从"路由之后"挪到"路由之前"），不是另一条独立起源的工作。

## 实现差异

不同变体之间可能不同的地方：latent 投影的位置、维度与归一化，专家输入/输出 shape，路由粒度与路由所在维度，恢复投影，以及 head 数、通信阶段与并行切分。判别一个实现属于哪一支，主要看两点：token 投影是否为方阵（`W_in ∈ R^{d×d}` 表示总维不变，属 Multi-Head）、router 输入维是 `d` 还是 `d_h`。

## 量化关联

**结构事实**

- NVIDIA 变体压的是**专家输入维**：routed 专家权重在 latent 空间（`W_FC1, W_gate ∈ R^{m×ℓ}`、`W_FC2 ∈ R^{ℓ×m}`），router 与 shared 专家仍在 `d` 维。
- Multi-Head 变体把 token 投影切成 `N_h` 个 `d_h` 维子 token（惯例 `N_h·d_h = d`），每个子 token 有自己的 router 与专家，router 作用在 `d_h` 维。

**量化推断（本库推断）**

- 压缩/恢复投影会把量化误差从 latent 空间映射回主干空间，两个投影不能只看单层误差。
- latent 分布可能比 residual stream 更窄或更尖，需单独选择校准轴与 scale，不能套用原维 dense FFN 的默认值。
- 专家选择与 latent 投影误差耦合，应同时记录路由一致率、latent reconstruction error 与 block 输出误差。

## 结果边界

- 本页只做术语分流，不定义任何单一实现；上文"量化推断"是跨变体的结构推断，不是任何一份来源的论文结论。
- 两份来源都不含量化或低比特实验，"latent 是否更耐量化"在本库没有证据。
- 本页未做独立数值或运行验证；具体结构与量化边界以两个变体页及其来源为准。

## Agent 如何使用

首轮读本页即可判断"LatentMoE"这个术语指向哪条工作，再决定继续读哪一页。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| NVIDIA LatentMoE 的结构式、成本对照与实验 | [NVIDIA LatentMoE](latent-moe-nvidia.md) |
| Multi-Head LatentMoE 的结构式与 Head Parallel | [Multi-Head LatentMoE](multihead-latent-moe.md) |
| MoE 路由与负载均衡的通用机制 | [Sparse MoE](moe.md) |
| 目标实现的 latent 维与路由契约 | 该实现附带的配置与源码（本库不提供） |

回答 LatentMoE 量化问题时必须说明：指的是哪条工作、latent 指哪个张量、router 输入维是 `d` 还是 `d_h`、并行方式与目标后端。

## 来源与证据

- [LatentMoE](../../sources/papers/paper-latentmoe.md)：NVIDIA 变体的结构式（LMoE-eff/LMoE-acc）、成本对照与实验。
- [Multi-Head LatentMoE](../../sources/papers/paper-multihead-latentmoe.md)：多 head 结构式、Head Parallel 与 IO-aware 算子。

本页为术语分流页；具体结构与量化边界以两个变体页及其来源为准。
