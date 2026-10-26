---
{
  "id": "paradigm-linear-attention",
  "title": "Linear Attention and Gated State Updates",
  "type": "paradigm",
  "tags": [
    "linear-attention",
    "state-update",
    "gdn",
    "calibration",
    "quantization"
  ],
  "related": [
    "paradigm-ssm",
    "component-attention",
    "component-kv-cache"
  ],
  "sources": [
    "paper-gdn",
    "paper-kimi-linear",
    "repo-kimi-linear"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "线性注意力",
    "linear attention"
  ],
  "updated": "2026-10-26"
}
---

# Linear Attention 与门控状态更新

## 一句话理解

线性注意力把 attention 的 `T×T` 分数矩阵换成一条随 token 递推的有限状态：`state_t = f(state_{t-1}, k_t, v_t, gate_t)`，输出由 query 从状态里读出来。状态大小固定、不随序列长度增长，因此它和 KV cache 的语义不同——这是它与 softmax attention 的根本差别。

## 状态递推与数据流

各类线性模型都写成矩阵值状态 `S_t`（每个 head 一个矩阵；形状约定随论文而异——Gated DeltaNet 写作 `d_v×d_k`，KDA 写作 `d_k×d_v`），区别只在更新规则：

```text
Mamba2（对照）     S_t = α_t · S_{t-1} + v_t k_tᵀ
DeltaNet          S_t = S_{t-1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
Gated DeltaNet    S_t = S_{t-1} ( α_t (I − β_t k_t k_tᵀ) ) + β_t v_t k_tᵀ
KDA（Kimi Linear） S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t-1} + β_t k_t v_tᵀ
```

读出统一为 `o_t = S_t q_t`（Kimi Linear 的 KDA 写作 `o_t = S_tᵀ q_t`）。三个门的作用不同：Gated DeltaNet 的 `α_t ∈ (0,1)` 是**数据相关标量**，控制状态衰减（`α_t → 0` 快速清空记忆、`α_t → 1` 退化成纯 delta 规则）；`β_t ∈ (0,1)` 是写入强度（脚注允许扩到 `(0,2)` 以启用负特征值、解锁 state tracking）；KDA 把门升级成**逐通道对角矩阵** `Diag(α_t)`，对记忆做更细粒度的衰减控制。

训练与推理走两条不同的路径：训练用 chunkwise 形式做并行扫描（delta 族用 WY 表示 + UT transform，Mamba2 走 SSD），推理逐 token 更新状态。Kimi Linear 的预填用 chunk kernel、解码切到 recurrent kernel。门控规则也可以从在线学习角度看——delta 更新等价于对 `½‖S_t k_t − v_t‖²` 做 test-time SGD，`β_t` 是学习率，门控相当于再乘一个自适应权重衰减。

**混合结构**是这一类模型的常见落地方式：Gated DeltaNet 给出 H1（线性层与滑窗注意力交替）与 H2（Mamba2 + Gated DeltaNet + SWA 混排）；Kimi Linear 让 KDA 与 MLA 按 **3:1** 逐层交替，并把 Full MLA 层设为 NoPE，把位置和 recency 信息整体交给 KDA 层。

## 变体对照

| 变体 | 状态更新 | 门 | 出处/规模 |
| --- | --- | --- | --- |
| Mamba2（对照） | `α_t S + v kᵀ` | 标量衰减 | 作为 Gated DeltaNet 的前身对照 |
| DeltaNet | `S(I − β k kᵀ) + β v kᵀ` | 无门，`β_t` 为写入强度；逐条（软）替换旧 KV | GDN 论文 §2.2 的对照变体 |
| Gated DeltaNet | `S(α(I − β k kᵀ)) + β v kᵀ` | 数据相关标量 `α_t` | 1.3B / 100B token 等条件对照 |
| KDA（Kimi Linear） | `(I − β k kᵀ)Diag(α)S + β k vᵀ` | 逐通道对角 `Diag(α_t)` | 48B 总参 / 3B 激活；主实验 1.4T token，最终发布 checkpoint 为 5.7T token / 1M 上下文；KDA:MLA = 3:1 |

## 实现差异

不同实现之间可能不同的地方：state 形状（Kimi Linear 取 `d_k=d_v=128`，每 head 一个矩阵）、门的参数化（标量门 / 逐通道门 / 低秩投影）、`q`、`k` 前面的短卷积与 L2 归一化、训练侧 chunk size 与 UT transform、预填与解码是否走不同 kernel、以及层间混合比例（Kimi Linear 选 layerwise 的 3:1，而非 headwise）。**state 的接口与 KV cache 不同**：不能用 cache 的 page、scale 或更新接口直接替代它。

## 量化关联

**结构事实**

- state 是跨 token 的递推量，大小固定、不随序列增长（Kimi Linear 为每 head `128×128`）。
- 门控与衰减引入了除法与倒数：Kimi Linear 提到两条数值约束：一般 DPLR 的细粒度衰减需要全精度二级 chunking（否则 half-precision matmul 用不满），累计衰减项的倒数可能数值不稳定；KDA 的受限 DPLR 形式把衰减提到 Householder 之外，因而免掉了这层二级 chunking，并用 `q/k` 的 L2Norm 稳定特征值。

**量化推断**

- state 的更新误差会沿 token 方向累积，短序列上的局部误差代表不了长上下文。
- 门控的非线性与 delta 更新对异常值、scale 漂移和舍入误差敏感——论文给的是数值稳定性线索，不是量化误差实验。
- 投影权重、门、state、state→output 读出与 chunk 边界是几条彼此不同的误差路径。
- 混合 attention / linear 层要分别选校准与量化策略，统一配置可能掩盖真正的瓶颈。

**未知项**

- 三份来源都没有量化或低比特实验。

## 结果边界

- 各递推式的形式与符号来自对应论文；其中 Mamba2 那条是 Gated DeltaNet 论文里的**对照写法**，代表不了 Mamba2 报告自身的完整结构。
- Kimi Linear 官方仓库当前只有 `license`、`README`、`tech_report.pdf` 与 `figures/`，**没有代码**，不能作为实现依据。
- KDA 声称的"算子效率提升约 100%、长序列约 2× 速度"绑定它自己的 chunk 实现与硬件，不能外推。
- 三份来源都不含量化实验，本页的量化内容全部是结构推断。

## Agent 如何使用

首轮读本页即可判断目标模型是否属于线性注意力/门控状态更新，以及它的状态是什么、门怎么参数化。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| Gated DeltaNet 的更新规则、chunkwise 训练、消融与实验 | [`paper-gdn`](../../sources/papers/paper-gdn.md) |
| KDA 的递推、参数化、复杂度与混合结构 | [`paper-kimi-linear`](../../sources/papers/paper-kimi-linear.md) |
| Kimi Linear 官方仓库里到底有什么 | [`repo-kimi-linear`](../../sources/repos/repo-kimi-linear.md)（只记录"没有代码"这一事实） |
| state 与 KV cache 的语义差别 | [KV Cache](../components/kv-cache.md)、[Attention](../components/attention.md) |
| 目标 checkpoint 的层比 / state 维与后端 kernel | 该 checkpoint 的 `config.json` / Flash Linear Attention kernel / 运行时（本库不提供） |

回答线性注意力量化问题时必须说明：用的是哪个变体与更新规则、state 的形状与 dtype、门与 chunk size、测试序列长度是否覆盖长上下文，以及实现是否真的走 linear/scan kernel。

## 来源与证据

- [Gated DeltaNet](../../sources/papers/paper-gdn.md)：带门 delta 规则、chunkwise 形式、矩阵值状态与 1.3B 对照实验。
- [Kimi Linear](../../sources/papers/paper-kimi-linear.md)：KDA 的逐通道门、参数化与低秩输出门、chunkwise 算法与混合比例消融。
- [MoonshotAI/Kimi-Linear](../../sources/repos/repo-kimi-linear.md)：固定 commit 下的仓库内容（无代码）。

本页的更新规则与结构描述以上述来源为限；量化结论均为结构推断。
