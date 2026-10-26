---
{
  "id": "model-mamba2",
  "title": "Mamba-2",
  "type": "model",
  "tags": [
    "ssm",
    "mamba",
    "state-update",
    "quantization"
  ],
  "related": [
    "paradigm-ssm",
    "paradigm-linear-attention",
    "component-kv-cache"
  ],
  "sources": [
    "paper-mamba2",
    "repo-mamba"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "state space model",
    "状态空间模型"
  ],
  "updated": "2026-10-26"
}
---

# Mamba-2

## 一句话理解

Mamba-2 是选择性 SSM 的第二代：论文用 State Space Duality（SSD）证明标量衰减 SSM 的矩阵形式与结构化 masked attention 等价，据此把 Mamba-1 的 `A` 从逐通道对角**收紧为 `scalar × identity`**（每个 head 一个输入相关的标量衰减），从而能用矩阵乘法做 chunked 计算，核心层比 Mamba-1 的 scan 快 2–8×。它不是带 K/V cache 的 Transformer。

## 结构规格表

| 项 | Mamba-2 |
| --- | --- |
| 核心层 | selective SSM，`A = aI`（每 head 一个输入相关标量衰减） |
| 状态 / head 维 | 消融设定 `N=64`、`P=64`（§9.4.2，**非通用约束**） |
| head pattern | 采用 **MVA**（`B`、`C` 为单 head、跨所有 `X` head 共享），论文称该 pattern 消融最优；Mamba-1 的 S6 = `P=1` + MIS/MVA（multi-input SSM / multi-value attention） |
| block | block 开头用**一次投影并行产生** `x, z, Δ, B, C`；`A` 不经投影，是静态参数 `A = -exp(A_log)`（论文 §7.1 文字写作 `A, X, B, C` 并行产生，但 §8 与官方实现都与此不符）；输出投影前加一个归一化层（LayerNorm/GroupNorm/RMSNorm，NormFormer 灵感，**初步实验**） |
| chunk | chunked 算法 Theorem 6.1 设定 `P = N`；§6.3 为给出复杂度界再取 `N = P = Q`（状态维 = head 维 = chunk 长度，属理论设定，非结构约束；官方实现 `chunk_size=256` 与此无关） |
| 复杂度 | 训练 `O(TN²)` FLOPs、推理 `O(TN)` FLOPs、推理显存 `O(N²)`（上述设定下） |
| 混合架构 | 350M / 48 层 / 7B token 消融中约 **10%** 层设为 attention 最优 |
| 已知实现默认 | `repo-mamba` 的 `Mamba2`：`headdim=64`、`ngroups=1`、`chunk_size=256`（代码默认值，非 checkpoint 配置） |

表中 `N/P` 与"10% attention"都是消融设定，`headdim/ngroups/chunk_size` 是官方代码默认值；三者都不能当作某个 checkpoint 的实际配置。

## 结构与数据流

SSD 把 SSM 写成半可分矩阵运算 `M = SSS(A,B,C)`（`M_ji = C_jᵀ A_{j:i} B_i`）；当 `A` 退化为标量（`A = aI`）时特化为

```text
M = L ∘ (C Bᵀ),   L = 1SS(a)
```

此时"物化 T×T 半可分矩阵再做二次乘法"与 quadratic masked kernel attention 完全等价——线性时间的 SSM 算法与二次时间的 kernel attention 是同一对象的两面（§5、§6）。

block 数据流（§7.1、§8）：`u` 经一次投影并行产生 `x, z, Δ, B, C` → `x_c = conv1d(x)`（depthwise、沿 `d` 独立）→ `y = SSM_{A,B,C,Δ}(x_c)` → 门控 `y_g = y · φ(z)`（如 SiLU）→ 归一化 `y_n = groupnorm(y_g)` → 输出投影。并行投影省掉 Mamba-1 的串行投影，略减参数量、更适配张量并行。

chunked 算法（§6，Theorem 6.1）把半可分矩阵做块分解：对角块用二次（attention）模式算，非对角块用半可分秩结构归约成更小的递推，计算以矩阵乘法为主。多 head 视角（§7.2）把状态维 `N` 类比 attention 的 QK head 维、head 维 `P` 类比 V head 维，模型维 `D` 增大时增加 head 数而不是加大 `N`/`P`。`B`、`C` 分支可套一个可选 kernel feature map `ψ`（默认逐元素 Swish / SiLU）。推理时每层维护固定大小的 SSM 状态，**没有 KV cache**。

官方实现（repo-mamba）：`Mamba2`（mamba2.py）默认 `headdim=64`、`ngroups=1`、`chunk_size=256`，`d_inner = expand · d_model`、`nheads = d_ssm // headdim`，带 `RMSNormGated`，支持张量并行；前向里 `A = -exp(A_log)`。SSD 核心走 Triton（`ssd_bmm`/`ssd_chunk_scan`/`ssd_chunk_state`/`ssd_state_passing`），单步更新在 `ops/triton/selective_state_update.py`。

## 量化关联

**结构事实**

- state 是跨 token 的递推量，大小固定（每 head `N × P` 级）、不随序列增长；推理时无 KV cache，只维护每层固定状态。
- `A` 被收紧为 `scalar × identity`，结构上是 SISO（输入通道各自独立）＋标量衰减，有效状态规模为 `N · D`。

**量化推断**

- state 误差随步数递推累积，长序列、状态重置与 chunk 边界都要单独覆盖；短输入评测容易漏掉漂移。
- 输入投影、局部卷积、门控、state update 与 readout 的校准轴不同，应分别统计误差。
- 后端只有 fp32/fp16/bf16 kernel；在参考实现上做 fake quant 不能证明目标低比特后端可用。

**未知项**

- 论文不含量化/低比特内容；官方仓库没有量化路径，已知 CUDA selective-scan kernel（Mamba-1 路径）为 fp32/fp16/bf16。

## 结果边界

- `N = P = Q` 是定理/实现为凑矩阵乘法形状而设的设定，不是 Mamba-2 的通用结构约束；`N = P = 64` 只是 §9.4.2 消融表的取值。
- "SSD 比 Mamba-1 scan 快 2–8×"是相对 Mamba-1 fused scan 实现的**核心层**加速（原文理由是该 scan 没用上 matmul 单元），不是端到端，也依赖硬件与 kernel。
- 混合模型的 attention 层位置（9/18/27/36/45/56）来自小规模经验，论文自己说只要分散且不在首尾、位置影响不大，不能当最优配置。
- "输出前加归一化层提升稳定性"原文是**初步实验**观察，不是已验证的普遍结论。
- 代码默认值（`headdim=64` 等）不代表论文或某个 checkpoint 配置；`repo-mamba` 里的 Mamba-3 属仓库 HEAD，不在 Mamba-2 论文口径内。
- 本页只做论文与源码的静态阅读，未编译或运行 scan/chunk kernel；论文与仓库都不含量化实验，量化内容均为结构推断。

## Agent 如何使用

首轮读本页即可判断目标是 Mamba-2（`A` 为标量衰减、无 KV cache、SSD chunked kernel）还是 Mamba-1/Mamba-3，以及推理时维护哪些状态。判定某实现属于 Mamba-2 的判据：`A` 收缩为 `scalar × identity`（每 head 一个标量衰减）；`ngroups=1`、`B/C` 跨 X head 共享是论文消融选出的默认，与 `A` 的结构是两条独立设计轴；核心层走 Triton SSD kernel、推理只维护固定 state 而无 KV cache。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| SSD 对偶、`A` 的结构限制、chunked 算法与 head pattern | [`paper-mamba2`](../../sources/papers/paper-mamba2.md) |
| 官方模块参数默认值、状态字段与 Triton/CUDA kernel 分布 | [`repo-mamba`](../../sources/repos/repo-mamba.md) |
| 与 Mamba-1/Mamba-3、与线性注意力的关系 | [Selective State Space Model](../paradigms/ssm.md)、[Linear Attention](../paradigms/linear-attention.md) |
| 目标 checkpoint 的真实配置与后端 kernel | 该 checkpoint 的 `config.json` / 官方或融合实现 / 运行时（本库不提供） |

回答 Mamba-2 量化问题时必须说明：state 的形状与 dtype、`A` 的参数化、chunk size 与 kernel、测试序列长度是否覆盖长上下文与状态重置。

## 来源与证据

- [Transformers are SSMs（Mamba-2 / SSD）](../../sources/papers/paper-mamba2.md)：SSD 对偶、`A` 的结构限制、chunked 算法与复杂度、block 设计、head pattern 与架构消融（§5–§9）。
- [state-spaces/mamba](../../sources/repos/repo-mamba.md)：固定 commit 下 `Mamba2` 的默认参数、状态字段与 Triton kernel 分布。

本页结构以上述来源为限；论文与仓库都不含量化实验，量化内容除注明者外均为本库推断。
