---
{
  "id": "paradigm-ssm",
  "title": "Selective State Space Model",
  "type": "paradigm",
  "tags": [
    "ssm",
    "mamba",
    "state-update",
    "quantization"
  ],
  "related": [
    "paradigm-linear-attention",
    "component-residual-mhc",
    "check-calibration-axes"
  ],
  "sources": [
    "paper-mamba",
    "paper-mamba2",
    "repo-mamba"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "state space model",
    "状态空间模型"
  ],
  "updated": "2026-10-24"
}
---

# Selective State Space Model（SSM）

## 一句话理解

SSM 用一条线性递推维护序列状态：离散化后 `h_t = Ā h_{t-1} + B̄ x_t`、`y_t = C h_t`，Mamba 一系进一步把 `Δ、B、C` 变成输入的函数（选择性），模型随之从时不变变成时变。状态大小固定、不随序列增长，这是它与 attention / KV cache 的根本差别。

## 状态递推与数据流

连续形式与零阶保持（ZOH）离散化：

```text
连续　 h'(t) = A h(t) + B x(t),   y(t) = C h(t)
ZOH　  Ā = exp(ΔA),   B̄ = (ΔA)⁻¹ (exp(ΔA) − I) · ΔB
递推　 h_t = Ā h_{t-1} + B̄ x_t,   y_t = C h_t
```

**选择性**（Mamba-1）：`Δ = softplus(Linear₁(x))` 广播到 `D` 维，`B = Linear_N(x)`、`C = Linear_N(x)`；因为这三个量都带上了序列长度维，模型由时不变变为时变，失去与卷积的等价，只能走递归 scan——这也是 Mamba 要自研 fused kernel 的原因。`A` **没有**做成选择性：`A` 的形状是每通道一个 `N` 维对角参数（记作 `(D, N)`），论文的理由是它只通过 `Ā = exp(ΔA)` 与 `Δ` 交互，让 `Δ` 选择性就足以让 `Ā`、`B̄` 选择性；但严格说 `B̄` 的 ZOH 表达式里也含 `A`，论文并未做 `A` 选择性的消融。

`Δ` 的语义是 RNN 门控的推广：`Δ` 大表示"重置状态、关注当前输入"，`Δ` 小表示"保留状态、忽略当前输入"；多条序列拼接时可以在边界把状态重置（对应 `Δ_t → ∞`）。

**Mamba-1 block**：输入投影 → 因果短卷积 `conv1d` → SiLU → selective scan → 输出投影，扩展因子 `E` 在实验里固定为 2、用两个 block 堆叠来匹配 Transformer 的参数量；SSM 本身参数很少，单块参数主要在三个线性投影上（合计 `3ED²`）。

**Mamba-2 把 `A` 收紧为 `scalar × identity`**（每个 head 一个输入相关的标量衰减），换来 SSM 与 masked attention 的对偶（`M = SSS(A,B,C)`，标量情形下 `M = L ∘ (C Bᵀ)`）以及 chunked 高效实现：训练 `O(TN²)` FLOPs、推理 `O(TN)` FLOPs、推理显存 `O(N²)`。它的 block 用一次投影并行产生 `x, z, Δ, B, C`（论文 §7.1 文字写作 `A, X, B, C`，但 `A` 在官方实现里是静态参数 `A = -exp(A_log)`），并在最终输出投影前多加一个归一化层。

推理时 Mamba-1 每层维护固定大小的 SSM 状态，单步递推走 `step`，**没有 KV cache**；block 里短卷积在推理态的缓存行为论文未描述，需回实现核验。

## 变体对照

| 变体 | `A` 参数化 | 官方实现默认 | 算子 |
| --- | --- | --- | --- |
| Mamba-1（S6） | 每通道一个 `N` 维对角参数 `(D, N)` | `d_state=16`、`d_conv=4`、`expand=2`、`dt_rank=ceil(d_model/16)` | `csrc/selective_scan/` 的 CUDA kernel（fp32/fp16/bf16） |
| Mamba-2（SSD） | `scalar × identity` | `headdim=64`、`ngroups=1`、`chunk_size=256` | Triton（`ssd_bmm`/`ssd_chunk_scan`/`ssd_chunk_state`/`ssd_state_passing`） |
| Mamba-3 | 数据相关 `A`（heavy-tail activation 生成） | `d_state=128`、`headdim=64`、`expand=2` | Triton + TileLang/CuTe |

表中默认值是**官方代码的默认值**，不代表论文或某个 checkpoint 的实际配置；引用具体模型规模要看对应 config 与权重。

## 实现差异

不同实现之间可能不同的地方：state size、`A` 的参数化、`Δ` 的生成方式（Mamba-1 还给出低秩推广 `s_Δ(x)=Linear_D(Linear_R(x))`）、`B`/`C` 的生成与共享方式、卷积宽度、scan 或 chunk 的 kernel、以及 head pattern。Mamba-2 用 MHS/MCS/MES/MIS 四个记号区分 head 结构，默认取 MVA（`B`、`C` 为单 head、跨所有 `X` head 共享），并指出 **Mamba-1 的 S6 等价于 `P=1` 加 MIS/MVA**。张量并行下 Mamba-2 用 `ColumnParallelLinear`/`RowParallelLinear` 切分，组归一化的组数要能被并行度整除。

## 量化关联

**结构事实**

- state 是跨 token 的递推量，大小固定、不随序列增长；推理时每层同时维护卷积状态与 SSM 状态。
- selective scan 由 `Δ、A、B、C` 与 state 共同决定，其中 `Δ` 经 softplus、`Ā = exp(ΔA)`。
- 范围事实：两篇论文与官方实现都不含量化或低比特路径（CUDA kernel 只覆盖 fp32/fp16/bf16）。

**量化推断**

- state/cache 的量化是运行时状态问题，不能只靠权重量化结果推断。
- 递推误差可能随序列长度累积；长序列、状态重置与 chunk 边界都要单独验证。
- `Δ` 门控路径、局部卷积与输出投影的异常值和校准轴不同，应分开统计。
- 需要用逐步 reference 实现对照 fused scan 的输出与数值误差，不能只看最终指标。

## 结果边界

- Mamba-2 的复杂度结论（训练 `O(TN²)`、推理 `O(TN)`、推理显存 `O(N²)`）建立在状态维、head 维与 chunk 长度取同一 `N` 的设定下，不能外推。
- "SSD 层比 Mamba-1 的 scan 快 2–8×"与"Mamba 推理吞吐比同规模 Transformer 高 4–5×"绑定硬件、实现与 batch（后者论文归因于没有 KV cache、可以用更大 batch），不能当作通用倍率。
- "约 10% 的层设为 attention 最优"是 350M / 48 层 / 7B token 消融的结论，不是通用配置。
- `A` 不做选择性的理由只是论文的假设（hypothesize），论文只消融了 `Δ`、`B`、`C`，**没有** A 选择性的实验证据。
- `repo-mamba` 里的 Mamba-3 是该仓库当前 HEAD 的实现，不属于 Mamba/Mamba-2 论文口径；论文与官方实现都不含量化实验。
- 本页只做源代码与论文的静态阅读，未编译或运行 scan/chunk kernel，也未做任何数值验证；量化内容全为结构推断。

## Agent 如何使用

首轮读本页即可判断目标模型是否属于 SSM 系、用的是哪一代、推理时维护哪些状态。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 选择性机制、block 结构、`Δ` 的语义与消融 | [`paper-mamba`](../../sources/papers/paper-mamba.md) |
| SSD 对偶、`A` 的结构限制、chunked 算法与 head pattern | [`paper-mamba2`](../../sources/papers/paper-mamba2.md) |
| 官方实现的默认参数、状态字段与 kernel 布局 | [`repo-mamba`](../../sources/repos/repo-mamba.md) |
| 与线性注意力的关系 | [Linear Attention](../paradigms/linear-attention.md) |
| 目标 checkpoint 的真实配置与后端 kernel | 该 checkpoint 的 `config.json` / 官方或融合实现 / 运行时（本库不提供） |

回答 SSM 量化问题时必须说明：Mamba 的哪一代、state 的形状与 dtype、scan/chunk 走哪条 kernel、测试序列长度是否覆盖长上下文与状态重置。

## 来源与证据

- [Mamba](../../sources/papers/paper-mamba.md)：连续/离散递推、选择性机制、`Δ` 的门控语义、block 结构与 fused scan。
- [Mamba-2 / SSD](../../sources/papers/paper-mamba2.md)：状态空间与结构化 masked attention 的对偶、`A` 的结构限制、chunked 算法与 head pattern。
- [state-spaces/mamba](../../sources/repos/repo-mamba.md)：固定 commit 下 Mamba-1/2/3 的默认参数、状态字段与 Triton/CUDA kernel 分布。

本页的递推与结构描述以上述来源为限；量化结论均为结构推断。
