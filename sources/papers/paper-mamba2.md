---
{
  "id": "paper-mamba2",
  "title": "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality",
  "category": "paper",
  "url": "https://arxiv.org/abs/2405.21060",
  "version": "v1（2024-05-31）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex"
  ],
  "read_locator": "摘要、§3.1–3.3（SSM 矩阵形式、半可分矩阵、SSM=SSS）、§5.1（scalar-identity SSM 与二次形式的等价）、§6 与 Theorem 6.1（chunked 算法与复杂度）、§7.1 块设计、§7.2 head pattern 与 Proposition 7.2、§7.3 kernel feature map、§8（张量并行版 block 数据流）、§9.4.2（架构消融默认超参）、§9.1（MQAR）、§9.2 语言建模（zero-shot 表、SSD+attention 表、2.7B 混合模型表）、§9.3 速度基准",
  "topic": "architectures",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

## 这份来源是什么

Mamba-2 / State Space Duality（SSD）论文（arXiv 2405.21060，v1）。它用半可分矩阵（semiseparable matrices）把 SSM 与结构化 masked attention 连成一个框架（SSD），并据此设计 Mamba-2：核心层是 Mamba selective SSM 的改进版，比 Mamba-1 的核心算法快 2–8×，语言建模上与强 Transformer 配方有竞争力。是本库 `model-mamba2`、`paradigm-ssm` 判断"Mamba-2 与 Mamba-1 的结构差别"的直接依据。

论文不涉及量化或低比特。

## 已读定位与阅读形式

以 arXiv e-print 的 LaTeX 源为主（`structure.tex` 及其 `\input` 的 `structure/*.tex`），版本号取自 abs 页 Submission history（该页只提供元数据，不是论文正文；该文只有 v1）。下文 § 号按 v1 的 `\input` 顺序（§3 SSM 与结构矩阵、§5 SSD、§6 高效算法、§7 Mamba-2 架构、§8 系统优化、§9 实证）；本文的表/图号散落在多个 `\input` 文件里，本页因此以章节 + 表格内容定位。

## 关键事实

- 论文的主结果是：SSM 的矩阵形式 `M = SSS(A,B,C)`（`M_ji = C_jᵀ A_{j:i} B_i`）与结构化 masked attention 在特定结构化情形下**重合**，线性时间的 SSM 算法与二次时间的 kernel attention 算法是同一对象的两面对偶（§5、§6）。
- 关键特化（§5.1）：当 `A_j` 退化为标量时（`A = aI`），`M = L ∘ (C Bᵀ)`，其中 `L = 1SS(a)`；此时"物化 T×T 半可分矩阵再做二次乘法"与 quadratic masked kernel attention 完全等价。
- 这对应 Mamba-2 相对 Mamba-1 的**结构限制**：`A` 由 Mamba-1 的 **diagonal（每个通道一个 N 维对角阵）收紧为 scalar×identity**——每个 head 一个输入相关的标量衰减因子，换来线性与二次两种算法可互换，以及分块（chunked）高效实现（§2.4 Overview 与 §5.1；该说法在 Related Work 一节亦有明确表述）。
- 高效算法（§6，Theorem 6.1）：SSD 的 chunked 算法把半可分矩阵做块分解——对角块用二次（attention）模式算，非对角块用半可分的秩结构归约成更小的递推；在与状态维相同的设定下取 `N = P = Q`（状态维 = head 维 = chunk 长度），复杂度是训练 `O(TN²)` FLOPs、推理 `O(TN)` FLOPs、推理显存 `O(N²)`，且计算以矩阵乘法为主。
- Mamba-2 block（§7.1，架构图）：去掉 Mamba-1 的串行线性投影，`A, X, B, C` 在 block 开头由**同一个投影并行产生**（类比 attention 里 Q/K/V 并行产生）；并在最终输出投影前**额外加一个归一化层**（LayerNorm / GroupNorm / RMSNorm，灵感来自 NormFormer，作用在乘法门控分支**之后**）。论文称这是**初步实验**中发现的大模型不稳定性的缓解手段，并行投影还略微减少参数量、更适配 Megatron 式张量并行切分。
- 张量并行下的 block 数据流（§8）：`x = uW^{(x)}`、`z = uW^{(z)}`、`Δ,B,C = projection(u)`（直接从 `u` 而不是 `x_c` 投影，以便切分、且整个 block 只需一次 all-reduce）、`x_c = conv1d(x)`（depthwise、沿 `d` 独立）、`y = SSM_{A,B,C,Δ}(x_c)`、`y_g = y·φ(z)`（门控，如 SiLU）、`y_n = groupnorm(y_g)`（组数需被张量并行度整除）、最终输出投影。（源文件末行写的是 `y_g W^{(o)}`，而上一行刚定义 `y_n`；按上下文应为 `y_n`，原文如此。）
- 多 head 视角（§7.2）：SSM 的状态维 `N` 与 head 维 `P` 分别类比 attention 的 QK head 维与 V head 维；实践中取常数（典型 `P={64,128}`），当模型维 `D` 增大时增加 head 数，而不是增大 `N`/`P`。
- head pattern（§7.2，定义 MHS/MCS/MES/MIS，分别对应 MHA/MQA/MKA/MVA）：论文给出 Proposition 7.2，指出 **Mamba-1 的 S6 层等价于"head 维 P=1 ＋ MIS/MVA（B、C 在所有 X 通道间共享）"**；Mamba-2 默认同样采用 MVA（B、C 为单 head、跨 X 的 head 共享），论文称消融显示该 pattern 表现最好。该 pattern 可再推广为 grouped-input SSM (GIS) / GVA（类似 GQA）。
- 消融中的默认超参（§9.4.2 Head Structure，Table 5 题注）：模型统一取 state expansion factor `N=64`、head size `P=64`；`A` 的 head 数总等于总 head 数 `H`，即每个 head 有各自输入相关的 `A` 衰减因子。
- kernel feature map（§7.3）：Mamba-2 在 `B`、`C` 分支（对应 attention 的 K、V）套一个可选的 kernel feature map `ψ`，也可选地作用于 `X`（V）分支；**默认取逐元素 Swish / SiLU**。论文另给出加入分母归一化项（把 `X` 增广一列 1，即 kernel attention 的 `G/G·11ᵀ` 形式，要求 `ψ` 恒正）的做法，但把它放在消融里——结论是对多数 `ψ` 反而引入不稳定，只对 ReLU 略有改善；各类 kernel 近似（cosFormer、RFA、Performer 等）都没有胜过简单的逐点非线性。
- 混合架构（§9.2）：SSD 层与 attention 层互补。350M / 48 层 / 7B token 的消融中，注意力层数为 0（纯 Mamba-2）时困惑度 8.60，6 层时最好（8.26），纯 Transformer++ 为 8.68——即**约 10% 的层设为 attention 最优**。
- 2.7B / 64 层 / 300B token 的对比（§9.2 混合模型表）：Transformer++ 6.13、Mamba-2 6.09、Mamba-2-MLP 6.13、**Mamba-2-Attention 5.95**（58 个 SSD 层 + 6 个 attention 层，位于 9/18/27/36/45/56 层）、Mamba-2-MLP-Attention 6.00。论文称加 MLP 会降低质量，但可以换来训练/推理速度，并更易上采到 MoE。
- 零样本评测（§9.2）：Mamba-2-780M 的 Pile 困惑度 7.26、平均 53.5，优于 Mamba-790M（7.33 / 53.0）与 Pythia-1B（7.82 / 49.0）；论文称 Mamba-2 大致对标两倍规模的 Pythia。
- 速度（§9.3、摘要）：SSD 核心层比 Mamba-1 的 scan 实现快 2–8×，并与 FlashAttention-2 做对比。
- 状态规模（§9.1）：在 MQAR（multi-query associative recall）任务上，SSD 层加改进架构允许 Mamba-2 使用远大于 Mamba-1 的 state size，表现优于 Mamba-1 甚至 vanilla attention。

## 不能推出的结论

- SSD 的等价性只在 `A` 退化为标量的特定结构化情形下成立；不能把"SSM 就是注意力"推广到任意 SSM 或任意 `A` 结构。论文还明确写了反向的边界：**SSD 并不推广标准 softmax attention**，也不能推广到任何没有有限维 feature map `ψ` 的变换。
- SSD 是 **SISO（输入通道各自独立处理）＋ scalar-identity** 的 selective SSM，有效状态规模是 `N·D`；不能把它当作 MIMO 或稠密 `B`/`C` 的 SSM。
- "2–8× faster" 是相对 Mamba-1 的 fused associative scan 实现的**核心层**加速（原文主要理由是该 scan 没有利用 matmul 单元），不是端到端加速，也依赖具体硬件与 kernel 实现，不能外推到其它后端。
- `N = P = Q`（状态维 = head 维 = chunk 长度）是定理/实现里为凑齐矩阵乘法形状而设的设定；不能把该相等关系当成 Mamba-2 的通用结构约束，也不能把 `N=P=64` 当作所有 Mamba-2 配置的取值（那只是消融表的设定）。
- MVA head pattern 是消融后选定的默认值，不能当作所有 SSM/序列变换的通用最优 pattern。
- 混合模型的注意力层位置（9/18/27/36/45/56）来自小规模经验，论文自己说只要分散且不在首尾、位置影响不大；不能把该具体位置当作最优配置。
- "加 MLP 降低质量"不等于 MLP 有害，论文同时给出其效率与上采 MoE 的动机。
- Mamba-2-780M 与 Mamba-790M 的参数量并不相同（780M vs 790M），该对比不是严格等参。
- "额外归一化层能提升稳定性"在原文中是**初步实验**观察，不能当作已验证的普遍结论。
- 论文不含量化或低比特内容，不能从本篇推出量化误差路径、校准需求或后端约束。
