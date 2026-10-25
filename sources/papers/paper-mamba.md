---
{
  "id": "paper-mamba",
  "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
  "category": "paper",
  "url": "https://arxiv.org/abs/2312.00752",
  "version": "v2（2024-05-31）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex"
  ],
  "read_locator": "摘要、§2 Eq.(1)–(4)、§3.1、§3.2 Algorithm 1–2、§3.3、§3.4（Mamba block 架构图）、§3.5 Theorem 1 与 Eq.(5)、§3.6、§4.1、§4.2、§4.5、§4.6、§5 discussion、附录 D",
  "topic": "architectures",
  "excerpt_sha256": null
}
---

## 这份来源是什么

Mamba 论文（arXiv 2312.00752，v2）。它提出**选择性状态空间模型（selective SSM，S6）**——让 SSM 的 `Δ, B, C` 成为输入的函数据此按内容选择性地保留/遗忘信息——并把它放进一个**既没有 attention、也没有独立 MLP block** 的简化架构（Mamba）。是本库 `paradigm-ssm` 判断「SSM 层里没有 attention、推理不需要缓存序列」这类结构问题的直接依据。

边界：论文只讨论选择性 SSM 的算法与结构，全篇不含量化/低比特实验或方法（§5 Discussion 只把 quantization 列为 LLM 已有、SSM 待考察的能力之一）；论文里也没有描述「推理态里保留卷积态」，本库此前关于「只有卷积态与 SSM state」的说法须回到实现核验（见「不能推出的结论」）。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源（`main.tex` 与 `\input` 的 `src/*.tex`，含附录 D）。版本号取自 abs 页 Submission history（该页只提供元数据，不是论文正文）。
- 版本：v1 2023-12-01、v2 2024-05-31（v2 为该文最后版本）；`main.tex` 里 `\toggletrue{arxiv}`，即编号按 arXiv 版。
- 本地缓存的 ar5iv HTML 是 503 错误页、不可用，本页未据它做任何复核。
- 下文 §、Eq.、Algorithm、Theorem、Figure 号按 v2 LaTeX 源逐文件计数。

## 关键事实

- SSM 的参数为 `(Δ, A, B, C)`，分两阶段定义序列变换：连续形式 `h'(t) = A h(t) + B x(t)`、`y(t) = C h(t)`（§2，Eq.(1)）；经离散化（ZOH：`Ā = exp(ΔA)`、`B̄ = (ΔA)⁻¹(exp(ΔA) − I)·ΔB`，Eq.(4)）得到递归形式 `h_t = Ā h_{t-1} + B̄ x_t`、`y_t = C h_t`（Eq.(2)），以及等价的卷积形式（Eq.(3)）。Eq.(4) 只在 arXiv 分支编号。
- `A` 的形状是**每通道一个 N 维对角参数**：Algorithm 1/2 里写作 `A : (D, N) ← Parameter`，注释说明它表示结构化的 `N×N` 矩阵（§3.2）。
- 选择性机制（§3.2）：把 `Δ`、`B`、`C` 变成输入的函数（`s_B(x) = Linear_N(x)`、`s_C(x) = Linear_N(x)`、`s_Δ(x) = Broadcast_D(Linear_1(x))`、`τ_Δ = softplus`），参数张量因此多出长度维 `L`，模型从**时不变变为时变**（Algorithm 1 → Algorithm 2 的对照）。时变之后失去与卷积的等价，只能用递归（scan）——这是论文必须自研高效算子的原因（§3.2、§3.3）。
- `s_Δ` 可推广（§3.6）：把广播维从 1 扩到 `R`（`R` 取 `D` 的一小部分，参数可忽略），等价地写成低秩投影 `s_Δ(x) = Linear_D(Linear_R(x))`。
- `A` **没有被做成选择性**：论文的理由是 `A` 只通过 `Ā = exp(ΔA)` 与 `Δ` 交互起作用，因此让 `Δ` 选择性就足以让 `Ā`、`B̄` 选择性（§3.5）。注意严格说 `B̄` 的 ZOH 表达式里也含 `A`（§2，Eq.(4)）；论文并未做 A 选择性的消融。
- 高效实现（§3.3）：靠 kernel fusion、parallel scan、recomputation 三个经典技术，把状态尽量留在片上高速存储（SRAM）里，避免把完整状态物化到 HBM（参数从 HBM 载入 SRAM，离散化与递推都在 SRAM 内完成，只把输出写回 HBM）。附录 D 给出 fused kernel 的算法级描述，并称 IO 量降低 `O(N)` 倍、实测加速 20–40×，且在 32K 长度上 scan 比 attention 快 up to 7×。
- Mamba block（§3.4 架构图，按源序是 Figure 3）：把 H3 block 与常见 MLP block **合并成一个同质重复的 block**，块内有**短卷积（conv1d）**后接 SSM（论文只写 convolution / short convolution，“因果”来自官方 `causal_conv1d` 实现；§3.4 架构图、附录 D 把 block 组成列为 input projection, convolution, activation, scan, output projection）；相比 H3 把第一个乘性门换成激活函数。扩展因子 `E` 是可控的，**实验里一律固定 `E=2`**、用两个 block 堆叠以匹配 Transformer 中 MHA+MLP 的 `12D²` 参数量；单块参数主要在三个线性投影上，合计 `3ED²`（`2ED²` 输入 + `ED²` 输出），SSM 本身参数少得多。激活用 SiLU/Swish（即 gated MLP 的 SwiGLU 变体），并在 block 里额外加一个归一化层（论文选 LayerNorm，动机来自 RetNet）。
- `Δ` 的语义（§3.5）：`Δ` 泛化了 RNN 的 gate——`Δ` 大表示「重置状态、关注当前输入」，`Δ` 小表示「保留状态、忽略当前输入」。
- Theorem 1（§3.5，Eq.(5)）：取 `N=1、A=−1、B=1、s_Δ = Linear、τ_Δ = softplus` 时，选择性 SSM 的递归退化为经典门控 RNN `g_t = σ(Linear(x_t))`、`h_t = (1−g_t) h_{t-1} + g_t x_t`；论文据此主张 SSM 的离散化是启发式门控机制的原理性基础。
- 边界重置（§3.5）：多条独立序列拼在一起时，选择性 SSM 可以在边界把状态重置（对应 `Δ_t → ∞`，或上面的 `g_t → 1`），这是 LTI 模型做不到的。
- 初始化的具体取值（§3.6）：默认真数值状态——复数值用于感知类模态，论文在语言/DNA 等离散模态上默认实数；复数初始化用 S4D-Lin（第 n 个元素为 `−1/2 + n·i`），实数初始化用基于 HIPPO 的 S4D-Real（`−(n+1)`）。`Δ` 被当作 bias 项初始化成 `τ_Δ^{-1}(Uniform([0.001, 0.1]))`。
- 合成任务（§4.1）：Selective Copying 上 S6 层本身就足以解任务（No-gate 架构下 97.0，Mamba 下 99.8），而此前的 LTI SSM 即使配更强的架构也解不了；Induction Heads 上 Mamba 可外推到训练长度的 4000×（百万级序列），其它方法不超过 2×。
- 语言建模（§4.2）：在 Pile 上按 GPT-3 的深度/宽度规格比较，Mamba 是**第一个在缩放律上追平强 Transformer++ 配方（RoPE + SwiGLU + RMSNorm、去 bias、更高学习率）的无注意力模型**，且序列变长（2k / 8k 上下文）时优势更明显；对比范围约 125M–1.3B 参数。
- 消融（§4.6）：`Δ` 是最重要的选择性参数（论文归因于它与 RNN gating 的联系，见 Theorem 1）；把 `Δ`、`(B,C)` 的投影从静态改成选择性收益最大；N 增大只有在 `B`、`C` 也选择性时才明显带来收益；语言建模上简单的实对角初始化（S4D-Real）比复数值初始化（S4D-Lin）更好，随机初始化也可用。
- 速度与内存基准（§4.5）：benchmark 用 state expansion `N=16`；自研 SSM scan 在序列长度 **2K 以上快于 FlashAttention-2**、比 PyTorch 标准 scan 实现快 **20–40×**；Mamba 推理吞吐比同规模 Transformer 高 **4–5×**，论文把主因归为**没有 KV cache 因而能用大得多的 batch**（举例：未训练的 Mamba-6.9B 吞吐高于小 5 倍的 Transformer-1.3B）。摘要里写作 5×。
- 论文自述的规模边界（§5 discussion）：实证只覆盖小规模，低于多数强开源 LLM（如 Llama）以及 RWKV、RetNet 等已在 7B+ 评估过的模型，更大规模下 Mamba 是否仍占优仍待评估。

## 不能推出的结论

- 摘要的「5× 吞吐」与正文的「4–5×」把主因归为**无 KV cache 带来的更大 batch**，是在论文自己的 scan 实现、特定硬件与序列长度下测得的核心层/端到端结果；不能当成任意后端、任意 batch 下的通用结论，也不能把「线性扩展」与「吞吐优势」混为一谈（前者是相对卷积型 SSM 的伪线性而言的算法性质）。
- 「推理不需要缓存之前的元素」是论文 §1 的算法层结论；本库原先写的「推理态只有卷积态与 SSM state」把 block 里的短卷积也算进去了，属**实现层**描述——论文没有写「卷积态」，两者字面上并不一致，需要回到官方实现核验卷积窗口与缓存行为。
- 论文的参数只有 `(Δ, A, B, C)`，**没有 D（feed-through / skip）项**；若某处写 `y_t = C_t h_t + D x_t`，那是实现或其它文献的口径，不能当作本论文的结构。
- 「论文明说不做 A 的选择性」不等于「经验上证明 A 的选择性无用」：论文只给了假设（hypothesize）和为简化而省略的理由，§4.6 只消融了 `Δ`、`B`、`C`，没有任何 A 选择性的实验。
- Theorem 1 只在 `N=1、A=−1、B=1、τ_Δ=softplus` 这组特定取值下成立，不能说选择性 SSM 一般等价于门控 RNN。
- Mamba-3B 对齐的 Transformer 是论文自己的训练配方与数据（且实证只到 125M–1.3B，§5 自述），不能概括为「SSM 全面超过 Transformer」，也不能外推到更大规模、其它数据或其它模态。
- 「小推理态」不等于「对量化更鲁棒」：论文不含量化或低比特分析，本库不能据此推断量化误差路径或校准需求。
- 论文只给到**算法级**的 kernel 描述（附录 D 的 fused kernel 流程），没有给实现级的权重布局（哪些投影融合/拆分）、CUDA 算子或 checkpoint 的 tensor 名称与形状；不能据此推断某个 checkpoint 的实际张量与形状。
- `E=2`、state size `N=16` 等是论文实验的固定取值，不是结构约束。
