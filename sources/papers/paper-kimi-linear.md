---
{
  "id": "paper-kimi-linear",
  "title": "Kimi Linear: An Expressive, Efficient Attention Architecture",
  "category": "paper",
  "url": "https://arxiv.org/abs/2510.26692",
  "version": "v2（2025-11-01）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1、§2.2、§3（Eq.1、§3.1 Eq.2–9、§3.2）、§4、§5.1–5.6（Table 1–5）、§6.1–6.3（Table 6）、§7.1（Table 7）、附录 A–D（含 appendix:results）",
  "topic": "architectures",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---
## 这份来源是什么

Kimi Linear 技术报告（arXiv 2510.26692，v2，Moonshot AI）。核心是 **Kimi Delta Attention（KDA）**：把 Gated DeltaNet 的标量遗忘门 `α_t` 换成**逐通道对角门 `Diag(α_t)`**，并给出配套的高效 chunkwise 并行算法；架构上把 KDA 层与 Full MLA 层按 **3:1** 逐层交替（NoPE 只作用于 MLA 层），预训练出 48B 总参 / 3B 激活的稀疏 MoE 模型。

是本库 `models/kimi-linear`、`paradigm-linear-attention` 与 `component-attention`（MLA+NoPE、混合层排布）的一手材料：KDA 的更新规则与参数化、门控从"per-head 标量"到"per-channel 对角"这一步、混合比例与长上下文证据都来自这里。与相邻来源的关系：`paper-gdn` 提供被改进的 gated delta rule 与 chunkwise 记法，`paper-deepseek-v2`/`paper-deepseek-v3` 提供 MLA。

范围边界：全文不含量化或低比特**实验**，也没有 state 数值分布统计；但多处讨论**数值精度/稳定性**（细粒度衰减的除法需全精度二级 chunking、`1/Γ` 不稳定、q/k 的 L2Norm 用于特征值稳定、训练/推理引擎精度不匹配会拖垮 RL），属量化相关线索而非量化和低比特结果。效率数据只有自研 kernel 的自报实测，且**未披露评测硬件**。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源（`main.tex` 与 `\input` 的章节、`figures/`、`table/`、`appendix/`），另读 arXiv HTML 渲染版核对摘要与关键数值。v1 2025-10-30、v2 2025-11-01，本地源码与最新 e-print 逐字一致。
- **编号口径**：按 v2 浮动体出现顺序，全文 **Figure 1–7、Table 1–9**；源里三个 `figure*` 只有图没有 `\caption`，**不占图号**；DPLR/KDA 伪代码放在一个带两个 `subfigure` 的 `figure` 浮点里（子图 label `listing:dplr`/`listing:KDA`，正文以 `Listing~\ref{...}` 引用），该浮点无独立父级 `\caption`。KDA 递推为 **Eq.1**（§3），分块展开 Eq.2–9，复杂度 Eq.13/14（§6.3）。

## 关键事实

- **KDA 递推**（§3 Eq.1）：`S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t-1} + β_t k_t v_tᵀ`（`S_t ∈ R^{d_k×d_v}`）、`o_t = S_tᵀ q_t`。相对 Gated DeltaNet 的标量门，KDA 把门升级成**逐通道对角矩阵 `Diag(α_t)`**（思路类比 GLA），对有限状态 RNN 的记忆做更细粒度的衰减控制。
- **参数化**（§4）：`q_t, k_t = L2Norm(Swish(ShortConv(W x_t)))`、`v_t = Swish(ShortConv(W_v x_t))`、`α_t = f(W_α^↑ W_α^↓ x_t) ∈ [0,1]^{d_k}`（**低秩投影，rank 等于 head dim**）、`β_t = Sigmoid(W_β x_t) ∈ [0,1]`；`d_k = d_v = 128`（全部实验）。输出 `o_t = W_o( Sigmoid(W_g^↑ W_g^↓ x_t) ⊙ RMSNorm(KDA(...)) )`——输出门同样低秩，论文称可在与全秩门性能相当的同时缓解 Attention Sink。
- **chunkwise 算法**（§3.1 Eq.2–9）：chunk 展开后用 WY 表示压缩、UT transform 降低非 matmul FLOPs；状态更新 `S_{[t+1]} = Diag(γ^C) S_{[t]} + (Γ^{i→C} ⊙ K)ᵀ(U − W S_{[t]})`，输出拆成 inter-chunk 与 intra-chunk 两项（Eq.9）。
- **KDA 是受限 DPLR**（§3.2、§6.1–6.3、§7.1；Table 6 递归/并行等价形式对照、Table 7 更新规则总表）：一般 DPLR 为 `S_t = (D − a_t b_tᵀ) S_{t-1} + k_t v_tᵀ`；KDA 取 `D = Diag(α_t)`、`a_t = β_t k_t`、`b_t = k_t ⊙ α_t`（等价于把 `a = b = k` 绑定）。这样可把衰减项提到 Householder 变换之外，**免掉 DPLR 需要的二级 chunking**（二级 chunk 矩阵计算 4→2 次），并在 inter-chunk/输出阶段**再消掉约 3 次矩阵乘**，算子效率相对一般 DPLR 提升约 **100%**；kernel 实测到 64k 长度约为 DPLR 的 **2× 速度**（Figure 2）。
- **复杂度**（§6.3 Eq.13/14）：单 head、chunk size `C = 64`、序列长 `T` 时 `FLOPs_KDA = 6Td_h² + 3TCd_h + TC²`；full attention 主项 `FLOPs_Attn = 2T²d_h`。KDA 的 state 固定为 `d_k × d_v` / head（128×128），**不随序列增长**；预填用 chunk kernel，解码切到 recurrent kernel。
- **数值精度/稳定性线索**（§3.2、§4、§5.4.2、§6.2）：细粒度衰减引入的除法需要全精度二级 chunking（GLA 做法，阻碍 half-precision matmul 充分利用）；累计衰减项的倒数 `1/Γ` 可能数值不稳定；q/k 的 L2Norm 用于特征值稳定；§5.4.2 提到训练/推理引擎的精度不匹配会拖垮 RL，故引入 truncated importance sampling。
- **"可学习位置编码"视角**（§2.2、§6.1）：带门控的 delta 规则可写成数据相关的**乘性位置编码**，放宽了 RoPE 的正交约束（原文该式括号内下标有笔误）；论文因此把**所有 Full MLA 层设为 NoPE**，把位置/recency 信息整体交给 KDA 层（KDA 的逐通道门对应 RoPE 的逐维频率）。
- **混合结构与规模**（§1、§4、§5.4）：主干沿用 Moonlight；KDA 与 MLA 按 **3:1** 逐层交替（选 layerwise 而非 headwise，理由为基建简单与训练稳定）；MoE 每 token 激活 **256 中的 8 个专家、含 1 个共享专家（7 路由 + 1 共享）**，首层是 dense 层 → **48B 总参 / 3B 激活**。
- **合成任务**（§5.1 Figure 4）：固定为 2 层 2 head、head dim 128，训练 ≤20000 步，lr 网格 `{5e-5, 1e-4, 5e-4, 1e-3}`，长度 256→2048（Palindrome、MQAR、Stack）。KDA 随长度增长在三个任务上都保持最高准确率，Palindrome 与 MQAR 上**收敛显著快于 GDN**；论文明确说 **Mamba2 在该设置下三个任务全部失败**。
- **组件消融**（§5.2 Table 1，16 head/16 层，同 FLOPs 预算）：混合比例 **3:1 最优**（train/val PPL 9.23/5.65），纯 MLA（0:1）最差（9.45/5.77）；7:1 验证更差（5.70），1:1 验证接近（5.66）但推理开销更大，15:1 明显更差（9.34/5.82）。输出门 **Sigmoid（默认）> 无门 > Swish**；去掉 depthwise 短卷积后 9.29/5.70，仍不可少。
- **NoPE vs RoPE**（§5.2、§5.5 Table 5）：Kimi Linear（NoPE）长上下文平均 **54.5**，同配置但给 MLA 加 RoPE 的版本只有 51.8；短上下文两者接近。论文解释为 RoPE 把过强的显式相对位置信号压在 global 层、与线性层的隐式位置偏置不匹配。
- **scaling law**（§5.3 Table 2、Figure 5）：8/64 专家、Muon 优化器，五个规模（653M–1.7B 激活、16–24 层、38.8B–128.0B token、ctx 4096）；相对 MLA 基线约 **1.16× 算力效率**，论文自述未做充分超参调优。
- **主结果：预训练与 SFT**（§5.5 Table 3/4，1.4T token、4k 上下文、同架构同参数量）：预训练阶段 **Kimi Linear > GDN-H > MLA**，MMLU 73.8、MMLU-Pro 51.0、BBH 72.9 均最高，唯一失手 EvalPlus（GDN-H 63.1 > 60.2）；SFT 后 GPQA-Diamond 62.1、AIME 2025 21.3、PolyMath-en 43.6、LiveCodeBench v6 26.0 最高，例外有 MATH500（GDN-H 83.0 > 81.2）、EvalPlus（MLA 62.6 > 61.0）、LiveBench（GDN-H 46.4 > 45.2）。
- **长上下文**（§5.5 Table 5，128k）：RULER **84.3**、RepoQA **68.5**、平均 **54.5** 全表最高；GDN-H 在长上下文掉到 MLA 之后（51.2 vs 52.2）。例外是 LongBench V2（35.0 < 36.1）、Frames（58.8 < 60.5）、Long Code Arena/Commit（32.7 < 33.2）。
- **RL**（§5.5 Figure 6）：用 K2 的数学 RLVR 数据、算法与超参，与 MLA 完全相同；Kimi Linear 在训练集与 AIME 2025/MATH500 上收敛更快且更高。
- **效率**（摘要、§5.6、§6.3 Figure 7、Figure 1b）：KV cache 相对 full attention 降低 **up to 75%**；prefill 相对 MLA 在 512k 为 2.3×、1M 为 2.9×；解码 TPOT（batch=1）标 512k **1.8×**、1M **2.2×**；同一 1M 点另有 §6.3 正文 2.3×、§7 "over 2×"、§5.6 正文 6×、Figure 1b 图注/§6.3/结论 6.3×（1.84ms vs 11.48ms，来自更大 batch）。
- **开源与 5.7T 版本**：KDA kernel/vLLM 实现与 checkpoint 见摘要（`fla-org/flash-linear-attention` 的 `fla/ops/kda`、`moonshotai/Kimi-Linear-48B-A3B-Instruct`）；最终发布 checkpoint 训到 **5.7T token**、支持 **1M** 上下文，1M RULER 上得 **94.8**（附录 D）。

## 不能推出的结论

- 论文不含量化/低比特实验，也不能据它推出一套量化方案；其数值精度/稳定性讨论都在 kernel 实现与训练稳定性层面，不是量化误差分析。"逐通道门"是**表达力**设计，不是量化友好性证据；"state 是每 head 一个 128×128 矩阵、不随序列增长"只是结构事实，不能直接推出该 state 可低比特存储或误差可控。
- "算子效率 +100% / kernel 2× / prefill 2.3–2.9× / 解码 6.3×"都是自研 kernel 的自报实测，且**论文未披露评测硬件**；Figure 7 的 batch size=1 只适用于该图，Figure 1b 的 6.3× 来自更大 batch。不能当算法固有开销或线上吞吐；KV cache 的 75% 是 3:1 混合的结构性减少，不是压缩/量化收益。
- 同一 1M 解码点至少有 5 种口径（Figure 7b 2.2×、§6.3 2.3×、§7 "over 2×"、§5.6 6×、§6.3/结论/Figure 1b 图注 6.3×）；引用须写明出处、曲线与 batch 设置，不能合并成一个数。
- Table 1 的消融（3:1 最优、Sigmoid>无门>Swish、卷积不可少）来自 16 head/16 层、同 FLOPs 预算的训练（原文未披露是否多次或多种子），是该规模下的经验折中，不能当作跨规模普适最优。
- "Mamba2 在该设置下全部失败"只在 2 层 2 head、给定 lr 网格与步数下成立（原文限定 "in our model settings"），不能外推为 Mamba2 普遍无效。
- §5.5 的"全面领先"有明确例外（预训练 EvalPlus；SFT MATH500/EvalPlus/LiveBench；长上下文 LongBench V2/Frames/Commit），且都基于同一内部评测框架与 1.4T token 口径；不能跨来源混用或外推到其它 checkpoint。
- NoPE 结论是"**在混合结构里**把位置信号交给 KDA 层"，不能简化为"MLA 不需要位置编码"；摘要"for the first time 在公平对比下超过 full attention"是论文自述，本页未做独立复现。scaling law 的 1.16× 也是论文自述的拟合结果，不是确定的增益上限。
- 本页只是论文级证据，不证明任何具体 checkpoint 的行为，也不能替代目标实现上的运行验证。
