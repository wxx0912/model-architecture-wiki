---
{
  "id": "model-kimi-linear",
  "title": "Kimi Linear",
  "type": "model",
  "tags": [
    "linear-attention",
    "state-update",
    "gdn",
    "mla",
    "quantization"
  ],
  "related": [
    "paradigm-linear-attention",
    "paradigm-ar",
    "paradigm-ssm",
    "component-kv-cache",
    "component-attention"
  ],
  "sources": [
    "paper-kimi-linear",
    "paper-gdn",
    "repo-kimi-linear"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "线性注意力",
    "linear attention"
  ],
  "updated": "2026-10-24"
}
---

# Kimi Linear

## 一句话理解

Kimi Linear 是混合注意力语言模型：**Kimi Delta Attention（KDA）** 的线性状态更新层与 **Full MLA** 层按 **3:1** 逐层交替，MLA 层全部设为 **NoPE**，把位置 / recency 信息整体交给 KDA。目标是在长上下文推理中降低 cache / 计算开销，同时保留部分传统 attention 层。

## 结构规格表

| 项 | Kimi Linear |
| --- | --- |
| 总参 / 激活 | 48B / 3B |
| 层排布 | KDA 与 Full MLA 按 **3:1** 逐层交替；首层 dense |
| KDA 维度 | head dim `d_k = d_v = 128`，state `S ∈ R^{d_k×d_v}`（每 head 128×128，不随序列增长） |
| KDA 门控 | `α_t` 逐通道对角（低秩，rank = head dim）、`β_t` sigmoid 标量；输出门低秩 sigmoid |
| chunk | `C = 64`；prefill 走 chunk kernel、decode 走 recurrent kernel |
| MLA | 所有 Full MLA 层 **NoPE** |
| MoE | 每 token 激活 256 中的 8 个（7 路由 + 1 共享） |
| 上下文 | 论文主实验 1.4T token / 4k（Table 3/4 的短上下文口径）；另有 128k 长上下文结果（Table 5）；发布 checkpoint 训到 5.7T token、支持 1M |

## 结构与数据流

KDA 递推（§3，Eq.1）：

```text
S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t-1} + β_t k_t v_tᵀ,   o_t = S_tᵀ q_t
```

相对 Gated DeltaNet 的**标量**门，KDA 把门升级成**逐通道对角矩阵 `Diag(α_t)`**。参数化（§4）：`q_t, k_t = L2Norm(Swish(ShortConv(W x_t)))`、`v_t = Swish(ShortConv(W_v x_t))`、`α_t = f(W_α^↑ W_α^↓ x_t) ∈ [0,1]^{d_k}`、`β_t = Sigmoid(W_β x_t) ∈ [0,1]`；输出 `o_t = W_o(Sigmoid(W_g^↑ W_g^↓ x_t) ⊙ RMSNorm(KDA(...)))`。chunkwise 算法用 WY 表示压缩 + UT transform（§3.1）。

KDA 是**受限 DPLR**（§3.2、§6.1–6.3）：取 `D = Diag(α_t)`、`a_t = β_t k_t`、`b_t = k_t ⊙ α_t`，把衰减项提到 Householder 变换之外，把二级 chunk 矩阵计算从 4 次降到 2 次。论文指出带门控的 delta 规则可看成数据相关的乘性位置编码，据此对 global full attention 采用 NoPE。

## 量化关联

**结构事实**

- KDA 的 state 固定为 `d_k×d_v` / head、不随序列增长；推理时模型同时维护 KDA 线性状态与 MLA KV cache 两类运行时状态。
- 论文多处讨论**数值精度 / 稳定性**：细粒度衰减的除法需要全精度二级 chunking、累计衰减倒数 `1/Γ` 可能不稳定、q/k 的 L2Norm 用于特征值稳定、训练/推理引擎精度不匹配会拖垮 RL。
- 范围事实：论文不含量化或低比特实验；官方仓库本身不含代码。

**量化推断（本库推断）**

- state 误差随序列递推，校准与验证必须覆盖长序列、chunk 边界与连续 decode。
- gate、delta update、state readout 与 MLA cache 是不同数值路径，需分别量化和诊断。
- 混合层结构支持按层型分组评估，但任何层型专用 bit 配置都要通过目标 kernel / 导出契约验证。

## 结果边界

- KDA 的"逐通道门"是**表达力**设计，不是量化友好性证据；state 大小固定也不等于可以低比特存储。
- 效率数字（prefill 2.3×/2.9×、解码 1.8×/2.2×、KV cache 降 up to 75%）是自研 kernel 的自报实测、**未披露评测硬件**；KV 的 75% 是 3:1 混合的**结构性减少**，不是压缩/量化收益。
- 长上下文并非全面最优：128k 平均 54.5 最高，但 LongBench V2（35.0 < 36.1）、Frames（58.8 < 60.5）、Long Code Arena/Commit（32.7 < 33.2）低于对照。
- 官方仓库在所读 commit 下**只有 README、技术报告 PDF 与图片，没有代码**；KDA 的 kernel 实现在 Flash Linear Attention，checkpoint 在 HuggingFace（本库当前未把这两处登记为来源），本页实现层描述以论文为限。
- 论文不含量化实验；本页量化内容除已注明者外均为结构推断，本库未运行任何量化或评测。

## Agent 如何使用

首轮读本页即可判断目标是 Kimi Linear 这类混合线性注意力模型、KDA/MLA 的层比例、state 维度与位置编码策略。判定一个实现属于 Kimi Linear 的判据：KDA 与 Full MLA 按 3:1 交替、门 `α_t` 是逐通道对角（而非 GDN 的标量门）、state 为每 head 128×128、MLA 层为 NoPE。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| KDA 递推、参数化、chunkwise 与 3:1 混合 | [`paper-kimi-linear`](../../sources/papers/paper-kimi-linear.md) |
| 被改进的 gated delta rule 基线 | [`paper-gdn`](../../sources/papers/paper-gdn.md) |
| 官方仓库现状（无代码） | [`repo-kimi-linear`](../../sources/repos/repo-kimi-linear.md) |
| 线性/注意力混合的通用机制 | [Linear Attention](../paradigms/linear-attention.md) |
| 目标 checkpoint 的真实配置与后端 kernel 支持 | 该 checkpoint 的 `config.json` / Flash Linear Attention kernel / 运行时（本库不提供） |

回答量化问题时必须说明：KDA/MLA 层比例、KDA state 维度与 chunk size、位置编码（NoPE）、MLA cache 精度、测试序列长度，以及目标后端 kernel 支持。

## 来源与证据

- [Kimi Linear](../../sources/papers/paper-kimi-linear.md)：KDA 递推与参数化（§3–4）、chunkwise、3:1 混合与 NoPE（§4、§5.2）、效率与长上下文结果（§5.5–5.6）。
- [Gated Delta Networks](../../sources/papers/paper-gdn.md)：被 KDA 改进的 gated delta rule 与 chunkwise 记法。
- [MoonshotAI/Kimi-Linear](../../sources/repos/repo-kimi-linear.md)：固定 commit 下的仓库文件树（无代码）。

本页结构以上述来源为限；论文来源不含量化实验，官方仓库来源无代码；量化内容均为本库推断。
