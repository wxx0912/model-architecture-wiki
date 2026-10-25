---
{
  "id": "paper-transformer",
  "title": "Attention Is All You Need",
  "category": "paper",
  "url": "https://arxiv.org/abs/1706.03762v7",
  "version": "v7（2023-08-02）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "§3.1–3.5（结构与公式）、§4 Table 1、§5.2–5.4（训练设置）、§6.1 Table 2、§6.2 Table 3、§6.3 Table 4",
  "topic": "core",
  "excerpt_sha256": null
}
---

## 这份来源是什么

Transformer 原始论文（arXiv 1706.03762，v7）。它给出 encoder-decoder 堆叠结构、scaled dot-product attention 与 multi-head attention、position-wise FFN、正弦位置编码，以及 embedding 与输出层共享权重，是本库 `component-attention`、`component-mlp`、`component-embedding-lm-head`、`component-kv-cache` 和 `paradigm-ar` 的结构依据。

论文的实验只有两项：机器翻译（§6.1–6.2）与英语成分句法分析（§6.3）。全文不涉及量化、低比特或部署。

## 已读定位与阅读形式

以 arXiv e-print 的 LaTeX 源为主（`model_architecture.tex`、`why_self_attention.tex`、`training.tex`、`results.tex`），再用 ar5iv HTML（`https://ar5iv.labs.arxiv.org/html/1706.03762`）复核章节号与数值。下文章节号、公式号、表号均按 v7。

## 关键事实

- Encoder 与 decoder 各由 N=6 个相同层堆叠；每个子层的输出是 `LayerNorm(x + Sublayer(x))`，即归一化在残差相加之后；d_model=512（§3.1）。
- Scaled dot-product attention 为 `Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V`（§3.2.1，Eq.1）。除以 √d_k 的理由是 d_k 变大时点积幅度随之变大、softmax 会落入梯度极小的区域；脚注给出在 q、k 各分量相互独立、均值为 0、方差为 1 的假设下，q·k 的方差为 d_k。
- Multi-head attention 用 h=8 组并行投影，`d_k = d_v = d_model/h = 64`，输出为 `Concat(head_1,…,head_h)·W^O`（§3.2.2）。该式在 LaTeX 源里是无编号的 `align*` 环境；投影矩阵形状为 W^Q_i, W^K_i ∈ R^(d_model×d_k)、W^V_i ∈ R^(d_model×d_v)、W^O ∈ R^(h·d_v×d_model)。
- Decoder 的 self-attention 在 softmax 前把非法连接（上三角）置为 −∞，使位置 i 只能依赖位置 <i 的输出（§3.1 decoder 段、§3.2.3）。
- FFN 为 `FFN(x) = max(0, xW_1 + b_1)W_2 + b_2`（§3.3，Eq.2）；d_model=512、d_ff=2048，同一层内所有位置共享同一组参数，层与层之间不共享。
- 输入 embedding、输出 embedding 与 pre-softmax 线性层共享同一权重矩阵，embedding 权重额外乘 √d_model（§3.4）。
- 正弦位置编码为 `PE(pos,2i) = sin(pos/10000^(2i/d_model))`、`PE(pos,2i+1) = cos(pos/10000^(2i/d_model))`；论文称它与 learned positional embedding 的结果 "nearly identical"（§3.5，Table 3 行 E）。
- 复杂度对比（§4，Table 1）：self-attention 每层 O(n²·d)、最少顺序操作 O(1)、最大路径长度 O(1)；recurrent 为 O(n·d²)、O(n)、O(n)；convolutional 为 O(k·n·d²)、O(1)、O(log_k n)。
- 训练设置（§5）：8×P100；base 模型 100K 步约 12 小时（0.4 s/步），big 模型 300K 步约 3.5 天（1.0 s/步）；Adam 取 β₁=0.9、β₂=0.98、ε=1e-9，warmup=4000（学习率公式为 Eq.3）；base 与 EN-FR big 用 dropout 0.1（其余 big 配置 0.3），label smoothing 统一 0.1（§5.2–5.4）。
- 机器翻译结果（§6.1，Table 2）：EN-DE BLEU 为 base 27.3、big 28.4；EN-FR 表中 big 为 41.8，但同一段正文写成 41.0，v7 仍保留这处不一致。解码用 beam size 4、length penalty α=0.6、最大输出长度 input+50。
- 消融（§6.2，Table 3）：base 为 N=6、d_model=512、d_ff=2048、h=8，dev PPL 4.92、BLEU 25.8、65M 参数；big 为 N=6、d_model=1024、d_ff=4096、h=16，PPL 4.33、BLEU 26.4、213M 参数。
- 句法分析（§6.3，Table 4）：4 层 Transformer、d_model=1024，在 Penn Treebank 的 WSJ 40K 句上训练；WSJ-only 得 91.3 F1，半监督得 92.7 F1。这一节用 beam size 21、α=0.3、最大输出长度 input+300，与机器翻译的解码配置不同。

## 不能推出的结论

- 论文的层是"归一化在残差之后"（Post-LN）。不能据此推断某个具体实现也用 Post-LN；本库讨论的 Pre-LN、RMSNorm 等实现不属于本篇内容。
- 本篇只覆盖 2017 年的 MHA、正弦位置编码与 FFN。RMSNorm、RoPE、GQA/MQA/MLA、SwiGLU、MoE 都不在论文内，相关结论不能引到这篇。
- Table 2 的 EN-FR 数值在表（41.8）与正文（41.0）中不一致，引用时必须指明出处，不能当作单一确定值。
- Table 1 是理论复杂度量级，不含常数项、访存与 kernel 实现细节；不能据此推断实际速度或显存占用。
- Table 2 的训练 FLOPs 是按 GPU 数 × 训练时间 × 估算单卡算力（K80 2.8、K40 3.7、M40 6.0、P100 9.5 TFLOPS）推算的估算值，不是实测。
- 论文明确说明 label smoothing 会恶化 PPL 但改善 BLEU；不能只用 PPL 判断某个训练配置的好坏。
- 论文不含任何低比特或量化内容，不能从本篇推出量化误差路径、校准需求或后端约束。
