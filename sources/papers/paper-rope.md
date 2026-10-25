---
{
  "id": "paper-rope",
  "title": "RoFormer: Enhanced Transformer with Rotary Position Embedding",
  "category": "paper",
  "url": "https://arxiv.org/abs/2104.09864",
  "version": "v5（2023-11-08）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "§3.1 Eq.(11)、§3.2 Eq.(12)–(16)、§3.3 Eq.(17)–(19)、§3.4.1–3.4.3（推导、逐元素实现 Eq.(34) 一带、long-term decay Figure 2）、§4.1 Table 1、§4.2–4.3 Figure 3 与 Table 2、§4.4、§4.5 Table 3–5 与局限",
  "topic": "core",
  "excerpt_sha256": null
}
---

## 这份来源是什么

RoPE（Rotary Position Embedding）原始论文（arXiv 2104.09864，v5）。它把相对位置编码成 query/key 上的**旋转矩阵**（乘性），使 attention 内积只依赖相对位置，并给出与线性注意力结合的方式、逐元素高效实现与长程衰减的证明。是本库 `component-position-encoding` 判断"位置信息可用旋转注入、且旋转不改变向量范数"的直接依据。

论文自身不涉及量化或低比特；讨论的是位置编码的数学构造与 NLP 任务上的收敛/精度。

## 已读定位与阅读形式

以 arXiv e-print 的 LaTeX 源 `roformer_arxiv.tex` 为主，公式号与表号用 ar5iv HTML（`https://ar5iv.labs.arxiv.org/html/2104.09864`）逐条复核；版本号取自 abs 页 Submission history（最新 v5）。下文 §、Eq.、Table、Figure 号按 v5 编译后的编号（与 ar5iv 标签一致）。

## 关键事实

- 目标（§3.1，Eq.(11)）：要求 query/key 的内积只以相对位置为变量，即 `⟨f_q(x_m,m), f_k(x_n,n)⟩ = g(x_m, x_n, m−n)`。
- 2D 情形的解（§3.2.1，Eq.(12)）：`f_q(x_m,m) = (W_q x_m)·e^{imθ}`、`f_k(x_n,n) = (W_k x_n)·e^{inθ}`，内积为 `Re[(W_q x_m)(W_k x_n)* e^{i(m−n)θ}]`，θ 为预设非零常数；几何含义是把仿射变换后的 embedding 按位置索引乘角度旋转。
- 一般形式（§3.2.2，Eq.(14)–(15)）：d 取偶数，把 d 维空间切成 `d/2` 个 2 维子空间，用分块对角旋转矩阵 `R^d_{Θ,m}` 左乘：`f_{q,k}(x_m,m) = R^d_{Θ,m} W_{q,k} x_m`。
- **维度配对固定为相邻两维**：`R^d_{Θ,m}` 是块对角矩阵，第 i 块作用在 `(x_{2i−1}, x_{2i})` 这一对上（§3.2.2，Eq.(15)；§3.4.2 的逐元素实现同样以相邻两维成对）。
- 角度参数 Θ（§3.2.2 正文，非独立编号公式）：写作 `θ_i = 10000^{−2(i−1)/d}`，`i ∈ [1,…,d/2]`。§3.3 的 Long-term decay 段与 §3.4.3 反而写作 `θ_i = 10000^{−2i/d}`（沿袭 Vaswani），且 §3.4.3 的求和下标是 `i = 0 … d/2−1`；令 `j = i−1` 后两式是同一个序列（分别对应 1-indexed 与 0-indexed），论文只是没写这句换算。
- 内积结果（§3.2.2，Eq.(16)）：`q_mᵀk_n = x_mᵀ W_qᵀ · R^d_{Θ,n−m} · W_k x_n`，其中 `R^d_{Θ,n−m} = (R^d_{Θ,m})ᵀ R^d_{Θ,n}`；`R^d_Θ` 是**正交矩阵**，作者据此称位置编码过程稳定（旋转不改变向量范数）。注意原文该式最后一步的 `W_q` 漏了转置（原文写作 `xᵀW_qR^d_{Θ,n−m}W_kx_n`），本页写的是更正后的形式。
- **符号约定在论文内部不统一**：§3.1、§3.2 的推导用相对位置 `m−n`，而 Eq.(16) 与 §3.4.1 的结果式用 `n−m`。
- 与既有方法的区别（§3.2.2 末；"不改变范数"见 §3.3）：正弦/学习式位置编码是**加性**的（在输入上加 p_i），RoPE 是**乘性**的，通过旋转矩阵乘积天然带入相对位置，且不改变 hidden 表示的范数。
- 长程衰减（§3.3 提出，证明见 §3.4.3，Figure 2）：在 `θ_i = 10000^{−2i/d}` 下可证明内积随相对位置增大而衰减。
- 与线性注意力结合（§3.3，Eq.(17)–(19)）：把 attention 写成 `Σ sim(q_m,k_n)v_n / Σ sim(q_m,k_n)` 的一般形式后，用 query 侧 `φ(q_m)` 与 key 侧 `φ(k_n)`（原文用 `\phi` / `\varphi` 两个符号）的内积替换相似度；因为 RoPE 只旋转、不改变范数，可直接把旋转矩阵乘在 `φ(q)`、`φ(k)` 的输出上。作者**刻意保持分母不变**以避免除零，并承认此时 value 的权重不严格是概率归一化的。
- 逐元素高效实现（§3.4.2，Eq.(34) 一带）：利用 `R` 的稀疏分块结构，把 `R^d_{Θ,m}x` 写成"逐对元素乘 cos + 相邻元素交换后乘 sin"的形式，避免直接做 `d×d` 矩阵乘。
- 实验环境（§4）：论文自述跑在 "two cloud severs with 4 x V100 GPUs"（原文拼写 severs，未说明是每台 4 张还是合计 4 张）。
- 机器翻译（§4.1 Table 1）：WMT2014 EN-DE（约 4.5M 句对、37k BPE 词表、beam 4 + length penalty 0.6、取最后 5 个 checkpoint 平均）；Transformer-base 27.3 BLEU，RoFormer 27.5。
- 预训练（§4.2 Figure 3 左）：用 RoPE 替换 BERT 的正弦位置编码，bert-base-uncased、batch 64、max len 512、100k steps、AdamW lr 1e-5；RoFormer 的 MLM loss 收敛更快。
- GLUE 微调（§4.3 Table 2，max len 512、batch 32、lr ∈ {2,3,4,5}e-5、3 epochs）：RoFormer 在 MRPC 89.5（BERT 88.9）、STS-B 87.0（85.8）、QQP 86.4（71.2）上更高，但在 SST-2 90.7（93.5）、QNLI 88.0（90.5）、MNLI 80.2/79.8（84.6/83.4）上更低。§4.3 正文自称"6 个数据集里 3 个显著超过 BERT"，而摘要与 §1 的说法更强（"consistently overcomes/consistently achieves better performance"）——与 Table 2 的事实不符。
- Performer + RoPE（§4.4 Figure 3 右）：Enwik8 上 12 层 char-level PerFormer（768 维、12 head），lr 1e-4、batch 128、max len 1024；加 RoPE 后收敛更快、loss 更低。
- 中文长文本（§4.5）：在 WoBERT 上把绝对位置编码换成 RoPE（Table 3 给出 BERT=char/abs.、WoBERT=word/abs.、NEZHA=char/rel.、RoFormer=word/RoPE 的对照），约 34GB 中文语料分多阶段预训练（Table 4 的 6 个阶段 max seq length / batch / steps / loss / accuracy，accuracy 63.4%–67.4%）；CAIL2019-SCM（Table 5，8964 组三元组、6:2:2 划分，准确率）：RoFormer-512 test 68.29% vs BERT-512 67.77%、WoBERT-512 68.10%；最大输入长度提到 1024 后 RoFormer-1024 得 test 69.79%，正文称比（512 截断的）WoBERT 绝对提升 1.5%，但按 Table 5 数值实为 1.69%。
- 论文自述局限（§4.5 末）：（1）缺少"为什么比使用其它位置编码的基线收敛更快"的完整解释；（2）长程衰减性质相似却在长文本上更好，也没有令人信服的解释；（3）RoFormer 仍建立在 Transformer 基础设施上，预训练需要硬件资源。

## 不能推出的结论

- 论文没有写出 Θ 两种写法（1-indexed 与 0-indexed）的换算，但两者在数学上是同一序列；**真正需要外部核对的是旋转的维度配对/布局约定**（本篇固定相邻两维成对，具体实现是否采用同一布局不能默认一致）。
- 论文内部相对位置符号在 `m−n` 与 `n−m` 之间不统一，引用内积式时必须说明采用哪一个约定。
- Eq.(16) 末式原文漏了 `W_q` 的转置，属论文笔误；不能把原文写法当作正确推导。
- GLUE 上 RoFormer 并非全面超过 BERT（SST-2、QNLI、MNLI 更低）；摘要与 §1 的 "consistently overcomes" 口径与 Table 2 冲突，不能只引摘要。
- WMT2014 上的 27.5 vs 27.3 是单个模型的单次结果，论文未给方差或多轮统计，不能当作稳健优势；Table 1 也只对比 Transformer-base 一种基线。
- 长程衰减是内积幅度随相对距离衰减的数学性质，论文没有给出实际注意力分布的实测，不能据此断言模型实际只依赖近邻 token。
- `R^d_Θ` 正交只保证旋转不改变向量范数，不能推断数值精度、长序列外推能力或对量化的鲁棒性。
- 2D 推导依赖"内积只依赖相对位置"的假设，并要求 d 为偶数；不能套用到任意维度划分或任意配对方式。
- 论文不含任何量化或低比特内容，不能从本篇推出量化误差路径或后端约束。
