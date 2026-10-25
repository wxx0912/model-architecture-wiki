---
{
  "id": "paper-rmsnorm",
  "title": "Root Mean Square Layer Normalization",
  "category": "paper",
  "url": "https://arxiv.org/abs/1910.07467",
  "version": "v1（2019-10-16）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "§3 Eq.(2)–(3)、§4 Eq.(4) 与 Table 1、§4.1 Eq.(5)–(6)、§4.2 Eq.(8)–(9)、§5、§6.1 Table 2、Table 4 与 Table 5、§6.3 Table 8、§6.4、摘要与 §7",
  "topic": "core",
  "excerpt_sha256": null
}
---

## 这份来源是什么

RMSNorm 原始论文（arXiv 1910.07467，v1，NeurIPS 2019）。它提出按均方根（RMS）统计量归一化 summed inputs，去掉 LayerNorm 的 re-centering、只保留 re-scaling 不变性，并给出只估计部分元素的 pRMSNorm 变体。是本库 `component-normalization` 判断"层归一化可以不做减均值"的直接依据。

论文自身不涉及量化、低比特或推理部署，只在结论里预期其效率改进与低精度算术、GPU kernel 融合"正交"。

## 已读定位与阅读形式

以 arXiv e-print 的 LaTeX 源 `paper.tex` 为主，公式号与表号用 ar5iv HTML（`https://ar5iv.labs.arxiv.org/html/1910.07467`）复核；版本号取自 abs 页 Submission history（该文只有 v1）。下文 §、Eq.、Table 号均按 v1。

## 关键事实

- 论文假设 LayerNorm 成功的原因是 **re-scaling 不变性**而非 re-centering 不变性，据此提出 RMSNorm：只按 RMS 正则化 summed inputs（§1、§4）。
- LayerNorm 定义为 `ā_i = (a_i − μ)/σ · g_i`，其中 `μ = (1/n)Σa_i`、`σ = sqrt((1/n)Σ(a_i − μ)²)`（§3，Eq.(2)–(3)）。
- RMSNorm 定义为 `ā_i = a_i / RMS(a) · g_i`，`RMS(a) = sqrt((1/n)Σ a_i²)`（§4，Eq.(4)）；论文指出当 summed inputs 均值为 0 时 RMSNorm 与 LayerNorm 完全相等。
- RMSNorm 的通用形式仍保留加性 bias：`y = f( (Wx)/RMS(a) ⊙ g + b )`（§4.1，Eq.(5)）；gain `g` 初始化为 1、bias `b` 通常初始化为 0（§3）。即本篇的 RMSNorm 并没有去掉 bias——这与许多后续实现里省略 bias 的 RMSNorm 不同。
- RMS 把 summed inputs 约束到半径按 √n 缩放的单位球上；论文称欧氏范数（只与 RMS 差 √n 因子）不 work（§4）——§6.1 Table 2 里 L2-Norm 的 Test14/Test17 为 20.7/22.0，低于 Baseline 的 21.7/23.4。
- RMS 的线性性 `RMS(αx) = α·RMS(x)`（Eq.(6)）使 RMSNorm 对权重矩阵缩放与输入缩放不变；因为不做 re-centering，它没有对应的平移不变性（§4.1）。
- 不变性对照（§4.1 Table 1）：RMSNorm 对 weight matrix re-scaling ✔、weight matrix re-centering ✘、weight vector re-scaling ✘、dataset re-scaling ✔、dataset re-centering ✘、single training case re-scaling ✔；与 LayerNorm 的唯一差别是 weight matrix re-centering（LayerNorm 为 ✔）。
- 梯度性质（§4.2，Eq.(8)）：`∂L/∂b = ∂L/∂v`、`∂L/∂g = (∂L/∂v) ⊙ Wx/RMS(a)`，两者对 x 与 W 的缩放都不变，且 ∂L/∂g 与归一化后的 summed inputs 成正比。
- `∂L/∂W` 中含矩阵项 `R = (1/RMS(a))·(I − (Wx)(Wx)ᵀ/(n·RMS(a)²))`，使 ∂L/∂W 对输入缩放不变、但与权重缩放负相关；论文把该负相关解释为"隐式学习率调节"（§4.2，Eq.(9)）。
- pRMSNorm 只用前 p% 元素估计 RMS：`RMS̄(a) = sqrt((1/k)Σ_{i≤k} a_i²)`，`k = ⌈n·p⌉`；该估计有偏，论文报告元素数过少时梯度不稳定，但机器翻译里取 p=6.25%、CIFAR-10 里取 p=12.5% 均可正常收敛（§5、§6.4）；Table 1 显示 pRMSNorm 与 RMSNorm 的不变性完全相同。
- 实验覆盖 TensorFlow / PyTorch / Theano 三种实现，模型含 GRU-based RNNSearch、Transformer、attentive reader、order-embedding、ConvPool-CNN-C；除特别说明外速度在单张 TITAN X (Pascal) 上测、取 3 次平均（§6）；其中 BatchNorm 走 cuDNN 实现，论文注明其耗时不能与其它实现直接比较（§6.2 脚注）。
- 机器翻译（§6.1 Table 2，RNNSearch，TensorFlow/Nematus；Test14/Test17，时间=秒/1k 训练步）：Baseline 21.7/23.4（399 s）、LayerNorm 22.6/23.6（665 s）、RMSNorm 22.4/**23.7**（501 s，比 LayerNorm 快 24.7%）、pRMSNorm **22.6**/23.1（493 s，快 25.9%）。
- Transformer（§6.1 Table 4，base 设置、300K 步，V100 计时）：不归一化时训练失败、BLEU 记 0；LayerNorm 26.6/27.7（248 s）、RMSNorm **26.8**/27.7（231 s，快 6.9%）、pRMSNorm 26.5/**27.8**（225 s，快 9.3%）。
- CIFAR-10（§6.4）：BatchNorm 最好；LayerNorm 比 Baseline 收敛快却把测试误差弄差 1.53%；RMSNorm 比 Baseline 好 0.013%、比 LayerNorm 省约 20.5% 训练时间；pRMSNorm 再省 2.6% 但测试精度损失 1.54%。
- 论文用 decoder GRU 的 hidden-to-hidden 映射统计量说明"只做 re-scaling 也够稳"（§6.1 Table 5，逐位置 M/S，ALL 列为全位置平均）：Baseline 的 ALL 为 M −1.60 / S 3.04，LayerNorm 为 M −0.51 / S 1.51，RMSNorm 为 M −0.73 / S 1.50——RMSNorm 的均值没有被归一到 0，但标准差与 LayerNorm 相当，且远小于未归一化的 Baseline。
- 整体加速区间（摘要与 §7 写作 **7%~64%**）原文未逐一指明两端来源；按数值对应，下端近于 Transformer（Table 4 的 6.9%~9.3%）、上端近于 order-embedding 的 pRMSNorm（§6.3 Table 8 记 63.9%，RMSNorm 为 40.8%），此对应为本库推断。论文明确实际提升取决于框架、硬件、网络结构与其它组件的相对开销（摘要、§7）。

## 不能推出的结论

- 论文实验全在训练侧，没有任何量化、低比特或推理 kernel 实测；结论里"效率改进与低精度算术、kernel 融合正交"是作者预期，不是实验结论。
- 加速比依赖框架/硬件/架构（论文自述 7%~64%），不能把某个模型上的加速比套到别的模型或后端。
- Table 1 只有 6 列，只覆盖 weight matrix（scaling + centering）、weight vector（仅 scaling）、dataset（scaling + centering）、single training case（仅 scaling）这几种组合，不含 bias 或表列以外的扰动形式；其中两个"re-centering"列衡量的是对位移/噪声的鲁棒，不能把"不做 re-centering"读成"对噪声完全不敏感"。
- "均值为 0 时 RMSNorm 等于 LayerNorm"是等式层面的结论；实际网络中 summed inputs 均值并不为 0，不能据此认为两者在实践中等价。
- pRMSNorm 的 RMS 是有偏估计，论文自述没有一致观察到它的实际加速，不能当作"等价且更快"的替代。
- 在 CIFAR-10 上 BatchNorm 仍最好、LayerNorm/RMSNorm 更差；不能从本篇推出 RMSNorm 全面优于其它归一化。
