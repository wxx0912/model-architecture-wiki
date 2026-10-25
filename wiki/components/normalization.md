---
{
  "id": "component-normalization",
  "title": "RMSNorm and Normalization Boundaries",
  "type": "component",
  "tags": [
    "normalization",
    "rmsnorm",
    "quantization"
  ],
  "related": [
    "paradigm-ar",
    "paradigm-dit",
    "paradigm-ssm",
    "component-diffusion-timestep",
    "component-residual-mhc"
  ],
  "sources": [
    "paper-rmsnorm",
    "paper-transformer",
    "paper-dit",
    "repo-dit",
    "repo-huggingface-transformers"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "root mean square normalization",
    "均方根归一化"
  ],
  "updated": "2026-10-24"
}
---

# RMSNorm 与归一化边界

## 一句话理解

归一化在每层把 hidden 重标定到稳定尺度。RMSNorm 只做 **re-scaling**（除以均方根）不做 re-centering（不减均值），再乘可学习 gain；它比 LayerNorm 少一步求均值，也少一份平移不变性。

## 结构与数据流

```text
LayerNorm: ā_i = (a_i − μ)/σ · g_i,  μ = (1/n)Σa_i,  σ = sqrt((1/n)Σ(a_i − μ)²)
RMSNorm  : ā_i = a_i / RMS(a) · g_i,  RMS(a) = sqrt((1/n)Σa_i²)
```

- 原始 Transformer（§3.1）每个子层输出 `LayerNorm(x + Sublayer(x))`（Post-LN），`d_model=512`。
- RMSNorm（§4.1，Eq.5）的通用形式**仍保留加性 bias**：`y = f((Wx)/RMS(a) ⊙ g + b)`；gain `g` 初始化为 1、bias `b` 通常初始化为 0。后续实现常省略 bias，属工程惯例，非本篇内容（本库引申）。
- 由 `RMS(αx) = α·RMS(x)`（Eq.6）得 RMSNorm 对权重 / 输入缩放不变；因为不做 re-centering，它没有平移不变性；当 summed inputs 均值为 0 时 RMSNorm 与 LayerNorm 完全相等。

DiT 的 adaLN（paper-dit §3.2；实现细节见 repo-dit）：block 内的 `norm1/norm2` 用 `LayerNorm(elementwise_affine=False, eps=1e-6)`，条件（timestep + class 之和）经 SiLU + 线性层生成 attention 与 MLP 各一组的 shift/scale/gate 调制量（共 6 个）；adaLN-Zero 再回归逐维 `α`，并把调制 MLP 的**末层线性**（`adaLN_modulation[-1]`）与 `final_layer` 对应线性层零初始化。

## 变体对照

| 变体 | 是否减均值 | 要点 | 来源 |
| --- | --- | --- | --- |
| LayerNorm | 是 | `(a−μ)/σ·g`，Post-LN 接在子层输出后 | Transformer §3.1 |
| RMSNorm | 否 | `a/RMS(a)·g`，通用形式仍可带 bias | paper-rmsnorm §4 |
| pRMSNorm | 否 | 只用前 `p%` 元素估 RMS（有偏估计） | paper-rmsnorm §5 |
| adaLN（DiT） | 是（基于 LayerNorm） | 条件生成 shift/scale/gate，可零初始化 | paper-dit §3.2 |

## 实现差异

不同实现之间可能不同的地方（工程归纳，非来源事实）：norm 类型与 `eps`、是否 affine、参数 dtype、norm 的**位置**（Pre-LN 还是 Post-LN，原始 Transformer 用 Post-LN）、norm 与残差 / kernel 的融合方式，以及是否用 FP32 累加。不同实现的 `RMSNorm` 名称不保证 `eps` 和累加器相同，必须以目标实现为准。

## 量化关联

**结构事实**

- RMSNorm 只做 re-scaling，通用形式仍保留 additive bias（Eq.5），gain 初始化 1、bias 初始化 0。
- 原始 Transformer 用 Post-LN：`LayerNorm(x + Sublayer(x))`。
- DiT 的 adaLN 把条件变成 shift/scale/gate 调制量，adaLN-Zero 额外回归 `α`，并把调制 MLP 末层与 `final_layer` 对应线性层零初始化（`paper-dit` §3.2 与 `repo-dit` 实现）。
- 当 summed inputs 均值为 0 时 RMSNorm 与 LayerNorm 相等；实际网络中均值并不为 0。

**量化推断**

- norm 会重标定输入，量化前后的异常值与 scale 不能脱离前后残差路径解释。
- `eps`、累加精度与 fused kernel 会在低比特或长序列下影响稳定性。
- 对 adaLN 类调制，scale/shift/gate 与 norm 输出要分开统计，不能只量化 norm weight。

## 结果边界

- paper-rmsnorm 的实验全在训练侧，没有任何量化、低比特或推理 kernel 实测；结论里"效率改进与低精度算术、kernel 融合正交"是作者**预期**，不是实验结论。
- 论文自述加速比 7%~64% 取决于框架、硬件、网络结构，不能把某个模型上的加速比套到别的模型或后端。
- "均值为 0 时两者相等"是等式层面的结论，不能据此认为两者在实践中等价；CIFAR-10 上 BatchNorm 仍最好。
- 本库未运行任何量化或评测；本页量化内容均为结构推断。

## Agent 如何使用

首轮读本页即可判断目标是 LayerNorm 还是 RMSNorm、是否 affine、norm 的位置与是否有条件调制。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| RMSNorm 定义、不变性与 pRMSNorm | [`paper-rmsnorm`](../../sources/papers/paper-rmsnorm.md) |
| 原始 Transformer 的 Post-LN 与子层结构 | [`paper-transformer`](../../sources/papers/paper-transformer.md) |
| DiT 的 adaLN / adaLN-Zero 调制 | [`paper-dit`](../../sources/papers/paper-dit.md) |
| DiT 的 `norm1/norm2` 实现细节 | [`repo-dit`](../../sources/repos/repo-dit.md) |
| 具体模型类的 norm 实现与量化接口 | [`repo-huggingface-transformers`](../../sources/repos/repo-huggingface-transformers.md) |
| 目标模型的 norm 实现与融合 kernel | 该 checkpoint 的 config / 源码 / 运行时（本库不提供） |

回答量化问题时必须说明：norm 类型与 `eps`、是否 affine、norm 位置与前后残差路径、是否条件调制，以及执行精度与目标后端。

## 来源与证据

- [RMSNorm](../../sources/papers/paper-rmsnorm.md)：定义（Eq.4–5）、不变性（Table 1）、pRMSNorm 与训练实验。
- [Attention Is All You Need](../../sources/papers/paper-transformer.md)：`LayerNorm(x+Sublayer(x))` 的 Post-LN 结构（§3.1）。
- [DiT](../../sources/papers/paper-dit.md)：adaLN / adaLN-Zero 的调制与零初始化（§3.2 定义；§5 为相应实验对照）。
- [facebookresearch/DiT](../../sources/repos/repo-dit.md)：固定 commit 下 `DiTBlock` 的 `LayerNorm` 与 `adaLN_modulation` 零初始化。
- [huggingface/transformers](../../sources/repos/repo-huggingface-transformers.md)：固定 commit 下模型类与量化 / cache 接口入口。

本页结构以上述来源为限；来源不含量化实验，量化内容均为结构推断，本库未做数值或运行验证。
