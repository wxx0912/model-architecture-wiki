---
{
  "id": "paradigm-dllm",
  "title": "Diffusion Language Model",
  "type": "paradigm",
  "tags": [
    "dllm",
    "diffusion",
    "timestep",
    "calibration",
    "quantization"
  ],
  "related": [
    "component-diffusion-timestep",
    "paradigm-ar",
    "check-calibration-axes"
  ],
  "sources": [
    "paper-llada",
    "repo-llada"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "diffusion language model",
    "diffusion lm",
    "扩散语言模型"
  ],
  "updated": "2026-10-24"
}
---

# Diffusion Language Model（dLLM）

## 一句话理解

dLLM 把生成写成对整段序列的迭代去掩码：从全掩码出发，每一步用双向注意力同时预测所有未知位置，再按置信度重新掩上一部分。它与 AR 逐 token 因果解码并列：一步可以解出多个 token，代价是必须给定采样步数与生成长度上限。

## 生成过程与状态

以 masked diffusion（LLaDA 的路线）为例。训练目标是给被掩码的位置去噪：

```text
L(θ) = −E_{t, x₀, xₜ} [ (1/t) · Σ_i 1[xₜ^i = M] · log p_θ(x₀^i | xₜ) ]
```

前向过程里每个 token 以概率 `t ~ U[0,1]` 独立被替换成掩码 `M`；损失**只对被掩码位置**算交叉熵，并用 `1/t` 加权。论文证明它是模型分布负对数似然的上界。

生成（反向过程）从全掩码的 response 出发（条件生成时 prompt 不掩码），在 `N` 个均匀时间步上从 `t = 1` 走到 `t = 0`：

1. 每一步把当前序列 `xₜ` 整段送进 mask predictor，同时预测所有掩码位置；
2. 只保留置信度最高的若干个预测，其余重新掩码。均匀步长下从 `t` 走到 `s = t − 1/N`，期望重掩 `s/t` 的预测、保留 `1 − s/t`；LLaDA 默认把置信度最低的位置重掩（低置信度 remasking，置信度取 argmax 预测的概率 `c^i = p_θ(x₀^i | x_t)`），随机重掩是它的对照方案，论文中低置信度明显更好；
3. 走满 `N` 步（到 `t = 0`）即停止；换用自回归或 block 采样时改由 `EOS` 结束。每步解出的 token 数由步数与生成长度共同决定：论文的吞吐实验固定输出 256 token，32 / 64 / 128 / 256 步对应每次前向解出 8 / 4 / 2 / 1 个 token。

任一步的状态由当前序列（未定位置为掩码 `M`）和掩码率 `t` 决定；训练与 SFT 共用同一套目标（SFT 只掩码 response）。同一份权重不重训即可在 autoregressive、block diffusion（含固定块长的 block diffusion LLaDA）与纯扩散之间切换——这是**采样期**的选择，不是不同的训练目标。

## 与 AR 的关键差异

| 维度 | AR | dLLM（masked diffusion） |
| --- | --- | --- |
| 解码顺序 | 左到右，逐 token | 整段迭代去掩码，位置顺序由置信度决定 |
| 注意力 | 因果 mask | 双向 / 非因果，无 causal mask |
| 单步产出 | 1 个 token | 一批 token（步数少于生成长度时；两者相等则每步 1 个） |
| 运行时状态 | KV cache 随长度增长 | 无 KV cache（原文称 incompatible with KV caching；"不做增量缓存"为本库推断） |
| 训练损失 | next-token 交叉熵 | 只在掩码位置、按 `1/t` 加权的交叉熵 |
| 停止条件 | 生成 `EOS` 或到长度上限 | 纯扩散走满 `N` 步；自回归 / block 采样按 `EOS` 结束 |

## 实现差异

不同 dLLM 实现、甚至同一实现的不同 sampler，在掩码比例与时间表、单 token 还是 block 更新、置信度排序与 remasking 规则、训练时的 attention mask、推理 sampler 参数上都可以完全不同。以 LLaDA 官方仓库为例：它只放推理与评测脚本（其中的默认 sampler 参数只是示例值，不是论文评测配置），模型定义在 HuggingFace Hub 的权重仓库里、加载时需要 `trust_remote_code=True`。dLLM 的一步不能等同于 AR 的一次 decode。

## 量化关联

**结构事实**

- 每一步都是整段（或整块）的非因果前向，没有 KV cache；以 LLaDA 为代表的主干使用 RMSNorm / SwiGLU / RoPE。
- 范围事实：LLaDA 论文与官方仓库都不含量化实现，论文只在附录给出吞吐 / 显存对照（吞吐实验为 bf16）。

**量化推断**

- 不同 mask ratio、step 和 token 位置会产生不同的激活分布，校准集必须覆盖这些过程状态和难例，不能只用一段完整自然文本做一次 forward。
- 量化误差会改变 token 的置信度排序，进而改变后续 mask 轨迹，误差可能沿离散选择路径被放大。
- 单步 logits、位置选择与排序、完整生成轨迹和最终任务质量需要分开测量；AR 的 next-token loss 与 prefill/decode 校准覆盖不了这条路径。
- 没有 KV cache 只说明运行时状态的低比特压力小，不代表激活更好量化。

## 结果边界

- 本页机制描述来自 LLaDA 的 masked diffusion；block diffusion、其它 remasking 方案和后续变体必须以其自身的 sampler 与源码为准，不能推广成所有 dLLM 的规则。
- 论文和官方仓库没有量化实现，本页所有量化内容都是结构推断，不构成精度或速度证据。
- 吞吐 1.5× / 1.8× 绑定特定步数配置与 A100 / batch-1 / bf16 条件，不能当成固定倍率，也不能外推到任意长度的端到端延迟。
- “32 / 64 / 128 / 256 步 = 每步 8 / 4 / 2 / 1 token”只在论文固定输出 256 token 的吞吐实验下成立，换生成长度必须重新换算。

## Agent 如何使用

首轮读本页即可判断一段生成是否属于 masked diffusion 的“整段迭代去掩码”，以及它在状态、注意力与缓存上和 AR 的差别；具体采样配置要到模型页和实现里取。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 训练目标、采样规则、实验配置 | [`paper-llada`](../../sources/papers/paper-llada.md) |
| 官方推理脚本、默认 sampler 参数、加载方式 | [`repo-llada`](../../sources/repos/repo-llada.md) |
| 某个具体模型的规格（层数 / 头数 / 缓存） | 对应模型页（如 [LLaDA](../models/llada.md)）与该 checkpoint 的 `config.json` |
| 其它 dLLM 变体的 sampler | 该变体自己的论文与源码 |
| 目标 checkpoint 的实际 mask / 采样配置与后端 kernel | 该 checkpoint 的 config / 源码 / 运行时（本库不提供） |

回答 dLLM 相关问题时，必须说明目标实现及其采样模式（纯扩散、分块迭代、块间自回归等哪一种）、步数、生成长度上限和 remasking 规则——这些因实现而异，缺了就无法谈量化影响。

## 来源与证据

- [LLaDA 论文](../../sources/papers/paper-llada.md)：前向 / 反向过程、`1/t` 加权训练损失、`s/t` remasking 与低置信度规则、三种采样策略及其边界。
- [LLaDA 官方仓库](../../sources/repos/repo-llada.md)：官方推理 / 评测脚本与默认参数；模型定义在 HuggingFace Hub，不在该仓库。

本页以 masked diffusion（LLaDA）为 `source-reported` 依据；其它 dLLM 变体必须以其自身 sampler 与源码为准，不能引到 LLaDA。
