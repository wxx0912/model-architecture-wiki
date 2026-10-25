---
{
  "id": "paradigm-dit",
  "title": "Diffusion Transformer (DiT)",
  "type": "paradigm",
  "tags": [
    "dit",
    "diffusion",
    "timestep",
    "attention",
    "mlp",
    "quantization"
  ],
  "related": [
    "component-attention",
    "component-mlp",
    "component-diffusion-timestep",
    "component-normalization",
    "check-calibration-axes"
  ],
  "sources": [
    "paper-dit",
    "repo-dit"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "dit",
    "diffusion transformer",
    "diffusion transformers"
  ],
  "updated": "2026-10-24"
}
---

# Diffusion Transformer（DiT）

## 一句话理解

DiT 用 Transformer 当扩散模型的去噪主干：把 VAE latent patch 化成 token，在给定 timestep 和条件下预测噪声（论文的实现里输出头同时给出噪声与方差），去噪循环反复调用同一个主干。它替换的只是 U-Net 主干，VAE、scheduler、采样器和评测套件仍是独立部件；主干不能单独决定端到端质量（工程判断）。

## 去噪过程与条件注入

1. **训练**：对 latent `x_0` 采样时间步、加噪得到 `x_t`，让主干预测噪声与方差；
2. **采样**：从纯噪声出发，每一步把当前 `x_t` 和 timestep 送进主干，按 scheduler 更新到前一个噪声水平；
3. 迭代到噪声水平归零（论文主结果用 250 步 DDPM），再由 VAE 解码器还原成像素。

每一步调用的是同一个主干，改变的只有输入 `x_t` 与 timestep。

主干的数据流（以 256×256 图像、下采样 8 为例）：latent `z` 为 `32×32×4` → patchify 线性嵌入成 `T=(I/p)²` 个 `d` 维 token → 加固定的正弦/余弦位置编码 → `N` 个 DiT block → final layer（adaLN 变体里是自适应 LayerNorm）→ 线性解码成 `p×p×2C` 张量 → 重排回空间布局。

条件注入是 DiT 的核心设计：`c = t_emb + y_emb`（timestep embedding 与类别 embedding 之和）送进每个 block 的 adaLN 调制。**timestep 侧**用 256 维频率 embedding + 两层 MLP（SiLU 激活），输出维度等于 hidden size；**每个 adaLN 层**的输出维度是 hidden size 的 4×（adaLN）或 6×（adaLN-Zero），后者对应 attention 与 MLP 各一组的 scale/shift/gate（共 6 个）。分类器无关引导（CFG）在训练时随机把 `c` 换成学到的 null embedding，推理时按

```text
ε̂ = ε(x_t, ∅) + s · ( ε(x_t, c) − ε(x_t, ∅) )
```

组合；论文只在 latent 的前 3 个通道上做引导。

## 条件注入方式对照

四种 block 变体在 XL/2、256×256 下的前向 Gflops（论文 §5 / Figure 5 的数字；分辨率取自附录 Table 4）：

| 方式 | 做法 | Gflops |
| --- | --- | --- |
| in-context | 把 `t`、`c` 的 embedding 当两个额外 token 拼进序列，block 不改动 | 119.4 |
| cross-attention | `t`/`c` 拼成长度 2 的独立序列，self-attention 后再加一层 cross-attention | 137.6 |
| adaLN | 用 `t`、`c` 的 embedding 之和回归每个 block 的 `γ`、`β`，取代直接学习的 scale/shift | 118.6 |
| adaLN-Zero | 在 adaLN 之上再回归逐维 `α`，作用在每个残差连接之前，并把 `α` 的 MLP 初始化为恒输出 0 | 118.6 |

adaLN 是其中唯一"对所有 token 施加同一函数"的机制；adaLN-Zero 训练全程优于 cross-attention 与 in-context（论文归因于 conditioning 机制本身，且它最省算力），并显著优于 vanilla adaLN（论文归因于把 block 初始化为恒等映射）。

## 实现差异

不同 DiT 实现之间可能不同的地方：latent 分辨率与下采样倍数、patch size、空间/时间 token 排列、位置编码（论文用固定正弦/余弦，官方实现为 2D sin-cos；未探索可学习、RoPE 或相对位置）、条件注入方式（adaLN / cross-attention / in-context / 直接拼接条件 token）；预测目标、scheduler 与采样步数属通用工程经验、不在本页来源内。这些不能只凭 "DiT" 这个名字推断。官方实现的结构基础件来自 timm（`PatchEmbed`、`Attention`、`Mlp`），adaLN-Zero 靠把 `adaLN_modulation` 与 `final_layer` 的最后一层零初始化实现，`learn_sigma=True` 时输出通道翻倍、同时预测噪声与方差。

## 量化关联

**结构事实**

- 推理是同一主干被反复调用 `T` 步，每一步的输入 `x_t` 与 timestep 都不同；主干包含 attention、MLP、adaLN 调制（scale/shift/gate）与线性输出头。
- 范围事实：论文与官方仓库都不含量化实现。

**量化推断**

- timestep / 噪声水平会改变激活范围，校准必须覆盖采样轨迹上的多个时间点，固定一个 `t` 代表不了整条轨迹。
- adaLN 的调制向量与 gate 会作用到整层，误差随去噪循环反复放大；attention、MLP、输出头与 VAE/连接器要分开评估。
- 视频或长序列实现还要按空间/时间 token 与显存路径验证，单帧结果不能代表视频质量。
- 最终 FID / 重建质量与逐层误差不总是单调对应，要同时保留轨迹级与端到端指标。

## 结果边界

- 本页结构描述来自 class-conditional ImageNet 的 DiT（论文 + 官方实现）；文本条件 DiT/MMDiT 与视频 DiT 的条件注入方式和质量结论不能由此推出——论文把文本条件列为未来工作，并未做实验。
- 论文的 Gflops 是给定分辨率与 patch size 下的前向复杂度、不含 VAE，不是模型固有属性；FID 不能跨评测实现或跨 VAE 解码器比较（论文 scaling 分析用 ft-MSE、最终指标用 ft-EMA）。
- 论文与官方仓库不涉及量化或低比特，本页的量化内容都是结构推断，不构成精度证据。

## Agent 如何使用

首轮读本页即可判断目标模型是否为 DiT 类主干、条件从哪条路径注入、去噪循环如何调用主干。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 主干、adaLN-Zero、条件注入对比、timestep embedding 规格 | [`paper-dit`](../../sources/papers/paper-dit.md) |
| 官方实现里 block 结构、零初始化与配置表 | [`repo-dit`](../../sources/repos/repo-dit.md) |
| timestep / mask 状态的量化语义 | [Diffusion Time/Noise 与离散 Mask 状态](../components/diffusion-timestep.md) |
| 目标模型的 latent 分辨率、patch size、scheduler | 目标仓库的源码与 config |

回答 DiT 量化问题时必须说明：条件注入方式、latent 分辨率与 patch size、采样步数与 scheduler、端到端是否含 VAE/连接器，以及目标后端。

## 来源与证据

- [Scalable Diffusion Models with Transformers](../../sources/papers/paper-dit.md)：patchify 主干、四种条件注入 block 的对比、timestep embedding 规格、scaling 与 FID 口径。
- [facebookresearch/DiT](../../sources/repos/repo-dit.md)：固定 commit 下 `models.py` 的类结构、零初始化写法与 `DiT_models` 配置表。

本页主干与条件注入的描述以这两份来源为限；论文不含量化实验，所有量化结论均为结构推断。
