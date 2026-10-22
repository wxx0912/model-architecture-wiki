---
id: paradigm-dllm
title: Diffusion Language Model
type: paradigm
tags: [dllm, diffusion, timestep, calibration, quantization]
related: [component-diffusion-timestep, paradigm-ar, check-calibration-axes]
sources: [paper-llada, repo-llada]
confidence: source-reported
evidence_scope: mixed
validated_at: 2026-10-22
---

# Diffusion Language Model（dLLM）

## 结构概述

dLLM 通过对 token 进行掩码、逐步预测和重新掩码来生成序列，而不是只沿左到右路径解码。Masked Diffusion、block diffusion 和 remasking 方案在训练目标、更新顺序、并行度和停止条件上存在差异。

## 结构与数据流

输入包含已知 token、mask token 和当前 mask/step 状态。模型通常仍使用 Transformer 主干，但 attention mask、位置处理和输出选择由扩散过程控制。每一步产生 token logits 或置信度，采样器选择要保留、填充或重新掩码的位置，重复若干步直到达到终止条件。

## 实现差异

核对 mask ratio/schedule、单 token 还是 block 更新、置信度排序、remasking、KV cache 是否可复用、训练时 attention mask 和推理时 sampler。不能把 dLLM 的每一步直接等同于 AR decode。

## 量化关联

- 不同 mask ratio、step 和 token 位置会产生不同激活分布；校准集应覆盖过程状态和难例。
- 量化误差可能改变 token 置信度排序，进而改变后续 mask 轨迹，形成离散的误差放大。
- 需要分别测量单步 logits、位置选择/排序、完整生成轨迹和最终任务质量。
- 普通 AR 的只看 next-token loss、prefill/decode 校准不足以覆盖 dLLM 的行为。

## Agent 检查

1. 读取训练配置和 sampler，记录 mask/step 状态的完整定义。
2. 确认 attention mask、位置编码、cache/state 复用和输出 head。
3. 按 mask ratio、step、序列长度和更新块大小分层采集校准数据。
4. 对 BF16、伪量化、导出回载和真实后端逐样本比较轨迹及结果。

## 来源与证据

LLaDA 论文和官方实现支持 masked diffusion 的基本流程；其他 dLLM 变体必须以其 sampler 和源码为准。

