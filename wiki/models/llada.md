---
{
  "id": "model-llada",
  "title": "LLaDA",
  "type": "model",
  "tags": [
    "dllm",
    "diffusion",
    "attention",
    "calibration",
    "quantization"
  ],
  "related": [
    "paradigm-dllm",
    "paradigm-ar",
    "component-diffusion-timestep",
    "component-attention",
    "component-embedding-lm-head"
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
  "updated": "2026-10-23"
}
---

# LLaDA

## 结构概述

LLaDA 是基于 Transformer 的 masked diffusion language model。它使用双向可见的 masked-token 预测目标，并通过迭代填充/重掩码生成文本；这使它在生成调度和误差传播上不同于仅做因果 next-token decode 的 AR 模型。

## 结构与数据流

文本 token 经 embedding 进入 Transformer 主干。训练按噪声/掩码过程随机遮蔽 token，模型预测被遮蔽位置的 token 分布。推理时 sampler 根据当前 mask 状态与模型输出决定每步填充哪些位置，并可能重新掩码低置信度位置；实际策略由官方代码的 sampler 参数决定。

## 实现差异

核对 checkpoint 的模型配置、attention mask、position embedding、训练 mask 分布、推理更新顺序、置信度/重掩码规则、采样步数和输出停止条件。不同 LLaDA checkpoint、推理代码版本和采样器不能混为一谈。

## 量化关联

- 校准需覆盖 mask ratio、迭代阶段、已知/未知 token 位置及不同长度。
- logits 的小变化可能改变位置置信度排序，造成后续 token 更新轨迹分叉。
- 除最终文本质量外，记录每步 logits、位置选择、mask 轨迹和停止行为，定位误差何时放大。
- AR 的 KV-cache-only 校准或 next-token 指标不足以代表 LLaDA 的生成质量。

## Agent 检查

1. 固定官方推理代码版本和 sampler 配置，逐步记录 mask 轨迹。
2. 按 mask ratio/step/位置分层收集校准样本和激活统计。
3. 比较 BF16 与量化模型的 logits、token 排序、逐步状态和最终输出。
4. 对导出权重和目标后端重复同一轨迹验证。

## 来源与证据

LLaDA 论文和官方仓库支持本页的训练与生成概述；实际采样行为以运行时版本和参数为准。
