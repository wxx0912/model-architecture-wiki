---
id: component-normalization
title: RMSNorm and Normalization Boundaries
type: component
tags: [normalization, rmsnorm, quantization]
related: [paradigm-ar, paradigm-dit, paradigm-ssm, component-residual-mhc]
sources: [paper-rmsnorm, repo-huggingface-transformers]
confidence: source-reported
evidence_scope: mixed
validated_at: 2026-10-22
---

# RMSNorm 与归一化边界

## 结构概述

RMSNorm 用输入的均方根进行缩放，再乘以可学习权重，省略 LayerNorm 的均值中心化。Pre-norm、post-norm、norm 前后的残差路径以及 fused norm kernel 会改变数值边界。

## 数据流与形状

对最后一维为 `d_model` 的 hidden state 做 `x / sqrt(mean(x^2)+eps) * weight`。常见 Transformer block 在 attention 和 FFN 前各放一次 norm；DiT 还可能把 timestep 条件转成 scale/shift/gate 注入 norm。

## 实现差异

核对 norm 类型、eps、参数 dtype、是否 affine、norm 位置、残差与 norm 的融合方式，以及是否使用 FP32 累加。不同实现的 `RMSNorm` 名称不保证 eps 和 accumulator 相同。

## 量化关联

- norm 会重标定输入，量化前后的异常值与 scale 不能脱离前后残差路径解释。
- eps、累加精度和 fused kernel 会在低比特或长序列下影响稳定性。
- 对 adaLN 类调制，scale/shift/gate 与 norm 输出要分开统计，不能只量化 norm weight。

## Agent 检查

1. 读取 norm 类型、eps、参数 dtype、fused kernel 和执行精度。
2. 记录每个 norm 的输入/输出范围及其前后 residual 分支。
3. 对 block 局部输出和最终 logits/样本做对比，验证 norm 低比特是否可部署。

