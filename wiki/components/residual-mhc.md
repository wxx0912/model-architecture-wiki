---
id: component-residual-mhc
title: Residual, Hyper-Connections and mHC
type: component
tags: [residual, mhc, quantization]
related: [paradigm-ar, paradigm-ssm]
sources: [paper-mhc, repo-tile-kernels]
confidence: source-reported
evidence_scope: mixed
validated_at: 2026-10-22
---

# Residual、Hyper-Connections 与 mHC

## 结构概述

普通 residual 将 block 输出加回主流。Hyper-Connections 将 residual stream 扩展为多个并行 stream，并通过 pre/post mixing 与跨层 residual mixing 组合它们。mHC 对 residual mixing 施加流形约束：论文描述的实践使用 Sinkhorn-Knopp 近似投影，使混合矩阵接近双随机结构，同时约束 pre/post mixing 的非负性。

## 数据流与形状

若有 n 个 stream，层输入可以抽象表示为 [B, T, n, d] 或等价重排形式；每层先聚合到 block 输入，再将输出散回 streams，并执行 residual mixing。该 shape 是解释性抽象，具体维度和布局必须以论文公式、目标模型源码或 TileKernels 调用约定确认。

## 量化关联

**结构事实：**

- mixing 矩阵、stream 聚合/散射、Sinkhorn/归一化和 block 输出构成额外数值路径。
- 跨层复合 mixing 会使误差传播不同于单 stream residual。

**量化推断：**

- 应单独检查 mixing 权重、归一化统计、非负性以及行列和约束；这不是普通 Linear 的完整替代。
- 低比特误差是否跨层放大需要运行验证，不能由“约束稳定”直接推导。
- 融合 kernel 的 dtype、导出和后端支持必须单独核对。

## Agent 检查

1. 确认 stream 数、mixing shape、归一化和初始化。
2. 比较 block 前后、聚合/散射和跨层 residual 的误差。
3. 检查实际实现是否使用 Sinkhorn、近似投影或融合 kernel；DeepSeek TileKernels 的 `tile_kernels/mhc/` 只能证明官方 kernel 实现，不等于某个 checkpoint 默认启用 mHC。
