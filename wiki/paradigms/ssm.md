---
id: paradigm-ssm
title: Selective State Space Model
type: paradigm
tags: [ssm, mamba, state-update, quantization]
related: [paradigm-linear-attention, component-residual-mhc, check-calibration-axes]
sources: [paper-mamba, paper-mamba2, repo-mamba]
confidence: source-reported
evidence_scope: mixed
validated_at: 2026-10-22
---

# Selective State Space Model（SSM）

## 结构概述

SSM 用状态空间递推建模序列。Mamba 通过输入相关的选择性参数和 selective scan 让状态保留或遗忘内容；Mamba-2 将状态空间计算与结构化状态空间对偶（SSD）联系起来，并提供分块实现。

## 结构与数据流

连续时间参数经离散化后形成 `state_t = A_t state_{t-1} + B_t x_t`、`y_t = C_t state_t + D x_t` 一类递推。实际 Mamba block 还包含输入/分支投影、门控、卷积或局部混合、scan 和输出投影。推理时维护每层的卷积状态与 SSM 状态。

## 实现差异

确认 state size、A 参数化、delta/步长、B/C 生成方式、卷积宽度、scan kernel、chunk size、状态初始化和推理 cache。Mamba 与 Mamba-2 的状态布局和 kernel 契约不能混用。

## 量化关联

- selective scan 中的 delta、A、B、C 和 state 共同决定递推稳定性；误差可能随序列长度积累。
- state/cache 的量化是运行时状态问题，不能仅靠权重量化结果推断。
- gate、局部卷积和输出投影可能有不同异常值与校准轴，应分开统计。
- 需要观察长序列、状态 reset、chunk 边界和真实 scan kernel 的差异。

## Agent 检查

1. 从源码确认 Mamba 版本、状态字段、scan 路径和 kernel。
2. 记录每个状态的 shape、dtype、初始化和生命周期。
3. 用逐步 reference 实现对照 fused scan 的输出和梯度/数值误差。
4. 以短序列和长序列分别评估状态量化、权重量化及端到端质量。

## 来源与证据

Mamba、Mamba-2 论文及官方实现支持选择性状态空间和 SSD 的结构描述；不同推理后端的融合实现需单独验证。

