---
id: model-kimi-linear
title: Kimi Linear
type: model
tags: [linear-attention, state-update, gdn, mla, quantization]
related: [paradigm-linear-attention, paradigm-ar, paradigm-ssm, component-kv-cache]
sources: [paper-kimi-linear, paper-gdn, repo-kimi-linear]
confidence: source-reported
evidence_scope: mixed
validated_at: 2026-10-22
---

# Kimi Linear

## 结构概述

Kimi Linear 是混合注意力语言模型，将 Kimi Delta Attention（KDA）等线性状态更新层与 MLA attention 层组合。其目标是在长上下文推理中降低 cache/计算开销，同时保留部分传统 attention 层的表达能力。此页面不把所有 Kimi 系列 checkpoint 都视为相同结构。

## 结构与数据流

KDA 将输入投影成 query/key/value 和门控项，以 delta-rule 风格递推更新有限状态；MLA 层维护其 latent KV 表示。block 排列、短卷积和门控细节依版本配置变化。推理时模型同时维护线性状态与 MLA cache 两类运行时状态。

## 实现差异

读取模型配置和官方源码确认层型排列、KDA head/state 维度、gate 形式、chunk size、MLA latent/cache shape、短卷积状态和状态重置。不要把 linear state 当成普通 KV cache，也不要假设不同后端使用相同的融合 scan kernel。

## 量化关联

- KDA 状态误差随序列递推，校准与验证必须覆盖长序列、chunk 边界及连续 decode。
- gate、delta update、state readout 和 MLA cache 是不同数值路径，需要分别量化和诊断。
- 混合层结构天然支持按层型分组评估，但任何层型专用 bit 配置都要通过目标 kernel/导出契约验证。
- 记录 state drift、长上下文质量、cache/state 内存和真实吞吐，不以短 prompt 的 logits 相似替代。

## Agent 检查

1. 由配置和源码生成逐层结构表，注明 KDA、MLA、卷积和 state/cache 字段。
2. 用真实长上下文 workload 采集每层状态与激活范围。
3. 单独测试 state 更新、reset、chunk 扫描和 MLA cache 的量化误差。
4. 确认导出格式和目标后端分别支持两种状态路径。

## 来源与证据

Kimi Linear 论文与官方仓库说明其混合线性注意力设计；目标 checkpoint 和后端仍需独立核对。

