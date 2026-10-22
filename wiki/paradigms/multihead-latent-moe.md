---
id: paradigm-multihead-latent-moe
title: Multi-Head LatentMoE
type: paradigm
tags: [latent-moe, moe, routing, quantization]
related: [paradigm-latent-moe, paradigm-latent-moe-nvidia, paradigm-moe]
sources: [paper-multihead-latentmoe]
confidence: source-reported
evidence_scope: mixed
validated_at: 2026-10-22
---

# Multi-Head LatentMoE

## 结构概述

Multi-Head LatentMoE 是另一种以 latent 表示和 head parallel 为核心的研究方案，不能与 NVIDIA LatentMoE 或 Kimi 的 Stable LatentMoE 视为同义词。其通信和 head 组织方式是结构分析的重点。

## 量化关联

Agent 需要先确认 latent head 数、投影/恢复路径、专家分配和通信布局，再确定量化边界。head 间共享 scale、跨设备通信、低维专家权重和恢复投影都需要独立验证；论文描述不等于某个推理后端已经支持。

## Agent 检查

1. 绑定论文版本、代码实现和目标模型；未绑定时保持 `unknown`。
2. 记录 head/latent/expert shape、通信阶段、packing 和并行切分。
3. 对每个 head、专家和恢复路径做局部数值与端到端验证。

## 来源与证据

本页仅使用 Multi-Head LatentMoE 论文作为结构来源，不复用其他 LatentMoE 变体的参数或量化规则。

