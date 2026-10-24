---
{
  "id": "component-attention-residuals",
  "title": "Attention Residuals (AttnRes)",
  "type": "component",
  "tags": [
    "attnres",
    "residual",
    "attention",
    "quantization"
  ],
  "related": [
    "component-residual-mhc",
    "paradigm-ar",
    "check-quantization-preflight"
  ],
  "sources": [
    "paper-attnres",
    "repo-attention-residuals"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "attention residuals",
    "attention residual",
    "注意力残差"
  ],
  "updated": "2026-10-23"
}
---

# Attention Residuals（AttnRes）

## 结构概述

Attention Residuals 用沿网络深度方向的、输入依赖的 softmax attention 替代固定的 residual 累加，使当前层能够选择性聚合历史层表示。Full AttnRes 关注所有历史输出；Block AttnRes 将层分成若干 block，只在 block 表示上做聚合以降低开销。

## 数据流与实现差异

每层除了当前 block 输出，还要访问历史 hidden states、生成聚合权重并做跨层加权。Block AttnRes 的 block 数、缓存方式、两阶段计算和通信实现必须以具体版本源码/配置确认；官方仓库目前主要提供论文和说明资料，并非完整模型实现。

## 量化关联

- 历史表示缓存、softmax 路由权重、跨层聚合和 block 边界是新的误差传播路径。
- 量化不能只看当前层 attention/FFN；应观察历史 state、聚合 logits/weights、聚合输出和跨层误差累积。
- cache 的精度、内存布局和通信 kernel 是部署约束；论文结构结论不等于已有量化 kernel 支持。

## Agent 检查

1. 确认目标模型是否真的实现 AttnRes，以及 Full/Block 变体和 block 数。
2. 记录历史表示 cache、聚合权重、block 边界和跨设备通信路径。
3. 分别做单层、跨层聚合、长深度和真实后端验证。

## 来源与证据

Kimi Team 的论文和官方资料支持 AttnRes 的机制；官方仓库未提供完整模型运行实现，因此目标 checkpoint 仍需源码确认。
