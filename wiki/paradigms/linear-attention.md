---
{
  "id": "paradigm-linear-attention",
  "title": "Linear Attention and Gated State Updates",
  "type": "paradigm",
  "tags": [
    "linear-attention",
    "state-update",
    "gdn",
    "calibration",
    "quantization"
  ],
  "related": [
    "paradigm-ssm",
    "component-attention",
    "component-kv-cache"
  ],
  "sources": [
    "paper-gdn",
    "paper-kimi-linear",
    "repo-kimi-linear"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "线性注意力",
    "linear attention"
  ],
  "updated": "2026-10-23"
}
---

# Linear Attention 与门控状态更新

## 结构概述

线性注意力通过可结合的特征映射或递推状态，避免显式构造完整的 `T×T` attention 矩阵。GLA、Gated DeltaNet 和 KDA 等变体使用门控、delta update 或 key-dependent 规则维护有限状态；它们与普通 KV cache 的语义不同。

## 结构与数据流

抽象形式是 `state_t = Update(state_{t-1}, k_t, v_t, gate_t)`，再由 query 和 state 产生输出。状态可能是矩阵、低秩张量或分块结构；训练可并行扫描，推理通常逐 token 更新。Kimi Linear 等混合架构还会在不同层交替使用 linear attention 与 MLA/attention。

## 实现差异

核对 state shape、chunk/block scan、gate 参数化、数值归一化、初始状态、训练/推理 kernel 和层间混合模式。不能用 KV cache 的 page、scale 或更新接口直接替代 state 接口。

## 量化关联

- state 是跨 token 的递推量，更新误差可能长期积累；短序列局部误差不足以代表长上下文。
- gate 的非线性和 delta 更新对异常值、scale 漂移和舍入误差敏感。
- 需要区分投影权重、gate、state、state→output 读出和 chunk 边界误差。
- 混合 attention/linear 层应分别选择校准和量化策略，统一配置可能掩盖瓶颈。

## Agent 检查

1. 识别每层是 attention、GLA、Gated DeltaNet、KDA 还是其他实现。
2. 记录 state dtype/layout、更新频率、chunk size、初始状态和 reset 边界。
3. 做短/长上下文、连续 decode、chunk 边界和状态重置测试。
4. 对真实后端的 scan/update kernel 做数值与吞吐验证。

## 来源与证据

Gated DeltaNet、Kimi Linear 论文支持门控状态更新及混合层的结构描述；Kimi Linear 官方仓库不含代码、不是实现依据，具体 state layout 必须核对实现。
