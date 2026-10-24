---
{
  "id": "model-mamba2",
  "title": "Mamba-2",
  "type": "model",
  "tags": [
    "ssm",
    "mamba",
    "state-update",
    "quantization"
  ],
  "related": [
    "paradigm-ssm",
    "paradigm-linear-attention"
  ],
  "sources": [
    "paper-mamba2",
    "repo-mamba"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "state space model",
    "状态空间模型"
  ],
  "updated": "2026-10-23"
}
---

# Mamba-2

## 结构概述

Mamba-2 是选择性状态空间架构，论文以 State Space Duality（SSD）联系 SSM 与结构化注意力形式，并据此构造可并行的 chunked 算法。该模型不是带常规 K/V cache 的 Transformer。

## 结构与数据流

典型 block 将输入投影到多路通道，经局部卷积/门控和参数化的 delta、B、C 等项，再由 SSD/scan 核心维护状态，最后投影回 hidden size 并残差相加。推理路径保留每层的卷积状态与 SSM state；确切参数布局依 implementation/config。

## 实现差异

核对 state dimension、head/group 数、chunk size、卷积宽度、delta 参数化、初始化、fused scan kernel、推理状态 dtype 和训练/推理代码版本。

## 量化关联

- 递推 state 和 delta 的量化误差可能随 token 数累积，短输入评测容易漏掉漂移。
- 对权重、输入投影、局部卷积、门控、state update 和 state readout 分别统计误差。
- 真实 SSD/scan kernel 的累加精度与低比特支持是部署约束，reference fake quant 不能证明后端可用。

## Agent 检查

1. 依据 config 与源码列出各状态 tensor 的形状、dtype 和生命周期。
2. 对照 reference 与 fused scan，在短序列、长序列和 chunk 边界比较输出。
3. 比较权重量化和状态量化的独立/联合影响。
4. 在目标推理后端测量质量、内存和吞吐。

## 来源与证据

Mamba-2 论文及官方 Mamba 仓库提供 SSD 和实现参考；不同 checkpoint 的层参数与后端融合需核验。
