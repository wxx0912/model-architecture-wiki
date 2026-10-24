---
{
  "id": "component-multimodal-connector",
  "title": "Multimodal Encoder and Connector",
  "type": "component",
  "tags": [
    "multimodal",
    "embedding",
    "quantization",
    "multimodal-connector"
  ],
  "related": [
    "paradigm-multimodal",
    "component-embedding-lm-head"
  ],
  "sources": [
    "paper-qwen2-vl",
    "repo-qwen2-vl"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-23"
}
---

# 多模态 Encoder 与 Connector

## 结构概述

Encoder 把原始模态转换为 feature，connector 把 feature 投影、重采样或对齐到语言/扩散主干的 hidden space。Connector 可以是线性层、MLP、resampler、cross-attention 或模态专用融合模块。

## 数据流与实现差异

记录输入分辨率/时长、patch 或 frame 数、feature shape、投影维度、token 顺序、mask、位置编码和 connector 的参数共享。processor 的预处理和 padding 也属于结构契约的一部分。

## 量化关联

- Encoder 与主干的动态范围和误差目标不同，连接处应设置独立校准边界。
- connector 的误差可能造成跨模态语义错位，即使主干逐层误差很小也可能影响任务结果。
- 量化候选必须受目标后端支持的输入 shape、layout、dtype 和跨设备拷贝限制约束。

## Agent 检查

1. 核对 processor、encoder、connector 与主干的真实调用链。
2. 在单模态和组合模态下记录 connector 输入/输出统计。
3. 先做 connector 局部消融，再评估主干量化和端到端质量。
