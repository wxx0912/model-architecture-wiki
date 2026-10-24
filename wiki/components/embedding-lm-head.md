---
{
  "id": "component-embedding-lm-head",
  "title": "Token Embedding and LM Head",
  "type": "component",
  "tags": [
    "embedding",
    "lm-head",
    "ar",
    "dllm",
    "quantization"
  ],
  "related": [
    "paradigm-ar",
    "paradigm-dllm",
    "paradigm-multimodal"
  ],
  "sources": [
    "paper-transformer",
    "paper-llada",
    "repo-huggingface-transformers"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-23"
}
---

# Token Embedding 与 LM Head

## 结构概述

Embedding 将离散 token id 映射到 hidden space；LM head 将最终 hidden states 投影到 vocabulary logits。许多 decoder-only 模型在输入 embedding 与输出 head 之间共享权重，但这不是所有模型的约束。

## 数据流与形状

Embedding 权重通常为 `[V, d_model]`，输入 `[B,T]` 得到 `[B,T,d_model]`；LM head 产生 `[B,T,V]` logits。多模态模型还可能有视觉/音频 embedding 和 connector；dLLM 的 head 仍可能在每一步输出全部或部分位置的 logits。

## 实现差异

确认权重是否 tied、词表 padding、logit scale、bias、分片方式、特殊 token、输入/输出 vocab 是否一致，以及多模态 projector 是否复用 embedding 空间。

## 量化关联

- 大 vocabulary 的 LM head 影响 logits、top-k 和 loss，权重/激活量化的代价与中间 block 不同。
- tied embedding 量化会同时影响输入表示和输出 logits，必须作为联合约束分析。
- dLLM 位置选择与置信度依赖 logits；多模态 connector 的输出范围会改变 head 的校准分布。

## Agent 检查

1. 从权重 key 和 config 确认 embedding/head 是否共享及其分片。
2. 比较量化前后 logits、top-k、token 置信度和生成/填充决策。
3. 验证导出回载后 tied weight、词表和特殊 token 仍一致。
