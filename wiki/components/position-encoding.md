---
{
  "id": "component-position-encoding",
  "title": "RoPE and Position Encoding",
  "type": "component",
  "tags": [
    "rope",
    "attention",
    "quantization"
  ],
  "related": [
    "component-attention",
    "paradigm-ar",
    "paradigm-dit",
    "paradigm-dllm"
  ],
  "sources": [
    "paper-rope",
    "repo-huggingface-transformers"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "rotary position embedding",
    "旋转位置编码"
  ],
  "updated": "2026-10-23"
}
---

# RoPE 与位置编码

## 结构概述

Rotary Position Embedding（RoPE）把位置相关的旋转施加到 query/key 的成对通道，使注意力内积携带相对位置信息。长上下文变体可能改变频率、缩放或位置索引映射。

## 数据流与形状

对每个位置 `p` 生成 cos/sin，按 head_dim 的偶/奇通道旋转 Q、K，再参与 attention logits。实现要确认 RoPE 是在 Q/K 投影后、cache 写入前还是其他位置执行，以及 partial rotary 的维度范围。

## 实现差异

检查 rope_theta、max_position_embeddings、scaling/rope_type、位置索引、交错或连续 half 的旋转布局、dtype 和 cache 中保存的是旋转前还是旋转后 K。

## 量化关联

- Q/K 的量化误差会直接改变旋转后的 logits；长上下文和高位置索引应单独校准。
- cos/sin 表、位置索引和 scaling 通常不是普通权重量化对象，需遵守后端实现。
- GQA/MLA、滑窗和多模态位置编码可能使用不同的 position path，不能共享默认校准。

## Agent 检查

1. 从配置和源码确认 RoPE 类型、旋转维度、索引和 cache 边界。
2. 在短/长位置比较旋转前后 Q/K、logits 和 cache 读取结果。
3. 核对导出格式和 kernel 对 rope scaling、低比特 Q/K 的支持。
