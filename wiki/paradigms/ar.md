---
{
  "id": "paradigm-ar",
  "title": "Autoregressive Transformer",
  "type": "paradigm",
  "tags": [
    "ar",
    "attention",
    "kv-cache",
    "rope",
    "quantization"
  ],
  "related": [
    "component-attention",
    "component-kv-cache",
    "component-normalization",
    "component-position-encoding",
    "check-quantization-preflight"
  ],
  "sources": [
    "paper-transformer",
    "repo-huggingface-transformers"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-23"
}
---

# Autoregressive Transformer

## 结构概述

AR 模型按条件概率逐 token 生成。典型 decoder-only Transformer 由 token embedding、因果 self-attention、归一化、门控 FFN 和 LM head 组成；训练时可并行计算整段序列，生成时按 prefill 和 decode 两个阶段运行。

## 结构与数据流

输入通常是 `[B, T]` 的 token id，embedding 后得到 `[B, T, d_model]`。每个 block 依次执行 norm、因果 attention、残差、norm、FFN 和残差，最后经 norm 与 LM head 产生 `[B, T, V]` logits。prefill 为整段上下文写入 cache；decode 每步只计算新 token 的 query，并读取历史 K/V。

## 实现差异

需要从配置和源码确认 pre-norm/post-norm、MHA/GQA/MLA、RoPE 位置、QKV 是否融合、KV cache layout、滑动窗口、张量并行切分和 LM head 是否与 embedding 权重共享。FlashAttention 等 kernel 会改变中间张量是否物化，但不改变结构语义。

## 量化关联

- prefill 与 decode 的激活分布、序列长度和 batch 形态不同，校准不能只覆盖其中一项。
- Q/K/V、attention logits、softmax、cache、FFN gate/up/down 和 LM head 是不同的误差路径；统一 scale 需要后端契约支持。
- cache 量化属于运行时状态量化，必须验证 cache 写入、读取、分页和长上下文行为。
- 逐层误差应同时看 block 输出、最终 logits 和真实后端生成，不能只用权重 MSE 判断。

## Agent 检查

1. 记录模型版本、配置、源码 commit、模块 prefix、权重 key 和张量 shape。
2. 确认 prefill/decode、cache dtype/layout、RoPE、GQA/MLA 和并行切分。
3. 为短上下文、长上下文、不同 batch 和真实 decode 分别建立校准/验证切片。
4. 量化后执行 fake-quant、导出回载和目标后端输出一致性检查。

## 来源与证据

Transformer 论文描述了 decoder self-attention 和因果建模；具体 AR 模型的模块名、融合方式和 cache 契约必须以目标实现为准。
