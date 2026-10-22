---
id: component-attention
title: "Attention: MHA, GQA and MLA"
type: component
tags: [attention, mha, gqa, mla, rope, kv-cache]
related: [paradigm-ar, paradigm-moe, component-kv-cache, model-deepseek]
sources: [paper-transformer, paper-deepseek-v2, paper-deepseek-v3, repo-huggingface-transformers]
confidence: source-reported
evidence_scope: mixed
validated_at: 2026-10-22
---

# Attention：MHA、GQA 与 MLA

## 结构概述

标准多头注意力（MHA）为每个 query head 配置独立的 K/V head。GQA 让多个 query head 共享一个 K/V head，以降低 KV cache 和带宽开销。DeepSeek 的 MLA 进一步将 key/value 信息压缩到 latent 表示，再在注意力计算中恢复所需表示；具体实现和缓存布局需要按模型源码确认。

## 数据流与形状

输入 hidden states 通常为 [B, T, d_model]，经过 Q/K/V 投影得到 head 维表示，再执行位置编码、缩放点积、mask、softmax 和输出投影。MHA、GQA 的差异主要在 q_heads 与 kv_heads 的关系；MLA 还增加 latent compression/decompression 路径。

## 实现差异

核对 QKV 是否 fused、head_dim、q_heads/kv_heads、RoPE 插入位置、KV cache layout、MLA latent 维度，以及 tensor parallel 的切分方式。不能仅凭模块名推断 Q/K/V 已融合或共享 scale。

## 量化关联

**结构事实：**

- Q/K/V/O 投影、RoPE、softmax、KV cache 和输出投影是不同数值路径。
- GQA/MLA 改变 KV cache 的形状和更新方式。
- MLA 的 latent cache 不等于普通 K/V cache。

**量化推断：**

- 需要分别观察 Q、K、V、O、latent projection、RoPE 输入和 cache 的范围。
- 低比特误差可能改变 attention logits、softmax 分布和长上下文行为；应同时做 prefill/decode 验证。
- QKV 统一量化或融合约束只能由具体后端和 packing 规则决定，不能写成通用硬约束。

## Agent 检查

1. 从配置、源码和权重 key 确认 MHA/GQA/MLA。
2. 记录 QKV/O 的模块路径、形状、融合关系和并行切分。
3. 记录 cache 的 dtype、layout、更新 kernel 和上下文长度。
4. 用固定输入比较单层 attention 输出、logits 和真实后端结果。
