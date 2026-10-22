---
id: component-kv-cache
title: KV Cache
type: component
tags: [kv-cache, attention, gqa, mla, quantization]
related: [component-attention, paradigm-ar, paradigm-linear-attention]
sources: [paper-transformer, paper-deepseek-v2, paper-kimi-linear]
confidence: source-reported
evidence_scope: mixed
validated_at: 2026-10-22
---

# KV Cache

## 结构概述

自回归 attention 在 decode 时缓存历史 key/value，避免每个新 token 重算完整上下文。MHA、GQA、MLA 的 cache 内容、head 数和布局不同；线性 attention 或 SSM 通常维护另一类递推状态，不应称为普通 KV cache。

## 量化关联

**结构事实：**

- cache 是运行时状态，不是权重 checkpoint 的简单附属字段。
- cache 量化涉及存储格式、scale 更新、page/layout、prefill 到 decode 的交接和 kernel 支持。
- MLA 的 latent cache 与 MHA/GQA 的 K/V cache 需要分别分析。

**量化推断：**

- 校准要覆盖 prefill/decode、上下文长度、batch 和 cache growth。
- 需要比较 cache 量化前后的 attention logits、长上下文任务和真实 decode latency。
- 伪量化结果不能证明目标后端的 cache kernel 已生效。

## Agent 检查

1. 识别 cache 的张量语义、dtype、page size 和 layout。
2. 核对 prefill 写入、decode 读取和跨 batch/请求复用。
3. 固定上下文和随机种子，比较 BF16、伪量化和真实后端。
