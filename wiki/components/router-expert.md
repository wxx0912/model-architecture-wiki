---
{
  "id": "component-router-expert",
  "title": "MoE Router, Dispatch and Expert MLP",
  "type": "component",
  "tags": [
    "moe",
    "routing",
    "mlp",
    "quantization"
  ],
  "related": [
    "paradigm-moe",
    "paradigm-latent-moe",
    "component-mlp"
  ],
  "sources": [
    "paper-switch-transformer",
    "paper-mixtral",
    "paper-deepseek-v2",
    "repo-deepseek-v3"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "mixture of experts",
    "mixture-of-experts",
    "混合专家"
  ],
  "updated": "2026-10-23"
}
---

# MoE Router、Dispatch 与 Expert MLP

## 结构概述

MoE block 可拆为 router、token dispatch、专家计算和 combine。把它们作为独立组件记录，有助于区分连续数值误差与 top-k 路由变化。

## 数据流与形状

Router 对 `[N,d_model]` hidden states 产生 `[N,E]` 分数，选出每个 token 的 k 个专家；dispatch 形成按专家分组的 token batch，专家 MLP 输出后按原 token index 和 routing weight combine。实现还可能有 shared experts、capacity 和 all-to-all 通信。

## 量化关联

- router 分数的微小变化会造成离散专家切换，不能只用 router 输出的 MSE 评价。
- dispatch/combine 的索引、权重和通信必须保持精确语义；专家权重与激活应按专家访问分布校准。
- fused grouped GEMM 的 group size、packing、scale layout 和专家维度是导出前硬约束。

## Agent 检查

1. 记录 router 输入/输出、top-k、capacity、token drop 和专家负载。
2. 保存 token→expert 映射用于量化前后路由一致性比较。
3. 确认专家权重 key、维度、shared/routed 区分和目标 kernel 支持。
