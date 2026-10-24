---
{
  "id": "paradigm-moe",
  "title": "Sparse Mixture of Experts",
  "type": "paradigm",
  "tags": [
    "moe",
    "routing",
    "mlp",
    "quantization"
  ],
  "related": [
    "component-mlp",
    "component-router-expert",
    "paradigm-latent-moe",
    "check-quantization-preflight"
  ],
  "sources": [
    "paper-switch-transformer",
    "paper-mixtral",
    "paper-deepseek-v2",
    "paper-deepseek-v3",
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

# Sparse Mixture of Experts（MoE）

## 结构概述

稀疏 MoE 用 router 为每个 token 选择少量专家，扩大参数容量而控制每个 token 的计算量。常见专家是独立的 gated FFN，也可能叠加 shared experts；router、dispatch、expert GEMM 和 combine 构成不同于 dense FFN 的数据流。

## 结构与数据流

输入 hidden states 经 router 得到 `[B,T,E]` 的 logits 或概率，按 top-k 选择专家并生成 token-to-expert 映射。dispatch 后每个专家处理自己的 token 子集，结果按 routing weight combine 回原顺序，再与 residual 相加。分布式实现还包含 all-to-all、capacity、token dropping 或 grouped GEMM。

## 实现差异

确认 top-k、专家数、shared/routed expert、auxiliary loss、capacity、负载均衡、专家权重布局、通信顺序和 token 重排。DeepSeekMoE 等实现的 shared expert 与细粒度专家切分不能直接套用 Mixtral 配置。

## 量化关联

- 专家访问频率通常不均衡；全局校准可能忽略低频但关键专家。
- router logits 的小误差可能改变 top-k，产生离散路由差异；router 通常应与专家 MLP 分开评估。
- 每个专家的 activation、权重和负载分布不同，per-expert 或分组 scale 需要后端支持。
- 量化后要同时检查路由一致率、专家输出误差、通信/packing 以及端到端质量。

## Agent 检查

1. 建立 token→expert 直方图，并覆盖不同数据域、长度和 batch。
2. 记录 router、dispatch、expert MLP、combine 和 shared expert 的模块路径。
3. 先筛除后端不支持的专家布局、group size、packing 和通信格式。
4. 将路由变化、专家输出误差和最终任务指标分开记录。

## 来源与证据

Switch Transformer、Mixtral 及 DeepSeek 技术报告支持稀疏路由的通用结构；具体专家布局和 kernel 以目标实现为准。
