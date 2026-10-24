---
{
  "id": "paradigm-latent-moe",
  "title": "LatentMoE Terminology and Variants",
  "type": "paradigm",
  "tags": [
    "latent-moe",
    "moe",
    "routing",
    "quantization"
  ],
  "related": [
    "paradigm-moe",
    "paradigm-latent-moe-nvidia",
    "paradigm-multihead-latent-moe",
    "component-mlp"
  ],
  "sources": [
    "paper-latentmoe",
    "paper-multihead-latentmoe"
  ],
  "confidence": "inferred",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-23"
}
---

# LatentMoE Terminology and Variants（术语与变体）

## 结构概述

“LatentMoE”不是唯一、统一的架构名称。本库已登记两种来源明确的变体：NVIDIA/研究论文的 LatentMoE（`paradigm-latent-moe-nvidia`）与 Multi-Head LatentMoE（`paradigm-multihead-latent-moe`）。Agent 命中该术语时必须同时显示组织、论文/模型和版本，再选择具体页面；未登记来源的变体名不得据术语推断结构。

## 结构与数据流

不同变体的共同抽象可能是 `x -> projection/latent path -> router/experts -> recovery/merge`，但路由所在空间、专家计算空间、head 并行和通信方式并不相同。

## 实现差异

确认 latent projection 的位置、维度、归一化、共享关系、专家输入/输出 shape、路由粒度和恢复投影。不要把 MLA 的 latent KV cache 或一般低秩适配器自动标记成 LatentMoE。

## 量化关联

- 压缩/恢复投影会把量化误差映射到 latent 空间和主干空间，两个投影不能只看单层误差。
- latent 分布可能比 residual stream 更窄或更尖，需单独选择校准轴和 scale。
- 专家选择与 latent 投影误差耦合，必须同时记录路由一致率、latent reconstruction error 和 block 输出误差。

## Agent 检查

1. 首先要求用户或来源提供该术语对应的论文、仓库或配置。
2. 画出投影、路由、专家和恢复投影的实际计算图。
3. 只有确认结构后才生成量化候选；未知部分标记为 `unknown`。

## 来源与证据

本页只确认术语歧义和变体分流；具体结构与量化边界应使用下列变体页面及其对应来源，不应把本页当作单一实现定义。
