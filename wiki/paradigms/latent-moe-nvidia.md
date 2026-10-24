---
{
  "id": "paradigm-latent-moe-nvidia",
  "title": "NVIDIA LatentMoE",
  "type": "paradigm",
  "tags": [
    "latent-moe",
    "moe",
    "routing",
    "quantization"
  ],
  "related": [
    "paradigm-latent-moe",
    "paradigm-moe",
    "component-mlp",
    "component-router-expert"
  ],
  "sources": [
    "paper-latentmoe"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-23"
}
---

# NVIDIA LatentMoE

## 结构概述

论文中的 LatentMoE 先把 token 从原始 hidden 维度投影到低维 latent 空间，再在 latent 空间执行 routed expert 计算，最后投影回原始维度。路由仍在原始 hidden 表示上进行；latent 维度是控制专家计算与参数成本的主要旋钮。

## 数据流

抽象路径为 `x -> router(x), W_down(x) -> latent expert MLP -> W_up(.) -> combine/residual`。需要区分原维 router、低维专家权重、down/up projection、token dispatch 和恢复投影。

## 量化关联

- W_down/W_up 与 latent expert 的 shape、scale 和 group size 不应套用原维 dense FFN 的默认值。
- router 在原维工作，路由误差与 latent reconstruction error 要分开记录。
- 低维投影、专家通信和导出 packing 必须先通过目标后端约束筛选。

## Agent 检查

1. 确认论文/模型版本及 latent dimension、expert count、top-k。
2. 建立原维 router、latent path、experts 和恢复投影的模块表。
3. 分层统计 router 一致率、latent 误差、expert 输出误差和端到端质量。

## 来源与证据

本页对应 Elango 等人的 NVIDIA/研究 LatentMoE 论文。当前不把论文作为 NVIDIA 独立实现或后端支持的证明；部署仍需目标模型代码和 kernel 证据。
