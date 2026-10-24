---
{
  "id": "component-mlp",
  "title": "MLP and Gated FFN",
  "type": "component",
  "tags": [
    "mlp",
    "swiglu",
    "geglu",
    "quantization"
  ],
  "related": [
    "paradigm-ar",
    "paradigm-moe",
    "component-attention"
  ],
  "sources": [
    "paper-transformer",
    "paper-deepseek-v2",
    "repo-huggingface-transformers"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-23"
}
---

# MLP 与 Gated FFN

## 结构概述

Transformer 的 FFN 通常先把 hidden states 投影到更宽的中间维度，再投影回 d_model。SwiGLU/GeGLU 使用 gate 和 up 两条投影，经过激活和逐元素乘法后进入 down 投影。不同实现可能拆开或合并 gate/up 权重。

## 数据流与形状

输入通常是 [B, T, d_model]；gate/up 产生 [B, T, d_ff]，门控后由 down 投影回 [B, T, d_model]。MoE 中这组投影会对每个专家分别存在，并可能使用 fused 3D 权重和 grouped GEMM。

## 实现差异

核对激活函数、gate/up 是否 fused、权重布局、bias、张量并行切分和中间张量是否物化。实现名称相同不代表权重 shape 或融合方式相同。

## 量化关联

**结构事实：**

- gate/up 共享输入，但非线性和逐元素乘法会改变误差传播。
- 中间激活可能有异常值；down 投影会把误差重新注入 residual stream。
- MoE expert 的 MLP 还受到 token 路由和专家负载影响。

**量化推断：**

- 应分别统计 gate、up、激活乘法前后和 down 的误差。
- gate/up 是否需要同精度取决于具体 fused kernel 和量化格式，不应默认强制统一。
- 只做单 Linear MSE 可能无法反映 gated FFN 的 block 输出质量。

## Agent 检查

1. 确认 gate/up/down 的实际 module prefix 和权重布局。
2. 检查 fused gate-up kernel 的输入、scale 和 packing 契约。
3. 对门控分支与 down 输出分别做局部消融。
4. 在 dense 与 MoE 场景分别记录激活覆盖。
