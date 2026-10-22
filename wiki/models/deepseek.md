---
id: model-deepseek
title: DeepSeek-V2 / DeepSeek-V3
type: model
tags: [ar, attention, mla, moe, routing, quantization]
related: [paradigm-ar, paradigm-moe, component-attention, component-router-expert, component-mlp, component-kv-cache]
sources: [paper-deepseek-v2, paper-deepseek-v3, repo-deepseek-v3, repo-huggingface-transformers]
confidence: source-reported
evidence_scope: mixed
validated_at: 2026-10-22
---

# DeepSeek-V2 / DeepSeek-V3

## 结构概述

DeepSeek-V2/V3 是 decoder-only autoregressive 模型，代表性结构包括 Multi-head Latent Attention（MLA）和 DeepSeekMoE。V3 技术报告还记录了辅助损失自由的负载均衡策略及 Multi-Token Prediction（MTP）训练/预测模块。不同尺寸和发布版本可能有不同配置，不应从系列名推断单个 checkpoint 的每项特征。

## 结构与数据流

主干由 embedding、多个 Transformer block、最终 norm 和 LM head 组成。attention 路径通过 MLA 的低秩 latent 表示压缩 KV 信息；MoE block 的 router 为 token 选择 routed experts，并配有 shared experts。V3 的 MTP 相关模块用于训练目标/预测扩展，推理图是否包含及如何导出需按具体 checkpoint 确认。

## 实现差异

核对 model config、attention/MLP 权重 key、latent rank、rope/nope 维度、专家数量与分组、shared expert、router top-k、MTP 配置、张量并行和权重 layout。Hugging Face、官方参考实现和推理引擎可能采用不同融合模块名与权重映射。

## 量化关联

- MLA latent KV 路径与普通 K/V cache 不同，cache 量化对象和后端 kernel 需按实际实现定义。
- 专家频率差异要求 per-expert coverage；router top-k 变化应与专家 MLP 输出误差分开测量。
- MTP 分支若保留在推理图中，应独立检查其权重和输出；若仅用于训练，则不能把它当作推理量化对象。
- mHC 是独立的残差结构研究，不能因同属 DeepSeek 团队就推断某个 V2/V3 checkpoint 默认使用 mHC。

## Agent 检查

1. 读取 checkpoint config 和权重索引，锁定模型版本及实现路径。
2. 对照源码确认 MLA、专家布局、MTP、融合 QKV/MLP 和 cache/state 契约。
3. 记录路由直方图、latent/cache 统计、各 expert 覆盖和长上下文误差。
4. 在目标后端执行导出回载及真实推理验证。

## 来源与证据

V2/V3 技术报告提供系列架构信息；仓库或具体 checkpoint 配置提供实现事实。页面不代表每个变体均包含相同模块。

