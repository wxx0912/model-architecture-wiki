---
id: paradigm-multimodal
title: Multimodal Transformer and Omni Models
type: paradigm
tags: [multimodal, ar, diffusion, quantization]
related: [paradigm-ar, paradigm-dit, component-embedding-lm-head, check-quantization-preflight]
sources: [paper-qwen2-vl, repo-qwen2-vl, repo-huggingface-transformers]
confidence: source-reported
evidence_scope: mixed
validated_at: 2026-10-22
---

# 多模态与 Omni 模型

## 结构概述

多模态模型把视觉、音频、视频或其他模态编码为 token/latent，再通过 projector、cross-attention、token 拼接或更复杂 connector 接入语言或扩散主干。VLM 与 Omni 模型的模态数量、时序建模和输出头可能完全不同。

## 结构与数据流

典型路径为 `raw modality -> encoder -> resampler/projector -> modality tokens -> fusion -> language/diffusion backbone -> output heads`。需要记录各模态 token 的 shape、位置编码、时间轴、mask、融合层和是否共享主干。视觉编码器和语言主干可能使用不同精度与后端。

## 实现差异

核对图像分辨率/patch、视频帧采样、音频窗口、connector 类型、token 排列、跨模态 attention mask、特殊 token、processor 和输出 head。模型名相同也可能有不同 processor 与 checkpoint 变体。

## 量化关联

- 模态 encoder、connector 和语言主干的数据分布不同；统一校准集或统一 bit 容易造成某一模态退化。
- 跨模态 connector 的小误差可能改变语言/扩散主干的输入语义，需单独看模态对齐误差。
- 视频/音频有时间轴和长序列状态，校准应覆盖帧率、时长和缺失模态等边界。
- 必须分别报告模态理解质量、生成质量、首 token/解码延迟和真实后端支持。

## Agent 检查

1. 解析 processor、encoder、connector、fusion 和输出 head 的真实路径。
2. 为每个模态记录 token shape、范围、mask 和校准样本分层。
3. 先按后端能力过滤 encoder/connector 的可量化模块，再搜索主干配置。
4. 做单模态、组合模态、长视频/音频和缺失模态回归。

## 来源与证据

Qwen2-VL 论文/仓库可作为 VLM 结构案例；“Omni”不是单一结构，具体模型必须绑定官方配置和代码。

