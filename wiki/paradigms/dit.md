---
id: paradigm-dit
title: Diffusion Transformer (DiT)
type: paradigm
tags: [dit, diffusion, timestep, attention, mlp, quantization]
related: [component-attention, component-mlp, component-diffusion-timestep, component-normalization, check-calibration-axes]
sources: [paper-dit, repo-dit]
confidence: source-reported
evidence_scope: mixed
validated_at: 2026-10-22
---

# Diffusion Transformer（DiT）

## 结构概述

DiT 用 Transformer 作为扩散模型的去噪主干。图像或视频先被 patch/token 化，模型在给定 timestep 和条件的情况下预测噪声、速度或重建目标。DiT 本身不是一个完整的图像系统，VAE、tokenizer、scheduler 和采样器也会影响端到端结果。

## 结构与数据流

输入 latent 或 patch 经 PatchEmbed 得到 `[B, N, d_model]`，timestep embedding 与类别/文本条件经调制模块注入 block。典型 block 包含 attention、MLP 和 adaLN-Zero 风格的 scale/shift/gate；最后的 modulation 和输出投影恢复 patch 维度，再由解码器得到样本。视频实现还会引入时间轴、空间轴或时空联合 attention。

## 实现差异

核对 latent 分辨率、patch size、空间/时间 token 排列、2D/3D 位置编码、条件注入位置、预测目标和 scheduler。不同实现可能使用 cross-attention、拼接条件 token 或仅使用 adaLN；不能仅由“DiT”名称推断。

## 量化关联

- timestep/noise level 会显著改变激活范围，校准必须覆盖采样轨迹，而不是一个固定时间点。
- adaLN 的调制向量和 gate 会把误差传播到整层；attention、MLP、输出头和 VAE/连接器要分开评估。
- 视频模型还需按空间/时间 token 和长序列显存路径验证；单帧结果不能代表视频质量。
- 最终 FID、重建或视频质量与逐层误差不总是单调对应，需保留轨迹级和端到端指标。

## Agent 检查

1. 识别 latent/token shape、patch/时间布局、条件路径和预测目标。
2. 记录 scheduler、采样步数、timestep 分布和 classifier-free guidance 配置。
3. 在早期、中期、晚期 timestep 采集 activation/误差统计。
4. 对图像和视频分别验证逐步输出、最终质量、显存和后端 kernel。

## 来源与证据

DiT 论文和官方实现支持上述主干与 adaLN 结构；具体模型的条件注入和输出目标需要核对版本源码。

