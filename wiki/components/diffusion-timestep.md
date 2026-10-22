---
id: component-diffusion-timestep
title: Diffusion Time/Noise and Discrete Mask State
type: component
tags: [timestep, diffusion, dit, dllm, mask-state, calibration]
related: [paradigm-dit, paradigm-dllm]
sources: [paper-dit, paper-llada, repo-dit]
confidence: source-reported
evidence_scope: mixed
validated_at: 2026-10-22
---

# Diffusion Timestep 与条件注入

## 结构概述

图像/视频扩散模型使用 timestep/noise level 描述连续去噪阶段，并将其编码为条件向量，注入 normalization、attention、MLP 或输出头。DiT 常用 adaLN 风格调制。dLLM 的 mask ratio、生成 step、block 状态和 remasking 是离散迭代状态，不能自动等同于连续 diffusion timestep/noise。

## 量化关联

timestep/noise 或 mask/step 都可能改变激活分布，但它们属于不同架构语义。校准集应按目标模型分别记录连续去噪轴或离散 mask/迭代轴；单个 timestep 或单个 mask ratio 的局部误差不能代表完整采样轨迹。

## Agent 检查

1. 记录 timestep/noise 或 mask/step 的采样分布。
2. 确认条件向量注入位置、dtype 和共享关系。
3. 固定 scheduler、steps、mask 策略和随机种子，比较逐步轨迹与最终质量。
