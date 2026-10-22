---
id: check-quantization-preflight
title: Quantization Preflight
type: check
tags: [quantization, calibration, deployment, quantization-preflight]
related: [check-structure-identification, check-calibration-axes, check-deployment-contract]
sources: [repo-huggingface-transformers, repo-kernelwiki]
confidence: inferred
evidence_scope: mixed
validated_at: 2026-10-22
---

# Quantization Preflight

## 目的

在搜索量化算法和 bit 配置前，把模型结构、可量化边界、必须联合的模块、校准轴和后端能力登记清楚。该页是流程检查单，不是对任意模型的默认结论。

## 必须登记的事实

- checkpoint、模型配置、代码仓库与 commit、运行时/后端版本；
- 模块 prefix、权重 key、tensor shape、dtype、参数共享和融合关系；
- Attention/MLP/MoE/状态更新/connector 等模块的输入输出边界；
- 可量化模块、暂不量化模块、必须联合量化模块及原因；
- 校准数据域、长度、batch、timestep/mask/state/路由等动态轴；
- 目标 bit、scheme、group size、scale/zero-point、packing、导出格式和 kernel 支持；
- fake quant、导出回载、reference 与真实后端的验证结果。

## 非法候选过滤

Agent 在搜索前应拒绝或标记以下候选：模块维度不满足 group/pack 要求、后端不支持该 dtype/scheme、fused 权重无法拆分、共享权重被分配冲突配置、状态/cache 无对应 kernel、导出后无法回载或模型路径被改变。

## 输出

生成一份机器可读的 preflight record，至少包含 `model_id`、`source_revision`、`module_inventory`、`constraints`、`calibration_axes`、`candidate_exclusions` 和 `evidence`。每项标记 `confirmed`、`source-reported`、`inferred` 或 `unknown`。

## 验收

Preflight 通过只表示候选在结构和接口层面可执行，不表示量化精度达标；精度、性能、显存和端到端质量仍需实验验证。

