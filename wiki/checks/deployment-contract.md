---
id: check-deployment-contract
title: Quantization Deployment Contract
type: check
tags: [quantization, deployment, packing, calibration]
related: [check-quantization-preflight, component-kv-cache, component-router-expert]
sources: [repo-huggingface-transformers, repo-deepseek-v3, repo-mamba]
confidence: inferred
evidence_scope: mixed
validated_at: 2026-10-22
---

# 量化部署契约

## 需要固定的字段

对每个候选配置登记：权重/激活/状态 bit 和 dtype、对称性、per-tensor/per-channel/per-group 轴、group size、scale/zero-point dtype、零点布局、packing 顺序、融合算子、设备、kernel、导出格式和回载方式。

## 结构相关约束

- QKV、gate/up、专家权重、MLA latent、connector 和状态 tensor 可能有不同的 shape/packing 规则；
- cache/state 需要读写 kernel、page/chunk layout、prefill→decode 或 reset 交接；
- tied weight、共享 scale、tensor/expert parallel 切分不能因导出而失去一致性；
- dynamic shape（序列、帧、token、专家负载）必须在实际后端支持范围内。

## 验证顺序

1. schema 与维度检查；
2. 导出后独立回载并比较权重/scale 元数据；
3. 单算子与单 block 数值对比；
4. reference fake-quant 与真实 kernel 对比；
5. 端到端质量、内存、延迟、吞吐和失败日志归档。

部署契约通过只说明“可运行且接口一致”，不能替代精度验收。

