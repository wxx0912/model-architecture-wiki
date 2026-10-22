---
id: check-structure-identification
title: Model Structure Identification
type: check
tags: [quantization, calibration, attention, moe, multimodal, ssm]
related: [check-quantization-preflight, component-attention, component-router-expert]
sources: [repo-huggingface-transformers, repo-kernelwiki]
confidence: inferred
evidence_scope: mixed
validated_at: 2026-10-22
---

# 模型结构识别

## 目的

把“模型名称”转换成可量化的实际计算图。结构识别必须以当前 checkpoint、配置、源码和运行图为证据，不能只匹配类名或论文架构图。

## 识别顺序

1. 锁定 checkpoint、配置、代码版本、权重索引和 tokenizer/processor。
2. 从 module tree、state dict key 和实际 forward hook 构建模块清单。
3. 识别 block 类型、attention 变体、FFN/gate、norm、residual、router、state/cache、connector 和输出头。
4. 记录每个边界的输入/输出 shape、dtype、布局、融合/共享关系和并行切分。
5. 用最小输入运行一次 reference forward，确认识别结果覆盖真实执行路径。

## 特别容易误判的结构

- fused QKV、fused gate/up 与拆分权重；
- GQA/MLA 的 cache 与普通 MHA cache；
- linear attention/SSM state 与 KV cache；
- MoE 的 shared/routed expert、专家布局和 token 重排；
- DiT 的 timestep/adaLN 与 dLLM 的 mask/step；
- tied embedding/LM head 与多模态 connector；
- mHC/Hyper-Connections 等跨层 residual mixing。

## 输出

输出带证据链接的结构图、模块表、未知项清单和量化边界。对于未能从源码或运行图确认的字段使用 `unknown`，不要用模型系列的默认值填充。

