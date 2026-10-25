---
{
  "id": "check-deployment-contract",
  "title": "Quantization Deployment Contract",
  "type": "check",
  "tags": [
    "quantization",
    "deployment",
    "packing",
    "calibration"
  ],
  "related": [
    "check-quantization-preflight",
    "component-kv-cache",
    "component-router-expert",
    "component-multimodal-connector",
    "model-deepseek",
    "model-mamba2"
  ],
  "sources": [
    "repo-huggingface-transformers",
    "repo-deepseek-v3",
    "repo-mamba"
  ],
  "confidence": "inferred",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-24"
}
---

# 量化部署契约

## 一句话理解

部署契约登记的是"可运行且接口一致"：导出、回载与算子接口的一致性。它不回答精度是否达标，也不等于后端已经真正以低比特计算。

## 需要固定的字段

本页的字段清单与验证顺序是本库为部署契约定义的约定（工程归纳，非单一来源事实）。字段包括：权重 / 激活 / 状态 bit 与 dtype、对称性、per-tensor / per-channel / per-group 轴、group size、scale / zero-point dtype、零点布局、packing 顺序、融合算子、设备、kernel、导出格式、回载方式。

## 结构相关约束

不同结构的张量有不同的 shape 与 packing 规则，契约必须逐类登记：

- QKV、gate/up、专家权重、MLA latent、connector 与状态 tensor 的 packing 规则可能不同；
- cache/state 需要读写 kernel 与分块布局（见 Mamba2 来源页记录的 `chunk_size`（实现默认值）、transformers 来源页记录的 `StaticCache` 预分配），以及 prefill→decode 的交接；
- 权重共享（如 embedding 与输出头 tied）与共享 scale 不能因导出失去一致性，张量并行切分需在导出后保持（见 Mamba2 来源页记录的 `ColumnParallelLinear`/`RowParallelLinear`）；
- dynamic shape（序列长度、token 数、专家负载）必须在实际后端支持范围内。

这些约束是本库从 transformers 的 cache 接口、DeepSeek 的 fp8/KV 布局与 Mamba 的状态实现归纳出的（工程归纳，非单一来源事实）。

## 验证顺序

1. schema 与维度检查；
2. 导出后独立回载并比较权重 / scale 元数据；
3. 单算子与单 block 数值对比；
4. reference fake-quant 与真实 kernel 对比；
5. 端到端质量、内存、延迟、吞吐与失败日志归档。

## 结果边界

- 部署契约通过只说明"可运行且接口一致"，不能替代精度验收；某字段未知时应给出具体待验证项，而不是补默认值。
- 本库未运行任何导出、回载或后端，未对任何目标格式作部署认证。

## Agent 如何使用

首轮读本页即可知道一份部署契约需要登记哪些字段、哪些结构约束最容易在导出时被破坏。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| cache 的运行时契约与量化接口入口 | [huggingface/transformers](../../sources/repos/repo-huggingface-transformers.md)、[KV Cache](../components/kv-cache.md) |
| MLA latent 与 fp8 路径的实际字段 | [DeepSeek-V3 官方实现](../../sources/repos/repo-deepseek-v3.md)、[DeepSeek 页](../models/deepseek.md) |
| SSM 状态形状与 kernel 布局 | [state-spaces/mamba](../../sources/repos/repo-mamba.md)、[Mamba-2 页](../models/mamba2.md) |
| 专家布局与并行切分 | [Router 与 Expert](../components/router-expert.md) |
| 目标格式在真实后端的可执行性 | 目标模型源码 / 导出器 / 后端 kernel 的独立验证（本库不提供） |

回答部署问题时必须说明：目标格式与作用张量、packing 与 scale 布局、后端与 kernel、以及验证停在"接口一致"还是"精度已验收"。

## 来源与证据

- [huggingface/transformers](../../sources/repos/repo-huggingface-transformers.md)：`Cache`/`DynamicCache`/`StaticCache` 与 `QuantizedCache`、`HfQuantizer` 等接口入口。
- [DeepSeek-V3 官方实现](../../sources/repos/repo-deepseek-v3.md)：MLA 的 cache buffer（`kv_cache`/`pe_cache`）与 fp8 推理路径。
- [state-spaces/mamba](../../sources/repos/repo-mamba.md)：SSM 的固定状态与 CUDA/Triton kernel 分布。

本页是 `inferred` 级的方法页（工程归纳），不构成任何格式的部署认证。
