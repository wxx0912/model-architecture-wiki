---
{
  "id": "check-calibration-axes",
  "title": "Calibration Axes by Architecture",
  "type": "check",
  "tags": [
    "calibration",
    "quantization",
    "timestep",
    "routing",
    "state-update"
  ],
  "related": [
    "check-quantization-preflight",
    "paradigm-ar",
    "paradigm-dit",
    "paradigm-dllm",
    "paradigm-moe",
    "paradigm-multimodal",
    "paradigm-ssm"
  ],
  "sources": [
    "paper-dit",
    "paper-llada",
    "paper-qwen2-vl",
    "paper-switch-transformer",
    "paper-transformer",
    "paper-mamba2",
    "repo-huggingface-transformers"
  ],
  "confidence": "inferred",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-24"
}
---

# 按架构组织校准轴

## 一句话理解

校准样本不只由文本 / 图像内容定义，还要覆盖会改变数值分布或离散路径的**运行时轴**。轴候选从结构页生成，再按目标模型与预算选择分层采样，而不是由静态模型名自动确定。

## 轴候选

| 架构 | 候选轴 | 结构依据 |
| --- | --- | --- |
| AR | prefill/decode、序列长度、batch、位置、cache growth | [Attention](../components/attention.md)、[KV Cache](../components/kv-cache.md)、[RoPE 与位置编码](../components/position-encoding.md)、[Autoregressive Transformer](../paradigms/ar.md) |
| DiT | timestep/noise、采样阶段、分辨率、空间/时间 token、guidance | [Diffusion Transformer](../paradigms/dit.md)、[扩散时间步](../components/diffusion-timestep.md) |
| dLLM | mask ratio、迭代 step、更新 block、置信度排序、重掩码 | [Diffusion Language Model](../paradigms/dllm.md) |
| MoE | router top-k、专家访问频率与负载分布、capacity、并行布局 | [Mixture of Experts](../paradigms/moe.md)、[Router 与 Expert](../components/router-expert.md) |
| Linear/SSM | state 形状与初值、序列长度、chunk 边界、reset、state dtype | [Selective State Space Model](../paradigms/ssm.md)、[Mamba-2](../models/mamba2.md) |
| Multimodal | 模态、分辨率、帧数/时长、connector 输出、缺失模态 | [Multimodal Transformer](../paradigms/multimodal.md)、[Qwen2-VL](../models/qwen2-vl.md) |

本表是本库基于各结构页归纳的轴候选（工程归纳，非来源事实），"结构依据"指向描述该轴的结构页；轴本身不能替代在目标数据上实测的敏感度。

## 采样与记录

每个样本携带轴的 metadata，并保存代表性覆盖率、极端值与未覆盖区域。少量平均 activation 统计不证明所有轴已覆盖；未覆盖轴明确降级为未知风险。

校准集通过覆盖审计后，再进入算法 / 配置搜索；搜索结果必须回指使用过的轴和样本版本，以便复现与比较。

## 结果边界

- 轴候选随结构而变，不能跨架构套用；某轴是否敏感取决于目标模型与数据，本库未做任何实测。
- 用少量平均 activation 统计不能证明轴已覆盖，未覆盖轴只能标为未知风险。
- 本库未运行任何校准或评测。

## Agent 如何使用

首轮读本页即可知道目标架构有哪些候选校准轴，以及每个轴的来源。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 各架构的计算图与动态轴 | [Paradigms](../paradigms/ar.md) 与同库各 paradigm 页 |
| 各组件引入的运行时状态（cache、timestep、mask、router、state） | [Components](../components/attention.md) 与同库各 component 页 |
| 校准记录与非法候选过滤 | [量化前检查](quantization-preflight.md) |
| 目标模型上各轴的实际分布与敏感度 | 目标数据上的分层采样与实测（本库不提供） |

回答校准问题时必须说明覆盖了哪些轴、哪些轴降级为未知风险，并区分候选轴清单与实际采样结果。

## 来源与证据

- [Diffusion Transformer](../../sources/papers/paper-dit.md)、[LLaDA](../../sources/papers/paper-llada.md)、[Qwen2-VL](../../sources/papers/paper-qwen2-vl.md)、[Switch Transformer](../../sources/papers/paper-switch-transformer.md)、[Attention Is All You Need](../../sources/papers/paper-transformer.md)、[Mamba-2](../../sources/papers/paper-mamba2.md)：各结构的结构与过程事实；本页的动态轴候选是基于这些事实的工程归纳，不是论文给出的校准事实。

本页是 `inferred` 级的方法页（工程归纳），不提供任何模型的实测敏感度结论。
