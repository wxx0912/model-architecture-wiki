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
    "paper-mamba2"
  ],
  "confidence": "inferred",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-23"
}
---

# 按架构组织校准轴

## 核心原则

校准样本不只由文本/图像内容定义，还要覆盖会改变数值分布或离散路径的运行时轴。Agent 应先从结构页生成轴的候选，再根据目标模型和预算选择分层采样。

## 建议轴

| 架构 | 需要显式记录的轴 |
|---|---|
| AR | prefill/decode、序列长度、batch、位置、cache growth、模态组合 |
| DiT | timestep/noise、采样阶段、分辨率、空间/时间 token、guidance |
| dLLM | mask ratio、迭代 step、更新 block、置信度排序、重掩码 |
| MoE | token domain、router top-k、专家访问频率、capacity、并行布局 |
| Linear/SSM | state 初值、序列长度、chunk 边界、reset、state dtype |
| Multimodal | 模态、分辨率、帧数/时长、connector 输出、缺失模态 |

## 采样与记录

每个样本应携带轴的 metadata，并保存代表性覆盖率、极端值和未覆盖区域。不能以少量平均 activation 统计证明所有轴已覆盖；对未覆盖轴明确降级为未知风险。

## 验收

校准集通过覆盖审计后，再进入算法/配置搜索。搜索结果必须回指使用过的轴和样本版本，以便复现和比较。
