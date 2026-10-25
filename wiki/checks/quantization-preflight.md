---
{
  "id": "check-quantization-preflight",
  "title": "Quantization Preflight",
  "type": "check",
  "tags": [
    "quantization",
    "calibration",
    "deployment",
    "quantization-preflight"
  ],
  "related": [
    "check-structure-identification",
    "check-calibration-axes",
    "check-deployment-contract"
  ],
  "sources": [
    "repo-huggingface-transformers",
    "repo-kernelwiki"
  ],
  "confidence": "inferred",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-24"
}
---

# 量化前检查

## 一句话理解

量化前检查是结构识别与量化搜索之间的一份固定结构记录：它说明目标模型"是什么结构、证据到哪一级、还有哪些字段必须回运行时确认"。它是前置记录，不是量化结论，也不声称可量化边界已由本工具生成。

## 记录字段

按 `data/preflight.schema.json`（`schema_version = 1`），顶层必填 `schema_version`、`profile_id`、`knowledge_page`、`model`、`implementation`、`runtime`、`structure`、`evidence`、`checklist`、`required_runtime_confirmation`；其中四组主体字段为：

- `model`：`name`、`checkpoint`、`model_type`、`architectures`，可选 `num_hidden_layers`、`hidden_size`、`attention_layout`、`moe_layout`、`parameter_sharing`、`special_paths`；
- `implementation`：`repository`、`revision`；
- `runtime`：`source_revision`、`device`、`dtype`、`input_signature`；
- `structure`：`features`、层数/头数/专家数等字段、`attention_layout`、`moe_layout`、`module_dependencies`、`parameter_sharing`、`special_paths`、`tensor_flow`。

另含 `evidence`（每条带证据层级）、`checklist` 与 `required_runtime_confirmation`。

## 证据层级

`evidence.classification` 定义 7 种取值：`source-reported`、`source-confirmed`、`config-confirmed`、`state-dict-confirmed`、`forward-confirmed`、`inferred`、`unknown`；compact 视图把运行时证据分为 `config-confirmed` / `state-dict-confirmed` / `forward-confirmed` / `inferred` 四级。它们表示证据来源，不能改写成量化效果或性能结论。

## 候选排除判据

以下条件会否定一个量化候选（本库方法归纳，`preflight.py` 未实现自动过滤）：模块维度不满足 group/pack 要求；后端不支持该 dtype/scheme；fused 权重无法拆分；共享权重被分配冲突配置；状态/cache 无对应 kernel；导出后无法回载或模型路径被改变。这些判据依赖目标 checkpoint、实现版本与部署后端，在它们确认之前不得据此过滤量化格式或模块。

## 后续阶段字段

compact 视图含 `quantization_checks`、`quantization_constraints` 与 `runtime_gaps`：`quantization_checks` 是待执行的风险检查建议，不是已确认约束；`quantization_constraints.status = not_assessed` 或其中的 `unknown` 不能解释成"没有约束"。可量化/暂不量化边界、部署约束与校准轴属后续量化与部署契约阶段，需要从结构记录中识别其待补字段，并不由本工具生成。

## 结果边界

- `--strict-runtime` 通过只表示 checkpoint、源码版本、关键结构字段、runtime 元数据（device/dtype/input_signature）以及 state_dict/reference-forward（含 shape/dtype/layout 元数据）证据齐全；它不表示候选在部署接口上可执行，也不表示量化精度达标。
- 精度、性能、显存与端到端质量仍需实验验证；本库未运行任何量化或评测。

## Agent 如何使用

首轮读本页即可知道"进入量化搜索前，结构记录需要哪些字段、每个字段属于哪一级证据"。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 如何得到模块清单与结构字段 | [模型结构识别](structure-identification.md) |
| 校准轴候选与采样要求 | [按架构组织校准轴](calibration-axes.md) |
| 部署契约字段与验证顺序 | [量化部署契约](deployment-contract.md) |
| 记录 schema 与校验脚本 | [`data/preflight.schema.json`](../../data/preflight.schema.json)、[`scripts/preflight.py`](../../scripts/preflight.py) |
| 目标候选在真实后端的可执行性与精度 | 目标模型源码 / 导出器 / 运行时的独立验证（本库不提供） |

回答量化前检查问题时必须说明证据层级，并区分"结构证据齐全"与"量化效果已验证"。

## 来源与证据

- [huggingface/transformers](../../sources/repos/repo-huggingface-transformers.md)：模型实现、config 字段与量化/cache 接口入口。
- [mit-han-lab/KernelWiki](../../sources/docs/repo-kernelwiki.md)：来源页/综合页分离、逐条 provenance 与证据边界的组织参考。

本页是 `inferred` 级的方法页（工程归纳）；`preflight.schema.json` 与本库脚本是本仓库自身的工具契约，不是模型结论。
