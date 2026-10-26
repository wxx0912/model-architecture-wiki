---
{
  "id": "check-structure-identification",
  "title": "Model Structure Identification",
  "type": "check",
  "tags": [
    "quantization",
    "calibration",
    "attention",
    "moe",
    "multimodal",
    "ssm"
  ],
  "related": [
    "check-quantization-preflight",
    "component-attention",
    "component-router-expert"
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

# 模型结构识别

## 一句话理解

结构识别把"模型名称"转换成可量化的实际计算图。判定依据只能是当前 checkpoint、config、源码与运行图——类名匹配或论文架构图都不构成证据。识别的产物是带证据链接的结构图、模块表、未知项清单与量化边界。

## 判定依据

| 层次 | 需要的证据 | 不足以证明 |
| --- | --- | --- |
| 权重与配置 | checkpoint、`config.json`、权重索引、tokenizer/processor | config 字段只说明声明值，不证明运行路径 |
| 模块清单 | module tree、state_dict key、forward hook | 类名 / 目录名不证明实际分支命中 |
| 结构字段 | block 类型、attention 变体、FFN/gate、norm、residual、router、state/cache、connector、输出头 | 论文架构图不证明该 checkpoint 采用 |
| 边界与布局 | 每个边界的输入/输出 shape、dtype、布局、共享/融合关系、并行切分 | 静态 shape 不证明运行时数值 |
| 运行确认 | 最小输入的 reference forward | 单次 forward 不覆盖全部路径 |

`data/preflight.schema.json` 的 `evidence.classification` 定义 7 种取值：`source-reported`、`source-confirmed`、`config-confirmed`、`state-dict-confirmed`、`forward-confirmed`、`inferred`、`unknown`；`--quantization-view` 把运行时证据分为 `config-confirmed` / `state-dict-confirmed` / `forward-confirmed` / `inferred` 四级。它们是证据层级，不能被改写成量化效果结论。

## 容易误判的结构

fused QKV、fused gate/up 与拆分权重；GQA/MLA cache 与普通 MHA cache；linear-attention/SSM 的 state 与 KV cache；MoE 的 shared/routed expert、专家布局与 token 重排；DiT 的 timestep/adaLN 与 dLLM 的 mask/step；tied embedding/LM head 与多模态 connector；mHC/Hyper-Connections 等跨层 residual mixing。这些差别决定后续量化边界与校准轴，不能靠模型系列名默认。

未能从源码或运行图确认的字段记为 `unknown`，不用模型系列的默认值填充。

## Profile 辅助

`scripts/preflight.py <profile-id> --config <checkpoint>/config.json --checkpoint <name-or-path> --source-revision <commit-or-tag>` 以指定 profile 为起点（profile 注册表已冻结、只做页面路由：省略 `profile-id` 时仅在已登记 profile 内按 `model_type` 与 config 签名匹配，未命中或多义即报错，须显式指定 `profile-id`，并用 `--list` 核对；不要新增 profile 适配新模型），只把配置文件直接给出的字段标为 `config-confirmed`；由 head 数关系推导的 MHA/GQA 与由专家字段推导的 MoE 标为 `inferred`。`state-dict-confirmed` 与 `forward-confirmed` 分别由 `--state-dict-keys` 与 `--forward-manifest` 产生，只给 `--config` 时记录不会出现这两级。

## 结果边界

- config 字段不能证明运行时代码路径，仍须核对加载类、权重 key 与 reference forward；推断出的量化风险都需要实验验证。
- `--strict-runtime` 通过只说明结构证据齐全，不说明量化精度或部署性能已达标。
- 本库未对任何 checkpoint 运行 forward 或量化，本页只给出判定框架，不提供某模型的实测结构。

## Agent 如何使用

首轮读本页即可知道"识别一个 checkpoint 需要哪些证据、哪些字段属于哪一级证据"。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 各组件的结构事实（attention 变体、FFN/gate、norm、residual、router、cache、connector） | [Components](../components/attention.md) 与同库各 component 页 |
| 各范式的计算图与动态轴（AR / DiT / dLLM / MoE / SSM / Multimodal） | [Paradigms](../paradigms/ar.md) 与同库各 paradigm 页 |
| 记录字段与候选排除判据 | [量化前检查](quantization-preflight.md) |
| 目标 checkpoint 的真实结构 | 该 checkpoint 的 `config.json`、`modeling_*.py`、state_dict 与运行时 forward（本库不提供） |

回答结构识别问题时必须说明用的是哪一级证据，并区分 config 声明值、state_dict/源码事实与运行时观测。

## 来源与证据

- [huggingface/transformers](../../sources/repos/repo-huggingface-transformers.md)：模型实现落点、类名、config 字段与 cache/量化接口入口。
- [mit-han-lab/KernelWiki](../../sources/docs/repo-kernelwiki.md)：来源页/综合页分离、逐条 provenance 与证据边界的组织参考。
- [`data/preflight.schema.json`](../../data/preflight.schema.json)、[`scripts/preflight.py`](../../scripts/preflight.py)、[`data/architecture-profiles.json`](../../data/architecture-profiles.json)：本页工具行为的契约与实现。

本页是 `inferred` 级的方法页（工程归纳），不提供任何模型的实测结构结论。
