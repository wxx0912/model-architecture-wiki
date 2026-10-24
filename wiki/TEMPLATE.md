# 页面模板

以下头部以真实字段替换后放在页面最开始。`related`、`prerequisites` 只填写已存在的页面 ID，`sources` 必须登记对应来源页；`confidence` 与 `evidence_scope` 取值见 `CONVENTIONS.md` 第 2.5 节。

```json
{
  "id": "<type>-<stable-name>",
  "title": "页面标题",
  "type": "paradigm",
  "tags": ["controlled-tag"],
  "aliases": ["中文别名"],
  "related": ["other-page"],
  "sources": ["paper-name"],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-23"
}
```

## 结构概述

解决什么问题、核心结构是什么。区分通用定义与具体实现。

## 结构与数据流

说明模块、张量语义/shape、计算顺序和状态边界；未知 shape 标明依赖具体配置。

## 实现差异

列出版本、配置、代码路径、融合/共享、并行和运行时差异。

## 量化关联

分开记录来源事实、配置确认（`config-confirmed`）、结构推断和需要实验验证的假设；说明可能影响的校准轴、误差路径和后端约束。

## Agent 检查

给出可执行的源码/配置/运行图核对项，不把推断写成硬约束。

## 来源与证据

说明 sources 中的一手资料支持哪些事实。记录适用模型版本、论文章节或代码路径；目标模型仍需独立核验。

## 结构档案（可选）

代表性结构族登记在 `data/architecture-profiles.json`。档案只保存结构特征和证据入口；`checkpoint`、`model_type`、`architectures`、代码 revision 未锁定时必须写 `unknown`。目标模型的完整 preflight 应由 `scripts/preflight.py` 生成并补齐运行时证据。

