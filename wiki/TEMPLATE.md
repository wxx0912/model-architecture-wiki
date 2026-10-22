---
id: <type>-<stable-name>
title: <page title>
type: paradigm | component | model | check
tags: [controlled-tag, another-tag]
related: [page-id]
sources: [source-id]
confidence: verified | source-reported | inferred | experimental
evidence_scope: source-reported | inferred | experimental | mixed
validated_at: YYYY-MM-DD
---

# <Page title>

## 结构概述

解决什么问题、核心结构是什么。区分通用定义与具体实现。

## 结构与数据流

说明模块、张量语义/shape、计算顺序和状态边界；未知 shape 标明依赖具体配置。

## 实现差异

列出版本、配置、代码路径、融合/共享、并行和运行时差异。

## 量化关联

分开记录来源事实、结构推断和需要实验验证的假设；说明可能影响的校准轴、误差路径和后端约束。

## Agent 检查

给出可执行的源码/配置/运行图核对项，不把推断写成硬约束。

## 来源与证据

说明 sources 中的一手资料支持哪些事实。记录适用模型版本、论文章节或代码路径；目标模型仍需独立核验。

