---
name: model-architecture-wiki
description: 查询 AR、DiT、dLLM、MoE、线性注意力、SSM、多模态模型及其结构组件，为量化前的模型结构识别和约束分析提供可追溯资料。
argument-hint: "[自然语言问题] | [--tag 标签 --type 类型] | [page-id]"
---

# Model Architecture Wiki

## 何时使用

当任务需要识别模型范式、模块组成、张量数据流、特殊结构组件，或判断某个结构可能带来的量化检查项时使用。它不替代实际模型源码检查、数值敏感度分析或 PTQ 执行。

## 查询路径

```bash
python3 scripts/query.py "<关键词>" --limit 10
python3 scripts/query.py --tag <标签> --type <paradigm|component|model|check>
python3 scripts/get_page.py <page-id>
python3 scripts/validate.py
```

先用关键词、标签和别名缩小范围，再读取具体页面。模型页面用于确认整体组合，组件页面用于理解共性，check 页面用于生成结构核查清单。页面级 `confidence` 是证据总体强度；`evidence_scope: mixed` 表示正文同时含来源事实和结构推断，必须按章节标签继续区分，不能把量化推断当成已验证效果。

## 输出契约

回答结构问题时应说明：

1. 页面支持的结构事实及来源；
2. 当前目标模型仍需从配置、源码、权重名或运行图确认的内容；
3. 可能影响量化的模块、数据分布、误差传播和部署接口；
4. 不应直接当作硬约束的推断或经验。

`confidence` 分为 `verified`、`source-reported`、`inferred`、`experimental`；`evidence_scope` 标记 `source-reported`、`inferred`、`experimental` 或 `mixed`。不要把 `source-reported` 或 `inferred` 改述成已在目标模型上运行验证。
