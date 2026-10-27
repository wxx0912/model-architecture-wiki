---
name: model-architecture-wiki
description: 查询模型范式、结构组件、数据流及其量化关联，为理解新模型和核查结构提供可追溯知识。
argument-hint: "[关键词] | [--tag 标签 --type 类型] | [page-id]"
---

# Model Architecture Wiki

## 何时使用

需要理解模型结构、组件关系、实现差异或结构对量化的影响时使用。本库提供知识与核查线索；目标模型确认、算法选择和实验编排由调用方完成。

## 如何查询

以下命令在本 Skill 所在仓库根目录执行；路径相对于本文件。先搜具体模型，未命中或覆盖不足时继续查范式与组件；命中模型页后也可沿相关链接补读组件。检查页用于了解需要核实什么。

```bash
python3 scripts/query.py "LLaDA"
python3 scripts/query.py --tag dllm --type paradigm
python3 scripts/get_page.py model-llada
python3 scripts/get_page.py model-llada --follow-sources
```

检索先匹配 id、标题、aliases 和 tags，无结果才搜索正文并标为弱相关。多词采用 AND；长问题先提炼为短关键词。`--tag` 按词表筛选，忽略大小写；`--type` 限定页面类型，`--limit` 控制数量，`--json` 提供结构化结果。可用分类见 [生成索引](index.generated.md)。未命中时换关键词或标签；仍无相关资料则说明知识缺口，不为命中而新增或强套 profile。

先读相关 Wiki 页；需要精确规格、实现细节或证据时，再用 `--follow-sources` 展开登记的本地来源记录。它不会下载原文，也不会自动展开所有 related 页面。

需要整理具体模型的结构证据时，可选用 `preflight.py`：`python3 scripts/preflight.py --list` 列出已有 profile，`python3 scripts/preflight.py llada-family` 读取对应模板。profile 是位置参数，不是查库前提；没有合适的 profile 仍可正常检索。输入 config、权重名与外部 forward 记录的用法见 [README](README.md)。工具不自动采集 forward，输出的检查项及 `not_assessed`/`unknown` 不代表已确认约束或没有约束。

## 如何理解知识

- 来源记录保留版本、阅读定位和“不能推出的结论”；Wiki 中的结构事实、实现事实、量化推断与未知项分别理解。页面级 confidence/evidence_scope 不能代替具体结论的证据。
- 论文规格与典型实现不能直接当成当前 checkpoint 的事实。config 提供初步结构线索，权重名不能证明实际执行图，forward 观测只覆盖相应输入与路径。
- 结构风险是待核实线索，不等于已测得的敏感层、格式限制或后端支持。知识命中和 preflight 通过都不能证明量化效果。

## 如何回答

按当前问题说明相关结构知识、必要来源与适用边界。涉及具体模型时，区分已有证据与待核实项；无需每次生成完整结构报告或量化计划。

新增或修改知识时读 [AUTHORING](AUTHORING.md)；索引、校验与工具参数见 [README](README.md)，日常查询不需要运行维护流程。
