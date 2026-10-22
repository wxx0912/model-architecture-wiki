# Model Architecture Wiki

面向量化 Agent 的模型结构知识库。目标是让 Agent 在校准、敏感度分析和 PTQ 搜索前，先理解模型模块、数据流、实现差异和需要核查的结构约束。

首批覆盖 AR、DiT、dLLM、MoE、线性注意力、SSM、多模态，以及 Attention、MLP、Residual、KV Cache、mHC 等组件。量化算法本身另行建设；本仓库记录结构对量化的影响、检查项和待验证问题。

## 目录

```text
SKILL.md                         # Agent 入口、查询方法和输出契约
index.md                         # 人工导航
wiki/paradigms/                  # 模型范式
wiki/components/                 # 可复用结构组件
wiki/models/                     # 具体模型案例
wiki/checks/                     # 结构识别和量化前检查
sources/                         # 论文、官方文档、源码来源索引
data/aliases.yaml                # 中英文别名
data/tags.yaml                   # 受控标签
scripts/query.py                 # 确定性检索
scripts/get_page.py              # 按 ID 读取页面
scripts/validate.py              # 页面和引用检查
scripts/build_index.py           # 生成确定性元数据索引
.github/workflows/validate.yml   # 推送/PR 时运行校验
```

## 使用

```bash
python3 scripts/query.py "latent attention" --type component
python3 scripts/query.py --tag moe --type paradigm
python3 scripts/get_page.py component-attention
python3 scripts/validate.py
python3 scripts/build_index.py --check
```

查询结果只是候选知识。Agent 必须继续检查当前模型的配置、源码、权重命名和实际计算图，并在结构报告中区分已确认事实、来源结论、推断和未知项。

## 页面约定

每页使用 YAML frontmatter，至少包含 `id`、`title`、`type`、`tags`、`sources`、`confidence`、`evidence_scope`、`validated_at`。正文按“结构概述→数据流→实现差异→量化关联→Agent 检查→来源与证据”组织。`confidence` 只描述证据强度，不代表量化效果已经验证；`evidence_scope: mixed` 表示页面同时包含来源事实和结构推断。

## 贡献原则

优先引用论文、官方技术报告、官方源码和具体版本；不要把典型实现当成所有模型的硬约束。新增页面时补充来源 ID、相关页面和别名，并运行 `scripts/validate.py`。
