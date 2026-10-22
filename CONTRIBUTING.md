# 贡献指南

新增页面前先查 `index.md`、别名和来源索引，避免重复概念。每页使用 `wiki/TEMPLATE.md`，分配稳定且唯一的 ID；标签必须来自 `data/tags.yaml`，来源必须登记在 `sources/README.md`。

优先引用论文原文、官方技术报告、官方仓库和对应版本源码。写清来源支持的具体事实，不从论文结构直接推断目标模型运行时实现。量化影响按来源事实与推断分开记录，并给出能验证推断的检查或实验。

新增页面后运行：

```bash
python3 scripts/validate.py
python3 scripts/query.py --tag <tag>
python3 scripts/build_index.py
```

可选地用 `python3 scripts/validate.py --check-external` 检查外部链接是否可访问；网络封锁或来源站点限流时，先确认 URL 语法和来源 ID，再在可联网环境重试。
