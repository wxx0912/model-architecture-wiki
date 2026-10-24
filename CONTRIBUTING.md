# 内容与维护约定

新增页面前先查 `index.md`、别名展开表和来源层，避免重复概念。每页使用 `wiki/TEMPLATE.md`，分配稳定且唯一的 ID；标签必须来自 `data/vocabulary.yaml`，来源必须登记为 `sources/` 下的来源页（规范见 `sources/SPEC.md`）。

优先引用论文原文、官方技术报告、官方仓库和对应版本源码。写清来源支持的具体事实，不从论文结构直接推断目标模型运行时实现。量化影响按来源事实与推断分开记录，并给出能验证推断的检查或实验。

`updated` 只表示页面最近编辑日期。不要把页面级 `confidence` 或 `evidence_scope` 当成章节级结论：来源报告、配置确认、结构推断和待实验验证的假设必须分别表述。

新增或修改页面后运行：

```bash
python3 scripts/validate.py
python3 scripts/build_index.py --check
python3 -m unittest discover -s tests -v
python3 scripts/check_shared.py --peer ../quantization-algorithm-wiki
```

可选地用 `python3 scripts/validate.py --check-external --proxy <proxy>` 检查外部链接是否可访问；URL 可达不代表内容已复核，不可达也不代表来源无效。
