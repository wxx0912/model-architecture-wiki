---
{
  "id": "repo-kernelwiki",
  "title": "mit-han-lab/KernelWiki（GPU kernel 知识库，用于组织方式参考）",
  "category": "doc",
  "url": "https://github.com/mit-han-lab/KernelWiki",
  "version": "b6b4301f15e8ce6955a56776690643ce5db369e6",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "README.md、SKILL.md、CLAUDE.md、index.md、references/{primer,schema,examples}.md、scripts/{query.py,get_page.py,grep_wiki.py,repo_status.py}、data/version-claims.yaml、queries/（7 个 by-*.md 索引）、artifacts/**/PROVENANCE.yaml、wiki/ 与 sources/{prs,docs,blogs,contests} 的目录划分、candidates/{cutlass,sglang,vllm,flashinfer,pytorch,deepgemm}.yaml",
  "topic": "organization",
  "excerpt_sha256": null
}
---

## 这份来源是什么

MIT HAN Lab 的 KernelWiki——一个 NVIDIA Blackwell/Hopper GPU kernel 优化的知识库，**也是本两库组织方式的参考对象**。本库登记它是知识级（`category: doc`）来源，用途是"知识库组织参考"：它示范了面向 LLM agent 检索的知识库该长什么样（分层目录、来源页与综合页分离、自动生成索引、每个证据钉 provenance）。它**不提供任何模型结构或量化算法的技术结论**。

## 已读定位与阅读形式

- 阅读形式：GitHub 仓库文件树 + `README.md`、`SKILL.md`、`CLAUDE.md`、`index.md`（`repo-tree`），按 commit `b6b4301f15e8ce6955a56776690643ce5db369e6` 读取。
- 已读定位：根目录与 `references/`、`scripts/`、`queries/`、`data/`、`artifacts/`、`wiki/`、`sources/` 的文件清单。

## 关键事实

- 仓库根**就是**一个 Claude Code skill：`SKILL.md` 在根目录，README 说明 clone 到 `~/.claude/skills/` 即自动注册；查询脚本自行解析 wiki root（或在缺失 PyYAML 时用内置纯 Python 回退），不需要环境变量或安装包。
- 目录分层：`wiki/`（综合页）与 `sources/`（上游来源页）分离；`artifacts/` 存**逐字上游资产**（PR patch、完整 kernel 文件），用 `PROVENANCE.yaml` 钉到上游 SHA；`candidates/` 是带 include/defer/exclude 决策的候选台账；`queries/` 是自动生成的 7 个交叉索引（by-architecture / problem / technique / hardware-feature / repo / kernel-type / language）。
- `data/version-claims.yaml` 是"版本声明注册表"：页面用 `version_sensitive: <id>` 指针指向集中登记，并校验双向一致。
- 检索工具三件套：`scripts/query.py`（关键词 + `--tag/--type/--architecture/--compact`，支持别名）、`scripts/get_page.py`（按 id 取页，`--follow-sources` 展开被引来源）、`scripts/grep_wiki.py`（正则搜索，`--only wiki|sources`）；`references/examples.md` 收录 10 个成品查询范式，`references/schema.md` 定义 frontmatter schema、confidence 规则与 reproducibility ladder。
- README 明确写出**证据边界**：审计过的上游 PR 语料截止 **2026-05-20**，文档/发布/榜单快照核实到 **2026-08-18 UTC**。

## 不能推出的结论

- 可借鉴的是**组织方式**（来源页/综合页分离、自动索引、逐条 provenance、证据边界声明），不是它的领域分类与结论；Blackwell/Hopper 的 hardware/technique 词表（sm100、tcgen05 等）不能搬进本库。
- README 自述的证据边界说明它**有意存在时效缺口**；引用其任何结论前须看该页自身的 provenance 与日期，不能把"库里有这页"当成"结论仍然最新"。
- 本页只读了目录结构、README/SKILL/CLAUDE/index 与工具清单，**没有逐页核对 wiki 内容质量**；"结构好"不等于"内容正确"。
- 它不是本库的技术来源：模型结构与量化算法的结论必须由论文/源码来源支撑。
