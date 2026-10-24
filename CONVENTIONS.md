# 组织规范

本文件规定知识库仓库的**组织形式**：目录骨架、页面契约、来源层、词表与 schema、工具
接口、领域差异落点，以及共享文件的一致性校验。它同时适用于
`quantization-algorithm-wiki` 与 `model-architecture-wiki`。

本文件只规定形式，不规定内容。两库的内容不同是预期结果：结构库解释模型结构与数据流，
算法库解释量化算法、量化约束、结构适用性与注意点。两库要让使用它们的 Agent 读一次就能
理解“这个库是干什么的、该怎么查、结果怎么用”，因此形式必须一致。

**本文件在两个仓库之间必须逐字节一致**，由 CI 校验，见第 8 节。仓库特有的信息（来源数量、
页面类型取值、标签）写在各自的 `data/vocabulary.yaml` 与 `README.md`，不得写进本文件。

来源页的细则见[来源页规范](sources/SPEC.md)，同为跨库共享文件。

## 1. 仓库骨架

两库使用同一套骨架。允许的差异只有第 7 节列出的落点。

```text
<wiki>/
├── SKILL.md                  # Agent 入口：何时使用、怎么查、输出契约
├── README.md                 # 人读：定位、覆盖范围、安装与检索命令
├── CONVENTIONS.md            # 本文件，共享
├── CONTRIBUTING.md           # 人读：内容维护约定
├── index.md                  # 人工导航与知识地图
├── index.generated.md        # 生成物，不可手改
├── data/
│   ├── vocabulary.yaml       # 受控词表，本仓库特有
│   ├── schemas.yaml          # 字段契约，校验器消费，本仓库特有
│   └── <domain>.json         # 领域机器可读事实源（可选）
├── wiki/
│   ├── TEMPLATE.md           # 页面模板
│   └── <页面>.md
├── sources/
│   ├── SPEC.md               # 来源页规范，共享
│   ├── README.md
│   ├── INDEX.generated.md    # 生成物
│   ├── topics/<topic>.md
│   └── <category>/<id>.md
├── scripts/
│   ├── wiki_common.py        # 共享库：读取、校验、检索的公共实现
│   ├── query.py              # 共享
│   ├── get_page.py           # 共享
│   ├── validate.py           # 共享
│   ├── build_index.py        # 共享
│   ├── check_shared.py       # 共享：共享文件一致性校验
│   └── <domain>.py           # 领域特有脚本
└── tests/
    ├── test_shared_tools.py  # 共享
    └── test_<domain>.py      # 领域特有
```

## 2. 页面契约

### 2.1 序列化

页面元数据使用 `---` 分隔的 **JSON** frontmatter，用标准库严格解析：禁止重复键，禁止
注释，禁止 JSON 之外的写法。不使用 YAML：自制 YAML 子集解析器无法表达嵌套结构，会让
“同一件事有第二套读法”。

```json
{
  "id": "example-page",
  "title": "页面标题",
  "type": "example",
  "tags": ["tag-a"],
  "aliases": ["别名"],
  "related": ["other-page"],
  "sources": ["paper-xxx"],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-23"
}
```

### 2.2 字段

必需字段必须全部存在；可选字段允许省略；无论必需或可选，都不允许出现表外字段。
字段集合由 `data/schemas.yaml` 声明，校验器不得硬编码枚举。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | 必需 string | 稳定标识，见 2.3 |
| title | 必需 string | 页面标题 |
| type | 必需 string | 页面类型，取值由 `data/vocabulary.yaml` 声明 |
| tags | 必需 array | 机制标签，元素属受控词表 |
| aliases | 必需 array | 该页的其他叫法，可为空数组 |
| related | 必需 array | 相关页面 ID，必须存在 |
| sources | 必需 array | 来源 ID，必须存在对应来源页 |
| confidence | 必需 string | 页面证据强度，见 2.5 |
| evidence_scope | 必需 string | 页面证据范围，见 2.5 |
| updated | 必需 string | 页面最近编辑日期，见 2.4 |
| structure_profile | 可选 string | 指向 `data/<domain>.json` 中的档案 ID |
| prerequisites | 可选 array | 阅读顺序上的前置页面 ID |
| version_sensitive | 可选 string | 指向版本敏感声明注册表的条目 ID |

### 2.3 标识

- 页面 ID 形如 `<type>-<slug>`，`slug` 为小写英文加连字符，正则 `^[a-z][a-z0-9-]+$`。
- 页面 ID 全库唯一。**更新标题不改变 ID。**
- 来源 ID 规则见[来源页规范](sources/SPEC.md)。
- 跨库引用使用完整 ID，不省略前缀。

### 2.4 日期

- `updated` 是**页面编辑日期**，不表示内容经过实测，也不表示证据已复核。
- 证据的时效由来源层的 `retrieved_at` 承担，页面不重复记录。
- 不设 `validated_at`：它与 `updated` 在语义上重叠，保留两个会让 Agent 误判。

### 2.5 证据强度

`confidence` 与 `evidence_scope` 取值属共享词表，两库取值一致：

| 字段 | 取值 | 含义 |
| --- | --- | --- |
| confidence | `verified` | 有官方文档或固定版本源码直接支持 |
| confidence | `source-reported` | 由权威来源报告，未在目标实现上核对 |
| confidence | `inferred` | 由多来源综合推导 |
| confidence | `experimental` | 未文档化的经验或技巧 |
| evidence_scope | `source-reported` | 正文以来源报告为主 |
| evidence_scope | `inferred` | 正文以推导为主 |
| evidence_scope | `experimental` | 正文以实验观察为主 |
| evidence_scope | `mixed` | 正文同时含来源事实与推导，须按章节继续区分 |

页面级 `confidence` 只描述整体证据强度，不得用它覆盖章节级的强弱差异。论文报告、源码
核对、项目实测和迁移假设必须分别表述，不合并成单一标签。

### 2.6 正文

正文以一级标题开头，随后按该库 `wiki/TEMPLATE.md` 定义的小节组织。**两库的正文小节可以
不同**，因为知识单元不同；但必须保证同一库内所有页面遵循同一模板，且模板中每一节都有
明确职责说明。

## 3. 来源层

来源页规范见[来源页规范](sources/SPEC.md)。要点：一来源一文件，`sources/<category>/<id>.md`，
frontmatter 记录版本锚与 `retrieved_at`，正文包含“不能推出的结论”，逐字引用片段带哈希锚。

来源层规则编号使用 `S` 前缀，页面规则使用 `P` 前缀，跨库一致性规则使用 `C` 前缀。

## 4. 词表与 schema

### 4.1 分工

| 文件 | 内容 | 差异 |
| --- | --- | --- |
| `data/vocabulary.yaml` | 页面类型、标签、检索别名展开表 | 本仓库特有 |
| `data/schemas.yaml` | 页面与来源页的字段契约 | 本仓库特有 |
| `scripts/wiki_common.py` | 共享枚举（证据强度、来源类别、阅读形式）与读取、校验、检索实现 | 共享 |

共享枚举定义在共享代码里，仓库的 `vocabulary.yaml` 只能扩展仓库特有部分，不得重新定义
证据强度等跨库语义。

### 4.2 校验器不得硬编码枚举

`query.py`、`get_page.py`、`validate.py`、`build_index.py` 的取值判断必须来自
`data/vocabulary.yaml`、`data/schemas.yaml` 或共享枚举。新增页面类型或标签时只改数据文件。

## 5. 工具接口

两库提供同一套命令与参数。公共选项必须同名同义，输出结构必须一致。

```bash
python3 scripts/query.py "<关键词>" [--tag 标签] [--type 类型] [--limit N] [--json]
python3 scripts/query.py --alias <别名键> [--tag 标签]
python3 scripts/get_page.py <page-id> [--follow-sources] [--json]
python3 scripts/validate.py [--check-external [--proxy URL]]
python3 scripts/build_index.py [--check]
```

约定：

- 检索是词面匹配，不是语义问答；空格分隔的多个词按 AND 处理。多词命中逻辑两库一致。
- `query.py --json` 输出对象数组，元素至少含 `id`、`title`、`path`、`score`。
- `get_page.py --follow-sources` 展开 `sources` 字段对应的**本地来源页正文**，使离线环境
  下不依赖外网；末尾附来源 URL 供人工核对。
- `validate.py` 的 `--check-external` 只检查来源 URL 可达性，默认关闭；可达不等于内容已
  复核。
- 领域特有脚本单独命名（例如结构库的 `preflight.py`），并在本库 `SKILL.md` 中登记。
- 领域特有的查询选项要在 `SKILL.md` 中登记选项名与语义；未实现的库遇到该选项时返回
  明确错误，不得静默忽略。结构库的 `--profile`、`--feature`、`--evidence` 由共享
  `query.py` 读取 `data/architecture-profiles.json` 实现；该文件缺失时这三个选项必须报错。

## 6. 领域差异落点

两库**只允许**在以下位置不同：

1. `wiki/` 下的页面内容与页面类型取值；
2. `data/vocabulary.yaml` 的页面类型、标签、别名展开表；
3. `data/<domain>.json` 的领域事实源（可选；结构库为结构档案）；
4. `scripts/<domain>.py` 与 `tests/test_<domain>.py` 领域脚本与测试；
5. `README.md`、`index.md`、`SKILL.md`、`CONTRIBUTING.md` 的内容；
6. `sources/topics/` 与来源页内容。

其余文件必须逐字节一致。

## 7. Agent 入口

两库的 `SKILL.md` 使用同一骨架，段落顺序固定：

1. `何时使用`：本库回答什么问题，不回答什么问题；
2. `查询路径`：可直接执行的命令；
3. `证据与边界`：证据分级、停止条件、不得越界的结论；
4. `输出契约`：回答时应给出的条目。

Agent 必须能从 `SKILL.md` 单独判断“这个库是干什么的、该不该用、用完得到什么”。

## 8. 共享文件与一致性校验

### 8.1 共享文件清单

```text
CONVENTIONS.md
sources/SPEC.md
scripts/wiki_common.py
scripts/query.py
scripts/get_page.py
scripts/validate.py
scripts/build_index.py
scripts/check_shared.py
tests/test_shared_tools.py
```

### 8.2 校验方式

`scripts/check_shared.py` 计算本仓库共享文件的 SHA-256，与对方仓库的同一清单比对。CI 中
对方仓库通过公开地址拉取；本地通过 `--peer <路径>` 比对。

```bash
python3 scripts/check_shared.py --peer ../model-architecture-wiki
```

不一致时失败并逐文件列出差异。修改共享文件必须同时更新两个仓库，否则 CI 阻断。

## 9. 校验规则编号

| 前缀 | 范围 | 定义位置 |
| --- | --- | --- |
| P | 页面契约与页面间引用 | 本文件第 2 节 |
| S | 来源页契约 | 来源页规范 |
| C | 跨库一致性与共享文件 | 本文件第 8 节 |

规则编号是稳定引用点：校验失败时输出编号，便于定位条款。

## 10. 变更流程

```bash
python3 scripts/validate.py
python3 scripts/build_index.py --check
python3 -m unittest discover -s tests -v
python3 scripts/check_shared.py --peer <另一仓库路径>
```

修改共享文件时，必须同步更新两个仓库，并同时更新本文件与[来源页规范](sources/SPEC.md)中的
对应条款；规范、校验实现和测试三者不得不同步。
