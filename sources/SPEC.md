# 来源页规范

本文件定义 `sources/` 下来源页的目录、字段、正文结构和校验规则。规范同时适用于
`quantization-algorithm-wiki` 与 `model-architecture-wiki`；两库使用同一套字段名与共享枚举
取值，序列化格式统一为 JSON frontmatter，见第 7 节与根目录 `CONVENTIONS.md` 第 2.1 节。

现状差异说明：按内容主题登记来源的四张表格（`core`、`reconstruction`、`transforms`、
`mixed-architecture`）将被一来源一文件的来源页取代，表格不再作为来源信息的维护位置。
主题分组不取消，改为来源页的 `topic` 字段加生成的索引；主题级说明移入 `sources/topics/`。

本规范只解决“来源怎么登记、证据怎么复核”。它不判断算法结论是否正确，也不替代正文页面。

## 1. 目录结构

```text
sources/
├── SPEC.md                  # 本规范
├── README.md                # 人工简介：入口指向 SPEC 与 INDEX，不含来源表格
├── INDEX.generated.md       # 生成物：按 topic 分组、含引用计数，不可手改
├── topics/<topic>.md        # 主题级说明与边界，人工维护
├── papers/<id>.md           # category=paper
├── repos/<id>.md            # category=repo
├── docs/<id>.md             # category=doc
└── reports/<id>.md          # category=report
```

文件名必须等于来源 `id`，例如 `paper-gptq` 对应 `sources/papers/paper-gptq.md`。

既有 ID 一律不改。页面 `sources` 字段已经引用的 ID 保持不变，避免破坏跨页面引用；
本规范只改变登记的载体，不改变 ID 命名。历史遗留的双 ID（同一篇论文分别登记 HTML 与
LaTeX，如 `paper-hawq2` 与 `source-hawq2`）允许保留，不做合并。

## 2. 字段

必需字段必须全部存在，标记为可选的字段允许省略；无论必需或可选，都不允许出现表外字段。

本表首列不使用反引号：`sources/` 下以竖线加反引号开头的行会被当作来源表格行处理，见 S11。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | 必需 string | 稳定 ID，小写英文加连字符，与文件名相同，全库唯一 |
| title | 必需 string | 来源标题。论文用正式标题，代码库用 owner/repo 加用途简述 |
| category | 必需 string | 受控：paper、repo、doc、report |
| url | 必需 string | 可定位入口 URL，必须以 http:// 或 https:// 开头 |
| version | 必需 string | 论文版本号、commit SHA、tag 或 revision；未知写字符串 unknown |
| version_kind | 必需 string | 受控：paper-version、commit、tag、revision、unknown |
| retrieved_at | 必需 string | 实际阅读日期，YYYY-MM-DD |
| read_form | 必需 array | 受控数组：latex、html、pdf、repo-tree |
| read_locator | 必需 string | 已读定位：章节、公式、表格、文件路径、函数或类名 |
| topic | 必需 string | 取值见本仓库 `data/vocabulary.yaml` 的 `source_topics` |
| requires_external_verification | 可选 bool | true 表示复核该来源必须联网或进入内网，默认 false |
| excerpt_sha256 | 可选 string | 正文逐字引用片段的 SHA-256；无片段时为 null |

字段语义说明：

- `retrieved_at` 是“实际读取这份来源的日期”，不是页面编辑日期，两者不得混用。
- `version` 钉住被读取的具体版本。论文记 arXiv 版本号或会议版本，代码记完整 commit 或
  revision。浮动分支只能作为 `url` 入口，不能写进 `version`。
- `read_form` 记录实际读取形式，用于区分“读了 HTML 正文”和“读了 LaTeX 源”。
  不得事后把 HTML 阅读改写成 LaTeX 阅读。
- `requires_external_verification` 为 `true` 时，无网络环境下该来源的结论必须被当作缺口，
  不得当作已确认事实使用。

## 3. 正文

正文必须依次包含以下四个二级标题，标题文字固定。

```markdown
## 这份来源是什么

## 已读定位与阅读形式

## 关键事实

## 不能推出的结论
```

- `## 这份来源是什么`：来源的性质、适用模型或范围、与相邻来源的关系。
- `## 已读定位与阅读形式`：与 `read_locator` 一致的分条说明，标明实际读取形式。
- `## 关键事实`：正文结论实际依赖的事实，逐条给出可定位引用（章节、公式、表格、函数）。
- `## 不能推出的结论`：不得为空。写明该来源不能支持的推广，例如实验条件不可外推、
  论文范围与当前实现不同、只有静态代码证据而没有运行证据。

只复述必要公式与短片段，不把整篇论文或上游代码复制进仓库。

## 4. 逐字引用片段

需要逐字证据时，在 `## 关键事实` 内使用哨兵包裹片段：

```markdown
<!-- excerpt-start -->
逐字引用内容
<!-- excerpt-end -->
```

全文允许 0 或 1 对哨兵。使用哨兵时，把哨兵之间的内容（不含哨兵行本身）按 UTF-8 编码
计算 SHA-256，小写十六进制写入 `excerpt_sha256`；不使用哨兵时该字段必须是 `null`。
片段必须与上游一致，改动片段即改动该字段。

## 5. 校验规则

两库的 `scripts/validate.py` 与共享库 `scripts/wiki_common.py` 须实现以下规则，编号即校验输出的前缀。

| 规则 | 内容 | 级别 |
| --- | --- | --- |
| S1 | 来源页位于 sources/ 下对应 category 目录，文件名等于 id | error |
| S2 | 必需字段齐全，且不出现第 2 节之外的字段 | error |
| S3 | category、version_kind、topic、read_form 元素均在受控词表内 | error |
| S4 | retrieved_at 为可解析的 YYYY-MM-DD | error |
| S5 | 四个二级标题齐全，且“不能推出的结论”一节内容非空 | error |
| S6 | 哨兵对数量为 0 或 1；为 1 时片段哈希等于 excerpt_sha256；为 0 时该字段为 null | error |
| S7 | url 以 http:// 或 https:// 开头 | error |
| S8 | 页面 sources 引用的每个 ID 都必须存在对应来源页 | error |
| S9 | 来源页 ID 全库唯一 | error |
| S10 | 无页面引用的来源页（引用计数为 0） | warning |
| S11 | sources/ 下非生成文件中出现以竖线加反引号开头的行 | error |

S10 为警告，不阻断校验；未采用的来源应显式留档为备查，而不是删除。

## 6. 其他约束

- 来源页不重复正文页面的结论。正文页承载算法或结构的解释，来源页承载“读到了什么、
  这份来源不能支持什么”。同一论断只在正文页维护一处。
- 论文、代码、配置、文档分别登记，不合并成一个条目。
- `retrieved_at` 只增不改。发现旧日期有误时改日期，并在 `sources/MIGRATION-AUDIT.md` 记明，
  不静默覆盖。
- 来源可达性检查（`validate.py --check-external`）与内容复核是两件事；URL 可达不表示
  页面内容已逐条核对。

## 7. 与组织规范的关系

来源层的字段名、取值词表和校验规则两库一致，与仓库骨架、页面契约、工具接口一并受
根目录 `CONVENTIONS.md` 约束。两份文件在两库之间必须逐字节一致，由 CI 校验。

本仓库的来源数量、来源主题取值和页面类型取值见 `data/vocabulary.yaml`，不写入本文件。

## 8. 迁移约定

从表格迁移到来源页时：

1. `id`、`url`、`version`、`read_locator` 从原表格直接搬运。
2. 原“类型”列拆成两个字段：资料类型写入 `category`（论文与会议论文为 `paper`，源码为
   `repo`，知识库与官方文档为 `doc`），阅读形式写入 `read_form`
   （原始论文 HTML 为 `html`，原始 LaTeX 为 `latex`，原始会议论文为 `pdf`，代码库为
   `repo-tree`）。
3. `topic` 取原表格所在文件的主题。
4. `retrieved_at` 按第 9 节口径回填。
5. `title` 需要新写；论文用正式标题，代码库用 owner/repo 加用途。
6. “不能推出的结论”需要新写，原表格文件尾部的边界说明可作为素材。

## 9. retrieved_at 回溯口径

历史来源的阅读日期从会话日志与仓库内既有记录回溯：

- 会话日志（`~/.codex/sessions/`）中的抓取命令带 UTC 时间戳，可定位到分钟，是主要依据。
- 仓库内既有记录作为交叉验证，例如迁移前 `sources/transforms.md` 的“阅读日期：2026-10-23”、
  迁移前 `sources/mixed-architecture.md` 的“核实日期：2026-10-23”。
- 日期写入仓库时统一按 Asia/Shanghai 本地日期，页面只写日期，不写时间戳。

不得填未经依据的日期。新增来源在抓取当场记录日期，不回填。

会话日志不在仓库内，因此回溯的逐条依据必须落盘为 `sources/MIGRATION-AUDIT.md`，否则
“已读”在日志清理后再次不可考。

## 10. 变更流程

新增或修改来源页后运行：

```bash
python3 scripts/validate.py
python3 scripts/build_index.py --check
```

修改本规范时，必须同步更新实现校验的脚本和本节命令。
