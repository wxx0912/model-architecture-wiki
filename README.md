# Model Architecture Wiki

面向量化 Agent 的模型结构知识库。目标是让 Agent 在校准、敏感度分析和 PTQ 搜索前，先理解模型模块、数据流、实现差异和需要核查的结构约束。

首批覆盖 AR、DiT、dLLM、MoE、线性注意力、SSM、多模态，以及 Attention、MLP、Residual、KV Cache、mHC 等组件。量化算法本身由 [Quantization Algorithm Wiki](https://github.com/wxx0912/quantization-algorithm-wiki) 建设；本仓库记录结构对量化的影响、检查项和待验证问题。

## 目录

```text
SKILL.md                         # Agent 入口：何时使用、查询路径、证据边界、输出契约
index.md                         # 人工导航与知识关系
index.generated.md               # 生成物，不可手改
CONVENTIONS.md                   # 组织规范，跨库共享
CONTRIBUTING.md                  # 内容维护约定
wiki/TEMPLATE.md                 # 页面模板
wiki/paradigms/                  # 模型范式
wiki/components/                 # 可复用结构组件
wiki/models/                     # 具体模型案例
wiki/checks/                     # 结构识别和量化前检查
sources/SPEC.md                  # 来源页规范，跨库共享
sources/README.md                # 来源层入口
sources/topics/                  # 主题级说明与边界
sources/<category>/<id>.md       # 一来源一文件（papers / repos / docs）
sources/INDEX.generated.md       # 生成物，不可手改
data/vocabulary.yaml             # 页面类型、标签、别名展开表、来源主题
data/schemas.yaml                # 页面与来源字段契约
data/architecture-profiles.json  # 代表性结构族与证据入口
data/preflight.schema.json       # 机器可读 preflight 输出契约
scripts/wiki_common.py           # 共享库：读取、校验、检索
scripts/query.py                 # 确定性检索（共享）
scripts/get_page.py              # 按 ID 读取页面（共享）
scripts/validate.py              # 页面、来源与引用校验（共享）
scripts/build_index.py           # 生成元数据索引（共享）
scripts/check_shared.py          # 共享文件一致性校验（共享）
scripts/preflight.py             # 从结构 profile 生成量化前检查清单
tests/test_shared_tools.py       # 共享工具回归测试（共享）
tests/test_architecture_wiki.py  # 本库内容与领域事实源测试
```

`CONVENTIONS.md`、`sources/SPEC.md`、`scripts/` 下除 `preflight.py` 外的脚本与 `tests/test_shared_tools.py` 属跨库共享文件，必须与算法库逐字节一致，由 `scripts/check_shared.py` 校验。

## 使用

```bash
python3 scripts/query.py "latent attention" --type component
python3 scripts/query.py --tag moe --type paradigm
python3 scripts/query.py --alias gqa --feature gqa
python3 scripts/query.py --profile mamba2-family
python3 scripts/get_page.py component-attention --follow-sources
python3 scripts/preflight.py deepseek-v3-family
python3 scripts/preflight.py attention-variants-reference --config path/to/config.json --checkpoint <checkpoint> --source-revision <commit-or-tag>
python3 scripts/preflight.py --config path/to/config.json --state-dict-keys path/to/keys.json --forward-manifest path/to/forward.json --source-revision <commit-or-tag> --strict-runtime
python3 scripts/preflight.py --config path/to/config.json --source-revision <commit-or-tag> --compact
python3 scripts/validate.py
python3 scripts/build_index.py --check
python3 scripts/check_shared.py --peer ../quantization-algorithm-wiki
```

`--profile`、`--feature`、`--evidence` 读取 `data/architecture-profiles.json`；`--alias` 读取 `data/vocabulary.yaml` 的别名展开表，未登记的键会报错而不是静默返回空。`get_page.py --follow-sources` 展开本地来源页，离线可用，末尾附来源 URL 供人工核对。

需要通过本机代理检查来源时，显式传入 `--proxy`，例如 `python3 scripts/validate.py --check-external --proxy socks5h://127.0.0.1:13659`。HTTP(S) 代理使用 Python 标准库（SOCKS 代理不被标准库支持）。该检查只验证 URL 可达性，不代表页面内容已经逐条复核。

查询结果只是候选知识。可将最接近的结构 profile 与本地 Hugging Face `config.json` 一起传入 `preflight.py`；对于没有 `model_type` 的实现（例如 Mamba2），profile 可用 `config_signatures` 按配置路径匹配。配置字段按 JSON key 记录为 `config-confirmed`，由字段关系推导的 Attention/MoE 布局标为 `inferred`；Agent 仍须核对加载类、state dict、源码版本和实际计算图。

`--state-dict-keys` 接受参数名 JSON 列表、Hugging Face `weight_map` 索引 JSON 或逐行 key 清单。只有 `{"keys": [...], "complete": true}` 或带 `weight_map` 的完整索引才会用连续索引填充总层数/专家数；普通列表和逐行清单只记录覆盖范围。它不能证明 tensor shape、执行顺序、融合关系或数值行为；这些仍需源码和 reference forward 验证。

`--forward-manifest` 接受一次 reference forward 导出的 `{"metadata": {"source_revision", "device", "dtype", "input_signature"}, "observations": [{"name", "shape", "dtype", "layout"}]}` 清单，将节点和运行元数据标为 `forward-confirmed`；元数据同时写入输出契约的 `runtime` 字段。严格模式要求这些元数据齐全；它只代表被观测到的输入和路径，不能替代多序列长度、cache、模态、timestep 或 sampler 场景验证。

`--strict-runtime` 是结构证据门槛：未锁定 checkpoint、源码 revision、关键结构字段，或缺少 state_dict/reference-forward 证据时返回失败并列出缺口。通过只表示结构证据齐全，不表示目标后端支持哪些格式/模块，也不表示量化精度已经验证。`--compact` 为 Agent 上下文保留 profile、结构、运行证据和量化检查；未经目标实现及后端验证的可量化模块、联合量化组、排除模块和格式/kernel 限制明确输出为 `unknown`。

## 页面约定

页面使用 JSON frontmatter（见 `CONVENTIONS.md` 第 2 节），必需字段为 `id`、`title`、`type`、`tags`、`aliases`、`related`、`sources`、`confidence`、`evidence_scope`、`updated`。正文以一级标题开头，按 `wiki/TEMPLATE.md` 的小节组织。`updated` 只是页面编辑日期；`sources` 指向 `sources/` 下的来源页，字段契约见 `sources/SPEC.md`。代表性结构族另有 JSON profile，生成的 preflight 记录遵循 `data/preflight.schema.json`。证据分类区分 `source-reported`、`source-confirmed`、`config-confirmed`、`state-dict-confirmed`、`forward-confirmed`、`inferred` 和 `unknown`；`confidence` 只描述页面证据强度，不代表量化效果已经验证。

## 贡献原则

优先引用论文、官方技术报告、官方源码和具体版本；不要把典型实现当成所有模型的硬约束。新增来源时写来源页而不是表格行，新增页面时补充来源 ID、相关页面和别名，然后运行 `scripts/validate.py`、`scripts/build_index.py --check` 与 `python3 -m unittest discover -s tests -v`。改共享文件时同步算法库并跑 `scripts/check_shared.py`。
