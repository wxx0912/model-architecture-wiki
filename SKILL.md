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
python3 scripts/get_page.py <page-id> --follow-sources
python3 scripts/preflight.py --list
python3 scripts/preflight.py <profile-id>
python3 scripts/preflight.py <profile-id> --config <checkpoint>/config.json --checkpoint <name-or-path> --source-revision <commit-or-tag>
python3 scripts/preflight.py --config <checkpoint>/config.json --source-revision <commit-or-tag>
python3 scripts/preflight.py --config <checkpoint>/config.json --state-dict-keys <checkpoint>/model.safetensors.index.json --source-revision <commit-or-tag>
python3 scripts/preflight.py --config <checkpoint>/config.json --state-dict-keys <checkpoint>/model.safetensors.index.json --forward-manifest <run>/forward.json --source-revision <commit-or-tag>
python3 scripts/preflight.py --config <checkpoint>/config.json --state-dict-keys <checkpoint>/model.safetensors.index.json --forward-manifest <run>/forward.json --source-revision <commit-or-tag> --strict-runtime
python3 scripts/preflight.py --config <checkpoint>/config.json --source-revision <commit-or-tag> --quantization-view
python3 scripts/validate.py
python3 scripts/build_index.py --check
python3 scripts/check_shared.py --peer ../quantization-algorithm-wiki
```

检索先匹配页面 id、标题、`aliases` 与 `tags`，没有命中再退回正文并标为弱相关，结果按 `type` 分组；`--tag` 按 `data/vocabulary.yaml` 的小写标签精确筛选，未登记的标签报错而不是静默返回空。`--follow-sources` 展开本地来源页正文，离线可用，末尾附来源 URL 供人工核对。来源可达性检查可显式指定代理：`python3 scripts/validate.py --check-external --proxy <proxy>`；HTTP(S) 代理使用 Python 标准库，SOCKS 代理不被支持，检查结果只表示 URL 可达。

先用关键词或标签缩小范围，再读取具体页面。模型页面用于确认整体组合，组件页面用于理解共性，check 页面用于生成结构核查清单。页面级 `confidence` 是证据总体强度；`evidence_scope: mixed` 表示正文同时含来源事实和结构推断，必须按章节标签继续区分，不能把量化推断当成已验证效果。页面 `sources` 指向 `sources/` 下的来源页；来源页记录实际读取形式、版本锚与“不能推出的结论”，不代表正文结论已被逐条复核。

代表性结构族保存在 `data/architecture-profiles.json`。处理具体 checkpoint 时，向 `preflight.py` 传入本地 `config.json` 和已知代码 revision；工具将直接读取的字段标为 `config-confirmed`，记录 JSON key 路径，并根据 query/KV head 数推断 MHA/GQA、根据专家字段提示 MoE。没有 `model_type` 的实现可通过 profile 的 `config_signatures` 按配置路径匹配。缺失字段保持 `unknown`；随后还要对照加载类、state dict、源码和 reference forward。profile 注册表已冻结：未登记 `model_type` 的模型必须显式传 profile ID（`preflight.py <profile-id>`，`--list` 可列出），不要靠新增 profile 适配新模型。profile 里的 `implementation.revision` 默认是 `unknown`，因此它是**结构候选模板**（路由到应读页面），不是某个实现版本的强证据。清单是**页面路由 + 待核实项**（每条含 `pages`），风险条目统一标“待核实”，不能直接当作精度结论。

`--state-dict-keys` 会从 JSON key 列表、Hugging Face `weight_map` 或逐行 key 清单提取模块和连续索引，并将证据标为 `state-dict-confirmed`。只有明确 `complete: true` 的 key 清单才可填充未知的总层数/专家数；不完整清单只报告覆盖范围。参数名不能证明真实执行图、融合、shape 或数值行为，必须继续做源码和 reference forward 核对。

`--forward-manifest` 接收 reference forward 导出的运行元数据（`source_revision`、`device`、`dtype`、`input_signature`）和节点观测（`name`、`shape`、`dtype`、`layout`），将其标为 `forward-confirmed`，并把元数据写入输出契约的 `runtime` 字段。严格模式要求运行元数据齐全；这只覆盖该次运行实际走过的路径，不能替代对其他长度、cache、模态、timestep 或 sampler 的验证。

`--strict-runtime` 在进入量化前检查 checkpoint、源码 revision、关键结构字段以及 state_dict/reference-forward 两类运行时证据；失败时必须先补齐缺口。通过该门槛不代表量化精度已经验证。

### 紧凑上下文与约束边界

给下游量化阶段传递结构结果时可用 `--quantization-view`（字段名换成 `quantization_checks` 等，体积不一定更小），再读取输出的 `knowledge_page` 对应页面。存在运行时证据时，该视图优先保留 config、state_dict、forward 和推断证据；页面级来源事实仍由知识页面提供。`quantization_checks` 是页面路由与待核实项（每条含应读页面 `pages`），既不是已确认的量化约束，也不是候选方案；`quantization_constraints.status=not_assessed` 或其中的 `unknown` 不能解释成“没有约束”。目标 checkpoint、实现版本和部署后端确认之前，不得据此过滤量化格式或模块。

## 输出契约

回答结构问题时应说明：

1. 页面支持的结构事实及来源；
2. 当前目标模型仍需从配置、源码、权重名或运行图确认的内容；
3. 可能影响量化的模块、数据分布、误差传播和部署接口；
4. 不应直接当作硬约束的推断或经验。

`confidence` 分为 `verified`、`source-reported`、`inferred`、`experimental`；`evidence_scope` 标记 `source-reported`、`inferred`、`experimental` 或 `mixed`。不要把 `source-reported` 或 `inferred` 改述成已在目标模型上运行验证。
