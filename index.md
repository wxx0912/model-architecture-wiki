# 模型结构导航

## Agent 调用路径

范式页说明主执行路径；组件页解释可复用模块；模型页说明具体模型如何组合这些结构；check 页把待核实项转成 preflight 记录。Agent 应先从模型配置确定结构特征（范式、注意力布局、位置编码、归一化、MoE、生成阶段等），再按 [index.generated.md](index.generated.md) 的"按结构特征检索"取页面，最后用加载类、state dict 和 reference forward 覆盖通用描述。只有目标源码和后端证据才能把待核实项升级为硬约束。

## 术语歧义

LatentMoE 是歧义术语，检索时必须区分：

- [NVIDIA LatentMoE](wiki/paradigms/latent-moe-nvidia.md)
- [Multi-Head LatentMoE](wiki/paradigms/multihead-latent-moe.md)
- [LatentMoE 术语与变体](wiki/paradigms/latent-moe.md)

## 机器可读入口

- `data/architecture-profiles.json`（**已冻结**，不再扩 profile）：只登记代表性结构族，`model_type` 未登记的模型需向 `preflight.py` 显式传 profile ID（位置参数，`--list` 可列出）。
- `python3 scripts/preflight.py <profile-id>`：**页面路由**。输出该结构应读的页面（每条检查项带 `pages`）与待核实项，不做风险判定，也不产出候选量化方案或过滤条件。
- `python3 scripts/validate.py`：一次校验页面契约、引用与链接；索引是否过期由 `build_index.py --check` 负责。
- [index.generated.md](index.generated.md)：全部页面的稳定 ID、类型与结构特征分组（生成物，勿手改）；来源清单见 [来源索引](sources/INDEX.generated.md)。
