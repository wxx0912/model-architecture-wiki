---
{
  "id": "model-deepseek-v4",
  "title": "DeepSeek-V4 / DeepSeek-V4-Pro / DeepSeek-V4-Flash",
  "type": "model",
  "tags": [
    "ar",
    "attention",
    "kv-cache",
    "moe",
    "routing",
    "mhc",
    "quantization"
  ],
  "aliases": [
    "DeepSeek-V4",
    "DeepSeek-V4-Pro",
    "DeepSeek-V4-Flash",
    "DSV4"
  ],
  "related": [
    "paradigm-ar",
    "paradigm-moe",
    "component-attention",
    "component-kv-cache",
    "component-residual-mhc",
    "component-router-expert",
    "component-position-encoding",
    "model-deepseek"
  ],
  "sources": [
    "paper-deepseek-v4",
    "repo-deepseek-v4"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-28"
}
---

# DeepSeek-V4 / DeepSeek-V4-Pro / DeepSeek-V4-Flash

## 一句话理解

DeepSeek-V4 是 decoder-only 自回归 MoE 模型系列，面向 100 万 token 上下文。相对 DeepSeek-V3 的三项升级是 **mHC** 强化残差、**CSA/HCA 交错**的混合注意力（KV 先按块压缩、CSA 再对压缩块做 sparse 选择）与 **Muon** 优化器；MoE 另把亲和度函数改成 `Sqrt(Softplus(·))`、最早几层 dense FFN 改成 **Hash 路由**。注意力既非 MHA/GQA 也非 MLA：KV 压缩成单头条目，另有滑窗分支与 attention sink。

## 结构规格表

| 型号 | 总参 / 激活 | 层数 | hidden | Q,KV heads | head_dim | 专家（总/激活） | MoE 中间维 | CSA top-k | o_groups | 前两层注意力 | context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Pro | 1.6T / 49B | 61 | 7168 | 128,1 | 512 | 384/6 | 3072 | 1024 | 16 | HCA | 1M |
| DeepSeek-V4-Flash | 284B / 13B | 43 | 4096 | 64,1 | 512 | 256/6 | 2048 | 512 | 8 | 纯滑动窗口 | 1M |

取数口径：论文 §4.2.1（Model Setups）逐项给出上述全部结构量，`repo-deepseek-v4` 的 `config.json` 与之一致（除论文未给逐层 `compress_ratios`）。两个系列另有公共取值：`q_lora_rank` 1536/1024、`o_lora_rank=1024`、`qk_rope_head_dim=64`、`index_n_heads=64`、`n_shared_experts=1`、`num_hash_layers=3`、`hc_mult=4`、`sliding_window=128`、`max_position_embeddings=1048576`。本表不能替代目标 checkpoint 的 `config.json`。

## 结构与数据流

- **主干**（§2）：保留 Transformer 主干、DeepSeekMoE 与 MTP；MTP 配置与 DeepSeek-V3 相同（`num_nextn_predict_layers=1`）。据 Figure 2 的整体架构示意，每层顺序为 mHC 残差 → 混合注意力（CSA 或 HCA）→ MoE。
- **Muon**（§2）：论文三项升级的第三项是优化器而非网络结构，本页不展开。
- **mHC**（§2.2）：残差流宽度扩大 `hc_mult` 倍（config 为 4），层内输入仍是 `d` 维；残差映射矩阵被约束为双随机矩阵（谱范数 ≤ 1，非扩张），输入/输出映射用 Sigmoid 限定非负，三个映射的参数按"动态 + 静态"生成，投影用 Sinkhorn-Knopp 迭代（config `hc_sinkhorn_iters=20`、`hc_eps=1e-6`）。
- **逐层配置**（§4.2.1）：Pro 前两层用 **HCA**，Flash 前两层用**纯滑动窗口注意力**，其后各层 CSA 与 HCA 交错；CSA 的 `m=4`、indexer query heads 64、indexer head dim 128、attention top-k 1024（Pro）/ 512（Flash）；HCA 的 `m′=128`；滑动窗口分支 `n_win=128`；全部 block 都用 MoE，前 3 个 MoE 层用 Hash 路由。
- **CSA**（§2.3.1）：先把每 `m` 个位置的两条 KV 序列按可学习压缩权重压成 1 个条目（压缩窗口重叠，序列长度 → `1/m`），再用 lightning indexer 给压缩块打分并取 top-k（`index_topk`，Pro 1024 / Flash 512）；选中的压缩条目同时作 key 与 value，按 MQA 做核心注意力。query 由压缩隐向量 `c_t^Q` 低秩上投影，且与 indexer query 共用该隐向量。
- **HCA**（§2.3.2）：压缩策略同 CSA 但压缩率 `m′ ≫ m`、不做重叠压缩、**不做稀疏选择**，在压缩后的 KV 上保持稠密注意力；同样使用 shared KV MQA、低秩 query 与分组输出投影（`o_groups`）。
- **公共注意力细节**（§2.3.3）：核心注意力前对 query 各 head 与压缩 KV 的唯一 head 做 RMSNorm；对 query、KV 条目、核心注意力输出的**最后 64 维**施加 partial RoPE（输出侧用位置 `-i` 抵消）；另有滑动窗口分支（`sliding_window=128`）提供最近若干未压缩 KV；softmax 分母加入可学习 sink logits（attention sink）。
- **MoE**（§2.1）：fine-grained routed experts + 共享专家；亲和度由 `Sigmoid` 改为 `Sqrt(Softplus(·))`；aux-loss-free 负载均衡 + 较轻的序列级平衡损失；去掉路由目标节点数量约束；最早的 3 层（`num_hash_layers=3`）用 Hash 路由按 token id 选专家。
- **位置与长度**（config）：`rope_theta=10000` 配合 YaRN（`factor=16`、`original_max_position_embeddings=65536`）到 `max_position_embeddings=1048576`；压缩分支另用 `compress_rope_theta=160000`。

## 量化关联

**结构事实**

- routed expert 参数使用 FP4 精度，后训练阶段对 MoE expert 权重与 indexer 的 QK 路径做 FP4 QAT（§1、§5）。
- attention 是 CSA 与 HCA 的交错混合，KV 侧经块压缩；CSA 额外做 top-k 稀疏选择（Pro 1024 / Flash 512），HCA 保持稠密。**前两层例外**：Pro 前两层是 HCA，Flash 前两层是纯滑动窗口注意力（§2.3.1、§2.3.2、§4.2.1）。
- 核心注意力的 key 与 value 是同一个压缩条目（shared KV MQA，§2.3.1），因此 KV 侧只有单头条目（config `num_key_value_heads=1`）。
- 注意力输出由三部分共同决定：压缩 KV（CSA 取 top-k / HCA 全取）、滑动窗口分支的未压缩 KV，以及只以 `Exp(z′_h)` 进入 softmax 分母、不提供 KV 信息的 attention sink（§2.3.1–§2.3.3）。
- 每层用 partial RoPE，只作用于最后 64 维（§2.3.3）。
- **异构 KV 存储**（§2.3.4）：KV 条目采用混合精度存储——RoPE 维用 BF16、其余维用 FP8；相对纯 BF16 存储 KV cache 体量约减半，论文另称 1M 上下文下约为 BF16 GQA-8 基线的 2%。
- 最初 3 个 MoE 层用 Hash 路由，按输入 token id 经预定义哈希函数决定目标专家（§2.1、config `num_hash_layers=3`）。

**实现事实**（来自固定版本权重仓 `config.json`，须按该 revision 核对）

- 类名 `DeepseekV4ForCausalLM` / `model_type=deepseek_v4`。
- Pro：61 层、hidden 7168、128 Q heads / 1 KV head、`head_dim=512`、384 专家 / top-6、`moe_intermediate_size=3072`、`index_topk=1024`、`o_groups=16`、`routed_scaling_factor=2.5`。
- Flash：43 层、hidden 4096、64 Q heads / 1 KV head、`head_dim=512`、256 专家 / top-6、`moe_intermediate_size=2048`、`index_topk=512`、`o_groups=8`、`routed_scaling_factor=1.5`。
- 两个 config 都声明 `quantization_config`：`fp8`、`fmt=e4m3`、`scale_fmt=ue8m0`、`weight_block_size=[128,128]`、`activation_scheme=dynamic`；并带 `expert_dtype=fp4`。
- `compress_ratios` 是逐层压缩比列表，Pro 62 项 / Flash 44 项（均为 `num_hidden_layers + 1`），取值 0 / 4 / 128。

**量化推断**

- routed expert 已经以 FP4 存储（`expert_dtype=fp4`）：量化对象与位宽应由 Agent 按输入权重、目标后端与实测结果决定，已发布精度不等于压缩空间耗尽；继续降位须按官方 kernel 支持核对。
- 权重仓声明的 FP8 契约是 `e4m3` + `ue8m0` scale、`weight_block_size=[128,128]`、`activation_scheme=dynamic`：可以确认的是**权重**按 128×128 分块，激活侧只有一个 dynamic 声明、粒度未在配置里给出，需另查实现；新方案要与这套权重格式对不齐的成本一并评估。
- `num_key_value_heads=1` 且 KV 是压缩后的单头条目：KV cache 的元素数由压缩率直接决定，cache 侧体量本身已被大幅压缩，KV cache 低比特化的收益空间比标准 GQA/MLA 小，而误差会直接进入唯一那条 KV 通路。
- CSA 的 top-k 选择（运行时按 indexer 分数动态决定读哪些块）会让被读取的 KV 块随输入变化：静态逐层敏感度分析可能无法覆盖被实际选中的压缩 KV 块。
- Hash 路由层用预定义哈希函数按 token id 决定**专家选择**，这一环节不依赖学习到的打分；但"是否仍计算门控分数对选中专家加权"报告未说明（见未知项），不能把"专家选择不用打分"读成"不保留门控打分"。共享专家各 MoE 层都有（§4.2.1 `n_shared_experts=1`），Hash 层也不例外。这几层的专家分布会随词表分布偏斜，专家级量化误差的覆盖需要按 hash 分桶核对。
- mHC 把残差流宽度扩大 4 倍且映射矩阵非扩张：残差通路的动态范围与标准残差不同，逐层激活量化（尤其对残差加和处）需要单独核对。
- attention sink 的 sink logits 与滑动窗口分支都会参与 softmax 归一化：只量化压缩 KV 而忽略 sink 与窗口分支，会改变归一化后的权重分布。
- KV 条目已是"RoPE 维 BF16 + 其余维 FP8"的混合存储（§2.3.4）：cache 侧已做过一轮按维度分精度的压缩，进一步低比特化要与这套布局对齐，不能把整条 KV 当作单一格式处理。

**未知项**

- Hash 路由的前 3 个 MoE 层：专家选择由 token id 的哈希决定；这些层的**门控加权实现**未核实——是否仍用门控分数对选中专家加权，报告未说明（§2.1 只描述专家选择）。共享专家不属未知项：§4.2.1 已给出每个 MoE 层 `n_shared_experts=1`。
- 报告是 preview 版本，且没有"低比特 vs 全精度 / 不同位宽"的对照精度实验：FP4 / FP8 / 低比特下的精度损失、校准方法与逐层敏感度均未披露。
- `compress_ratios` 列表比层数多 1 项，**多出项对应哪一层未核实**，引用单项时不能直接按层号索引（取值语义 `4`=CSA、`128`=HCA、`0`=无压缩已由 §4.2.1 佐证）。
- §4.2.1 已给出 `m=4`、`m′=128`、attention top-k 与 `n_win=128`，且 `n_win` 与 config `sliding_window=128` 同名同值；压缩窗口如何切分、indexer 布局仍须按官方推理实现核对。
- mHC 内部三个映射的量化敏感度、以及 Sinkhorn 迭代是否在推理期执行，均未核对实现。
- 官方推理实现（论文指向权重仓 `inference/` 目录）的 KV cache 异构布局与块压缩的切分方式未读。

## 结果边界

- 论文给的效率结论是**估计值**：1M 上下文下单 token FLOPs 与 KV cache 相对 DeepSeek-V3.2 的 27% / 10%（Pro）与 10% / 7%（Flash），不是实测吞吐或显存。
- `max_position_embeddings=1048576` 是配置声明上限，须配合 YaRN 使用；这不证明任何后端能在该长度下实际服务。
- 论文是训练侧报告，评测数字不代表量化后表现（被评测模型本身经过 FP4 QAT；权重仓声明 FP8 契约，但论文未说明评测时的实际计算精度）。
- 本库未加载权重、未运行量化或评测，未对具体 checkpoint 做数值验证。

## Agent 如何使用

首轮读本页即可判断：是否 MoE（是）、attention 是否为 CSA/HCA 混合（是，且 KV 侧为单头压缩条目）、是否含 mHC、是否有 Hash 路由层、KV cache 是否还有压缩分支以外的滑动窗口部分、是否有 attention sink。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| mHC 的定义、双随机约束与实现差异 | [`component-residual-mhc`](../components/residual-mhc.md) |
| attention 变体（MHA / GQA / MLA）与 V4 的差异 | [`component-attention`](../components/attention.md) |
| KV cache 的通用结构与压缩、滑动窗口分支的关系 | [`component-kv-cache`](../components/kv-cache.md) |
| MoE 路由、专家与 Hash 路由的共性结构 | [`component-router-expert`](../components/router-expert.md)、[`paradigm-moe`](../paradigms/moe.md) |
| partial RoPE 与 YaRN 扩展 | [`component-position-encoding`](../components/position-encoding.md) |
| DeepSeek-V2 / V3 的 MLA、DeepSeekMoE 与 MTP 基线 | [`model-deepseek`](deepseek.md) |
| CSA / HCA 的公式与压缩细节、效率口径 | [`paper-deepseek-v4`](../../sources/papers/paper-deepseek-v4.md) |
| 目标 checkpoint 的层数、专家数、`compress_ratios`、精度契约 | [`repo-deepseek-v4`](../../sources/repos/repo-deepseek-v4.md) |

回答量化问题时必须说明：具体型号（Pro / Flash）、层数与专家数、attention 是 CSA 还是 HCA 层、`head_dim=512`、KV 为单头压缩条目、是否含滑动窗口分支与 attention sink、`expert_dtype=fp4` 与 FP8 契约、以及 mHC 对残差通路的影响。

## 来源与证据

- [DeepSeek-V4 技术报告](../../sources/papers/paper-deepseek-v4.md)：§2 的三项升级（mHC、CSA/HCA、Muon）、§2.1 MoE 与 MTP 改动、§2.3.1–§2.3.3 CSA / HCA 与细节、§2.3.4 效率口径、§3.5.1 KV cache 结构、§4.2.1 两个模型的逐项配置、§1 与 §5.2.1 的 FP4 与 QAT 说明。
- [deepseek-ai/DeepSeek-V4 系列权重仓](../../sources/repos/repo-deepseek-v4.md)：Pro / Flash 的 `config.json` 字段与精度契约。

本页结构以上述来源为限；报告没有"低比特 vs 全精度 / 不同位宽"的对照精度实验，量化内容除注明者外均为本库推断。
