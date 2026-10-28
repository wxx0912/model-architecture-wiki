---
{
  "id": "model-glm5",
  "title": "GLM-5",
  "type": "model",
  "tags": [
    "ar",
    "attention",
    "mla",
    "moe",
    "routing",
    "rope",
    "quantization"
  ],
  "aliases": [
    "GLM-5",
    "智谱 GLM-5",
    "GlmMoeDsa"
  ],
  "related": [
    "paradigm-ar",
    "paradigm-moe",
    "component-attention",
    "component-kv-cache",
    "component-router-expert",
    "component-position-encoding",
    "component-normalization",
    "model-glm45"
  ],
  "sources": [
    "paper-glm5",
    "repo-glm5"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-28"
}
---

# GLM-5

## 一句话理解

GLM-5 是 decoder-only 自回归 MoE 模型，把 GLM-4.5 的专家数扩到 256、总参扩到 744B（激活 40B）。attention 从 GQA + partial RoPE 换成 **MLA**（576 维 latent KV cache），并做了两处改造：**Muon Split**（up-projection 按 head 拆开再正交化）与 **MLA-256**（head dim 192→256、head 数减 1/3）。长上下文通过**持续预训练引入 DSA**获得，作用于全部层。

## 结构规格表

| 项 | GLM-5 | 来源 |
| --- | --- | --- |
| 总参 / 激活 | 744B / 40B | 论文 §2.1 |
| 层数 | 78（3 dense + 75 MoE；论文 §2.1 另写"降到 80"） | 论文 §2.1、附录 A Table 10、权重仓 |
| hidden | 6144 | 论文 Table 10、权重仓 |
| attention | MLA：64 Q heads / 64 KV heads；`qk_head_dim=256`（nope 192 + rope 64）、`v_head_dim=256`、`kv_lora_rank=512`、`q_lora_rank=2048` | 论文 Table 10、权重仓 |
| DSA indexer | `index_n_heads=32`、`index_head_dim=128`、`index_topk=2048` | 论文 Table 10、权重仓 |
| 专家（总 / 激活 / 共享） | 256 / 8 / 1，专家中间维 2048 | 论文 §2.1、Table 10、权重仓 |
| 前若干层 dense | `first_k_dense_replace=3` | 权重仓 |
| MTP | checkpoint 含 1 层（`num_nextn_predict_layers=1`）；论文说训练时共享 3 层 MTP 参数 | 论文 §2.1、权重仓 |
| 位置编码 | `rope_theta=1e6`、`rope_type=default` | 权重仓 |
| 上下文 | `max_position_embeddings=202752` | 论文 §3.1、权重仓 |
| 词表 | 154880 | 论文 Table 10、权重仓 |

论文 §2.1 给参数量、专家数与 MLA / MLA-256 的取舍，附录 A Table 10 给出 hidden、专家中间维、QK/V head dim、Q/KV LoRA、head 数、indexer 参数与词表，权重仓 `config.json`（`repo-glm5`，固定 commit）给出目标 checkpoint 的精确字段。两者口径不完全相同：Table 10 的 QK Head Dim=192 对应 config `qk_nope_head_dim=192`（config `qk_head_dim=256` 另含 rope；此对应关系是本页推断），Table 10 的 "Routed Experts"=8 指每 token 激活数（config `n_routed_experts=256` 为专家总数）。Table 10 的 `# Key-Value Heads` 为 `–`（MLA 不适用），64 个 KV heads 只来自权重仓 config。层数在论文内部不一致（§2.1 写 80、Table 10 为 78）；引用时须说明口径，本表不能替代目标 checkpoint 的 `config.json`。

## 结构与数据流

- **主干**：embedding → 3 层 dense FFN block → 75 层 MoE block → final norm → LM head（`tie_word_embeddings=false`）；另有 1 层 MTP。
- **注意力（MLA）**（§2.1）：用压缩的 latent KV cache，论文表述为 **576 维**（对照 GQA-8 的 2048 维）。权重仓给出的字段是 `kv_lora_rank=512`、`qk_rope_head_dim=64`、`qk_nope_head_dim=192`、`v_head_dim=256`、64 个 Q head 与 64 个 KV head。解码时每个 head 做 576 维点积（高于 GQA 的 128 维），这是论文引入 **MLA-256** 的动机。
- **Muon Split**（§2.1）：原配方对 up-projection 矩阵 `W^{UQ}`、`W^{UK}`、`W^{UV}` 整体做矩阵正交化；GLM-5 改为按 head 拆成小矩阵分别正交化，使不同 head 的投影权重以不同尺度更新；论文报告这样 attention logits 尺度稳定、无需裁剪（§2.1；Table 1 给出对应结果）。
- **MLA-256**（§2.1）：head dimension 从 192 提到 256，attention head 数减少 1/3，保持训练计算量与参数量不变，降低解码计算；Table 1 显示与 MLA 性能相当。
- **DSA**（§2.1.1）：用基于内容的动态细粒度选择替换稠密 `O(L²)` 注意力，通过"dense warm-up + sparse adaptation"从稠密 MLA 基座持续预训练迁移——warm-up 训 1000 步（每步 14 条 202,752 token 序列、最大学习率 5e-3），sparse adaptation 训 20B tokens；论文称沿用标准 DSA 配方，但未明确 GLM-5 warm-up 是否冻结基座（该细节出自同节 GLM-4.7-Flash 实验）。论文报告长序列注意力计算降低约 1.5–2×、128K 上下文可用一半 GPU 成本，并称 lightning indexer 的 token 级稀疏不丢弃长程依赖，因此可用于全部层。
- **MoE**（§2.1、权重仓）：256 个路由专家 + 1 个共享专家，每 token 激活 8；路由打分函数为 `sigmoid`，`topk_method=noaux_tc`、`norm_topk_prob=true`、`routed_scaling_factor=2.5`、`n_group=topk_group=1`（不做分组限制路由）；前 3 层是 dense FFN，`moe_layer_freq=1` 表示其后每层都是 MoE。
- **MTP**（§2.1）：训练时共享 3 层 MTP 的参数（而不是训练 3 个独立 MTP 层），使 draft 模型内存与"只训 1 层 MTP"的 DeepSeek-V3 一致，同时提高接受率；Table 2 给出 GLM-5 接受长度 2.76、DeepSeek-V3.2 为 2.55（私有 prompt 集、4 步投机）。

## 量化关联

**结构事实**

- 论文 §2.4.3 在 SFT 阶段做 **INT4 量化感知训练（QAT）**，并开发了同时用于训练与离线权重量化的量化 kernel，使训练与推理位级一致。
- 论文 §5 给出面向 Ascend Atlas 的部署量化分配：**标准 Attention 与 MLP 块用 W8A8（INT8），MoE 专家用 W4A8（INT4）**；用 QuaRot 做 outlier 抑制、Flex_AWQ_SSZ 做 scaling 校准，工具为 msModelSlim；目标是放进单台 Atlas 800T A3。
- attention 是 MLA（576 维 latent KV cache），不是 GQA（§2.1）。
- DSA 作用于**全部层**，用 lightning indexer 做 token 级稀疏选择（§2.1.1）。
- MoE 为 256 路由专家 + 1 共享专家，每 token 激活 8（§2.1、权重仓）。
- 前 3 层是 dense FFN，其后各层都是 MoE（权重仓 `first_k_dense_replace=3`、`moe_layer_freq=1`）。
- MLA-256 的 head dim 为 256、head 数比 MLA 少 1/3（§2.1）。

**实现事实**（来自固定版本权重仓 `config.json`，须按该 revision 核对）

- `GlmMoeDsaForCausalLM` / `model_type=glm_moe_dsa`；`num_hidden_layers=78`、`hidden_size=6144`、`intermediate_size=12288`、`moe_intermediate_size=2048`。
- MLA 字段：`num_attention_heads=64`、`num_key_value_heads=64`、`head_dim=64`、`qk_head_dim=256`、`qk_nope_head_dim=192`、`qk_rope_head_dim=64`、`v_head_dim=256`、`kv_lora_rank=512`、`q_lora_rank=2048`、`attention_bias=false`。
- DSA 字段：`index_n_heads=32`、`index_head_dim=128`、`index_topk=2048`、`indexer_rope_interleave=true`。
- MoE 字段：`n_routed_experts=256`、`n_shared_experts=1`、`num_experts_per_tok=8`、`scoring_func=sigmoid`、`topk_method=noaux_tc`、`routed_scaling_factor=2.5`、`n_group=1`、`topk_group=1`。
- 其它：`max_position_embeddings=202752`、`vocab_size=154880`、`rms_norm_eps=1e-05`、`tie_word_embeddings=false`、`dtype=bfloat16`、`num_nextn_predict_layers=1`。

**量化推断**

- GLM-5 论文 §5 已在 Ascend 上采用 W4A8（专家 INT4、主干 INT8），但官方发布的权重 config 仍声明 `dtype=bfloat16`，两者不是同一份产物。量化方向应由 Agent 按输入权重、目标后端与精度目标决定：存在专家 INT4 部署路径只说明这条路线可行，不代表专家侧 PTQ 空间已耗尽；继续降位须按官方 kernel 与后端支持核对。
- MLA 的 latent KV cache 只有 576 维（远小于 GQA-8 的 2048 维）：cache 侧低比特化的收益空间比 GQA 小，而误差会直接进入唯一那条 latent 通路。
- DSA 的 top-k 选择随输入变化：被读到的 token 集合不固定，逐层静态敏感度分析可能覆盖不到实际命中的集合。
- head dim 提到 256（MLA-256）：单 head 元素更多，per-head 尺度估计与分组粒度需要按 256 维核对；动态范围是否随之变大需实测，不能由维度直接推断。
- 前 3 层是 dense FFN，没有专家分流：这几层的量化误差直接进入主干，与 MoE 层的误差结构不同。
- MTP 的 3 层共享参数：多步投机解码共用同一组权重，量化误差会跨投机步累积；若该层保留在推理图，须作为独立对象检查。
- MoE 256 专家、top-8：per-expert 覆盖与路由 top-k 稳定性是量化误差与部署格式的主要来源。

**未知项**

- 层数口径在论文内部不一致（§2.1 写 80、附录 A Table 10 为 78），差异来源未定位。
- 参数量口径不一致（§2.1 写 744B、§5 写 "750B"）。
- 论文不含逐层量化敏感度与精度对照：W4A8 只到"组件级精度分配 + 所用算法名"，没有精度对比数字与校准口径。
- 576 维 latent KV cache 由 `kv_lora_rank=512` 与 `qk_rope_head_dim=64` 相加得到，这是**推断**，config 未直接给出该组合关系。
- DSA 的 indexer 结构、稀疏选择细节论文未给，指向 DeepSeek-V3.2。
- §2.1.2 的注意力变体消融在 GLM-9B / GLM-4.7-Flash 上完成，不能外推到 GLM-5。

## 结果边界

- 论文是训练侧报告，评测数字不代表量化后表现；§5 的 W4A8 是面向 Ascend Atlas 单机部署的适配，不是通用量化结论。
- Table 2 的接受长度来自私有 prompt 集与 4 步投机设置，不是通用吞吐结论。
- 论文附录 A Table 10 给出架构超参，但与权重仓 `config.json` 存在口径差异；本页的表以论文 §2.1、Table 10 与权重仓 `config.json` 为限。
- 本库未加载权重、未运行量化或评测，未对具体 checkpoint 做数值验证。

## Agent 如何使用

首轮读本页即可判断：是否 MoE（是，256 专家 top-8）、attention 是否为 MLA（是，576 维 latent cache）、是否启用 DSA 稀疏注意力（是，作用于全部层）、前几层是否 dense（前 3 层）、是否含 MTP。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| MLA、GQA 与 MHA 的对照 | [`component-attention`](../components/attention.md) |
| latent KV cache 与压缩注意力的共性结构 | [`component-kv-cache`](../components/kv-cache.md) |
| MoE 路由、专家与 top-k 归一化 | [`component-router-expert`](../components/router-expert.md)、[`paradigm-moe`](../paradigms/moe.md) |
| RoPE 与长上下文扩展 | [`component-position-encoding`](../components/position-encoding.md) |
| 上一代 GQA + partial RoPE 的 MoE（GLM-4.5） | [`model-glm45`](glm45.md) |
| 规模、专家数、MLA / Muon Split / MLA-256、MTP 共享、§5 的 W4A8 部署量化 | [`paper-glm5`](../../sources/papers/paper-glm5.md) |
| 目标 checkpoint 的 `config.json` 字段 | [`repo-glm5`](../../sources/repos/repo-glm5.md) |

回答量化问题时必须说明：层数口径（§2.1 写 80、Table 10 与 config 为 78）、attention 是 MLA 且 latent cache 576 维、head dim 为 256、DSA 作用于全部层、专家数与 top-k、前 3 层为 dense、MTP 层情况，以及论文 §2.4.3 在 SFT 阶段做 INT4 QAT、§5 已在 Ascend 上采用 W4A8（专家 INT4）。

## 来源与证据

- [GLM-5 技术报告](../../sources/papers/paper-glm5.md)：§2.1 规模 / MLA / Muon Split / MLA-256 / MTP 共享（Table 1、Table 2）、§2.1.1 DSA 与 Table 3、§2.1.2 变体消融与 GLM-4.7-Flash DSA 验证（Table 6）、§2.4.3 INT4 QAT、§5 W4A8 与融合算子、附录 A Table 10 架构超参。
- [zai-org/GLM-5 权重仓](../../sources/repos/repo-glm5.md)：固定 commit 的 `config.json` 字段。

本页结构以上述来源为限；论文 §5 只给组件级量化分配与算法名、§2.4.3 只说明 SFT 阶段做 INT4 QAT，逐层敏感度与精度结论均为本库推断。
