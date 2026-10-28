---
{
  "id": "model-minimax-m2",
  "title": "MiniMax-M2",
  "type": "model",
  "tags": [
    "ar",
    "gqa",
    "rope",
    "rmsnorm",
    "moe",
    "routing",
    "quantization"
  ],
  "aliases": [
    "MiniMax-M2",
    "MiniMaxM2",
    "MiniMax M2 系列"
  ],
  "related": [
    "paradigm-ar",
    "paradigm-moe",
    "component-attention",
    "component-position-encoding",
    "component-router-expert",
    "component-normalization"
  ],
  "sources": [
    "paper-minimax-m2",
    "repo-minimax-m2"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-28"
}
---

# MiniMax-M2

## 一句话理解

MiniMax-M2 是 decoder-only 自回归 MoE 模型：62 层、229.9B 总参 / 9.8B 激活、192K 原生上下文。它**全层使用 full attention + GQA（48 Q / 8 KV heads）**，明确放弃 MiniMax-Text-01 的线性与全注意力混合；FFN 换成 **256 个细粒度专家的 MoE，每 token 激活 8 个**（config `shared_intermediate_size=0`，论文未提及共享专家），路由用 **sigmoid gating + 可学习 per-expert bias**；并带 1→3 个 **MTP** 模块用于投机解码。

## 结构规格表

| 项 | 值 | 来源 |
| --- | --- | --- |
| 总参 / 激活 | 229.9B / 9.8B | 论文 §2.1 |
| 层数 | 62 | 论文 §2.1、权重仓 |
| hidden | 3072 | 论文 §2.1、权重仓 |
| 注意力布局 | GQA，全层 full attention：48 Q heads / 8 KV heads、`head_dim=128` | 论文 §2.1、权重仓 |
| QK-Norm | 有（`use_qk_norm=true`、`qk_norm_type=per_layer`） | 权重仓 |
| 位置编码 | RoPE，`rope_theta=5e6`、`rotary_dim=64`（partial，factor 0.5） | 权重仓 |
| 专家（总 / 激活） | 256 / 8（config 另声明 `shared_intermediate_size=0`） | 论文 §2.1（总数/激活）、权重仓（`shared_intermediate_size`） |
| 每专家 FFN 中间维 | 1536 | 权重仓 |
| MTP | 训练单模块（K=1）后拷贝扩展到 3 个；config `num_mtp_modules=3`、`mtp_transformer_layers=1` | 论文 §2.3、权重仓 |
| 上下文 | 192K（`max_position_embeddings=196608`） | 论文 §2.1、权重仓 |
| 词表 | 200064 | 论文 §2.1、权重仓 |
| 预训练 token | 29.2T | 论文 §1、§3 |

取数口径：**层数、hidden、词表、专家数、48/8 heads、192K 上下文**取自论文 §2.1；论文**不含完整架构超参表**，`head_dim`、`rotary_dim`、`rope_theta`、每专家 FFN 中间维、QK-Norm 等来自权重仓 `config.json`（`repo-minimax-m2`，固定 commit）。上下文两个口径对齐：论文写 192K，config `max_position_embeddings=196608`（= 192×1024）。本表不能替代目标 checkpoint 的 `config.json`。

## 结构与数据流

- **主干**（§2.1）：embedding → 62 × block（attention → MoE FFN）→ final norm → LM head（`tie_word_embeddings=false`）。每个 block 都是"attention + MoE FFN"，没有单独区分 dense 层。
- **注意力**（§2.1、§2.2.2）：**全层 full multi-head attention**，48 个 query head、8 个 key-value head（GQA），RoPE 全模型应用。相对 MiniMax-Text-01 的"Lightning Attention 与 full attention 交错"混合设计，M2 回到全层 full attention；报告称没有子二次变体能在生产设定下稳定匹配 full attention 质量。
- **QK-Norm**（建模代码）：`use_qk_norm=true`；建模实现是对 `head_dim × num_heads`（Q）与 `head_dim × num_kv_heads`（K）整块做 RMSNorm，不是 per-head（config 的 `qk_norm_type=per_layer` 字段未被该建模文件引用）。
- **MoE**（§2.1、§2.2.1）：256 个细粒度专家、每 token 激活 8 个；路由先对 logits 取 **sigmoid**，再加可学习 `e_score_correction_bias` 选 top-k，最后用 sigmoid 值归一化作为专家权重；论文未提及共享专家，config 声明 `shared_intermediate_size=0`（含义见未知项）。三点改动：Fine-Grained Experts（更多更小的专家，提升路由组合多样性）、Sigmoid Gating（每个专家独立打分，去掉 softmax 的 zero-sum 约束）、Expert Bias（可学习 per-expert 偏置隐式调节负载，减少对辅助损失的依赖）。
- **MTP**（§2.3）：预训练用单个 MTP 模块（K=1，DeepSeek-V3 式，loss 权重 0.3→0.1）；decay 阶段用主干权重拷贝把 MTP 从 1 个扩到 3 个（K=3）以支撑多步投机解码，先冻结主干只训 MTP、再联合训练；推理时 3 个 MTP 模块产生 draft token、由主干一次前向验证。
- **长上下文**（§3）：常量阶段 19.9T tokens、decay 阶段 9.3T tokens，上下文从 8K 多阶段扩到 32K 再到 192K。

## 量化关联

**结构事实**

- attention 是**全层 full attention + GQA**（48 Q / 8 KV heads），不用 linear / sparse attention（§2.1、§2.2.2）。
- MoE 为 **256 个细粒度专家、每 token 激活 8**；路由用 sigmoid gating + 可学习 per-expert bias（§2.1、§2.2.1）。
- 权重仓 `config.json` 声明 `shared_intermediate_size=0`；论文与建模代码均未提及共享专家，该字段的真实语义须按具体推理后端核对（见未知项）。
- 每个 block 都是 attention + MoE FFN，**没有单独的 dense FFN 层**（§2.1）。
- MTP 训练时从 1 个模块扩展到 3 个模块（§2.3）。
- 报告明确记录：拒绝 linear / sparse attention 的原因之一是它在推理上**对低精度存储敏感**、缺原生 prefix caching、与投机解码集成不清（§2.2.2）。
- 原生上下文 192K（§2.1）。

**实现事实**（来自固定版本权重仓 `config.json` 与同 commit 建模代码，须按该 revision 核对）

- 类名 `MiniMaxM2ForCausalLM` / `model_type=minimax_m2`；62 层、hidden 3072、48 Q heads / 8 KV heads、`head_dim=128`。
- MoE：`num_local_experts=256`、`num_experts_per_tok=8`、`intermediate_size=1536`（建模确认即每专家 FFN 中间维）、`scoring_func=sigmoid`、`use_routing_bias=true`、`shared_intermediate_size=0`。
- 路由实现：`sigmoid(logits) + e_score_correction_bias` 选 top-k，权重用 sigmoid 值归一化。
- QK-Norm：`use_qk_norm=true`；建模实现为 RMSNorm 作用在整块 head 维（对 Q 为 `head_dim × num_heads`、对 K 为 `head_dim × num_kv_heads`）。
- MTP：config 声明 `use_mtp=true`、`num_mtp_modules=3`、`mtp_transformer_layers=1`；同 commit 的 HF 建模文件不实现 MTP。
- 其它：`rope_theta=5000000`、`rotary_dim=64`、`max_position_embeddings=196608`、`vocab_size=200064`、`tie_word_embeddings=false`、`sliding_window=null`、`rms_norm_eps=1e-06`。
- 发布量化配置：`quantization_config` 为 FP8（`float8_e4m3fn`、`weight_block_size=[128,128]`、动态激活），`modules_to_not_convert=["gate","e_score_correction_bias","lm_head"]`。

**量化推断**

- 专家占权重总量的绝大部分而激活稀疏（9.8B / 229.9B）：per-expert 覆盖与路由 top-k 稳定性是量化误差与部署格式的关键来源。
- 路由用 sigmoid + 可学习 bias：量化后路由分数分布变化会直接改变专家选择，须核对；发布 FP8 配置把 `gate` 与 `e_score_correction_bias` 排除在量化外；本库据此推断官方在这两处保留了精度。
- **若**该 checkpoint 确无共享专家（config `shared_intermediate_size=0`，待后端核对），则没有"每 token 都激活"的补偿通路，MoE 层的量化误差没有共享分支兜底；这是待实测的推断。
- GQA 有 8 个 KV head、48 个 query head、`head_dim=128`：每 token、每层的 KV cache 是 K、V 两份，各 8×128，共 2×8×128=2048 个元素，总量再乘序列长度与 batch；6 个 query head 共享同一个 KV head，每 KV head 承载更多读取，cache 量化敏感度需按通道实测，不能由 head 数直接推断离群程度。
- QK-Norm 作用在 Q/K 的整块 head 维上：可能限制 Q/K 的 scale 分布、降低 attention 对离群点的敏感度（推断）；QK-Norm 自身的权重也是量化对象。
- MTP 在 decay 阶段用主干权重拷贝把模块从 1 个扩到 3 个：这是**初始化**上的复用，不能等同于 3 个模块共享同一组参数（本库已读范围内的 §2.3 只记录权重拷贝扩展）。多步投机是否真正共用权重须按 checkpoint 命名核对；3 个 MTP 模块至少要分别作为独立量化对象检查，误差可能跨投机步累积。
- 每层都是 MoE、无 dense 层：MoE 的误差结构贯穿全部 62 层。

**未知项**

- 论文与权重仓都不含量化精度实验：FP8 config 只说明发布格式，精度损失、校准方法与逐层敏感度均未披露。
- config 声明 3 个 MTP 模块，但同 commit 的 `modeling_minimax_m2.py` 不实现 MTP，实际推理路径须按具体后端核对。
- `mlp_intermediate_size=8192`、`shared_intermediate_size=0`、`attn_type_list`（62 个 `1`）未在 `modeling_minimax_m2.py` 中被引用，含义须按具体推理后端核对。
- 论文只给"子二次注意力对低精度存储敏感"的定性判断，未给 M2 自身所用低精度方案（FP8 声明来自权重仓）。
- 论文不给完整架构超参表，`head_dim` 等以 config 为准；论文与 config 是否有口径差异未逐项比对。
- 3 个 MTP 模块是否都进入发布 checkpoint 的推理图、与 `num_mtp_modules` 字段的关系未核对；这 3 个模块之间是否共享参数（还是仅由主干权重拷贝初始化后各自独立）也未核对。

## 结果边界

- 论文是训练侧报告：Table 1（2B 消融）与 Table 2 / Table 3（full vs SWA）都是受控实验，不代表 M2 的绝对能力，也不代表量化后表现。
- 权重仓的 FP8 `quantization_config` 是发布格式声明，不代表精度已达标。
- 本页只覆盖 M2；系列另有 M2.5 / M2.7，其架构差异未读。
- `max_position_embeddings=196608` 是配置声明上限，不证明任何后端能在该长度下服务。
- 本库未加载权重、未运行量化或评测，未对具体 checkpoint 做数值验证。

## Agent 如何使用

首轮读本页即可判断：是否 MoE（是，256 专家 top-8；config 声明 `shared_intermediate_size=0`）、attention 类型（GQA，全层 full attention）、是否含 QK-Norm（是）、是否含 MTP（是，config 声明 3 模块）、发布 config 是否声明量化（是，FP8，排除 `gate` / `e_score_correction_bias` / `lm_head`）。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| MoE 路由、专家与 top-k 归一化 | [`component-router-expert`](../components/router-expert.md)、[`paradigm-moe`](../paradigms/moe.md) |
| GQA 与 QK-Norm 在 attention 中的位置 | [`component-attention`](../components/attention.md) |
| RoPE 与长上下文扩展 | [`component-position-encoding`](../components/position-encoding.md) |
| RMSNorm | [`component-normalization`](../components/normalization.md) |
| 架构取舍、MoE 消融、SWA 对比与 MTP 设计 | [`paper-minimax-m2`](../../sources/papers/paper-minimax-m2.md) |
| 目标 checkpoint 的 `config.json` 字段与发布量化格式 | [`repo-minimax-m2`](../../sources/repos/repo-minimax-m2.md) |

回答量化问题时必须说明：是否 MoE（专家数与 top-k、无共享专家）、attention 是 GQA 且全层 full、KV heads 与 `head_dim`、是否含 QK-Norm、是否含 MTP（3 模块）、上下文口径（192K），以及权重仓已声明 FP8 发布格式且把 `gate` / `e_score_correction_bias` / `lm_head` 排除在量化外。

## 来源与证据

- [MiniMax-M2 系列技术报告](../../sources/papers/paper-minimax-m2.md)：§2.1 规模与模块组成、§2.2.1 MoE 三点改动与 Table 1、§2.2.2 full attention 取舍与 Table 2 / Table 3、§2.3 MTP、§3 预训练数据与长上下文扩展。
- [MiniMaxAI/MiniMax-M2 权重仓](../../sources/repos/repo-minimax-m2.md)：固定 commit 的 `config.json`、建模代码字段语义与 `quantization_config`。

本页结构以上述来源为限；两个来源都不含量化精度实验，FP8 配置是发布格式声明，量化内容除注明者外均为本库推断。
