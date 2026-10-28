---
{
  "id": "model-qwen3",
  "title": "Qwen3 / Qwen3-MoE",
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
    "Qwen3",
    "Qwen3-MoE",
    "Qwen3Moe",
    "Qwen3-235B-A22B",
    "Qwen3-30B-A3B",
    "千问3"
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
    "paper-qwen3",
    "repo-qwen3"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-28"
}
---

# Qwen3 / Qwen3-MoE

## 一句话理解

Qwen3 是 decoder-only 自回归模型系列，dense 与 MoE 共用同一基础主干：GQA + SwiGLU + RoPE + pre-norm RMSNorm，并相对 Qwen2 **去掉 QKV-bias、加入 QK-Norm**。家族含 6 个 dense（0.6B–32B）与 2 个 MoE（30B-A3B、235B-A22B），MoE 为 128 专家、每 token 激活 8 且**不含共享专家**。注意力仍是 GQA，因此不涉及 MLA 的 latent 压缩，也不是线性注意力或 SSM。

## 结构规格表

| 型号 | 层数 | hidden | Q,KV heads | FFN / 专家中间维 | 总参 / 激活 | 专家（总/激活） | tie embedding | context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 28 | 1024 | 16,8 | 3072 | 0.6B | — | 是 | 32K |
| Qwen3-1.7B | 28 | 2048 | 16,8 | 6144 | 1.7B | — | 是 | 32K |
| Qwen3-4B | 36 | 2560 | 32,8 | 9728 | 4B | — | 是 | 128K |
| Qwen3-8B | 36 | 4096 | 32,8 | 12288 | 8B | — | 否 | 128K |
| Qwen3-14B | 40 | 5120 | 40,8 | 17408 | 14B | — | 否 | 128K |
| Qwen3-32B | 64 | 5120 | 64,8 | 25600 | 32B | — | 否 | 128K |
| Qwen3-30B-A3B | 48 | 2048 | 32,4 | 768 | 30B / 3B | 128/8 | 否 | 128K |
| Qwen3-235B-A22B | 94 | 4096 | 64,4 | 1536 | 235B / 22B | 128/8 | 否 | 128K |

取数口径：**层数、Q/KV heads、专家数、context** 取自论文 Table 1（dense）与 Table 2（MoE）；**hidden、FFN / 专家中间维、tie embedding** 取自 `repo-qwen3` 的 `config.json`（tie embedding 论文 Table 1 有、Table 2 无；hidden 与中间维论文未给）；**总参 / 激活**取自论文 §1 与模型命名。dense 与 MoE 共用"基础"架构（§2），但 KV heads（dense 8 / MoE 4）与 hidden 宽度不同，不是同一套超参。context 的 "32K / 128K" 是论文支持口径，官方 `config.json` 的 `max_position_embeddings` 为 40960；两者不是一个数。本表不能替代某个 checkpoint 的 `config.json`。

## 结构与数据流

- **主干**（§2）：embedding → N × block（pre-norm RMSNorm → attention → 残差 → pre-norm RMSNorm → FFN 或 MoE → 残差）→ final RMSNorm → LM head。dense 的 FFN 为 SwiGLU；tie embedding 只在 0.6B / 1.7B / 4B 为真（Table 1）。
- **注意力**（§2）：GQA；相对 Qwen2 **移除 QKV-bias**，并**加入 QK-Norm** 以稳定训练（该句在 dense 段落，MoE 侧由"共享同一基础架构"承接，config `attention_bias=false` 亦佐证）。`head_dim=128` 是显式配置而非推导值：以 235B-A22B 为例，`hidden_size=4096`、64 个 Q head，`64 × 128 = 8192`，所以 Q 投影输出宽度由 `head_dim` 决定，不能用 `hidden_size / num_heads` 估算。
- **位置编码**（§2、§3.2）：RoPE，`rope_theta=1e6`（长上下文阶段从 1e4 提到 1e6，ABF）；config 中 `rope_scaling=null`，128K 依赖推理期 YARN + Dual Chunk Attention 扩展。
- **MoE**（§2、Table 2）：与 dense 同一基础架构，仅把 FFN 换成 MoE；fine-grained expert segmentation；每层 128 专家、每 token 激活 8；**不含共享专家**（与 Qwen2.5-MoE 不同）；用 global-batch load balancing loss 促进专家专精。
- **两种推理模式**（§1、§4.3）：thinking 与 non-thinking 融合在同一权重内，另有 thinking budget 控制推理算力——这是后训练与推理行为，不是结构分支，也不改变模块组成。

## 量化关联

**结构事实**

- QK-Norm 是 Qwen3 相对 Qwen2 新增的注意力部件，论文给出的理由只是"稳定训练"（§2）。
- attention 不再使用 QKV-bias（§2；MoE 侧依据"共享同一基础架构"与 config `attention_bias=false`）。
- MoE 为 128 个专家、每 token 激活 8、**无共享专家**（§2、Table 2）。
- dense 与 MoE 共用同一**基础**架构（§2），但两者在 KV heads（dense 8 / MoE 4）与 hidden 宽度上不同，不是同一套超参。
- GQA 的 KV heads：dense 为 8，MoE 为 4（Table 1、Table 2）。

**实现事实**（来自固定版本权重仓 `config.json`，须按该 revision 核对）

- 类名：dense 为 `Qwen3ForCausalLM` / `model_type=qwen3`；MoE 为 `Qwen3MoeForCausalLM` / `model_type=qwen3_moe`。
- 读到的 8 个 config 公共取值：`attention_bias=false`、`head_dim=128`、`rope_theta=1e6`、`rope_scaling=null`、`max_position_embeddings=40960`、`rms_norm_eps=1e-6`、`vocab_size=151936`。
- MoE config：`num_experts=128`、`num_experts_per_tok=8`、`decoder_sparse_step=1`、`norm_topk_prob=true`；`moe_intermediate_size` 为 235B-A22B 1536、30B-A3B 768。
- 这些 config **不含 QK-Norm 字段**（不同于 GLM-4.5 权重仓的 `use_qk_norm`），也不含 MTP 相关字段。

**量化推断**

- KV heads 少（dense 8、MoE 4；`head_dim=128`）：每 token、每层的 KV cache 是 K、V 两份，各 `num_key_value_heads × head_dim`（dense 2×8×128=2048、MoE 2×4×128=1024 个元素），总量再乘序列长度与 batch；KV head 少意味着**更多 query heads 共享同一个 KV head**（如 8B 为 32:8、30B-A3B 为 32:4、235B-A22B 为 64:4），每 KV head 承载更多 query 读取，通道级离群程度需实测，不能由 head 数直接推断。
- attention 内新增 QK-Norm：若它作用在 Q/K 的 head 维上，会限制 Q/K 的 scale 分布，可能降低 attention 对 Q/K 离群点的敏感度（推断，未验证）；同时 QK-Norm 自身的权重也是量化对象。
- 无 QKV-bias：线性层量化对象是纯权重矩阵，不需要为 bias 单独安排校正或通道对齐。
- MoE 专家占权重总量绝大部分而激活稀疏：per-expert 覆盖与路由 top-k 稳定性是量化误差和部署格式的关键来源；无共享专家意味着没有"每 token 都激活"的补偿通路。
- 同仓库内 dense 与 MoE 的 `head_dim`、RoPE base、norm eps 一致，结构差异集中在 FFN（换成 MoE）与 KV heads 数，量化方案可按"主干与 attention 契约共用 + FFN / KV 分叉"组织。

**未知项**

- 论文与权重仓都不含任何量化实验：Qwen3 的 int4 / int8 / FP8 精度、校准方法与逐层结论均未披露。
- QK-Norm 的归一维度与实施位置、MoE gate 的具体形式、routing 实现细节均未核对建模代码。
- 各 dense 档的 `hidden_size` / `intermediate_size` 只核到 `config.json`；论文 Table 1 未给这些字段。
- 论文 §2 的词表 151,669 与权重仓 `vocab_size=151936` 不一致，差异来源未定位。

## 结果边界

- 论文是训练侧报告：Table 1 / Table 2 的 context（32K / 128K）与某个 checkpoint 的 `max_position_embeddings`（40960）不是一个数，不能互相替代。
- 论文评测（AIME、LiveCodeBench、BFCL 等）是论文训练设定下的结果，不代表量化后表现。
- 本页把整个 Qwen3 系列写在一页：dense 与 MoE 的差异只体现在规格表与 MoE 段，涉及具体型号时必须说明是哪个 checkpoint。
- 本库未加载任何权重、未运行量化或评测，未对具体 checkpoint 做数值验证。

## Agent 如何使用

首轮读本页即可判断：是否 MoE（30B-A3B / 235B-A22B 是）、attention 是否 GQA、是否含 QKV-bias（无）、是否含 QK-Norm（论文称有）、KV heads 数与 `head_dim`、是否 tie embedding。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 系列构成、Table 1 / Table 2 配置、QK-Norm 与 QKV-bias 改动、RoPE base 与长上下文扩展 | [`paper-qwen3`](../../sources/papers/paper-qwen3.md) |
| 目标 checkpoint 的 `config.json` 字段（层数、heads、专家数、`head_dim`） | [`repo-qwen3`](../../sources/repos/repo-qwen3.md) |
| MoE 路由与专家的共性结构 | [`component-router-expert`](../components/router-expert.md)、[`paradigm-moe`](../paradigms/moe.md) |
| GQA 与 QK-Norm 在 attention 中的位置 | [`component-attention`](../components/attention.md) |
| RoPE、YaRN / DCA 扩展 | [`component-position-encoding`](../components/position-encoding.md) |
| RMSNorm | [`component-normalization`](../components/normalization.md) |
| 目标 checkpoint 的真实配置与 runtime 行为 | 该 checkpoint 的 `config.json` / 官方源码 / 运行时（本库不提供） |

回答量化问题时必须说明：具体型号（dense 还是 MoE、哪一档）、层数与 KV heads、`head_dim`、是否含 QKV-bias、QK-Norm 情况、专家数与 top-k、是否 tie embedding、目标 context 与 RoPE 扩展方式。

## 来源与证据

- [Qwen3 技术报告](../../sources/papers/paper-qwen3.md)：§2 架构与 Table 1 / Table 2、§3.1–§3.2 预训练阶段（RoPE base、YARN / DCA）、§1 系列规模。
- [Qwen/Qwen3 系列权重仓](../../sources/repos/repo-qwen3.md)：8 个 checkpoint 的 `config.json` 字段。

本页结构以上述来源为限；两个来源都不含量化实验，量化内容除注明者外均为本库推断。
