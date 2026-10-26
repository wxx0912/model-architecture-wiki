---
{
  "id": "model-glm45",
  "title": "GLM-4.5 / GLM-4.5-Air",
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
    "GLM4.5",
    "GLM-4.5-Air",
    "Glm4Moe",
    "智谱 GLM"
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
    "paper-glm45",
    "repo-glm45"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-26"
}
---

# GLM-4.5 / GLM-4.5-Air

## 一句话理解

GLM-4.5 系列是 decoder-only 自回归 MoE 模型，结构取向是**"窄而深"**：减小 hidden 与路由专家数量、增加层数（92 层）。关键部件是 GQA + partial RoPE + QK-Norm 的注意力、160 路由专家 + 1 共享专家（每 token 激活 8）的 MoE，外加 1 层 MTP 用于投机解码。注意力是 GQA（论文 §2.1 明确），因此不涉及 MLA 的 latent 压缩或线性注意力的状态递推。

## 结构规格表

| 项 | GLM-4.5 | GLM-4.5-Air |
| --- | --- | --- |
| 总参 / 激活 | 355B / 32B | 106B / 12B |
| 层数 | 92（前 3 层 dense，其余 MoE） | 46（前 1 层 dense） |
| hidden | 5120 | 4096 |
| attention | GQA：96 Q heads / 8 KV heads / head_dim 128；partial RoPE 比例 0.5；QK-Norm 有 | GQA：96 / 8 / 128；QK-Norm 无 |
| dense FFN 中间维 | 12288 | 10944 |
| MoE 中间维 | 1536 | 1408 |
| 专家（总 / 激活 / 共享） | 160 / 8 / 1 | 128 / 8 / 1 |
| MTP 层 | 1 | 1 |
| 上下文 | 128K（`max_position_embeddings=131072`） | 128K |

GLM-4.5 列取自官方 `config.json`（commit `cbb2c7c`）与论文 Table 1，二者一致；GLM-4.5-Air 列取自论文 Table 1，**未核对 Air 的 config.json**。论文 Table 1 的参数量口径含 MTP 层、不含词嵌入与输出层。引用具体 checkpoint 必须用目标 `config.json` 核对，本表不能替代。

## 结构与数据流

主干是 embedding → 3 层 dense block → 89 层 MoE block → final norm → LM head（`tie_word_embeddings=false`，输入与输出不共享）；另有 1 层 MTP（`num_nextn_predict_layers=1`）。MTP 在计算图中的具体挂载位置论文未给，本页不作断言。

- **注意力**：GQA，`num_attention_heads=96`、`num_key_value_heads=8`、`head_dim=128`；**partial RoPE** 只对 head_dim 的 `partial_rotary_factor=0.5`（64 维）施加旋转，其余维度不带位置编码（比例取自官方 `config.json`，论文只写"partial RoPE"、未给比例）；论文报告加入 **QK-Norm** 以稳定 attention logits 范围（§2.1，实施位置未核对）；`rope_theta=1e6`（中训练从 1e4 调整而来）；`attention_bias=true`（论文未说明覆盖哪些投影）。
- **MLP**：`hidden_act=silu`；dense 层中间维 12288，MoE 专家中间维 1536。
- **MoE**（论文 §2.1 / §2.4）：每层 160 路由专家 + 1 共享专家，每 token 选 top-8；论文说明用 loss-free balance routing 与 sigmoid gates；路由 bias 更新率前 15T tokens 为 0.001、之后 0.0，另加权重 0.0001 的序列级 balance loss。官方 `config.json` 另给：`norm_topk_prob=true`、`routed_scaling_factor=2.5`、`n_group=topk_group=1`（即不做分组限制路由）。
- **MTP**：`num_nextn_predict_layers=1`，一层 MoE 结构，论文说明用于推理期投机解码。

## 量化关联

**结构事实**

- partial RoPE 比例 0.5：head_dim 128 中只有 64 维带位置旋转，其余 64 维是无位置编码通道（比例来自官方 `config.json`；论文只写"partial RoPE"）。
- MoE 每层 160 路由专家 + 1 共享；每 token 选 top-8，共享专家每 token 都参与。
- 论文报告加入 QK-Norm 以稳定 attention logits 范围（§2.1）。
- `attention_bias=true`（论文未说明覆盖哪些投影）。
- MTP 是独立的一层 MoE 结构。

**实现事实**（来自固定版本权重仓 `config.json`，须按该 revision 核对）

- `Glm4MoeForCausalLM` / `model_type=glm4_moe`；`num_hidden_layers=92`、`first_k_dense_replace=3`、`num_nextn_predict_layers=1`；`partial_rotary_factor=0.5`、`use_qk_norm=true`；`n_routed_experts=160`、`n_shared_experts=1`、`num_experts_per_tok=8`；`tie_word_embeddings=false`。

**量化推断**

- 160 专家、每 token 激活 8：专家权重占总量绝大部分但激活稀疏，per-expert 覆盖与路由稳定性（top-k 变化）是量化误差的关键来源；共享专家每 token 参与，其误差会直接进入主干。
- QK-Norm 若作用于 attention logits 之前，会使 Q/K 的 scale 受控，可能降低 attention 分支对 outlier 的敏感度（推断，未验证）。
- `attention_bias=true`：不能假设所有线性层 bias 可被吸收或忽略，bias 应作为独立的量化或校正对象。
- partial RoPE 的"旋转通道（64 维）/ 非旋转通道（64 维）"动态范围与量化敏感度可能不同，旋转类等价变换（如 QuaRot / SpinQuant 式）需按通道拆分核对，不能整套 head_dim 一把处理。
- MTP 层若保留在推理图需独立检查其权重与输出；若仅用于训练或投机解码，则是单独的量化对象。

**未知项**

- 论文与权重仓都不含任何量化实验：GLM-4.5 的 int4 / int8 / FP8 精度、校准方法与逐层结论均未披露。
- GLM-4.5-Air 的官方 `config.json` 未核对。
- sigmoid gate 的具体形式、QK-Norm 的实施位置、partial RoPE 的维度切分只核到 config 字段，**未按建模代码核对**。

## 结果边界

- 论文是训练侧报告：Table 1 的参数量口径（含 MTP、不含 embedding 与输出层）与 checkpoint 显存不是同一个数。
- `max_position_embeddings=131072` 是权重仓支持的上限；论文的序列扩展路径是 4096 → 32768 → 131072，RoPE base 在扩到 32K 时从 1e4 调到 1e6。
- 论文评测（MMLU / BBH 等）是论文训练设定下的结果，不代表量化后表现。
- 本库未运行任何量化或评测；未对具体 checkpoint 做数值验证。

## Agent 如何使用

首轮读本页即可判断：是否 MoE（是）、专家数与 top-k（160/8+1）、attention 是否 GQA + partial RoPE + QK-Norm、是否含 MTP、线性层是否带 bias。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| GLM-4.5 的架构取舍、Table 1 配置、超参数 | [`paper-glm45`](../../sources/papers/paper-glm45.md) |
| 目标 checkpoint 的 `config.json` 字段 | [`repo-glm45`](../../sources/repos/repo-glm45.md) |
| MoE 路由与专家的共性结构 | [`component-router-expert`](../components/router-expert.md)、[`paradigm-moe`](../paradigms/moe.md) |
| GQA、partial RoPE、QK-Norm | [`component-attention`](../components/attention.md)、[`component-position-encoding`](../components/position-encoding.md) |
| RMSNorm | [`component-normalization`](../components/normalization.md) |
| 目标 checkpoint 的真实配置与 runtime 行为 | 该 checkpoint 的 `config.json` / 官方源码 / 运行时（本库不提供） |

回答量化问题时必须说明：具体版本（4.5 / Air）、专家数与 top-k、attention 布局（GQA + partial RoPE + QK-Norm）、是否含 MTP、bias 与 QK-Norm 情况、cache 与后端支持。

## 来源与证据

- [GLM-4.5 技术报告](../../sources/papers/paper-glm45.md)：§2.1 架构、Table 1 配置、§2.4 超参数。
- [zai-org/GLM-4.5 权重仓](../../sources/repos/repo-glm45.md)：固定 commit 的 `config.json` 字段。

本页结构以上述来源为限；来源不含任何量化实验，量化内容除注明者外均为本库推断。
