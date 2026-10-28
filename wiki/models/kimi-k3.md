---
{
  "id": "model-kimi-k3",
  "title": "Kimi K3",
  "type": "model",
  "tags": [
    "ar",
    "linear-attention",
    "mla",
    "latent-moe",
    "attnres",
    "multimodal",
    "quantization"
  ],
  "aliases": [
    "Kimi-K3",
    "KimiK3",
    "月之暗面 Kimi K3"
  ],
  "related": [
    "paradigm-linear-attention",
    "component-attention",
    "component-attention-residuals",
    "paradigm-multihead-latent-moe",
    "component-router-expert",
    "component-multimodal-connector",
    "model-kimi-linear"
  ],
  "sources": [
    "paper-kimik3",
    "repo-kimik3"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-28"
}
---

# Kimi K3

## 一句话理解

Kimi K3 是原生多模态 MoE 模型：2.8T 总参、104B 激活、93 层、1M 上下文。它在三个维度上改信息流——序列维用 **3 个 KDA（Kimi Delta Attention，线性注意力）+ 1 个 Gated MLA** 的 3:1 混合；深度维用 **Attention Residuals** 取代均匀累加的残差；宽度维用 **Stable LatentMoE**（896 路由专家、每 token 激活 16 个、2 个全宽共享专家）。视觉由从零训练的 **MoonViT-V2** 提供，与文本共享同一骨干。

## 结构规格表

| 项 | Kimi K2 | Kimi K3 |
| --- | --- | --- |
| 层数 | 61 | 93 |
| 总参 / 激活 | 1.04T / 32.6B | 2.78T / 104.2B |
| hidden | 7168 | 7168 |
| attention heads | 64 | 96 |
| Latent MoE 维度 | — | 3584（0.5×） |
| 每专家 MoE 中间维 | 2048 | 3072 |
| 路由专家（总 / 激活） | 384 / 8 | 896 / 16 |
| 共享专家 | 1 | 2 |
| dense 层数 | 1 | 1 |
| 词表 | 160K | 160K |
| 训练上下文 | 128K | 1M |

层数、参数量、专家数、head 数、词表与上下文取自论文 Table 1（Kimi K2 与 K3 的结构对照）。权重仓 `config.json`（`repo-kimik3`）给出更精确的字段口径：MLA 的 `kv_lora_rank=512` / `q_lora_rank=1536` / `qk_nope_head_dim=128` / `qk_rope_head_dim=64` / `v_head_dim=128`，KDA 的逐层清单（`full_attn_layers` 24 层、`kda_layers` 69 层，对应 Table 1 的 69 KDA + 24 MLA）与 `gate_lower_bound=-5.0`（论文只给 `g_min=-5` 与 16-token tile 的定性描述），MoE 的 `routed_scaling_factor=1.0`，以及 `attn_res_block_size=12`（论文 §2.2 给 8 block × 12 层）。本表不能替代目标 checkpoint 的 `config.json`。

## 结构与数据流

- **混合注意力**（§2.1）：每 block 为 3 个 KDA 层 + 1 个 Gated MLA 层（3:1），模式在骨干中重复；**骨干末尾额外加一个 Gated MLA 层**，保证最后一层是全局注意力。权重仓的逐层清单与之一致：`full_attn_layers` 共 24 层（间隔 4 层，最后一层是 93）、`kda_layers` 共 69 层。
- **KDA**（§2.1.1）：在 delta-rule 递推上加逐通道遗忘门——`S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ`、`õ_t = S_tᵀ q_t`（Eq.1）。计算沿用 Kimi Linear 的分块并行形式：块间用入块状态递推、块内用掩码注意力并行（Eq.3–4）。另有两个设计点：**下界衰减**（K3 把衰减 logits 到逐步 log-decay 的映射从 Kimi Linear 的负 Softplus 改成 scaled sigmoid，使 log-decay 有下界 `g_min=-5`；按 16 token 次级 tile 分块沿用 Kimi Linear，config 字段 `gate_lower_bound=-5.0`）与**满秩门**（输出门，config `use_full_rank_gate=true`）。
- **Gated MLA**（§2.1.2）：MLA 把每 token 的 KV 压成低维隐向量 `c_t`，只缓存 `c_t` 再上投影重建 K/V；Kimi K3 只在全局注意力层保留它。与 Kimi K2 / K2.5 的两点差别是：**所有 MLA 层用 NoPE**（不做显式位置编码，位置敏感信息由中间的 KDA 层提供，因此扩上下文时无需重调 RoPE base 或用 YaRN）；给 MLA 增加**输入相关的逐通道满秩输出门** `y_t = W_o[Sigmoid(W_g x_t) ⊙ õ_t]`（Eq.7）。报告还说明训练时 attention 输出保持 FP32 以修正 flash attention 的有偏舍入误差。
- **Attention Residuals**（§2.2）：Full AttnRes 给每层一个可学习 pseudo-query `q_l = w_l`，key/value 取 token embedding 与各前序层输出，权重为 `exp(q_lᵀ·RMSNorm(k_i))` 归一化后加权求和（Eq.8–9），计算 `O(L²d)`、内存 `O(Ld)`；Block AttnRes 把 L 层分成 N 个 block，块内层输出先求和成单个表示，再只在 block 级表示之间做注意力以降低内存（config `attn_res_block_size=12`）。
- **Stable LatentMoE**（§2.3）：沿用 DeepSeekMoE 的共享 + 路由组织（Eq.11）——共享专家直接处理输入，路由路径先把 `x` 投到隐空间 `z = W↓x`、分发到被选专家、聚合后再经 `W↑` 回到 hidden，并在聚合与上投影之间插入 RMSNorm。三个组件：**Normalized LatentMoE**（聚合后、上投影前加 RMSNorm）、**SiTU-GLU**（Sigmoid Tanh Unit GLU，抑制激活爆炸）、**Quantile Balancing**（按匹配目标负载的路由分数分位数设置专家 bias）。config 对应字段为 `latent_moe_use_norm=true`、`hidden_act=situ`、`moe_router_activation_func=sigmoid`、`topk_method=noaux_tc`。
- **视觉**（§2.4）：MoonViT-V2 为 27 层、约 0.4B 参数的 ViT，采用 RMSNorm 且去掉线性层与注意力投影的全部 bias；**从零用 next-token prediction 训练**（K2.5 用 SigLIP 初始化，报告称从零训练是为稳定性）。图像与视频共享参数，注意力分解为帧内空间与帧间时间两遍，投影前用 2×2 pixel-shuffle 把视觉 token 降到 1/4。
- **Per-Head Muon**（§2.5）：矩阵参数用 Muon；对 Q / K / V 投影沿 head 维切分动量矩阵、每 head 单独做 Newton–Schulz 正交化。属优化器，不改变网络结构。
- **MTP**：论文 Table 1 与 §4.1.4 说明 K3 预训练带 1 层 MTP（并微调成 EAGLE-3 draft），而权重仓 `num_nextn_predict_layers=0`，即发布 config 不含 MTP 模块；两者口径不一致。

## 量化关联

**结构事实**

- 论文 §4.1.4 的 QAT：把占参数内存主体的 MoE 专家权重压到 **MXFP4**、激活用 **MXFP8**，attention 投影 / latent MoE 投影 / 共享专家 / MoE 路由器保持高精度；**SFT 与 RL 全阶段做 QAT**，RL 时 rollout 与训练共用同一量化方案（§4.1.1、§4.1.4）。
- MLA 层**不使用位置编码**（NoPE），且带满秩输出门（§2.1.2、config `mla_use_nope=true`、`mla_use_output_gate=true`）。
- Attention Residuals 在深度方向做注意力聚合：Full AttnRes 让每层读取 embedding 与所有前序层表示；K3 采用 **Block AttnRes**——层按 12 层分块（§2.2、config `attn_res_block_size=12`），块内层输出先求和成块表示，块间再对块表示（含 embedding）做注意力。
- Stable LatentMoE 每个 MoE 层 896 路由专家、每 token 激活 16、2 个全宽共享专家（首层为 dense，`first_k_dense_replace=1`）；路由分支带有 RMSNorm（§2.3）。
- 训练时 attention 输出保持 FP32（§2.1.2）。
- 视觉编码器从零训练、图像与视频共享参数、2×2 pixel-shuffle 降 token（§2.4）。
- **统一缓存布局**（§5.4.1）：KDA-aware prefix cache 把定长的 KDA 递归状态与按 token 分页的 MLA KV 打包进同一分页池（页字节大小统一，页内按 head 连续存储），两类 cache 共用一套分配 / 引用计数 / 淘汰逻辑。

**实现事实**（来自固定版本权重仓 `config.json`，须按该 revision 核对）

- 官方发布的 Kimi K3 checkpoint 自带 `quantization_config`：`Linear` 目标按 **MXFP4**（4-bit float、group size 32、对称、minmax 观察器）打包，`quant_method=compressed-tensors`；`ignore` 列表含 attention、共享专家、部分 `mlp.*_proj`、`lm_head`、视觉塔与投影器；该权重声明不带激活量化（`input/output_activations=null`），`kv_cache_scheme=null`。
- `KimiK3ForConditionalGeneration` / `model_type=kimi_k3`；文本侧 `KimiLinearForCausalLM` / `kimi_linear`；93 层、hidden 7168、96 Q heads / 96 KV heads、`first_k_dense_replace=1`。
- MLA：`kv_lora_rank=512`、`q_lora_rank=1536`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`。
- KDA：`linear_attn_config.head_dim=128`、`num_heads=96`、`use_full_rank_gate=true`、`gate_lower_bound=-5.0`、`short_conv_kernel_size=4`；层号清单 `kda_layers` 69 层、`full_attn_layers` 24 层。
- MoE：`num_experts=896`、`num_experts_per_token=16`、`num_shared_experts=2`、`moe_intermediate_size=3072`、`routed_expert_hidden_size=3584`、`moe_renormalize=true`。
- 视觉：27 层、`vt_hidden_size=1024`、`patch_size=14`、`merge_kernel_size=[2,2]`、`mm_projector_type=patchmergerv2`。
- `num_nextn_predict_layers=0`；`max_position_embeddings=1048576`。

**量化推断**

- 权重已经以 MXFP4 发布（专家侧 `Linear`）：attention（含 MLA 与 KDA 投影）、共享专家、`lm_head`、视觉塔与投影器都在 `ignore` 里，是相对未被压缩的部分；已发布精度不等于量化空间已定，逐层预算应由 Agent 按输入权重、目标后端与实测结果划分。
- KDA 是递推状态而非 KV cache：长序列下状态会被反复读写，量化误差沿时间累积；它与 MLA 的 latent KV cache 是两类不同的误差传播路径。
- MLA 层没有位置编码、且带逐通道满秩门：gate 的逐通道缩放会放大或抑制不同通道，per-channel 量化粒度需要与 gate 的分组对齐核对（推断）。
- Attention Residuals 让权重分布在深度方向被显式归一化（softmax over layers，key 先过 RMSNorm）：层间表示的量级差异被抑制，但这也意味着残差聚合路径上的量化误差会在多层之间混合。
- 896 专家、每 token 激活 16：per-expert 覆盖与路由 top-k 稳定性是误差与部署格式的主要来源；Quantile Balancing 的 bias 由分数分位数决定，量化后路由分数分布变化会直接改变专家选择。
- 视觉塔与投影器不在量化集合内，但它们在 `ignore` 名单里不等于对量化不敏感，只是当前发布格式没压缩它们。

**未知项**

- 论文不含量化后的精度对照：§4.1.4 只给 QAT 配方（专家 MXFP4 / MXFP8 激活、非专家组件保持高精度），没有校准方法、逐层敏感度或量化前后精度数字。
- `ignore` 名单里的 `re:.*mlp\.(gate|up|gate_up|down)_proj.*` 是否也覆盖路由专家内部的投影与 MoE router（论文 §4.1.4 说 router 保持高精度），以及 `routed_expert_hidden_size=3584` 与 `moe_intermediate_size=3072` 各自对应的维度，**须读权重映射与建模代码确认**。
- KDA 的满秩门与下界衰减的具体数值实现（tile 切分、门的下界如何在门控中施加）论文只给设计意图，未逐式核对。
- Quantile Balancing 的直方图估计与 bias 更新细节在附录，本页未核对。
- 论文与 config 都不给 `kv_cache_scheme`（为 `null`）：K3 的统一 cache 能否低比特存储、按什么粒度量化，没有来源支撑。§5.4.1 已给出 KDA-aware prefix cache 把两类状态打包进同一分页池，"能否并存"不是缺口，缺的是该统一格式的低比特方案。

## 结果边界

- 论文是训练侧报告，评测数字不代表量化后表现；MXFP4 既是权重发布格式也是 §4.1.4 的 QAT 配方，论文未给量化后精度，不代表精度已达标。
- Table 1 的 2.78T 是表内精确值，摘要的"2.8T"是同一量级的简写。
- "MoonViT-V2 与 SigLIP 初始化基线在视觉评测上相当"是论文在自身评测集上的消融结论，不能外推到其它视觉任务。
- `max_position_embeddings=1048576` 是配置声明上限，不证明任何后端能在该长度下服务。
- 本库未加载权重、未运行量化或评测，未对具体 checkpoint 做数值验证。

## Agent 如何使用

首轮读本页即可判断：是否 MoE（是，896 专家 top-16 + 2 共享）、attention 是否为线性与全局混合（是，3 KDA : 1 Gated MLA）、是否用 NoPE 的 MLA（是）、是否含 Attention Residuals（是）、是否原生多模态（是，MoonViT-V2）、发布格式是否已量化（是，专家侧 MXFP4 + 全阶段 QAT）。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| KDA 所属的线性注意力与门控状态更新范式 | [`paradigm-linear-attention`](../paradigms/linear-attention.md) |
| Attention Residuals 的定义与实现差异 | [`component-attention-residuals`](../components/attention-residuals.md) |
| MLA 与 GQA / MHA 的对照 | [`component-attention`](../components/attention.md) |
| LatentMoE 的术语与变体（含 Multi-Head LatentMoE） | [`paradigm-multihead-latent-moe`](../paradigms/multihead-latent-moe.md)、[`paradigm-latent-moe`](../paradigms/latent-moe.md) |
| MoE 路由与专家的共性结构 | [`component-router-expert`](../components/router-expert.md) |
| 多模态编码器与投影器 | [`component-multimodal-connector`](../components/multimodal-connector.md) |
| 上一代线性注意力模型 | [`model-kimi-linear`](kimi-linear.md) |
| KDA / AttnRes / Stable LatentMoE / MoonViT-V2 的原理、Table 1 与 §4.1.4 QAT 配方 | [`paper-kimik3`](../../sources/papers/paper-kimik3.md) |
| checkpoint 字段、逐层 KDA/MLA 清单、量化格式与 ignore 名单 | [`repo-kimik3`](../../sources/repos/repo-kimik3.md) |

回答量化问题时必须说明：是否 MoE（专家数与 top-k）、attention 的 KDA:Gated MLA 配比与 MLA 的 NoPE、KDA 是递推状态而非 KV cache、是否含 Attention Residuals、视觉塔与投影器是否在量化集合外，论文 §4.1.4 在 SFT+RL 全阶段做 QAT（专家 MXFP4 权重 / MXFP8 激活、非专家组件保持高精度），以及官方 checkpoint 已按 MXFP4 发布专家侧 `Linear`。

## 来源与证据

- [Kimi K3 技术报告](../../sources/papers/paper-kimik3.md)：§2.1 混合注意力与 KDA / Gated MLA、§2.2 AttnRes、§2.3 Stable LatentMoE、§2.4 MoonViT-V2、§2.5 Per-Head Muon、§3.2 Table 1 结构对照、§4.1.4 MXFP4 QAT 与 MTP draft 微调。
- [moonshotai/Kimi-K3 权重仓](../../sources/repos/repo-kimik3.md)：固定 commit 的 `config.json`、逐层注意力清单、MoE 字段与 `quantization_config`。

本页结构以上述来源为限；论文不含量化后的精度对照，MXFP4/MXFP8 是 QAT 配方与发布格式而非精度结论，量化内容除注明者外均为本库推断。
