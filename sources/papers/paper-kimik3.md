---
{
  "id": "paper-kimik3",
  "title": "Kimi K3: Open Frontier Intelligence",
  "category": "paper",
  "url": "https://arxiv.org/abs/2607.24653",
  "version": "v2（2026-08-07）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1、§2（§2.1 混合注意力与 §2.1.1 KDA、§2.1.2 Gated MLA、§2.2 Attention Residuals、§2.3 Stable LatentMoE、§2.4 Native Vision 与 MoonViT-V2、§2.5 Per-Head Muon）、§3.2 的 Table 1（K2 与 K3 结构对照）、§4.1.4（MXFP4 Quantization-Aware Post-Training）",
  "topic": "architectures",
  "excerpt_sha256": null
}
---
## 这份来源是什么

Kimi K3 技术报告（arXiv 2607.24653 v2，Kimi Team）。报告覆盖架构、预训练、后训练、基础设施与评测；本库只取结构相关部分：§2.1（混合注意力、KDA、Gated MLA）、§2.2（Attention Residuals）、§2.3（Stable LatentMoE）、§2.4（MoonViT-V2）、§2.5（Per-Head Muon）、§4.1.4（MXFP4 量化感知后训练），以及 §3.2 的 Table 1（Kimi K2 与 K3 的结构对照）。它是 `model-kimi-k3` 的结构依据，也为 `paradigm-linear-attention`、`component-attention-residuals`、`paradigm-multihead-latent-moe`、`component-attention`、`component-multimodal-connector` 提供"线性注意力 + 全局 MLA 混合、注意力残差、LatentMoE"的现实实例。

范围边界：只覆盖 Kimi K3 一个模型；报告的训练与 RL / 基础设施部分（MoonEP、KDA 系统协同设计等）本库只在不影响结构理解时引用。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2607.24653v2`），未读 LaTeX 源与 PDF。
- 定位：§2.1 的 3:1 混合比例与末尾额外 Gated MLA、§2.1.1 KDA 的 Eq.1 与分块形式 Eq.3–4、§2.1.2 Gated MLA 的 Eq.7 与 NoPE 说明、§2.2 的 Eq.8–9（Full AttnRes）与 Block AttnRes、§2.3 的 Eq.11 与三个组件、§2.4 的 MoonViT-V2 结构、§2.5 的 per-head 正交化；§3.2 的 Table 1 逐格读取。

## 关键事实

- **规模**（§1、Table 1）：2.8T（Table 1 写 2.78T）总参、104B（104.2B）激活参数、原生多模态 MoE、1M token 上下文；相对 Kimi K2 的整体 scaling 效率提升约 2.5×。Table 1 另给：层数 61→93、hidden 7168 不变、每专家 MoE 中间维 2048→3072、路由专家 384→896、每 token 激活专家 8→16、共享专家 1→2、attention heads 64→96、dense 层数 1→1、词表 160K 不变、训练上下文 128K→1M、MTP 层数 1→1（各 1 层）。
- **混合注意力**（§2.1）：按层混合线性注意力与全局注意力。**每个 block 为 3 个 KDA 层 + 1 个 Gated MLA 层（3:1）**，该模式在骨干中重复；**骨干末尾额外加一个 Gated MLA 层**，保证最后一层一定是全局注意力。
- **KDA**（§2.1.1）：在 delta-rule 递推上加入**逐通道遗忘门**。单头更新为 `S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ`，输出 `õ_t = S_tᵀ q_t`（Eq.1）。计算上沿用 Kimi Linear 的**分块并行**形式：块间用入块状态 `S_[t]` 递推，块内用掩码注意力并行（Eq.3–4，`inter-chunk` + `intra-chunk` 两项）。KDA 另有两个设计点：**下界衰减**（把累计衰减的倒数限制在有限数值范围内，按 16 token 的次级 tile 分块计算）与**满秩门**。
- **Gated MLA**（§2.1.2）：MLA 把每个 token 的 KV 压成低维隐向量 `c_t`，只缓存 `c_t` 再上投影重建 K/V。Kimi K3 只在周期性的全局注意力层保留 MLA；与 Kimi K2 / K2.5 不同的是：**所有 MLA 层使用 NoPE（不做显式位置编码）**，位置敏感信息由夹在中间的 KDA 层提供，MLA 层只做全局内容交互（这样扩上下文时不需要重调 RoPE base 或上 YaRN）；另外给 MLA 增加**输入相关的逐通道满秩输出门**，`y_t = W_o[Sigmoid(W_g x_t) ⊙ õ_t]`（Eq.7）。报告同时说明：为修正 flash attention 的有偏舍入误差，**训练时 attention 输出保持 FP32**。
- **Attention Residuals**（§2.2）：用注意力式的深度聚合替换均匀累加的残差。Full AttnRes 给每层一个可学习 **pseudo-query** `q_l = w_l`，key 与 value 取 token embedding 与各前序层输出，权重为 `exp(q_lᵀ·RMSNorm(k_i))` 归一化后加权求和（Eq.8–9）；计算量 `O(L²d)`、内存 `O(Ld)`。Block AttnRes 把 L 层分成 N 个 block，block 内层输出先求和成单个表示，再只在 block 级表示之间做注意力，以降低内存（与流水并行下的跨 stage 通信）。
- **Stable LatentMoE**（§2.3）：沿用 DeepSeekMoE 的共享 + 路由专家组织（Eq.11）：共享专家直接处理输入；路由路径先把 `x` 投到隐空间 `z = W↓x ∈ ℝ^ℓ`，分发给被选专家，聚合后再经 `W↑` 回到 `ℝ^d`，并在聚合与上投影之间插入 RMSNorm。Kimi K3 每个 MoE 层**固定 2 个全宽共享专家**（首层为 dense，`first_k_dense_replace=1`），路由专家 **896 个、每 token 激活 16 个**（稀疏度 56）。三个组件分别是：**Normalized LatentMoE**（聚合后、上投影前加 RMSNorm，降低路由分支尺度对专家组合的敏感性）、**SiTU-GLU**（Sigmoid Tanh Unit GLU，抑制激活爆炸）、**Quantile Balancing**（按"匹配目标负载的路由分数分位数"设置每个专家的 bias 来做负载均衡）。报告给出的动机是：路由分支由 `W↓`、门控多分支专家 FFN 与 `W↑` 串成近四段连续矩阵乘，这种病态结构在 2.8T 规模下会产生内部激活爆炸；同时近千个专家的负载均衡超出了既有 aux-loss-free bias 更新适用的范围。
- **原生视觉**（§2.4）：文本、图像、视频由**同一个共享骨干**在同一上下文中处理，没有后挂的模态对齐阶段。视觉编码器 **MoonViT-V2 为 27 层、约 0.4B 参数的 ViT**，采用 RMSNorm 并**去掉线性层与注意力投影的全部 bias**；与 Kimi K2.5 的关键差别是**完全从零用 next-token prediction 训练**（K2.5 用 SigLIP 对比学习权重初始化），报告称原因是稳定性——预训练编码器接到 LLM 上后联合优化不稳（MoonViT-3D 梯度范数持续偏高且频繁尖峰），从零训练的 MoonViT-V2 全程稳定，且在视觉评测上与 SigLIP 初始化基线相当。图像与视频**共享参数**（沿用 MoonViT-3D：注意力分解为帧内空间与帧间时间两遍，时间维度再池化）；投影前用 **2×2 pixel-shuffle** 下采样，把视觉 token 数降到 1/4，使最高 3584×3584 像素在 1M 上下文内可承受。
- **Per-Head Muon**（§2.5）：矩阵参数（含 attention 投影）用 Muon 优化器；对 Q / K / V 投影进一步用 **per-head 变体**——沿 head 维切开动量矩阵，对每个 head 的块单独做 Newton–Schulz 正交化，理由是整矩阵正交化会让梯度/动量尺度大的 head 主导共享更新方向。
- **量化感知后训练**（§4.1.4）：为降低部署内存与成本，把**占参数内存主体的 MoE 专家权重压到 MXFP4**、激活用 **MXFP8**，而**非专家组件（attention 投影、latent MoE 投影、共享专家、MoE 路由器）保持高精度**；**整个后训练阶段（SFT 与 RL）都做 QAT**，RL 时 rollout 与训练共用同一量化方案以消除 train–inference 失配（§4.1.1、§4.1.4）。§5 另写 K3 的大部分激活用 block-wise FP8 量化配合 offload。另有 1 层 MTP 被微调成 EAGLE-3 draft（§4.1.4）。

## 不能推出的结论

- 报告给出的是结构与训练配方，**不含量化后的精度对照实验**：§4.1.4 只给出 QAT 配方（专家 MXFP4 权重 / MXFP8 激活、非专家组件保持高精度），没有校准方法、逐层敏感度或量化前后精度数字，不能据此推断 Kimi K3 的量化表现。
- KDA 的"满秩门"与"下界衰减"给了设计意图与分块形式：下界衰减的 `g_min = -5` 固定、按 16-token 次级 tile 计算累计衰减，均由论文 §2.1.1 给出；门控的完整数值实现（下界如何施加、tile 切分细节）仍须按官方实现核对，权重仓 config 只给 `gate_lower_bound=-5.0` 等字段。
- §2.3 的 Quantile Balancing 只给了思想与分位数规则，直方图估计与 bias 更新的实现细节在附录，本页未复核附录推导。
- §2.4 的"MoonViT-V2 与 SigLIP 初始化基线视觉评测相当"是论文的消融结论，限定在其评测集上，不能外推到其它视觉任务。
- Table 1 的总参 2.78T 与摘要的 2.8T 是同一量级的不同精度写法，前者是表内精确值。
- 论文的评测（长程编码、agent、视觉等）是论文设定下的结果，不代表量化版本的表现。
- 本页只读 HTML 渲染版，未核对 LaTeX 源与附录；评测数字未逐表复核。
