---
{
  "id": "component-attention-residuals",
  "title": "Attention Residuals (AttnRes)",
  "type": "component",
  "tags": [
    "attnres",
    "residual",
    "attention",
    "quantization"
  ],
  "related": [
    "component-residual-mhc",
    "component-normalization",
    "paradigm-ar",
    "check-quantization-preflight"
  ],
  "sources": [
    "paper-attnres",
    "repo-attention-residuals"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "attention residuals",
    "attention residual",
    "注意力残差"
  ],
  "updated": "2026-10-26"
}
---

# Attention Residuals（AttnRes）

## 一句话理解

标准 residual 对历史层输出做**固定单位权重**的累加（PreNorm dilution：hidden 幅度随深度按 `O(L)` 增长）。AttnRes 把这条深度方向的累加改成**输入相关、跨层的 softmax attention**：当前层用自己学到的 query 去加权聚合历史层表示，权重大小由内容决定而不是固定为 1。

## 结构与数据流

Full AttnRes 的定义（§3，Eq.1–4）：

```text
h_l = α_{0→l}·h_1 + Σ_{i=1}^{l-1} α_{i→l}·f_i(h_i),   Σ α = 1
α_{i→l} = exp(q_lᵀ RMSNorm(k_i)) / Σ_j exp(q_lᵀ RMSNorm(k_j))
q_l = w_l（每层一个可学习 d 维向量，与输入无关）
k_i = v_i = h_1（token embedding）或 f_i(h_i)（第 i 层输出）
```

即：把"层输出的累加"看成深度方向的一次 attention，`q_l` 是伪 query，`k_i = v_i` 是各层表示，`RMSNorm` 加在 key 上防止输出幅度大的层支配权重。每层只新增一个 RMSNorm 与一个 `w_l ∈ R^d`（论文按"attention / MLP 各算一层"计；由此推出落到 Transformer block 上时，pre-attention 与 pre-MLP 各一套独立的伪 query + RMSNorm）。

**Block AttnRes**（§3.2，Eq.5–6）：把 `L` 层分成 `N` 个 block（每块 `S = L/N` 层），块内输出求和成一个块表示 `b_n`；跨块只对 `N` 个块表示 + embedding 做 attention，块内第 `i` 层再额外把部分和作为一个源。存储从 `O(L)` 降到 `O(N)`、计算从 `O(L²)` 降到 `O(N²)`。

**推理分两阶段**（§4.2，Algorithm 1）：伪 query 与输入无关，块内 `S` 层的跨块 attention 可以先合并成一次批量查询（Phase 1，返回 max 与 log-sum-exp），块内再顺序计算（Phase 2），两者用 online softmax 精确合并。每层内存访问降到 `(N/S + 3)d` 读、`2d` 写（Table 1 设定 `L=128, N=8, S=16, m=4`；Block 典型 `5.5d`、Full `24d`，对照标准 residual `3d`、mHC `34d`）。

## 变体对照

| 变体 | 深度方向聚合的源 | 存储 | 计算 | 备注 |
| --- | --- | --- | --- | --- |
| Full AttnRes | 所有先前层输出 + embedding | `O(Ld)` | `O(L²d)` | 每 token 需存全部历史表示 |
| Block AttnRes | `N` 个块表示 + embedding | `O(Nd)` | `O(N²d)` | `N=L` 退化 Full，`N=1` 退化标准 residual；经验 `N ≈ 8` 可恢复大部分收益 |
| 标准 residual | 等权累加（固定单位权重） | — | — | 不归一化、深度累积幅度 O(L)（AttnRes 的权重归一化 Σα = 1） |

论文在 §6.2 把先验残差变体统一解释为**深度方向的线性 attention**：标准 residual 的深度混合矩阵是全 1 下三角、Highway 是 1-semiseparable、(m)HC 是 `m`-semiseparable、Full AttnRes 给出稠密满秩矩阵、Block AttnRes 有效秩介于 `N` 与 `N+S` 之间；AttnRes 则是深度方向的 softmax attention。各变体的 update rule 见 Table 5。

## 实现差异

不同实现之间可能不同的地方：Full 还是 Block（以及 `N`、`S`）、伪 query 是学到的向量还是输入相关的 `d×d` 投影、是否在 key 上做 RMSNorm、聚合是 per-head 还是整维、块表示的求和方式与 block 边界,以及流水并行下跨 stage 缓存与激活重算的实现。论文的模型侧改动只有 residual：底座与 Kimi Linear 一致（KDA 与 MLA 按 3:1 交替、每层后接 MoE FFN）；一个关键约束是**所有伪 query 必须初始化为 0**（使初始权重均匀、退化等权平均），避免训练初期震荡。

## 量化关联

**结构事实**

- 每层新增一个 RMSNorm（作用在 key 上）、一个 `w_l ∈ R^d` 伪 query，以及一组 `Σα = 1` 的跨层 softmax 权重。
- 推理期每 token 额外持有 `N` 个 `d` 维块表示（Block 的固定 `N` 界定了这部分运行时状态的大小）；论文例子里 128K token、8 block 的 prefill 约需 15 GB，按序列维切到 `P` 个 TP 设备后约 1.9 GB/设备。

**量化推断**

- 历史表示 cache、softmax 聚合权重、跨层聚合与 block 边界是新的误差传播路径。
- 量化不能只看当前层的 attention/FFN：应观察历史 state、聚合 logits/weights、聚合输出与跨层误差累积。
- block 表示的缓存精度、内存布局与通信 kernel 是部署约束；论文的结构结论不等于已有低比特 kernel 支持。

**未知项**

- 论文不含量化、低比特、激活/权重数值分布统计；官方仓库在所读 commit 下**没有代码**。

## 结果边界

- 论文不含任何量化或低比特实验，也没有激活/权重分布统计；本页量化内容全为结构推断，不能推出"哪些张量对量化敏感""cache 该用多少 bit"。
- `N ≈ 8`、`S=2/4/8` 附近等价、`H/L_b ≈ 0.3` 最优、`d_model/L_b` 最优点从 60 移到 45，都是特定小规模（194M–528M 激活参数；消融用 Table 2 的 16-head 模型，caption 写 “16-layer”，本库按配置换算为 16 block = 32 层，`6.5e19` FLOPs 是 §5.4 容量扫描的固定算力）下的经验结果，不能外推。
- 开销数字（训练 <4%、推理 <2%、prefill <0.3 GB/设备）是作者在自身实现与硬件上的测得值，不是其它实现的保证。
- 结论来自**预训练 / 中训练后**的评测（未披露是否做过 RLHF / 指令微调）；Table 3 里 MMLU-Pro 是持平（52.2 对 52.2），摘要的 "across all evaluated tasks" 是整体表述，不能把每一项都当成有提升。
- §6.1–6.2 的"深度方向线性 attention"框架不是本文首创（论文注明对偶性来自 chen2026postlayernorm、半可分秩视角来自 mamba2）；这些原始论文如何自述，本库未逐篇核对。
- `repo-attention-residuals` 只含 README、技术报告 PDF 与图素材，不含代码；本页实现层描述以 `paper-attnres` 为限，不能证明某个 checkpoint 真的启用 AttnRes 或其 block 数。

## Agent 如何使用

首轮读本页即可判断目标是 Full 还是 Block AttnRes、block 数、以及每层新增了哪些张量与运行时状态。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 定义与权重公式、Block 机制、两阶段推理与全部消融 | [`paper-attnres`](../../sources/papers/paper-attnres.md) |
| 官方仓库现状（无代码，仅论文与配图） | [`repo-attention-residuals`](../../sources/repos/repo-attention-residuals.md) |
| 与 mHC 等残差变体的关系 | [Residual and mHC](./residual-mhc.md) |
| 每层归一化在量化中的处理 | [Normalization](./normalization.md) |
| 目标模型是否真的启用 AttnRes、block 数、伪 query 初始化 | 目标模型源码 / 配置（本库不提供） |

回答 AttnRes 量化问题时必须说明：Full 还是 Block、`N`/`S` 取值、伪 query 与 RMSNorm 的部署形式、每 token 需要缓存的块表示数量，以及目标后端与精度。

## 来源与证据

- [Attention Residuals](../../sources/papers/paper-attnres.md)：动机与定义（Eq.1–4）、Block 变体（Eq.5–6）、两阶段推理与内存访问（Table 1）、模型侧改动、scaling law 与消融（Table 2–4）、统一视角（§6，Table 5）。
- [MoonshotAI/Attention-Residuals](../../sources/repos/repo-attention-residuals.md)：固定 commit 下的仓库文件树（只有 README/PDF/素材，无代码）。

本页结构以上述来源为限；论文不含量化实验，量化内容均为结构推断。
