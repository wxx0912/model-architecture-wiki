---
{
  "id": "paper-deepseek-v4",
  "title": "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence",
  "category": "paper",
  "url": "https://arxiv.org/abs/2606.19348",
  "version": "v1（2026-04-26）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1、§2（§2.1 继承自 V3 的设计、§2.2 mHC、§2.3.1 CSA、§2.3.2 HCA、§2.3.3 其它细节、§2.3.4 效率讨论）、§3.5.1（KV Cache Structure and Management）、§4.2.1（Model Setups）、§4.3 Table 1（参数量对照）、§5.1.1（Reasoning Efforts 与 Table 2 / Table 3）、§5.2.1（FP4 QAT）",
  "topic": "architectures",
  "excerpt_sha256": null
}
---
## 这份来源是什么

DeepSeek-V4 系列技术报告（arXiv 2606.19348 v1，DeepSeek-AI）。报告覆盖架构、基础设施、训练与后训练、评测；本库只取结构相关部分：§2.1（继承 V3 的 MoE / MTP 及其改动）、§2.2（Manifold-Constrained Hyper-Connections）、§2.3.1–§2.3.3（混合注意力 CSA / HCA 与其细节）、§2.3.4（效率口径）。它是 `model-deepseek-v4` 的结构依据，也为 `component-attention`、`component-kv-cache`、`component-residual-mhc`、`component-router-expert`、`paradigm-moe` 提供"压缩 + 稀疏混合注意力、超连接、Hash 路由"的现实实例。

范围边界：报告描述的是 preview 版本，只覆盖 DeepSeek-V4-Pro 与 DeepSeek-V4-Flash 两个 MoE 模型；§4.2.1 给出两者的模型配置，但没有给逐层 `compress_ratios` 列表（该列表只在权重仓 config 里）；报告没有"低比特 vs 全精度 / 不同位宽"的对照精度实验。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2606.19348v1`），未读 LaTeX 源与 PDF；论文表格中的工具调用 schema 属论文内容，本页不取。
- 定位：§2.1 的 Mixture-of-Experts 与 Multi-Token Prediction 两段、§2.2 的 HC / mHC / 动态参数化、§2.3.1 CSA（Eq.9–19）、§2.3.2 HCA（Eq.20–26）、§2.3.3 的 Q/KV 归一化、partial RoPE、滑动窗口分支、attention sink（Eq.27）；§4.2.1 的 Model Setups（两个模型的结构量）；§5.1.1 的三种推理模式；§1 的系列规模与效率结论只在必要时引用。

## 关键事实

- **系列规模**（§1、§4.2.1）：DeepSeek-V4-Pro 总参 1.6T、激活 49B；DeepSeek-V4-Flash 总参 284B、激活 13B；两者都支持 100 万 token 上下文。摘要写预训练语料 "more than 32T tokens"，§1 细分为 Flash 32T、Pro 33T。
- **模型配置**（§4.2.1，逐项给出）：
  - **Flash**：43 层、hidden `d=4096`；**前两层用纯滑动窗口注意力**，其后各层 CSA 与 HCA 交错。CSA 的压缩率 `m=4`、indexer query heads `n_h^I=64`、indexer head dim `c^I=128`、attention top-k `512`；HCA 的压缩率 `m′=128`；两者 query heads `n_h=64`、head dim `c=512`、query 压缩维 `d_c=1024`；输出投影分组 `g=8`、每组中间输出维 `d_g=1024`；滑动窗口分支 `n_win=128`。全部 block 都用 MoE，前 3 个 MoE 层用 Hash 路由；每层 1 共享 + 256 路由专家、专家中间维 2048、每 token 激活 6；MTP 深度 1；mHC 扩展因子 `n_hc=4`、Sinkhorn-Knopp 迭代 `t_max=20`。
  - **Pro**：61 层、`d=7168`；**前两层用 HCA**，其后 CSA 与 HCA 交错。CSA 的 `m=4`、`n_h^I=64`、`c^I=128`、attention top-k `1024`；HCA 的 `m′=128`；`n_h=128`、`c=512`、`d_c=1536`；`g=16`、`d_g=1024`；`n_win=128`；每层 1 共享 + 384 路由专家、专家中间维 3072、每 token 激活 6；其余同 Flash。
- **继承与改动**（§2）：保留 Transformer 主干、DeepSeekMoE 与 MTP（MTP 配置与 DeepSeek-V3 完全相同），新增 mHC、混合注意力（CSA + HCA）与 Muon 优化器；其余未说明的细节沿用 DeepSeek-V3。
- **MoE 相对 V3 的改动**（§2.1）：亲和度分数从 `Sigmoid(·)` 改为 `Sqrt(Softplus(·))`；仍用 aux-loss-free 负载均衡，另加一个较轻的序列级平衡损失；**去掉了路由目标节点数量的约束**；把最初几个 Transformer block 的 dense FFN 换成**使用 Hash 路由的 MoE 层**，Hash 路由按输入 token id 经预定义哈希函数确定目标专家（§2.1 只说明专家选择的依据，未说明这些层是否仍计算门控分数对选中专家加权；共享专家不是缺口：§4.2.1 给出每个 MoE 层 `n_shared_experts=1`）。
- **mHC**（§2.2）：把残差流宽度扩大 `n_hc` 倍（Eq.1：`X_{l+1} = B_l X_l + C_l F_l(A_l X_l)`）；核心是把残差映射矩阵 `B_l` 约束到双随机矩阵（Birkhoff 多胞形，Eq.2），使谱范数不超过 1、残差变换非扩张，从而稳定深层堆叠；`A_l`、`C_l` 用 Sigmoid 约束为非负有界以避免信号抵消；三个映射的参数按"动态（输入相关）+ 静态"分解生成，双随机投影用 Sinkhorn-Knopp 迭代实现。
- **CSA**（§2.3.1）：先把每 `m` 个位置的 KV cache 压缩成 1 个条目——从两条 KV 序列 `C^a`、`C^b` 与对应压缩权重 `Z^a`、`Z^b` 出发（Eq.9–10），每 2m 个条目按压缩权重与可学习位置偏置加权成一个压缩条目（Eq.11–12，压缩窗口对 `C^a` 与 `C^b` 有重叠），序列长度压到 `1/m`；再按 DeepSeek Sparse Attention 选 top-k 个压缩条目（Eq.13–17，indexer 打分 `Σ_h w_{t,h}·ReLU(q_{t,h}·K_s)`）。
- **CSA 核心注意力**（§2.3.1）：选中的压缩条目同时作 key 与 value，按 MQA 做核心注意力（Eq.18–19）；query 由压缩隐向量 `c_t^Q` 低秩上投影得到，且与 indexer query 共用同一压缩隐向量；输出侧用**分组输出投影**（把 `n_h` 个 head 输出切成 `g` 组，每组先投到 `d_g` 再合并）降低投影开销。
- **HCA**（§2.3.2）：压缩策略与 CSA 类似，但压缩率 `m′ ≫ m`、**不做重叠压缩**、**不做稀疏选择**（在压缩后的 KV 上保持稠密注意力），序列长度压到 `1/m′`（Eq.20–23）；同样使用 shared KV MQA、低秩 query 与分组输出投影。
- **两类注意力的公共细节**（§2.3.3）：核心注意力之前对 query 的每个 head 与压缩 KV 的唯一 head 做 RMSNorm；对 query、KV 条目与核心注意力输出的**最后 64 维**施加 partial RoPE（对输出施加位置 `-i` 的 RoPE 以抵消 KV 携带的绝对位置）；除压缩分支外再加一条**滑动窗口分支**，为每个 query 额外提供最近 `n_win` 个未压缩 KV；核心注意力用**attention sink**，把可学习的 sink logits 以 `Exp(z′_h)` 加到 softmax 分母（Eq.27）。
- **效率口径**（§1、§2.3.4，均为估计值）：1M 上下文下，DeepSeek-V4-Pro 的单 token 推理 FLOPs 为 DeepSeek-V3.2 的 27%、KV cache 为 10%；DeepSeek-V4-Flash 分别为 10% 与 7%。
- **精度相关**（§1、§5）：routed expert 参数使用 FP4 精度；后训练阶段对 MoE expert 权重与 indexer 的 QK 路径做 **FP4 量化感知训练（QAT）**；论文同时指出当前 FP4×FP8 的峰值算力与 FP8 相同。
- **推理模式**（§5.1.1 Reasoning Efforts，Table 2、Table 3）：定义 Non-think、Think High、Think Max 三种模式，Think Max 会向 system prompt 注入一段特定指令。

## 不能推出的结论

- 报告是 **preview 版本**；§4.2.1 已给出层数、hidden、head 数、专家数、`head_dim`、`m`、`m′`、attention top-k、`n_win` 等，但**没有给逐层 `compress_ratios` 列表**——该列表只能从权重仓 `config.json` 取，且长度比层数多 1，多出项的含义未核实。
- 报告没有"低比特 vs 全精度 / 不同位宽"的对照精度实验：routed expert 用 FP4 存储、后训练含 FP4 QAT，被评测模型本身就运行在低比特路径上；**不能据此推断任意 checkpoint 在 FP4 / FP8 / int4 下的精度损失**，也不能推断校准方法与逐层敏感度。
- §1 / §2.3.4 的效率数字是**估计值**（estimated single-token FLOPs、accumulated KV cache size，Figure 1 右），不是实测吞吐或实测显存；论文给的 27% / 10% 是对比 DeepSeek-V3.2 的相对值，不能当作绝对开销。
- §2.3 只用符号 `m`、`m′`、`n_win`，具体取值在 §4.2.1：`m=4`、`m′=128`、`n_win=128`、attention top-k 为 512（Flash）/ 1024（Pro），与 `config.json` 一致；但压缩窗口如何切分、indexer 布局仍须按官方推理实现核对。
- §3.5.1 描述了 KV cache 的异构存储布局（SWA 的 state cache、未压缩尾部 token、磁盘存储等），但压缩窗口的具体切分方式与 indexer 布局仍须按官方推理实现核对，不能只凭本文推断。
- 论文评测（含 1M 上下文与 agent 任务）是论文训练与推理设定下的结果，不代表量化版本的表现。
- 本页只读 HTML 渲染版，未核对 LaTeX 源；评测数字未逐表复核。
