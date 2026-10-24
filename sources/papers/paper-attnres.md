---
{
  "id": "paper-attnres",
  "title": "Attention Residuals",
  "category": "paper",
  "url": "https://arxiv.org/abs/2603.15031",
  "version": "v1（2026-03-16）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1、§2.1、§3（Eq.1）、§3.1（Eq.2–4）、§3.2（Eq.5–6）、§4.1（Eq.7–8）、§4.2 与 Algorithm 1（Table 1）、§5.1（Table 2、Figure 4）、§5.2（Figure 5、Table 3）、§5.3（Table 4、Figure 6）、§5.4.1（Figure 7）、§5.4.2（Figure 8）、§6.1（Eq.9、Table 5）、§6.2（Eq.10、Figure 9）、§7、附录 A、附录 B（Eq.11–17）",
  "topic": "core",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---
## 这份来源是什么

Attention Residuals（AttnRes）的官方技术报告（Kimi Team / MoonshotAI；arXiv 2603.15031 v1，代码入口 `github.com/MoonshotAI/Attention-Residuals`）。它把深度方向的 residual 累加从"固定单位权重的加法递推"改成"对历史层输出做输入依赖的 softmax attention"，并给出可扩展的 Block 变体与配套的流水并行通信、两阶段推理方案。是本库 `component-attention-residuals` 的结构依据，也给出它与 (m)HC、DenseFormer、Highway、DDL 等残差扩展的关系（对照 `component-residual-mhc`）。

与相邻来源的关系：mHC(-lite) 在本报告中作为**对照方法**出现在 scaling law 表与消融里（`paper-mhc` 才是它的原始论文）；Kimi Linear / KDA + MLA 是本报告做 48B 主实验的底座架构。

范围边界：这是**结构与方法**报告，全文不含量化、低比特、激活范围统计或后端 kernel 精度讨论；实验规模是 194M–528M 激活参数的 scaling 组 + 48B/3B 激活的主模型，没有更大的部署侧证据。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源 `main.tex` 及其全部 `\input`（`0-abstract`–`8-conclusion` + 附录）；另读 arXiv abs 页取版本号与标题。**arXiv HTML 全文渲染版与 ar5iv 现均转换失败、无正文**，故 `read_form` 的 `html` 指 abs 页，全部编号与事实以 LaTeX 源为准；只有 v1（2026-03-16）。
- 编号提示：消融表用 `\captionof{table}` 写在一个 `figure` 浮动体里，故 **Table 4 = 组件消融、Figure 6 = block size 曲线**同处一个 float；**Table 5 = 残差变体对比表**（§6.1 末尾）。章节：§3 AttnRes（3.1 Full、3.2 Block）、§4 Infrastructure（4.1 Training、4.2 Inference）、§5 Experiments、§6 Discussions、§7 Related Work；公式 1–6 在 §3、7–8 在 §4、9–10 在 §6、11–17 在附录 B。

## 关键事实

- **动机**（§1–2.1）：PreNorm + 标准 residual `h_l = h_{l-1} + f_{l-1}(h_{l-1})` 展开后是对所有先前层输出做**固定单位权重**的累加，hidden state 幅度随深度按 `O(L)` 增长、稀释每层相对贡献（PreNorm dilution）；局限归成三条：无选择性访问、聚合后信息不可逆丢失、后层被迫生成越来越大的输出。
- **定义**（§3，Eq.1–4）：`h_l = α_{0→l}·h_1 + Σ_{i=1}^{l-1} α_{i→l}·f_i(h_i)`，权重 `Σ α = 1`；`α_{i→l} = exp(q_lᵀ RMSNorm(k_i)) / Σ_j exp(q_lᵀ RMSNorm(k_j))`（深度方向 softmax attention）。`q_l = w_l` 是**每层一个可学习 d 维向量**，`k_i = v_i = h_1`（token embedding）或 `f_i(h_i)`；核内 RMSNorm 防止输出幅度大的层支配权重。
- **Block AttnRes**（§3.2，Eq.5–6）：把 `L` 层分成 `N` 个 block（每块 `S = L/N` 层），块内输出求和成一个表示 `b_n`；跨块只对 `N` 个块表示 + embedding 做 attention，块内第 `i` 层额外把部分和作为一个源。存储从 `O(L)` 降到 `O(N)`、计算从 `O(L²)` 降到 `O(N²)`。两个端点：`N = L` 退化为 Full，`N = 1` 退化为标准 residual。经验上 `N ≈ 8` 就能恢复大部分收益（"每 token 只需存 8 个 hidden state"），固定的 `N` 也**界定了 KV cache 的大小**。
- **开销**（§3.1、§4.1）：Full 每 token `O(L²d)` 计算、`O(Ld)` 存储；普通训练里 `O(Ld)` 与反向已保留的激活重合、**不额外占内存**，但大规模训练的激活重算与流水并行下必须显式保活并跨 stage 传输。流水并行下跨 stage 缓存把每个块只传一次后，峰值从 `O(C)` 降到 `O(P)`（`V×` 改善）；激活检查点使**每层激活占用与标准架构完全相同**，流水并行端到端开销 **<4%**、无流水并行时可忽略。
- **推理两阶段**（§4.2，Algorithm 1，Table 1）：伪 query `w_l` 与输入无关，块内 `S` 层的跨块 attention 可合并成一次批量查询（Phase 1，返回 max 与 log-sum-exp），块内再顺序计算（Phase 2），两者用 online softmax 精确合并。每层内存访问降到 `(N/S + 3)d` 读、`2d` 写（Table 1 设定 `L=128, N=8, S=16, m=4`；Block 典型 `5.5d`、Full `24d`，对照标准 residual `3d`、mHC `34d`）；典型推理端到端延迟开销 **<2%**。prefill 需保存 `N·T·d` 个块表示（128K token、8 block 约 15 GB），按序列维切分到 `P` 个 TP 设备降到约 1.9 GB/设备，配 16K chunked prefill 进一步降到 0.3 GB 以下。
- **模型侧改动很小**（§5）：底座与 Kimi Linear 完全一致（KDA 与 MLA 按 3:1 交替、每层后接 MoE FFN），**唯一改动是 residual**；每层只增加一个 RMSNorm 和一个 `w_l ∈ R^d`（按论文"attention/MLP 各算一层"的口径；落到 Transformer block 上，pre-attention 与 pre-MLP 各有一套独立的伪 query + RMSNorm）。关键约束：**所有伪 query 必须初始化为 0**，使初始权重均匀（退化为等权平均）以避免训练初期震荡。
- **Scaling law**（§5.1，Table 2、Figure 4）：5 个规模（194M/241M/296M/436M/528M 激活参数，38.7B–119B token），统一 8192 上下文，同规模内共享同一套按 baseline 选的超参（论文称这**偏向 baseline**）。拟合：Baseline `1.891·C^-0.057`、Block `1.870·C^-0.058`、Full `1.865·C^-0.057`；5.6 PFLOP/s-days 处 Block 1.692 vs baseline 1.714，约 **1.25× 算力优势**；最大规模处 Full 与 Block 只差 0.001。Table 2 同时列 mHC(-lite) 作参考：Full AttnRes 优于 mHC，Block 在更低的每层内存 I/O 下追平 mHC。
- **主结果**（§5.2，Table 3、Figure 5）：48B 配置（27 block = 54 层、256 路由专家激活 8 + 1 共享、48B 总 / 3B 激活），Block AttnRes 取每块 6 层 → 9 block + embedding 共 10 个深度来源；训练沿用 Kimi Linear 1.4T token 配方（4096 上下文、Muon、WSD、global batch 8M，先 1T 预训练 + ~400B 中训练，再扩到 32K；MLA 用 NoPE 故上下文扩展不需 YaRN）。训练动态：验证损失全程更低；输出幅度 baseline 单调增长、Block AttnRes **有界周期性**；baseline 最早几层梯度异常大，AttnRes 的 softmax 竞争使梯度分布**更均匀**。论文称在所有基准上持平或更好——GPQA-Diamond 36.9→44.4、Minerva Math 53.5→57.1、HumanEval 59.1→62.2、MMLU 73.5→74.6、TriviaQA 69.9→71.8、C-Eval 79.6→82.5，MMLU-Pro 52.2 对 52.2（持平）。
- **消融**（§5.3，Table 4、Figure 6；caption 写 "16-layer"，按论文口径实为 16 block = 32 层）：baseline 1.766；DenseFormer（固定输入无关系数）1.767 **无收益**；mHC 1.747；Full 1.737、Block 1.746；滑窗聚合（只留最近 8 层 + embedding）1.764，明显不如 Full/Block（说明"选择性访问远处层"比"访问更多近处层"重要）。组件：输入依赖 query 更好（1.731）但每层多一个 `d×d` 投影且解码需顺序访存，故默认用学习到的 query；去掉 q/k 换输入无关标量变差（1.749）；softmax 换 sigmoid 变差（1.741）；per-head 聚合变差（1.752）；去掉 key 上的 RMSNorm 让 Full/Block 都变差（1.743/1.750）。block size：`S=1` 即 Full，`S=2/4/8` 都在 1.746 附近，`S=16/32` 向 baseline 退化，实践中固定约 8 个 block。
- **架构偏移与权重形态**（§5.4）：固定算力（约 `6.5e19` FLOPs）与激活参数（约 `2.3e8`）下扫 25 个配置，两种方法都在 `H/L_b ≈ 0.3` 最优，AttnRes 在全部 25 个配置上低 0.019–0.063；baseline 最优点在 `d_model/L_b ≈ 60`（1.847）、AttnRes 移到 ≈45（1.802），即同等预算下更偏好"更深更窄"（论文明确这不构成部署建议）。学到的权重：局部性被保留（每层最关注紧邻前一层），同时出现选择性非对角集中；embedding 始终保有不可忽略权重（pre-attention 层尤其明显）；pre-MLP 输入更依赖近处表示、pre-attention 感受野更宽。§6.2 把"某些层无论输入都获得高权重"称为**深度方向的 attention sink**。
- **统一视角**（§6.1–6.2，Eq.9–10、Table 5）：标准 residual 的深度混合矩阵 `M` 是全 1 下三角；Highway 是 1-semiseparable；(m)HC 是 `m`-semiseparable；Full AttnRes 给出稠密、秩为 `L` 的 `M`；Block AttnRes 的有效秩介于 `N` 与 `N+S` 之间。论文据此把这些残差变体归为**深度方向的线性 attention**，AttnRes 则是深度方向的 softmax attention；并指出当核 `φ(q,k)` 可分解时深度 attention 会塌回递推（MRLA–GLA、DDL–DeltaNet 的来源）。Figure 2 给出 Block AttnRes 的 PyTorch 风格伪代码；报告未提供完整模型实现。

## 不能推出的结论

- 论文不含任何量化、低比特或激活/权重数值分布统计；不能据此推出 AttnRes 下"哪些张量对量化敏感""cache 该用多少 bit"或后端 kernel 支持情况。本页只能提供结构与数值路径（每层一个 RMSNorm、一个 d 维 query、跨层 softmax 权重、每 token 需缓存 `N` 个块表示）。
- 论文没有给出 AttnRes 与标准 residual 在**质量层面锁定等显存或等延迟**的完整对比（只给每层内存访问量 Table 1 与端到端开销百分比）；scaling law 组共享同一套超参（论文自称偏向 baseline），开销数字（训练 <4%、推理 <2%、prefill 0.3 GB/设备）是作者在其实现与硬件上的测得值，不能当作其它实现的保证。
- `N ≈ 8`、`S=2/4/8` 附近等价、`H/L_b ≈ 0.3` 最优、`d_model/L_b` 最优点从 60 移到 45，都是在特定小规模（194M–528M 激活参数；消融用 16 block、固定算力 `6.5e19` FLOPs）下的经验结果，不能直接外推到其它规模、专家配置或数据配比；论文自己也声明"更偏好深度"不等于部署建议。
- 结论来自**预训练/中训练后**的基准评测（4096 上下文训练、后扩到 32K；论文只报告预训练与中训练，未披露是否做过 RLHF/指令微调），不能推出指令微调后、长上下文或量化部署下的相对优势。Table 3 里 MMLU-Pro 是持平（52.2 对 52.2），摘要的"across all evaluated tasks"是整体表述，引用时不能把每项都当成有提升。
- Block AttnRes 的"固定 `N` 界定 KV cache 大小"意味着推理时每 token 额外持有 `N` 个 `d` 维块表示（论文的 15 GB/128K token 例子即由此而来），这是**新增的运行时状态**，不能等同于普通残差架构的 cache 需求，也不能用它推断某个 checkpoint 的实际 cache 布局。
- 论文把先验残差变体统一解释为深度方向线性 attention（§6.1–6.2），但该框架不是本文首创：论文自己写明该对偶性来自 chen2026postlayernorm、深度混合矩阵的半可分秩视角来自 mamba2；(m)HC、Highway、DenseFormer 的原始论文并没有这样描述自己，引用该等价关系时应注明出处。
- 本页只是论文级证据，不能证明某个具体 checkpoint 是否真的启用 AttnRes、block 数是多少、伪 query 是否按论文初始化；这些必须回到目标实现核验。
