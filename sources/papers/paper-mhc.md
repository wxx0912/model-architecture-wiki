---
{
  "id": "paper-mhc",
  "title": "mHC: Manifold-Constrained Hyper-Connections",
  "category": "paper",
  "url": "https://arxiv.org/abs/2512.24880",
  "version": "v2（2026-01-05）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1（Eq.1–4）、§2.2、§3（Eq.5、Table 1）、§3.1（Figure 2–3）、§3.2（Table 2）、§4.1（Eq.6）、§4.2（Eq.7–9）、§4.3.1（Eq.10–19）、§4.3.2（Table 3、Eq.20）、§4.3.3（Figure 4）、§5.1、§5.2（Figure 5、Table 4）、§5.3（Figure 6）、§5.4（Figure 7–8）、§6、附录 A（Table 5）",
  "topic": "core",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

## 这份来源是什么

mHC（Manifold-Constrained Hyper-Connections）的原始技术报告（DeepSeek-AI；arXiv 2512.24880，v2 2026-01-05，v1 2025-12-31）。它针对 Hyper-Connections（HC）把 residual stream 拓宽成 `n` 条并行流之后"复合转移矩阵不受约束、破坏 identity mapping"的问题，把 residual 混合矩阵 `H^res` 投影到**双随机矩阵流形**（Birkhoff 多面体）上，并给出配套的 kernel fusion、重算与 DualPipe 通信重叠方案。是本库 `component-residual-mhc` 的结构依据（对照 `component-attention-residuals`：AttnRes 走的是"跨层 softmax 选择"，mHC 走的是"约束的多流混合矩阵"）。

范围边界：论文是**架构 + 基础设施**报告，实验是 3B/9B/27B 的 MoE 预训练对比（HC vs mHC vs baseline）；不含量化或低比特误差分析，也未定量评估精度对结果的影响（只规定了混合精度策略）。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源（单文件 `main.tex`，`deepseek.cls`，模板内 `nonumbering` 选项实际仍编号）为主；另读 arXiv abs 页（Submission history 与作者列表，用于确认版本 v2 = 2026-01-05、机构 DeepSeek-AI）与 arXiv HTML 渲染版（`https://arxiv.org/html/2512.24880v2`）交叉核对表号。
- 下文按源码的浮动体出现顺序：Table 1–4 在正文、Table 5 在附录 A（源码在文末有把附录浮动体重编号为 `A1/A2…` 的 `\setcounter` 命令，但它出现在该表之后，渲染版本仍显示为 Table 5）；Figure 1–8；公式 1–20（其中 Eq.10–19 是 §4.3.1 的融合 kernel 张量/精度规格，集中在同一个 `align` 里）。

## 关键事实

- **HC 的形式**（§1，Eq.3）：`x_{l+1} = H^res_l·x_l + H^{post,T}_l·F(H^pre_l x_l, W_l)`，其中 residual stream 从 `C` 维拓宽成 `n×C`（`n` 为 expansion rate），`H^res_l ∈ R^{n×n}` 在流内混合特征，`H^pre_l, H^post_l ∈ R^{1×n}` 负责读出/写回；这三者都由"输入相关的动态项 + 静态偏置"组成（§3，Eq.5，动态项经 `tanh`、并带初始化为小值的门控标量 `α`）。论文取典型扩张率 `n = 4`（原文 typical/e.g.）。
- **HC 的失效机制**（§1，Eq.4）：把 Eq.3 递归展开后，前向信号由复合映射 `∏ H^res` 支配，而该乘积**不保持特征全局均值**，导致信号无界放大或衰减。§3.1 用两个诊断量量化——复合映射行和绝对值的最大值（前向最坏放大）与列和绝对值的最大值（反向）——称其为 **Amax Gain Magnitude**。
- **实测不稳定**（§3.1，Figure 2–3，27B 模型）：HC 在约 12k step 出现 loss 突增，与梯度范数不稳定高度相关；Amax Gain Magnitude 峰值接近 **3000**，远离理想的 1。
- **HC 各映射的贡献**（§3，Table 1，消融）：全部关掉为基准 `0.0`；只启用 `H^res` 得到 `−0.022`（相对基准的绝对 loss 差）；再加 `H^pre` 为 `−0.025`；三者齐全为 `−0.027`。即收益主要来自 `H^res`。
- **系统开销**（§3.2，Table 2）：单层、每 token 的 residual 维护 I/O——标准 residual 为读 `2C`／写 `C`；HC 为读 `(5n+1)C + n²+2n`／写 `(3n+1)C + n²+2n`，即约按 `n` 倍增长，且 `H^pre/H^post/H^res` 的中间激活需为反向保留，流水并行下通信量也是 `n` 倍（论文指出这会造成更大的流水气泡）。
- **mHC 的核心约束**（§4.1，Eq.6）：把 `H^res` 约束到双随机矩阵集 `{H^res ∈ R^{n×n} : H^res·1 = 1, 1ᵀ·H^res = 1ᵀ, H^res ≥ 0}`（即 Birkhoff 多面体）。`n = 1` 时双随机退化为标量 1，**回到原始 identity mapping**。论文给出三条性质：谱范数 `‖H^res‖₂ ≤ 1`（非扩张，抑制梯度爆炸）、双随机矩阵对乘法封闭（任意深度的复合仍是双随机）、以及它是置换矩阵的凸包（可理解为置换的凸组合；论文称反复相乘"倾向于"随深度单调增加信息混合）。另外对 `H^pre`、`H^post` 施加非负约束（§4.1；具体参数化见 §4.2 Eq.8 的 `σ(·)` 与 `2σ(·)`），避免正负系数相消。
- **参数化**（§4.2，Eq.7–9）：先把 `x_l ∈ R^{n×C}` 展平成 `vec(x_l) ∈ R^{1×nC}`（保留完整上下文），再做动态投影 `φ^pre, φ^post ∈ R^{nC×n}`、`φ^res ∈ R^{nC×n²}`，加静态偏置与门控标量 `α`；`H^res = Sinkhorn-Knopp(H̃^res)`。注意这与 HC 不同——HC 是对 `n×C` 逐流做 RMSNorm 并用 `θ^{pre}, θ^{post} ∈ R^{1×C}`、`θ^{res} ∈ R^{n×C}`，mHC 展平到 `nC` 维后投影。
- **Sinkhorn-Knopp 投影**（§4.2，Eq.9）：以 `M⁰ = exp(H̃^res)` 为起点，交替做行归一化与列归一化 `M^t = T_r(T_c(M^{t-1}))`，收敛到双随机矩阵；实验取迭代次数 `t_max = 20`（附录 Table 5 同）。
- **精度与 kernel**（§4.3.1，Eq.10–19）：`φ` 用 **tfloat32**，`vec(x_l)` 用 **bfloat16**，偏置、门控标量与系数用 **float32**；把偏置与线性投影合并成 `b_l`/`φ_l`、把 RMSNorm 权重也吸收进 `φ_l`（融合后 `φ_l ∈ R^{nC×(n²+2n)}`、`b_l ∈ R^{1×(n²+2n)}`），并把除法重排序到矩阵乘之后以降低延迟（数学等价）；实现三类专用 kernel（融合两次 scan、融合小系数运算、把 Sinkhorn 迭代放进单个 kernel 并为反向写自定义 kernel 在片上重算中间量）。融合后 `F_post,res` 这个 kernel 的读从 `(3n+1)C` 降到 `(n+1)C`、写从 `3nC` 降到 `nC`。多数 kernel 用 TileLang 实现。
- **激活重算**（§4.3.2，Table 3、Eq.20）：只常驻保存每 `L_r` 层的第一层输入 `x_{l0}`（`nC` 元素）与逐层的 `F(H^pre x_l)`（`C` 元素），其余在反向重算；重算块内瞬态占用 `(n+2)C × L_r`。最优块长 `L_r* = argmin[nC·⌈L/L_r⌉ + (n+2)C·L_r] ≈ √(nL/(n+2))`；由于重算块不能跨流水 stage，实践中让重算边界与流水 stage 对齐。
- **DualPipe 通信重叠**（§4.3.3，Figure 4）：扩展 DualPipe 调度，把 MLP 层的 `F_post,res` kernel 放在独立的高优先级计算流上、并避免 attention 层使用长驻留 kernel；重算与流水通信解耦（每个 stage 的首个激活 `x_{l0}` 已本地缓存）。整体结果：`n = 4` 时额外时间开销 **6.7%**（§1 与 §4.3 各给一次）。
- **主实验**（§5.1）：MoE 架构参照 DeepSeek-V3；对比 baseline / HC / mHC，`n = 4`；主体是 27B 模型（数据量与参数成比例），外加 3B、9B 做算力 scaling，再加一个 3B/1T token 的 token scaling。
- **27B 结果**（§5.2，Figure 5、Table 4）：mHC 相对于 baseline 的最终 loss 降低 **0.021**，梯度范数也回到与 baseline 相当的水平。八项基准（baseline → mHC）：BBH 43.8→**51.0**、DROP 47.0→**53.9**、GSM8K 46.7→**53.8**、HellaSwag 73.7→**74.7**、MATH 22.0→**26.0**、MMLU 59.0→**63.4**、PIQA 78.5→**80.5**、TriviaQA 54.3→**57.6**；对应 HC 分别为 48.9 / 51.6 / 53.2 / 74.3 / **26.4** / 63.0 / 79.9 / 56.3。也就是说 mHC 在七项上优于 HC，**唯一例外是 MATH（26.0 vs HC 26.4）**；论文强调相对 HC 在 BBH 上 +2.1、DROP 上 +2.3。
- **scaling**（§5.3，Figure 6）：算力曲线覆盖 3B/9B/27B，相对 baseline 的 loss 优势随算力只**轻微衰减**；另有 3B 模型在 1T token 内的 token scaling 轨迹。
- **传播稳定性**（§5.4，Figure 7–8）：理想情况下 mHC 的复合映射应为精确双随机（前向增益与反向增益都为 1）；由于 Sinkhorn 只迭代 20 次，实测单层反向增益略有偏离，复合后偏离增大但**有界，最大约 1.6**，相比 HC 的近 3000 下降约三个数量级。Figure 8 的 heatmap 显示 HC 在增益大的地方其它元素也普遍偏大（全局不稳定），mHC 则一致稳定。
- **模型规格**（附录 A，Table 5）：27B——30 层（首 1 层 dense）、72 路由专家 + 2 共享专家、每 token 激活 6 个、dim 2560、FFN 1536、32 个 attention head（dim 128）、MLA（KV rank 512）、RoPE（dim 64、θ=10000）、序列 4096、batch 1280、50000 步/262B token、Loss-Free 负载均衡；`n = 4`、门控 `α` 初始化 0.01、Sinkhorn `t_max = 20`。3B/9B/3B-1T 变体同表（3B：12 层、dim 1280、FFN 896、16 head、2.97B 总参/612M 激活、39.3B token；3B-1T：batch 2560、100000 步、1.05T token；3B/9B 路由专家均 64）。

## 不能推出的结论

- 论文不含量化或低比特误差的实测实验，也未定量分析数值精度对结果的影响；但它规定了混合精度策略（§4.3.1：`φ` 用 tfloat32、`vec(x_l)` 用 bfloat16、偏置/门控/系数用 float32），这是精度设计约束而非量化误差分析。不能据此推出 mHC 结构下"哪些张量该量化、误差如何传播"。本页能提供的只是数值路径事实：多流 residual 引入 `H^res/H^pre/H^post` 三类系数张量及其激活、Sinkhorn 迭代、以及反向重算路径。
- 三条理论性质（谱范数 ≤ 1、乘法封闭、Birkhoff 凸包）只对**精确**双随机矩阵成立；实现用 20 次 Sinkhorn 迭代得到近似解，论文自述单层反向增益已偏离 1、复合后最大偏离约 1.6。不能把"双随机"当成实现里严格满足的等式。
- 6.7% 的额外时间开销是 DeepSeek 内部大规模训练（DualPipe 调度 + TileLang kernel + `n=4`）的测量值，依赖具体实现与硬件；不能当作其它实现或其它 `n` 的保证。
- 论文在 Figure 6 caption 里把 3B/9B/27B 三个点都称作各自规模下的 "compute-optimal configuration"，但 §5.1 用的规则是"数据量与参数量成比例"（27B 用 262B token，约 9.7 token/参数），不等于 Chinchilla 口径；引用 "compute-optimal" 时须同时交代论文自己的比例规则，且 3B/9B/27B 的结论不能外推到更大规模、更长序列或不同专家配置。
- Table 4 的结论是"mHC 优于 HC 的多数基准"，MATH 一项是 HC 更好（26.4 vs 26.0）；引用时不能笼统写成"全面超过 HC"。
- "Superior scalability" 依据的是**相对 baseline 的 loss gap**趋势（Figure 6a）与内部大规模训练的经验，而非绝对 loss 的缩放指数；不能当作已量化的 scaling law。
- Amax Gain Magnitude（~3000 vs ~1.6）是在 27B 模型上、对某条选定序列在所有 token 上平均后的诊断量，不能直接换算成精度或数值误差界。
- 论文的 `n = 4`、block 结构（每层一个 attention + 一个 FFN 视为两个独立层）等设定不是通用最优；`n = 1` 才退化为标准 residual（Eq.6 的说明），不能由 `n = 4` 的结果反推其它 `n`。
- 本页只是论文级证据，不证明某个 checkpoint 是否启用 mHC、`n` 取多少、Sinkhorn 迭代几次；这些必须回到目标实现的源码与配置核验（本库 `repo-tile-kernels` 只能证明官方有 kernel 实现）。
