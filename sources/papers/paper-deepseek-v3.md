---
{
  "id": "paper-deepseek-v3",
  "title": "DeepSeek-V3 Technical Report",
  "category": "paper",
  "url": "https://arxiv.org/abs/2412.19437",
  "version": "v2（2025-02-18）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "§1（含训练成本表）、§2.1.1、§2.1.2、§2.2、§3.2.1–3.2.3、§3.3 与 content/fp8.tex、§4.2、§4.3、§4.4 与 tables/base_evaluation.tex、§5.3 与 tables/chat_evaluation.tex、附录 B.1–B.2",
  "topic": "architectures",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---
## 这份来源是什么

DeepSeek-V3 技术报告（arXiv 2412.19437，v2）。671B 总参数 / 每 token 激活 37B 的稀疏 MoE 语言模型。基本架构沿用 DeepSeek-V2 的 **MLA + DeepSeekMoE**，在此之上加了三件事：**无辅助损失的负载均衡**（给每个专家一个只参与路由的 bias 项）、**多 token 预测（MTP）**训练目标、以及 **FP8 混合精度训练**（论文称首次在超大规模模型上验证）。

支撑本库 `paradigm-moe`（sigmoid affinity、无辅助损失均衡、node-limited routing、不丢 token）、`component-attention` 与 `component-kv-cache`（MLA 的缓存对象）、`models/deepseek`，以及**量化相关页面**——本页是"训练期低比特/FP8"的一手材料：量化粒度、累加精度、"激活梯度对量化高度敏感"的实测证据。

边界：论文的 FP8 是**训练精度框架**，不是推理期的权重量化或低比特 kernel 方案；论文没有给出 INT8/INT4 权重部署的实验。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源 `main.tex` 与 `\input` 的 `content/fp8.tex`、`tables/base_evaluation.tex`、`tables/chat_evaluation.tex`；版本号取自 abs 页 Submission history（该页只提供元数据，不是正文）。v1 2024-12-27、v2 2025-02-18。
- 下文 §、公式、表号按 v2 LaTeX 源的浮动体出现顺序（Table 1 即 §1 的训练成本表）。

## 关键事实

- **规模与成本**（§1）：671B 总参数 / 每 token 激活 37B；预训练 14.8T token；总训练成本 2788K H800 GPU 小时（预训练 2664K + 上下文扩展 119K + 后训练 5K），按 $2/GPU 小时折约 $5.576M；论文称预训练"remarkably stable"、全程没有不可恢复的 loss spike，也没有回滚。
- **MLA**（§2.1.1，与 V2 相同）：`c^{KV}_t = W^{DKV} h_t`，上投影出 `k^C_t = W^{UK} c^{KV}_t`、`v^C_t = W^{UV} c^{KV}_t`；解耦 key `k^R_t = RoPE(W^{KR} h_t)` 承载位置信息，`k_{t,i} = [k^C_{t,i}; k^R_t]`；query 侧低秩压缩 `c^Q_t = W^{DQ} h_t`、`q^C_t = W^{UQ} c^Q_t`、`q^R_t = RoPE(W^{QR} c^Q_t)`，论文明确 query 压缩只为降低**训练期激活显存**。推理时**只缓存 `c^{KV}_t` 与 `k^R_t` 两个量**，attention 分母是 `sqrt(d_h + d_h^R)`。
- **MoE 与无辅助损失均衡**（§2.1.2）：`h'_t = u_t + Σ_{i=1}^{N_s} FFN^{(s)}_i(u_t) + Σ_{i=1}^{N_r} g_{i,t} FFN^{(r)}_i(u_t)`；affinity 用 **sigmoid**（论文称与 V2 略不同），在选中的 affinity 之间归一化得 gate。给每个专家一个 bias `b_i`，**只加在 top-K 的路由选择上**（`s_{i,t} + b_i ∈ TopK`），真正乘到 FFN 输出的 gate 仍取原始 affinity；训练每 step 末按整个 batch 的专家负载动态调整——过载 `b_i ← b_i − γ`、欠载 `b_i ← b_i + γ`。仍保留序列级 `L_Bal = α Σ f_i P_i` 但 `α` 极小。**node-limited routing**：每个 token 最多送到 `M` 个节点，节点按其上"最高 `K_r/M` 个 affinity 之和"选出。因为均衡效果好，V3 在**训练和推理都不丢 token**（报告未说明 V2 是否丢 token）。
- **MTP**（§2.2）：用 `D` 个**串行**模块预测 `D` 个额外 token（与 Meta 的并行独立输出头不同），保持完整因果链。第 k 个模块 = 与主模型共享的 embedding 层、共享输出头、一个 Transformer block、以及投影 `M_k ∈ R^{d×2d}`；`h'^k_i = M_k[RMSNorm(h^{k-1}_i); RMSNorm(Emb(t_{i+k}))]`；损失 `L_MTP = (λ/D) Σ_{k=1}^{D} L^k_MTP`。推理时可**直接丢弃 MTP 模块**，也可**复用于投机解码**。
- **模型超参**（§4.2）：61 层、hidden 7168、参数按 std 0.006 初始化；`n_h=128`、`d_h=128`、`d_c=512`、`d'_c=1536`、`d_h^R=64`；**除前三层外所有 FFN 都换成 MoE 层**；每 MoE 层 **1 个共享专家 + 256 个路由专家**、专家中间维 2048、**每 token 激活 8 个路由专家**、每 token 最多送到 4 个节点；MTP 深度 `D=1`；像 V2 一样在压缩 latent 之后加 RMSNorm、在宽度瓶颈乘缩放因子 → 671B 总参 / 37B 激活。
- **训练超参**（§4.2）：14.8T token、最大序列 4K；AdamW `β=(0.9,0.95)`、`weight_decay=0.1`；学习率先 2K 步升到 `2.2e-4`，保持到 10T token，再用 4.3T token 余弦衰减到 `2.2e-5`，最后 500B token 中前 333B 保持 `2.2e-5`、后 167B 用 `7.3e-6`；梯度裁剪 1.0；batch size 在前 469B token 从 3072 升到 15360；`M=4`；`γ=0.001`（前 14.3T）后为 `0.0`（最后 500B）；`α=0.0001`；`λ=0.3`（前 10T）后为 `0.1`。
- **FP8 混合精度训练**（§3.3 与 `content/fp8.tex`）：三个 GEMM（`Fprop`/`Dgrad`/`Wgrad`）都用 FP8（输入 FP8、输出 BF16 或 FP32，理论比 BF16 快 2 倍，`Wgrad` 允许把激活按 FP8 缓存以省显存）；**保留高精度**的是 embedding、输出头、MoE gating、归一化算子、attention 算子，以及 master weight、权重梯度、优化器状态。**细粒度量化**：激活按 **`1x128` tile**（每 token 每 128 通道）、权重按 **`128x128` block** 分组缩放，标准 FP8 GEMM 不支持、需配合 FP32 累加自行实现。**累加精度**：H800 上 FP8 GEMM 累加只保留约 14 bit（远低于 FP32），因此每隔 `N_C=128` 个元素的 MMA 就提升到 CUDA Core 做高精度累加。**格式**：所有张量统一用 **E4M3**（靠细粒度分组共享指数位弥补动态范围，即 mantissa over exponents）。**在线量化**：每个 tile/block 都**在线**求 amax 再算 scale，不用历史 amax 的延迟量化。**低精度存储与通信**：AdamW 一阶/二阶矩用 BF16 跟踪（master weight 与梯度仍 FP32）；attention 后 Linear 的输入改用定制 **E5M6**、scale 取 2 的整数次幂、反向从 `1x128` tile 转 `128x1`；MoE 里 SwiGLU 输入按 FP8 缓存、反向重算输出；MoE 上投影前的激活与下投影前的激活梯度量化成 FP8 后再参与 dispatch 通信，而 forward/backward 的 **`combine` 组件仍保持 BF16** 以保住关键路径精度。
- **FP8 消融**（附录 B.1）：在约 16B/1.33T token 与约 230B/0.9T token 两个基线上对比，FP8 相对 BF16 的相对误差保持在 **0.25% 以下**。
- **激活梯度的精度敏感性**（附录 B.2）：把与 `Dgrad` 相关的张量改成 `128x128` block-wise 量化，会让一个约 16B 的 MoE 在约 300B token 的训练中**发散**；论文**推测**原因是激活梯度在 token 间高度不平衡、产生 token 相关的离群值，而 block-wise 量化无法处理这类离群值。
- **上下文扩展与基础设施**（§4.3、§1、§3）：预训练后用 YaRN 做两阶段扩展、各 1000 步，4K → 32K → 128K（`s=40`、`α=1`、`β=32`、缩放因子 `√t = 0.1 ln s + 1`），且**只作用于解耦共享 key `k^R`**。DualPipe 流水并行（气泡更少、把大部分通信与计算重叠）、自研跨节点 all-to-all kernel 吃满 IB 与 NVLink 带宽，并把显存优化到**不需要张量并行**即可训练。
- **评测**（§4.4/§5.3）：Base 版（对比 V2-Base / Qwen2.5-72B-Base / LLaMA-3.1-405B-Base）——MMLU 87.1、MMLU-Pro 64.4、BBH 87.5、HumanEval 65.2、MMMLU-non-English 79.4；Chat 版（对比 V2-0506 / V2.5-0905 / Qwen2.5-72B-Inst / LLaMA-3.1-405B-Inst / Claude-3.5-Sonnet-1022 / GPT-4o-0513）——MMLU 88.5、MMLU-Pro 75.9。两张表题注都强调用同一套内部评测框架；base 表逐行标 `# Shots` 与 metric，chat 表只标 metric、输出长度限 8K、少于 1000 样本的基准用不同 temperature 多次测试取稳健结果。§4.4 自述评测框架在过去几个月有变化、V2-Base 数字与先前报告略有差异。

## 不能推出的结论

- 论文的 FP8 是**训练**精度框架（GEMM 输入 FP8、累加高精度、部分算子刻意保持高精度），不能直接当作推理期权重量化或低比特部署方案；论文没有给 INT8/INT4 权重、激活量化的推理实验。
- 「FP8 与 BF16 相对误差 <0.25%」是附录 B.1 在 16B/230B MoE 基线、特定 token 数下的观测，且依赖"细粒度量化 + FP32 累加 + E4M3 + 在线量化"这一整套组合。论文只验证了把 Dgrad 换成 block-wise 量化会发散这一种改动，并未逐项测试其它改动，因此不能反推该结论对其它实现或其它改动同样成立，更不能当成"FP8 训练在任意实现下都安全"。
- 「激活梯度对 block-wise 量化敏感」的**原因**是论文的**假设**（token 间不平衡导致 token 相关离群值），不是已证明的机制；但"block-wise 量化 Dgrad 会导致该 16B MoE 训练发散"是论文报告的实测结果。
- 论文没有公开量化校准数据集、per-channel/per-token 的激活分布统计或 outlier 的逐层位置；本库要与量化实现对齐时必须另行在目标 checkpoint 上测量。
- MLA 推理只缓存 `c^{KV}` 与 `k^R` 是**算法层**事实；论文没有讨论对这两个缓存做低比特量化的误差，也不能据此推断其量化友好度。
- 「不丢 token」是 V3 在论文的均衡策略与部署下的结论，不能外推到其它 MoE 配置或实现（V2 训练期就丢 token）。
- MTP 深度 `D=1`、`γ`、`λ`、`α`、学习率表、batch 调度都是**这一次训练**的取值，不是结构约束；MTP 的收益是论文自述观测，本页未独立复现。
- 论文的对比数字来自各家不同的评测实现，不能跨来源混用；也不能由"V3-Base 是最强开源 base"推出它在任何具体下游任务或量化设置下都更优。后训练章节的 R1 蒸馏消融（LiveCodeBench-CoT 31.1→37.4、MATH-500 74.6→83.2）与 MTP 实测（第二 token 接受率 85–90%、解码 1.8× TPS）不能外推为推理期量化或低比特部署的误差特性。
