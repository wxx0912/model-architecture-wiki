---
{
  "id": "paper-llada",
  "title": "Large Language Diffusion Models",
  "category": "paper",
  "url": "https://arxiv.org/abs/2502.09992",
  "version": "v3（2025-10-18）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1（Figure 1–2）、§2.1–2.4（Eq.3–6）、§3 引言（Figure 3）、§3.2–3.3（Table 1–4）、§4、§5、附录 A（Eq.7–16、Algorithm 1–5）、B.2（Table 5）、B.3（Table 6）、B.4（Figure 4、Table 7–9）、B.5（Table 10）、B.6（Figure 5）、B.7（Table 11）、B.8（Table 12）、附录 C（Table 13–19）",
  "topic": "core",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---
## 这份来源是什么

LLaDA（Large Language Diffusion with mAsking，arXiv 2502.09992，v3）。它把**掩码扩散模型（MDM）**从零训练到 8B 规模，用来回答"自回归是不是 LLM 核心能力（可扩展性、上下文学习、指令跟随）的唯一路径"。做法是：前向过程按线性概率独立掩码 token，反向过程用一个**非因果** Transformer（mask predictor）同时预测所有被掩码位置，训练目标是对数似然下界。

是本库 `models/llada`、`paradigm-dllm` 的一手材料（本页是这两个页面最核心的算法依据）：前向/反向过程的精确形式、训练损失与它的 `1/t` 权重、采样时的 remasking 规则、以及扩散式采样与自回归采样的对比都来自这里。与相邻来源的关系：backbone 沿用 LLaMA 同类组件（RMSNorm/SwiGLU/RoPE，见 `paper-rmsnorm`、`paper-rope`），但 LLaDA 自身改用**非因果**注意力、不使用 causal mask。

对量化的直接相关性：LLaDA **没有 KV cache**（非因果注意力与 KV 缓存不兼容），推理期显存不随生成步数增长；论文也自述未做任何系统级优化。范围边界：不涉及量化或低比特（唯一与精度有关的是附录 B.7 的 bf16 吞吐/显存对照），也未做 RL 对齐。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源 `main.tex`（单文件，含 `math_commands.tex`），版本号取自 abs 页 Submission history（v1 2025-02-14、v2 2025-02-18、v3 2025-10-18）；本地源码与 arXiv 最新 e-print 逐字一致。
- 下文章节号、式号、表号、图号按 v3 浮动体出现顺序：**Figure 1–5、Table 1–19、Algorithm 1–5**。关键编号：训练损失 **Eq.3**、其上界关系 **Eq.4**、SFT 损失 **Eq.5**、条件似然评估式 **Eq.6**；附录 Algorithm 1/2 为预训练/SFT、3 为似然评估、4/5 为随机与低置信度 remasking。`\begin{wraptable}` 与非 `tab:` 前缀标签也占表号，表号按 `\caption` 实际出现顺序数。

## 关键事实

- **前向过程**（§2.1、附录 A.1 Eq.7）：每个 token 以概率 `t` 独立被替换为掩码 `M`、以 `1-t` 保持不变；`t ~ U[0,1]`，`t=1` 时整句全掩码。线性掩码率的动机是"文本信息量与 token 数成正比"。
- **训练目标**（§2.1 Eq.3）：`L(θ) = −E_{t,x_0,x_t}[ (1/t) · Σ_i 1[x_t^i = M] log p_θ(x_0^i | x_t) ]`——**只用被掩码位置的交叉熵**，并按 `1/t` 加权。论文证明它是模型分布负对数似然的上界（Eq.4）；并强调与 BERT 固定掩码率、以及 MaskGIT（少了 `1/t` 项）的区别。
- **模型与训练**（§2.2）：backbone 与 LLaMA 同类（RMSNorm + SwiGLU + RoPE），但**不使用 causal mask**；训练 1B 与 8B 两个规模。8B 在 **2.3T token** 上从零预训练、固定序列长度 4096，总成本 **0.13M H800 GPU 小时**（与同规模同数据量的 AR 模型相当）。学习率用 **WSD**（2000 步升到 `4e-4` 并保持，处理 1.2T token 后降到 `1e-4`，最后 0.3T token 内线性降到 `1e-5`）；另把 **1% 预训练数据设为随机长度**以支持变长。8B **只跑了一次、未做超参搜索**。
- **SFT**（§2.3 Eq.5）：只对被掩码的 response 位置算同一个 `1/t` 加权交叉熵（prompt 不掩码），形式与预训练完全兼容（把 `p_0‖r_0` 当 `x_0`、`p_0‖r_t` 当 `x_t`）。8B 用 **4.5M 对**数据训 3 个 epoch，lr 50 步升到 `2.5e-5`、最后 10% 迭代降到 `2.5e-6`。短样本用 `|EOS|` 补齐、采样时丢弃，因此**响应长度由模型决定**（但生成需给定长度上限）。
- **采样/反向过程**（§2.4、附录 A.2、Algorithm 4/5）：从全掩码定长序列出发，按 `N` 个均匀时间步从 `t=1` 走到 `0`；每步同时预测所有掩码位置，再按 `s/t` 的比例重新掩码。默认用**低置信度 remasking**（把置信度最低的 `s/t` 比例位置重新掩码，Algorithm 5），比纯随机（Algorithm 4）明显更好（Table 9）。
- **三种采样策略**（附录 B.4 Figure 4、Table 7/8）：同一套权重在预训练/SFT 后**无需再训练**即可支持 autoregressive、block diffusion、固定块长的 block diffusion LLaDA 与**纯扩散**采样。Base 上纯扩散整体最好（BBH 49.7、GSM8K 70.3、Math 31.4，自回归为 38.1/63.1/23.6），块长越大 block diffusion 越好。Instruct 上自回归几乎崩溃（GSM8K/Math/HumanEval/MBPP 为 0/9.5/0/0），纯扩散整体最好；而 block diffusion LLaDA 在 GSM8K（77.5）与 Math（42.2）上更高，块长取 8 时 GSM8K 升到 **78.6**。
- **似然评估**（§2.4 Eq.6、附录 A.2）：条件似然用等价的"定数掩码"形式 `−E_l[ (L/l) Σ_i 1[r_l^i=M] log p_θ(r_0^i|p_0,r_l) ]`（`l` 从 `{1..L}` 均匀采样）；该形式方差更低，**128 次蒙特卡洛就够了**，原始形式要 1000 次以上才稳定。
- **预训练 benchmark**（§3.2 Table 1，LLaDA 8B Base vs LLaMA3 8B Base）：MMLU **65.9** vs 65.4；GSM8K **70.3** vs 48.7；Math **31.4** vs 16.0；CMMLU **69.9** vs 50.7；C-Eval **70.5** vs 51.7；HumanEval 35.4 vs 34.8；TruthfulQA **46.1** vs 44.0。落后项：BBH 49.7 vs 62.1、ARC-C 45.9 vs 53.1、HellaSwag 70.5 vs 79.1、PIQA 73.6 vs 80.6、MBPP 40.0 vs 48.8、GPQA 25.2 vs 25.9。论文自述其在数学与中文任务上有优势。
- **SFT 后 benchmark**（§3.2 Table 2）：LLaDA 8B Instruct 只做 SFT（对手是 SFT+RL）：MMLU 65.5 vs 68.4、MMLU-Pro 37.0 vs 41.9、GSM8K 69.4 vs 78.3、Math 31.9 vs 29.6、GPQA 33.3 vs 31.9、HumanEval 49.4 vs 59.8、MBPP 41.0 vs 57.6、ARC-C **88.5** vs 82.4。论文承认整体略落后但差距不大，并指出 MMLU 等在 SFT 后反而下降（措辞为 possibly due to the suboptimal quality of the SFT data）。
- **可扩展性**（§3 引言 Figure 3；缩放对照表在附录 C Table 18/19）：在 1B 上让 LLaDA 与 ARM 共享架构/数据/配置；用 `6ND` 估训练 FLOPs，把缩放范围扩到 `10^20~10^23`。LLaDA 在 MMLU、GSM8K 上趋势更强，PIQA 等较弱任务随规模差距缩小；论文明确**不拟合定量曲线**。
- **反转推理**（§3.3 Table 4）：自建 496 组中文诗句对，零样本做"给上句续下句"与"给下句推上句"。GPT-4o 82.7/34.3、Qwen2.5-7B Instruct 75.9/38.0、LLaDA 8B Instruct 51.8/**45.6**——LLaDA 前后向几乎持平，反转项大幅超过两者，且论文称未为反转任务做特殊设计。
- **CFG**（§3.3、附录 B.3 Table 6）：支持无监督 CFG（`p̃ ∝ p(r_0|p_0,r_t)^{1+w} / p(r_0|m,r_t)^w`），在多数基准上提升（ARC-C 45.9→47.9、HellaSwag 70.5→72.5、GPQA 25.2→26.1；WinoGrande 74.8→74.8 持平，论文 caption 自称 "consistently improves" 与表体不符）；但**正文主结果一律不用 CFG**。
- **生成长度**（附录 B.5 Table 10）：长度取 256/512/1024 时结果不敏感（BBH 45.0/50.4/49.7，GSM8K 70.0/70.8/70.3 等）。
- **速度与显存**（附录 B.6/B.7 的 Figure 5 与 Table 11；单张 A100-80GB、batch 1、bf16；吞吐实验固定输出 256）：采样步数可调（32/64/128/256 步 = 每步解出 8/4/2/1 个 token），GSM8K 与 Math 上以与 LLaMA3 8B 相当的质量达到 **1.5×/1.8×** 吞吐，HumanEval 在吞吐对齐时可比、MBPP 落后；LLaDA **没有 KV cache**，显存不随采样步数变化，与"LLaMA3 8B 不开 KV cache"相当、比"开 KV cache"略高（输入 512/输出 2048：18.52 GB vs 20.00/16.73 GB）。
- **模型规格**（附录 B.2 Table 5）：LLaDA 1B 与 ARM baseline 1B 完全同构（22 层、`d=2048`、32 个注意力头、FFN 5634、词表 126,464、总参 1.49B）；**LLaDA 8B 为 32 层、`d=4096`、FFN 12,288、KV heads 32（vanilla MHA）**、总参 8.02B；LLaMA3 8B 为 32 层/FFN 14,336/KV heads 8/8.03B。
- **局限（§5 自述）**：生成长度是用户给定超参；没有为 LLaDA 设计专用注意力或位置编码，也没有任何系统级优化（如 KV cache）；采样算法仍是初步的；未做 RL 对齐；受算力限制，与 ARM 的严格同数据对比只在 `<10^23` FLOPs 内。

## 不能推出的结论

- 论文不含量化、低比特或数值精度实验（附录 B.7 的 bf16 只是把 LLaMA3 作为对照测吞吐与显存）。**不能**由"LLaDA 没有 KV cache、显存不随步数增长"推出"更容易量化/更省显存"：显存结论是 fp16/bf16、单卡 A100-80GB、batch 1、定长输入输出下测得的，且论文明确没做任何推理优化。
- LLaDA 的采样不是单次前向：一次生成要跑 `N` 步、每步一次全序列前向。因此"吞吐 1.5×/1.8×"是**特定步数配置**下的结果，不能当成固定倍率，也不能据此推断任意长度下的端到端延迟。
- Table 1/2 的对比里，对手数字来自各自论文（表中以 `†`/`¶` 标注），训练数据（LLaDA 2.3T vs LLaMA3 15T vs Qwen2.5 18T）与后训练流程（LLaDA 只有 SFT）都不同；不能当作同数据同算力下的严格对照，论文自己也说 8B 没有对应的同架构 AR 模型。
- §2.2 正文说"LLaDA 用 vanilla MHA 而非 GQA"，但**附录 Table 5 里 1B 的 ARM 与 LLaDA 都写 KV heads = 4**，只有 8B 的 LLaDA 是 32；引用具体 KV head 数须以 Table 5 为准。
- "打破 reversal curse"（Table 4）是在**自建中文诗句补全**（496 组）上零样本测得的，不能外推为 LLaDA 在所有顺序敏感任务上都反转友好；其 forward 分数（51.8）明显低于对手，说明"持平"建立在较弱的前向能力上。
- Table 7/8 里 Instruct 上自回归几乎为 0 是**SFT 数据的长度偏置**造成的（论文解释：SFT 数据是完整句子，模型倾向在给定长度内写完一整句），不能据此说"自回归采样对扩散模型根本无效"；Table 6 的 CFG 收益是每个任务在 `{0.5,1,1.5,2}` 里**各自挑最好 scale** 汇总的，正文主结果不含 CFG。
- 论文明确说 8B 预训练与 SFT **各只跑一次、没有调超参**，训练数据也未公开；其 benchmark 不能当作可复现的调优结果，也不能据此判断某个具体 checkpoint 的量化敏感层。
- 本页只是论文级证据，不证明任何具体 checkpoint 的行为，也不能替代在目标实现上的运行验证。
