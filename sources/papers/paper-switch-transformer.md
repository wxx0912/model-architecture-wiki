---
{
  "id": "paper-switch-transformer",
  "title": "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
  "category": "paper",
  "url": "https://arxiv.org/abs/2101.03961",
  "version": "v3（2022-06-16）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "§2.1 Eq.(1)–(2)、§2.2 专家容量 Eq.(3) 与辅助损失 Eq.(4)–(6)、§2.3 Table 1、§2.4 Table 2–4、§3 Figure 4–6、§4.3 Figure 7–8、§5.6 Table 9、附录 B/C",
  "topic": "architectures",
  "excerpt_sha256": null
}
---

## 这份来源是什么

Switch Transformer 论文（arXiv 2101.03961，v3，JMLR）。它把 MoE 路由简化为 **top-1**（每个 token 只路由到一个 expert），并给出专家容量、可微负载均衡损失与两项训练稳定性技术（selective precision、缩小初始化）及一项微调正则化手段（expert dropout），把稀疏模型训到 1.6T 参数。是本库 `paradigm-moe`、`component-router-expert` 判断"top-1 路由可行、容量因子与丢弃机制如何工作"的直接依据，也是 `check-quantization-preflight` 关注的 router 精度问题的来源。

## 已读定位与阅读形式

以 arXiv e-print 的 LaTeX 源 `21-0998.tex` 为主，公式号与表号用 ar5iv HTML（`https://ar5iv.labs.arxiv.org/html/2101.03961`）复核；版本号取自 abs 页 Submission history（最新为 v3）。下文 §、Eq.、Table 号按 v3。

## 关键事实

- 常规 MoE 路由：router 权重 `W_r` 产生 logits `h(x) = W_r·x`，在 N 个 expert 上做 softmax 得到 gate 值 `p_i(x) = e^{h(x)_i}/Σ_j e^{h(x)_j}`，取 top-k 组成集合 𝒯，层输出为 `y = Σ_{i∈𝒯} p_i(x)·E_i(x)`（§2.1，Eq.(1)–(2)）。router 的可微性来自这里的 gate 值 `p_i(x)`。
- Switch 的做法是把 k 固定为 **1**（top-1）：论文称这保持了模型质量、减少路由计算、使每个 expert 的 batch（专家容量）可以至少减半、并简化实现与通信（§2.1）。作者明确与"k>1 才能让路由函数拿到非平凡梯度（non-trivial gradients）"的既有猜想相反。
- 专家容量 `expert capacity = (tokens per batch / number of experts) × capacity factor`（§2.2，Eq.(3)）；capacity factor > 1 用于缓冲不均衡。路由溢出的 token（dropped tokens）**跳过该层计算、经残差直接传给下一层**；容量因子越大越浪费计算与内存。
- 论文报告其设定下 dropped token 通常 **<1%**，且没有观察到丢弃率对专家数的依赖；配合足够大的辅助损失系数即可获得良好负载均衡（§2.2）。
- 负载均衡辅助损失（§2.2，Eq.(4)–(6)）：`loss = α·N·Σ_i f_i·P_i`，其中 `f_i = (1/T)Σ_{x∈𝓑} 1{argmax p(x) = i}`（**不可微**）、`P_i = (1/T)Σ_{x∈𝓑} p_i(x)`（可微）。均匀路由时两者都为 1/N；乘 N 是为了让损失值不随专家数变化。论文取 `α = 1e-2`（并在 1e-1~1e-5 间扫过，称 1e-2 能快速均衡又不干扰主损失）。
- selective precision（§2.4，Table 2，32 专家）：bfloat16 训练会发散（Neg. Log Perp. −3.780，标注 diverged，1390 ex/s）；全 float32 质量 −1.718、1160 ex/s；把**局部路由运算**转 float32、其余保持 bfloat16 得 −1.716、1390 ex/s。即"混合精度训练不稳定时，只把 router 相关计算提升到高精度"。
- 缩小初始化（§2.4，Table 3）：权重用截断正态 `μ=0、σ=sqrt(s/n)`；把默认 `s=1.0` **缩小 10 倍**后，32 专家模型 3.5k 步的质量从 −3.60（std 0.68）改善到 −2.72（std 0.01，3 个随机种子）。
- expert dropout（§2.4，Table 4，属微调正则化）：微调时只在 expert 的 FFN 上加大的 dropout——非 expert 层保持 0.1、expert 层加到 0.4——论文称这让四个小任务（GLUE / CNNDM / SQuAD / SuperGLUE）都有改善；其中 GLUE 85.2 与 CNNDM 19.6 是该表最好值，SQuAD 83.7 与 SuperGLUE 73.0 仍低于 d=0.2 行的 83.9 / 73.2。单纯全局加大 dropout（d=0.3）反而更差。
- Switch vs MoE 对比（§2.3，Table 1，均为 128 专家、隔层放 expert、TPUv3、32 cores）：quality 为 100k 步后的 Neg. Log Perp.（阈值 −1.50），括号内为训练小时数与 ex/s。T5-Base −1.731（未达阈值）、T5-Large −1.550（131.1 h、470）；MoE-Base 三档容量因子 −1.547 / −1.559 / −1.572（68.7 h/840、80.7 h/790、80.1 h/860），Switch-Base −1.554 / −1.553 / −1.561（72.8 h/860、65.0 h/910、62.8 h/1000），Switch-Base+ −1.534（67.6 h/780）。
- Table 1 的口径：MoE Transformer 用 top-2、每个 token 走两个 FFN，算力更大；Switch 与 T5-Base FLOP-matched。论文强调 Switch 在**速度-质量**折中上更好，并在更小容量因子（1.0、1.25）下表现更好（§2.3）。
- 论文自己提示速度同时取决于算法与实现：Table 1 里 MoE 从 CF 2.0 到 1.25 反而变慢（840→790），作者在脚注里归因于底层实现差异（§2.3）。
- 扩展实验（§3）：等算力下把专家数从 2 一路翻倍到 256（对应 14.7B 参数）时质量持续改善（Figure 4）；64 专家的 Switch-Base 达到同一质量只需 T5-Base 的七分之一时间（Figure 5）；相对每 token 多用 3.5× FLOPs 的 T5-Large，Switch-Base 仍有 2.5× 墙钟加速（Figure 6）。摘要把这类收益概括为"同算力下最高 7× 预训练加速"，§3.1 给出按 step time 计约 7.5×。
- 最大模型 Switch-C（§5，Table 9）：1571B 参数（1.6T）、2048 个专家、15 层、每层都有 expert、`d_model=2080`、`d_ff=6144`、32 head、890B FLOPs/seq；C4 上 250k/500k 步的 Neg. Log Perp. 为 −1.096 / −1.043，作者称与 T5-XXL 同算力下达到固定困惑度快 4 倍。同表还给出 Switch-Base 7B / Switch-Large 26B / Switch-XXL 395B 的参数与配置。
- 预训练设定：C4 语料、masked language modeling，随机丢弃 15% token 并用一个 sentinel token 替换（§2.3）。
- 多语种（§4.3 Figure 7–8）：相对 mT5-Base，Switch 在全部 101 种语言上都有提升，平均 step 加速约 5×，91% 的语言达到 4× 以上。
- 蒸馏（§4.2 Table 6–8）：把稀疏教师蒸馏回稠密 T5-Base，可在压缩 95%+ 的同时保留约 30% 的质量增益。
- 丢弃的替代方案（附录 B，No-Token-Left-Behind）：迭代地把溢出 token 改路由到次高概率专家，可几乎不丢 token，但论文报告实验上没有带来收益。
- router 探索策略（附录 C，Table 11）：比较了确定性/argmax、softmax 采样、输入 dropout 与乘性 jitter，论文全程采用 input jitter，经验上最好。

## 不能推出的结论

- Table 1 不是同算力对比：MoE-Base 是 top-2、每 token 走两个 FFN，FLOPs 更大；Switch 的"更快更好"里包含"算得更少"，不能当作等算力下的质量优势证据。
- 论文明确说速度同时受算法与实现影响（MoE 在 CF 2.0→1.25 反而变慢即是例证），不能把单个配置的速度比外推到其它框架或实现。
- selective precision 是**训练期**稳定性手段（把 router 局部运算升到 float32）；论文没有做推理侧的低比特实验，"bfloat16 可训练"不能等同于"低比特推理可行"。
- dropped token 通常 <1% 且"未观察到与专家数的依赖"是论文对其**全部实验**的概括（§2.2），但仍是其数据与配置内的观测，不能外推到其它数据分布、batch 规模或实现。
- 专家容量公式假定张量形状在编译期静态确定（Mesh-Tensorflow/TPU），把它当作与框架无关的通用做法不合适。
- expert dropout（0.1/0.4）是**微调**正则化配置，不能当作预训练配置使用。
- 论文不含 INT8 及更低比特的量化结果，不能从本篇推出量化误差路径、校准需求或后端约束。
- Table 9 的 Switch-C 配置是"仅专家并行、无模型并行"的结果，其超参（d_model 2080 等）不能直接搬到带张量并行的实现上。
