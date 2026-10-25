---
{
  "id": "paper-mixtral",
  "title": "Mixtral of Experts",
  "category": "paper",
  "url": "https://arxiv.org/abs/2401.04088",
  "version": "v1（2024-01-08）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex"
  ],
  "read_locator": "§1、§2 Table 1 与 Figure 1、§2.1 门控与 top-2 输出式、§3 Table 2–3、§3.2 Figure 4、§5 Figure 7–8 与 Table 5",
  "topic": "architectures",
  "excerpt_sha256": null
}
---

## 这份来源是什么

Mixtral of Experts 技术报告（arXiv 2401.04088，v1，Mistral AI）。它给出 Mixtral 8x7B 这个**纯稀疏 MoE 解码器**的结构超参与路由细节（8 个专家、每 token top-2、SwiGLU 专家、32k 稠密上下文），并公开权重（Apache 2.0）。是本库 `paradigm-moe` 判断"每层 8 专家 / 每 token 激活 2 个"以及"**总参数（47B）vs 激活参数（13B）**"区分的直接依据。

报告本身不涉及量化或低比特；它提到的是推理侧 kernel（Megablocks、vLLM）与 Expert Parallelism 的部署方式。

## 已读定位与阅读形式

以 arXiv e-print 的 LaTeX 源 `main.tex` 为主；版本号取自 abs 页 Submission history（该页只提供元数据，不是论文正文；该文只有 v1）。下文 §、Table、Figure 号按 v1 LaTeX 源的浮动体顺序（其中 §3.3 的偏差基准是一个 `wrapfigure`，占的是 **Figure** 计数器，不是表）。

## 关键事实

- Mixtral 是 decoder-only 稀疏 MoE 模型：**每一层的 FFN 子块都被 MoE 层替换**（不是隔层替换），每个 token 由 router 从 8 组专家中选 2 组，输出相加；选中的专家在不同 layer/step 可以不同（§1、§2.1、Figure 1）。
- **两种参数量**（摘要、§2.1、§3）：每个 token 能访问 **47B** 参数（sparse / 总参数，随专家数 `n` 增长），但推理时只用到 **13B** 激活参数（随 `K` 增长，上限为 `n`）；论文用「47B 仍小于 LLaMA 2 70B」来论述显存开销。
- 结构超参（§2 Table 1 与正文）：`dim=4096`、`n_layers=32`、`head_dim=128`、`hidden_dim=14336`、`n_heads=32`、`n_kv_heads=8`、`context_len=32768`、`vocab_size=32000`、`num_experts=8`、`top_k_experts=2`。
- 门控定义（§2.1）：`G(x) := Softmax(TopK(x·W_g))`，其中 `TopK(ℓ)_i = ℓ_i` 当 `ℓ_i` 属于 logits 的 top-K，否则取 `−∞`；由此可以把 gate 为 0 的专家整个跳过，不做计算。
- 专家就是标准 FFN；Mixtral 用 SwiGLU 作为 `E_i(x)`，并取 K=2，因此单个 token 的输出为 `y = Σ_i Softmax(Top2(x·W_g))_i · SwiGLU_i(x)`（§2.1）。
- 与 GShard 的差别（§2.1）：Mixtral 替换**全部** FFN 子块，而 GShard 只替换隔层；GShard 对第二个专家用更复杂的门控策略。
- 除 MoE 替换与完整 32k 上下文外，其余改动论文只以引用方式交代为"与 Mistral 7B 相同"，**未在本报告中描述这些改动的具体内容**（§2）；预训练使用多语种数据（§1）。
- 质量对比（§3 Table 2，均为自家评测 pipeline 重跑）：Mixtral 激活 13B —— MMLU 70.6%、HellaSwag 84.4%、WinoGrande 77.2%、PIQA 83.6%、ARC-e 83.1%、ARC-c 59.7%、NQ 30.6%、TriviaQA 71.5%、HumanEval 40.2%、MBPP 60.7%、MATH 28.4%、GSM8K 74.4%；LLaMA 2 70B 对应为 69.9/85.4/80.4/82.6/79.9/56.5/25.4/73.0/29.3/49.8/13.8/69.6。可见在 HellaSwag、WinoGrande、TriviaQA 上 Mixtral 低于 LLaMA 2 70B。论文对"是否全面超过"的口径并不统一：摘要说 "across all evaluated benchmarks"、Figure 2 题注说 "on all benchmarks"，Table 2 题注说 "almost all popular benchmarks"，Figure 3 题注又说 "except on reading comprehension benchmarks"。
- 评测口径差异（§3）：MBPP 用的是手工校验过的子集，TriviaQA 不提供 Wikipedia 上下文——所以本表数字与 LLaMA 2 原始报告不可直接混用。
- 长上下文（§3.2 Figure 4）：passkey 检索任务中检索准确率与 passkey 位置、输入长度无关，达到 100%；在 proof-pile 子集上困惑度随上下文长度增加单调下降。
- 路由分析（§5，The Pile 验证集的 layers 0 / 15 / 31）：没有观察到明显的**按主题**的专家专业化——ArXiv（LaTeX）、PubMed Abstracts（生物）、PhilPapers（哲学）的专家分配分布很接近；只有 DM Mathematics 略有不同，作者**推测**（原文 `likely`）这是其合成数据性质与语言覆盖窄所致。
- 专家选择看起来更对应**句法**而非领域（§5，Figure 8 是逐 token 上色的样例）：Python 里的 `self` 与英文里的 `Question` 常走同一专家（尽管它们跨多个 token）；代码中的缩进 token 总是走同一批专家，尤其在首/末层。
- 相邻 token 复用专家的比例（§5 Table 5，只看 layers 0/15/31）：以 "First choice"（相邻 token 首次选择同一专家）为例，随机基线为 1/8 = 12.5%，而 layer 15 达 23.6%–28.4%、layer 31 达 19.7%–26.3%，layer 0 则接近随机；论文称这体现**局部性**，并指出它对 Expert Parallelism 意味着某些专家可能被过度订阅，也可以反过来用于缓存。（附录另有覆盖全部层的重复率结果。）
- 部署配套：§1 记录了向 vLLM 提交集成 Megablocks CUDA kernel 的改动；§2.1 说明 MoE 层可通过 Expert Parallelism（EP）分布到多 GPU，token 被路由到对应 GPU 处理再送回原位，并指出 EP 会带来负载均衡挑战。

## 不能推出的结论

- "8x7B" 是命名：总参数（sparse）是 **47B**，不是 8×7B=56B，也不是 13B 的稠密模型；激活 13B 只说明单 token 用到的参数子集大小，不代表算力/显存等价于 13B 稠密模型（路由与经验分布另有开销）。
- Table 2/3 与 Figure 2/3 是论文用**自家 pipeline** 重跑的对比（且 MBPP/TriviaQA 口径与 LLaMA 2 不同），与各家原始报告的数字不保证一致，跨来源混用会出错；论文内部对"是否全部超越"的三处表述本身也不一致，引用时应指明出处。
- §5 的路由分析论文自称 "a small analysis"：样本是 The Pile 验证集的一个子集、表 5 只看了 3 层，不能推出 Mixtral 的 router 在所有任务、语言或层上都无语义专业化（也不能反过来说"已验证有语义专业化"）。
- 相邻 token 的专家复用率是**该 router 在该数据上**的观测，不能外推到其它 MoE 实现的专家分配模式。
- passkey 100% 是合成检索任务的结果，不等于真实长文本理解能力；proof-pile 困惑度单调下降也只是延续预训练分布的表现。
- §2.1 的门控式（softmax over top-K logits）是论文采用的**一种**实现，论文自己列举了多种候选；不能把该式当作所有 MoE 的通用定义。
- 报告不含任何量化或低比特内容；Megablocks/vLLM 属于推理 kernel 与部署工程，不能据此推断量化误差路径、校准需求或低比特可行性。
- Table 1 的超参是 Mixtral 8x7B **这一个** checkpoint 的配置，不代表 MoE 结构的通用取值；论文也未描述它继承自 Mistral 7B 的那些改动的细节，这些需回到 Mistral 7B 论文核对。
