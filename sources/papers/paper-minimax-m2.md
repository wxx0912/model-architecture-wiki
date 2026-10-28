---
{
  "id": "paper-minimax-m2",
  "title": "The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence",
  "category": "paper",
  "url": "https://arxiv.org/abs/2605.26494",
  "version": "v2（2026-07-30）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1、§2.1（Overall Architecture）、§2.2.1（Mixture-of-Experts，含 Table 1）、§2.2.2（Attention，含 Table 2、Table 3）、§2.3（Multi-Token Prediction）、§3（Pre-Training Data：19.9T 常量阶段、9.3T decay、8K→32K→192K 扩展）",
  "topic": "architectures",
  "excerpt_sha256": null
}
---
## 这份来源是什么

MiniMax-M2 系列技术报告（arXiv 2605.26494 v2，MiniMax）。系列从 M2 迭代到 M2.5、M2.7，报告主体覆盖架构、预训练数据、后训练数据与 RL 系统、基础设施与评测；本库只取与结构相关的部分：§2.1（Overall Architecture）、§2.2.1（MoE，含 Table 1 消融）、§2.2.2（Attention，含 Table 2 / Table 3 的 full 与 SWA 对比）、§2.3（MTP）、§3（预训练数据阶段与长上下文扩展）。它是 `model-minimax-m2` 的结构依据，也为 `paradigm-moe`、`paradigm-ar`、`component-attention`、`component-position-encoding`、`component-normalization` 提供"全层 full attention + GQA + 细粒度 sigmoid-gating MoE"的现实实例。

范围边界：只覆盖 M2 的架构；报告的后训练数据、RL 系统（Forge）、基础设施与评测部分本库不取。报告不含任何量化实验。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2605.26494v2`），未读 LaTeX 源与 PDF。
- 定位：§2.1 的规模与模块组成；§2.2.1 的 Fine-Grained Experts / Sigmoid Gating / Expert Bias 三点改动与 Table 1；§2.2.2 的 full attention 取舍与 Table 2 / Table 3；§2.3 的 MTP 设计与权重拷贝扩展；§3 的 19.9T + 9.3T 与 8K→32K→192K 扩展；§4 之后的评测数字未逐一复核。

## 关键事实

- **规模**（摘要、§2.1）：旗舰 M2 为 **62 层 decoder-only Transformer**，**229.9B 总参 / 9.8B 激活/ token**，hidden **3,072**，词表 **200,064**，预训练 **29.2T tokens**，原生上下文 **192K**。
- **注意力**（§2.1、§2.2.2）：**全部层使用 full multi-head attention**，**48 个 query head、8 个 key-value head（GQA）**，全模型应用 RoPE。这**放弃了 MiniMax-Text-01 的混合设计**（Lightning Attention 与 full attention 交错的 hybrid）。报告称没有找到在生产设定（推理、编码、agent）下能稳定匹配 full attention 质量的子二次变体。
- **MoE**（§2.1、§2.2.1）：每层为 **256 个细粒度专家、每 token 激活 8 个**；路由用 **sigmoid gating + 可学习 per-expert bias**，去掉 softmax 的 zero-sum 约束、并以 bias 隐式调节专家利用率来大幅减少对辅助负载均衡损失的依赖。三点改动为 Fine-Grained Experts、Sigmoid Gating、Expert Bias。
- **Table 1（MoE / MTP 消融）**：在 **2B 激活 / 17.8B 总参、500B tokens** 规模上比 baseline、w/ MTP、w/ Fine-Grained 三档；Fine-Grained 把 `num_experts` 从 32 提到 128、`topk` 从 2 提到 8（MATH 19.6→24.1、HumanEval 29.7→32.5 等）。这是消融证据，不是 M2 本体的绝对指标。
- **SWA 对比证据**（§2.2.2、Table 2、Table 3）：报告在 M2 规模上广泛试过 hybrid Sliding Window Attention，预训练阶段所有变体在检索、多跳推理与上下文学习上均退化（Table 2：RULER 128K CWE 90.0→72.0、RULER 128K MQ 99.0→93.0、MTOB e-k ChrF 44.8→27.2）；SFT 后差距集中在超 32K 的长上下文任务（Table 3：SWE-verified 54.7→50.2、Terminal-Bench 26.7→23.8、τ²-Bench telecom 32.5→21.0），而 32K 以内有增有减。
- **拒绝子二次注意力的两条理由**（§2.2.2）：(i) 质量损失难可靠度量（大规模下多跳推理出现明显退化，代理指标与下游相关性脆弱）；(ii) **基础设施不成熟**——线性注意力在训练时也常是 memory-bound，推理上还面临**对低精度存储敏感、缺原生 prefix caching、与投机解码集成不清**。
- **MTP**（§2.3）：预训练时用**单个 MTP 模块（K=1）**，沿用 DeepSeek-V3 设计，MTP loss 权重从 0.3 退火到 0.1；在持续预训练的 decay 阶段**用主干权重拷贝把 MTP 从 1 个扩到 3 个（K=3）**以支持多步投机解码——先冻结主干只训 MTP 直到 loss 稳定，再切换到全模块联合训练（报告称全程冻结主干会让 MTP 质量更差）。推理时 3 个 MTP 模块生成 draft token、由主干一次前向验证。
- **预训练与长上下文**（§1、§3）：常量阶段训练 **19.9T tokens**，decay 阶段 **9.3T tokens**，合计 29.2T；上下文用**多阶段**从 **8K 扩到 32K 再到 192K**（长上下文语料以高质量代码拼接、天然长文 PDF 与主题相关的文档打包为主）。

## 不能推出的结论

- 报告是**训练侧**文档：不含任何量化位宽、校准、导出格式或 kernel 结论，不能据此推断 M2 在 FP8 / int4 / int8 下的精度或速度。
- 报告**不含完整架构超参表**：62 层、hidden 3072、词表 200064、256 专家 top-8、48/8 heads 与 192K 由 §2.1 给出；`head_dim`、`rotary_dim`、`rope_theta`、per-expert FFN 维、QK-Norm 等只在权重仓 config，须按 checkpoint 核对。
- Table 1 的 2B 消融、Table 2 / Table 3 的 full 与 SWA 对比都是**受控实验**，不能当作 M2 的绝对能力或量化后结论。
- 报告描述的 hybrid SWA 退化是**该报告评测设定下**的观测，不能外推为所有子二次注意力都不可用。
- 报告主体是 M2 系列；本页只取 M2 的结构口径，M2.5 / M2.7 的架构差异未读。
- 本页只读 HTML 渲染版，未核对 LaTeX 源；§8 的评测数字未逐表复核。
