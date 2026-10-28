---
{
  "id": "paper-qwen3",
  "title": "Qwen3 Technical Report",
  "category": "paper",
  "url": "https://arxiv.org/abs/2505.09388",
  "version": "v1（2025-05-14）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1、§2（架构，含 Table 1、Table 2）、§3.1–§3.2（预训练数据与阶段）、§4.3（thinking 模式融合，仅取结构相关）",
  "topic": "architectures",
  "excerpt_sha256": null
}
---
## 这份来源是什么

Qwen3 系列技术报告（arXiv 2505.09388 v1，Qwen Team）。报告覆盖 Qwen3 的架构、预训练、后训练与评测；本库只取结构相关部分：§2 架构（含 Table 1 / Table 2）、§3.2 预训练阶段（RoPE base 与长上下文扩展）、§1 与 §4.3（系列规模、双模式，仅作上下文）。它是 `model-qwen3` 的结构依据，也为 `paradigm-moe`、`paradigm-ar`、`component-attention`、`component-position-encoding`、`component-normalization` 提供"dense 与 MoE 共用主干、GQA + QK-Norm、无共享专家 MoE"的现实实例。

范围边界：只覆盖 Qwen3 系列（6 个 dense + 2 个 MoE）；报告不含任何量化实验，也不含推理与部署细节。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2505.09388v1`），未读 LaTeX 源与 PDF。
- 定位：§2 架构 + Table 1（dense 配置）+ Table 2（MoE 配置）；§3.1–§3.2 预训练数据与三阶段；§1 摘要（系列规模、tokenizer）；§4.3 只取"thinking / non-thinking 融合"的描述，§4 评测数字未逐表核对。

## 关键事实

- **系列构成**（§2）：6 个 dense 模型——Qwen3-0.6B、1.7B、4B、8B、14B、32B；2 个 MoE 模型——Qwen3-30B-A3B 与 Qwen3-235B-A22B。旗舰 235B-A22B 总参 235B、每 token 激活 22B。
- **dense 主干**（§2）：与 Qwen2.5 类似，使用 Grouped Query Attention（GQA）+ SwiGLU + RoPE + RMSNorm（pre-normalization）。相对 Qwen2 的两点改动：**移除 QKV-bias**；**引入 QK-Norm** 以稳定训练。
- **Table 1（dense，列为 层数 / Q,KV heads / tie embedding / context length）**：0.6B 为 28 / 16,8 / Yes / 32K；1.7B 为 28 / 16,8 / Yes / 32K；4B 为 36 / 32,8 / Yes / 128K；8B 为 36 / 32,8 / No / 128K；14B 为 40 / 40,8 / No / 128K；32B 为 64 / 64,8 / No / 128K。
- **MoE 主干**（§2）：与 dense 同一基础架构；follow Qwen2.5-MoE 做 fine-grained expert segmentation；共 **128 个专家、每 token 激活 8 个**；与 Qwen2.5-MoE 不同，**不含共享专家**；采用 global-batch load balancing loss 促进专家专精。
- **Table 2（MoE，列为 层数 / Q,KV heads / 专家 总数,激活 / context length）**：30B-A3B 为 48 / 32,4 / 128,8 / 128K；235B-A22B 为 94 / 64,4 / 128,8 / 128K。
- **tokenizer**（§2）：Qwen 自家 tokenizer，byte-level BPE（BBPE），词表大小 **151,669**。
- **预训练**（§1、§3.1–§3.2）：约 **36T tokens、119 种语言**；三阶段——General Stage（S1）>30T @4096、Reasoning Stage（S2）约 5T @4096、Long Context Stage @32768（长上下文语料 75% 长度落在 16,384–32,768）。长上下文阶段用 ABF 把 **RoPE base 从 10,000 提到 1,000,000**，并引入 **YARN 与 Dual Chunk Attention（DCA）** 实现推理期 4× 序列长度扩展。
- **后训练**（§1、§4.3）：把 thinking 模式与 non-thinking 模式融合进同一模型，并引入 thinking budget 控制推理期算力。

## 不能推出的结论

- 报告是**训练侧**文档：不含量化位宽、校准、导出格式或 kernel 结论，不能据此推断 Qwen3 在 int4 / int8 / FP8 下的精度或速度。
- Table 1 / Table 2 的 "Context Length"（32K / 128K）是支持口径，不等于某个权重仓 `config.json` 的 `max_position_embeddings`（后者为 40960）；128K 依赖推理期 YaRN / DCA 扩展，不能直接当作训练序列长度。
- QK-Norm 只给了名称：未给实施位置与归一维度，也未说明是否对 Q / K 分别归一，具体形式须按建模代码核对。
- MoE 的 gate 形式（softmax 还是 sigmoid）、top-k 概率是否归一化、routing 顺序在 §2 未给出，须按实现核对。
- 论文词表写 151,669，而官方权重仓 `config.json` 的 `vocab_size` 为 151,936；两者口径不同，不能混用。
- 论文评测（AIME、LiveCodeBench、BFCL 等）是论文训练设定下的结果，不代表任何量化版本的表现。
- 本页只读 HTML 渲染版，未核对 LaTeX 源；§4 评测数字未逐表复核。
