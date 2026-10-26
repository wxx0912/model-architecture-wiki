---
{
  "id": "paper-glm45",
  "title": "GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models",
  "category": "paper",
  "url": "https://arxiv.org/abs/2508.06471",
  "version": "v1（2025-08-08）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-26",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1、§2.1（架构）、§2.4（超参数）、Table 1、§4（评测）",
  "topic": "architectures",
  "excerpt_sha256": null
}
---
## 这份来源是什么

GLM-4.5 系列技术报告（arXiv 2508.06471 v1）。报告覆盖预训练、中训练、后训练（SFT / RL）与评测；本库只取与**结构**相关的部分：§2.1 架构、Table 1 配置、§2.4 超参数。它是 `model-glm45` 的结构依据，也为 `paradigm-moe`、`component-attention`、`component-position-encoding`、`component-normalization` 提供"窄而深 MoE + GQA + partial RoPE + QK-Norm"的现实实例。

范围边界：只覆盖 GLM-4.5 与 GLM-4.5-Air 两个稀疏 MoE 模型；报告不含任何量化实验，也不含推理与部署细节。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2508.06471v1`），未读 LaTeX 源。
- 定位：§2.1 架构（含 Table 1 配置）、§2.4 超参数（优化器、序列长度、RoPE base、路由与 MTP 设置）、§4 评测（只取架构相关对照，未逐表核对数字）。

## 关键事实

- **整体形态**（§2.1）：采用 MoE 架构。与 DeepSeek-V3、Kimi K2 不同，GLM-4.5 **减小宽度（hidden 与路由专家数量）、增加高度（层数）**，因为更深的模型推理能力更好。
- **注意力**（§2.1）：Grouped-Query Attention + **partial RoPE**；用 2.5 倍于常规的 attention heads（5120 hidden 用 96 heads）；加 **QK-Norm** 稳定 attention logits 范围。论文明确：增加 head 数不改善训练 loss，但稳定提升 MMLU、BBH 等推理基准。
- **MoE**（§2.1）：使用 loss-free balance routing 与 **sigmoid gates**。
- **MTP**（§2.1）：GLM-4.5 与 Air 都加一层 MoE 作为 MTP 层，用于推理期投机解码。
- **配置**（Table 1）：GLM-4.5 为 355B 总参 / 32B 激活、3 层 dense + 89 层 MoE、1 层 MTP、hidden 5120、dense 中间维 12288、MoE 中间维 1536、head dim 128、96 attention heads、8 KV heads、160 专家、每 token 激活 8、1 共享专家、QK-Norm 有；GLM-4.5-Air 为 106B / 12B、1 层 dense + 45 层 MoE、hidden 4096、dense 中间维 10944、MoE 中间维 1408、128 专家、QK-Norm 无。Table 1 的参数量口径**含 MTP 层、不含词嵌入与输出层**。
- **超参数**（§2.4）：Muon 优化器用于除 word embedding、bias、RMSNorm 权重外的全部参数（Newton-Schulz 步数 5、momentum 0.95）；最大序列长度预训练 4096，中训练扩到 32768 与 131072；扩到 32K 时把 **RoPE base 从 10,000 调到 1,000,000**；loss-free balance routing 的 bias 更新率前 15T tokens 为 0.001、之后 0.0，另加权重 0.0001 的序列级 balance loss；MTP loss 权重 λ 前 15T 为 0.3、之后 0.1；weight decay 0.1、无 dropout；学习率 warmup 0→2.5e-4 后余弦衰减到 2.5e-5。

## 不能推出的结论

- 报告是**训练侧**文档：不含量化位宽、校准、导出格式或 kernel 结论，不能据此推断 GLM-4.5 在 int4 / int8 / FP8 下的精度或速度。
- partial RoPE 只给了名称，**未给旋转比例**；sigmoid gate 只给了名称，**未给亲和度函数与 top-k 归一化细节**——具体实现需按代码核对。
- Table 1 的参数量口径（含 MTP、不含 embedding 与输出层）与"某个 checkpoint 的显存占用"不是一个数，不能直接换算。
- 论文评测（MMLU / BBH 等）是论文训练设定下的结果，不代表任何量化版本的表现。
- 本页只读 HTML 渲染版，未核对 LaTeX 源；§4 评测数字未逐表复核。
