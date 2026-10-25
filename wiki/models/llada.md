---
{
  "id": "model-llada",
  "title": "LLaDA",
  "type": "model",
  "tags": [
    "dllm",
    "diffusion",
    "attention",
    "calibration",
    "quantization"
  ],
  "related": [
    "paradigm-dllm",
    "paradigm-ar",
    "component-diffusion-timestep",
    "component-attention",
    "component-embedding-lm-head"
  ],
  "sources": [
    "paper-llada",
    "repo-llada"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "diffusion language model",
    "diffusion lm",
    "扩散语言模型"
  ],
  "updated": "2026-10-24"
}
---

# LLaDA

## 一句话理解

LLaDA 是从零训练的 masked diffusion 语言模型，论文含 1B 与 8B 两档。主干与 LLaMA 同类（RMSNorm + SwiGLU + RoPE），但改用非因果注意力、与 KV caching 不兼容（论文原话；由非因果注意力所致为本库推断）；它是 dLLM 范式在 8B 规模上的实例，本页给论文口径的规格与这个模型的量化相关事实。

## 结构规格

论文附录 Table 5 给出两档配置（同表还列了 1B/7B ARM baseline 与 LLaMA3 8B 作对照）：

| 规格 | LLaDA 1B | LLaDA 8B |
| --- | --- | --- |
| 层数 | 22 | 32 |
| hidden dim | 2048 | 4096 |
| attention heads | 32 | 32 |
| KV heads | 4 | 32 |
| FFN dim | 5634 | 12,288 |
| 词表 | 126,464 | 126,464 |
| 总参 | 1.49B | 8.02B |

除此之外，论文正文说明 LLaDA 使用非因果注意力、与 KV caching 不兼容（"因此没有 KV cache"为本库推断；这一项不在 Table 5 中）。引用规格时有两点要留意：

- 论文 §2.2 那段概括的是 **8B**：用 vanilla multi-head attention 而非 GQA，query heads 与 KV heads 同为 32；而 Table 5 里 **1B 的 LLaDA 记 KV heads = 4**。两处说的是不同档的模型，引用具体数字时要写明取自哪里。
- 正文把 1.49B 的模型称作“1B”、把 6.83B 的 ARM 称作“7B”，是名义标号，不是精确参数量。

## 结构与数据流

LLaDA 用双向可见的 mask predictor 取代因果解码：文本经 embedding 进入主干，主干组件与 LLaMA 同类，**不使用 causal mask**，论文称其与 KV caching 不兼容（由此没有增量解码为本库推断）。生成时把整段序列反复过同一个主干（掩码与重掩码规则见 [Diffusion Language Model](../paradigms/dllm.md)），因此它和 AR 模型的差别主要在生成范式（主干仍是 LLaMA 类，仅注意力改用非因果 mask）。

论文口径的训练与数据：8B 明确从零预训练（§1 写 from scratch），1B 论文未显式说明；8B 在 2.3T token 上训练、固定序列长度 4096，成本约 0.13M H800 GPU 小时，且只跑了一次、没有做超参搜索。SFT 只对被掩码的 response 位置算同一个 `1/t` 加权损失（prompt 不掩码），用 4.5M 对数据训 3 个 epoch；短样本用 `EOS` 补齐、采样时丢弃，所以响应长度由模型决定，但生成时必须给定长度上限。官方推理脚本的默认 sampler 参数是 `steps=128 / gen_length=128 / block_length=128`，属于示例值，不是论文评测配置。

## 量化关联

**结构事实**

- LLaDA 没有 KV cache；每一步是整段非因果前向；主干组件为 RMSNorm / SwiGLU / RoPE。
- 范围事实：论文与官方仓库都没有量化实现，论文只在附录给出吞吐 / 显存对照（吞吐实验为 bf16）。

**量化推断**

- 校准与误差传播的通用路径见 [Diffusion Language Model](../paradigms/dllm.md)，本页只补 LLaDA 自身的条件。
- 论文的精度对照是 LLaMA3 8B 这类不同架构、不同训练量的模型，且训练数据未公开；拿它当量化基线时要单独说明条件，不能当作同架构对照。

## 结果边界

- 本页规格是论文 Table 5 口径；目标 checkpoint（如 HuggingFace Hub 上的 `GSAI-ML/LLaDA-8B-Base` / `-Instruct`）的层数、头数、RoPE 细节以该仓库的 `config.json` 与 `modeling_*.py` 为准，本页不能替代。
- 论文对 KV head 的记载分两处（§2.2 文字只讲 8B，1B 的 KV head 数只能查 Table 5），引用时必须写明取自哪一处；论文自己也说明 8B 没有同架构的 AR 对照模型。
- 论文与官方仓库没有量化实现，不能据本页推断 LLaDA 的量化敏感层、低比特下 mask / remasking 的行为或后端支持。
- 本页是 `source-reported` 的论文级证据，不证明任何具体 checkpoint 的实际运行时行为。

## Agent 如何使用

首轮读本页即可拿到 LLaDA 的论文口径规格，并知道哪些字段必须回到具体 checkpoint 才能确认。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 模型规格、前向 / 反向过程、训练与评测口径 | [`paper-llada`](../../sources/papers/paper-llada.md) |
| 官方推理 / 评测脚本、默认参数、加载方式 | [`repo-llada`](../../sources/repos/repo-llada.md) |
| 生成范式与采样语义 | [Diffusion Language Model](../paradigms/dllm.md) |
| 目标 checkpoint 的层数 / 头数 / RoPE / 权重 key | 该 checkpoint 在 HuggingFace Hub 的 `config.json` 与 `modeling_*.py`（`trust_remote_code=True`，脚本注明需要 `transformers==4.38.2`） |

回答 LLaDA 相关问题时，必须说明用的是 Base 还是 Instruct、哪个 checkpoint、生成长度上限与采样步数；层数 / 头数这类字段要区分“论文 Table 5 口径”和“目标 checkpoint 实测”。

## 来源与证据

- [LLaDA 论文](../../sources/papers/paper-llada.md)：模型规格（附录 B.2 Table 5）、前向 / 反向过程、训练与采样口径。
- [LLaDA 官方仓库](../../sources/repos/repo-llada.md)：推理 / 评测脚本与默认参数；模型定义在 HuggingFace Hub，不在该仓库。

论文口径为 `source-reported`；层数、head 数、是否用 GQA 属于具体 checkpoint 的事实，必须回到 Hub 上的 `config.json` 与 `modeling_*.py` 核对。
