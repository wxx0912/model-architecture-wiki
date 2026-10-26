---
{
  "id": "component-embedding-lm-head",
  "title": "Token Embedding and LM Head",
  "type": "component",
  "tags": [
    "embedding",
    "lm-head",
    "ar",
    "dllm",
    "quantization"
  ],
  "related": [
    "paradigm-ar",
    "paradigm-dllm",
    "paradigm-multimodal",
    "check-calibration-axes"
  ],
  "sources": [
    "paper-transformer",
    "paper-llada",
    "repo-huggingface-transformers"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-26"
}
---

# Token Embedding 与 LM Head

## 一句话理解

Embedding 把离散 token id 映射到 hidden 空间，LM head 把最终 hidden 投影回词表得到 logits。二者处在同一套词表空间的两端；原始 Transformer 让它们**共享权重**（tied），但这是每个模型的配置，不是通例。

## 结构与数据流

```text
token ids [B,T] → Embedding [V, d_model] → hidden [B,T, d_model]
final hidden [B,T, d_model] → LM head [d_model, V] → logits [B,T, V] → softmax
```

原始 Transformer（§3.4）规定：输入 embedding、输出 embedding 与 pre-softmax 线性层**共享同一权重矩阵**，且 embedding 权重额外乘 `√d_model`，decoder 输出再经 softmax 得到下一 token 分布。

dLLM（LLaDA）沿用非因果 backbone：head 在每一步对**所有被掩码位置**同时输出 logits，再按 argmax 概率取置信度做 remasking；词表规模为 126,464（LLaDA 1B/8B），对照 LLaMA3 8B 为 128,000。

## 变体对照

| 变体 | 输入 embedding | 输出 head | 说明 |
| --- | --- | --- | --- |
| 原始 Transformer | 共享权重矩阵 | 共享权重矩阵（pre-softmax 线性层） | embedding 权重乘 `√d_model` |
| 现代 decoder-only | 独立或共享 | 独立或共享 | 是否 tied 由 `config.tie_word_embeddings` 与权重 key 决定，不能默认 |

## 实现差异

不同实现之间可能不同的地方：权重是否 tied、词表是否有 padding、是否有 logit scale / bias、head 的分片方式（张量并行下按词表切分）、特殊 token、输入与输出词表是否一致，以及多模态 projector 是否复用 embedding 空间。仅凭"decoder-only + LM head"不能判断是否 tied，必须看 `config.tie_word_embeddings`、以及 state_dict 里 `lm_head.weight` 是否与输入 embedding 同 key（同 tensor）。

## 量化关联

**结构事实**

- 原始 Transformer 共享 embedding/输出矩阵并给 embedding 乘 `√d_model`（§3.4）。
- LLaDA 的 head 对所有被掩码位置同时输出 logits，词表为 126,464（论文口径）。

**量化推断**

- 大 vocabulary 的 LM head 影响 logits、top-k 与 loss，其权重/激活量化的代价与中间 block 不同。
- tied embedding 量化会同时影响输入表示与输出 logits，必须作为联合约束分析。
- dLLM 的位置选择与置信度依赖 logits；多模态 connector 的输出范围会改变 head 的校准分布。

**未知项**

- 三份来源都不含量化实验；`repo-huggingface-transformers` 只记录量化接口入口（`HfQuantizer` 等），不含量化实验。

## 结果边界

- "共享权重 + `√d_model` 缩放"来自原始 Transformer（2017），不能推断所有 decoder-only 模型都 tied；是否 tied 以目标实现为准。
- LLaDA 的词表与 head 行为是论文口径，不能外推到其它 dLLM。
- 论文不涉及量化；`repo-huggingface-transformers` 只记录量化接口入口，不含量化实验。本页量化内容均为结构推断，且本库未运行任何量化或评测。

## Agent 如何使用

首轮读本页即可判断目标模型 embedding 与 head 是否共享、词表规模、head 如何取 logits。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 原始 Transformer 的 embedding/head 共享与 `√d_model` | [`paper-transformer`](../../sources/papers/paper-transformer.md) |
| dLLM 的 head 输出与词表规模 | [`paper-llada`](../../sources/papers/paper-llada.md) |
| 具体模型类的路径与量化接口入口 | [`repo-huggingface-transformers`](../../sources/repos/repo-huggingface-transformers.md) |
| 校准轴与 torch KV/量化接口 | [Calibration Axes](../checks/calibration-axes.md) |
| 目标实现里 embedding/head 是否 tied 与权重 key | 目标模型 config 与 state_dict（本库不提供） |

回答量化问题时必须说明：embedding 与 head 是否 tied、词表大小、是否有 logit scale/bias、head 是否张量并行切分，以及量化精度配置。

## 来源与证据

- [Attention Is All You Need](../../sources/papers/paper-transformer.md)：embedding / 输出层共享权重与 `√d_model` 缩放（§3.4）。
- [LLaDA](../../sources/papers/paper-llada.md)：非因果 head（§2.2）对所有掩码位置输出 logits（§2.4）、词表规模（附录 B.2 Table 5）。
- [huggingface/transformers](../../sources/repos/repo-huggingface-transformers.md)：固定 commit 下模型类路径与量化 / cache 接口入口。

本页结构以上述来源为限；来源不含量化实验，量化内容均为结构推断。
