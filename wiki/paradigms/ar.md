---
{
  "id": "paradigm-ar",
  "title": "Autoregressive Transformer",
  "type": "paradigm",
  "tags": [
    "ar",
    "attention",
    "kv-cache",
    "rope",
    "quantization"
  ],
  "related": [
    "component-attention",
    "component-kv-cache",
    "component-normalization",
    "component-position-encoding",
    "check-quantization-preflight"
  ],
  "sources": [
    "paper-transformer",
    "repo-huggingface-transformers"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "autoregressive",
    "decoder-only",
    "自回归"
  ],
  "updated": "2026-10-26"
}
---

# Autoregressive Transformer

## 一句话理解

AR 模型把整段序列的概率分解成逐 token 条件概率之积：训练时一次并行算完整段，生成时从左到右逐 token 解码。decoder-only 主干由 token embedding、因果 self-attention、归一化、FFN 和 LM head 组成（FFN 的具体形式随实现而不同）。本页只讲这条执行路径与它的量化边界，不含 MoE、dLLM、状态空间等并列范式。

## 生成过程与状态

训练目标就是自回归分解，每个位置只依赖它之前的 token：

```text
p(x) = Π_{t=1..T} p(x_t | x_{<t})
```

因果 mask 保证位置 `t` 看不到 `t` 之后的 token，因此整段可以一次前向、并行计算所有位置的损失。生成分两个阶段：

- **prefill**：把整段 prompt 一次送进模型，为每层写入 K/V（或该变体对应的压缩状态），产出第一个新 token；
- **decode**：每步只把新 token 送进模型，当前 token 计算 Q/K/V，新的 K/V 追加进 cache，Q 与 cache 中的历史 K/V 做 attention，每步吐 1 个 token。

运行时状态就是 KV cache 加上当前长度：每个 token 的 cache 增量由 attention 变体决定（MHA/GQA/MLA 的差异见 [Attention](../components/attention.md)，layout 与更新见 [KV Cache](../components/kv-cache.md)）。停止条件是生成 `EOS` 或到达长度上限；采样方式（greedy / temperature / top-k / top-p / beam）不改变结构，但会改变激活分布。

| 维度 | prefill | decode |
| --- | --- | --- |
| 输入 | 整段 prompt `[B, T]` | 每步 1 个新 token `[B, 1]` |
| attention | 整段参与，一次算出全部位置的输出 | 算新 token 的 Q/K/V，Q 与历史（含本步新增）K/V 做 attention |
| 对 cache 的动作 | 写入本层 K/V | 动态缓存追加 1 个位置并读全量历史；预分配（static）或滑窗实现的行为不同 |
| 产出 | 第一个新 token | 每步 1 个 token |

## 实现差异

同一个 AR 契约在不同实现之间可能不同的地方：pre-norm 还是 post-norm；attention 用 MHA / GQA / MLA；RoPE 的插入位置；QKV 是否融合；KV cache 的类型（动态增长、预分配 static、是否滑动窗口、以什么精度存）；张量并行与序列并行的切分方式；LM head 是否与 embedding 共享权重。FlashAttention 一类 kernel 只改变中间张量是否物化，不改变结构语义。这些要按目标实现逐项核对。

## 量化关联

**结构事实**

- prefill 的输入是整段、decode 的输入是单 token，两者的激活分布与 batch 形态不同。
- KV cache 是随序列长度增长的运行时状态；Q/K/V/O、attention logits 与 softmax、FFN 的 gate/up/down、LM head 是彼此不同的数值路径。

**量化推断**

- 校准要分别覆盖 prefill 与 decode；只覆盖其中一项代表不了另一项。
- cache 量化属于运行时状态量化，写入、读取、分页与长上下文下的行为都要单独验证。
- 逐层误差要同时看 block 输出、最终 logits 和真实后端生成；只看权重 MSE 或单层输出误差不足以下结论。
- 能否对 QKV 复用同一组 scale，取决于后端的 packing 与 kernel 契约，不是结构决定的。

## 结果边界

- 本页描述的是 decoder-only AR 的通用执行路径；具体模型的模块名、融合方式、cache 类型与精度以目标实现为准。
- Transformer 原论文是 encoder-decoder、Post-LN、正弦位置编码，不能直接当作现代 decoder-only LLM 的实现描述。
- 本页关于训练可整段并行、prefill 产出第一个 token 之类的表述，是对执行路径的结构性描述；`paper-transformer` 只给出因果 mask 的机制，没有描述现代实现的两阶段执行。
- 本页没有在目标后端运行过量化的模型，量化部分都是结构推断，不构成精度或速度证据。

## Agent 如何使用

首轮读本页即可判断目标模型是否属于 decoder-only AR，以及生成时的运行时状态是什么。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 因果 attention 与 decoder 的原始定义 | [`paper-transformer`](../../sources/papers/paper-transformer.md) |
| KV cache 的类型、接口与可选精度 | [`repo-huggingface-transformers`](../../sources/repos/repo-huggingface-transformers.md)（`cache_utils.py`） |
| attention 变体、cache layout、归一化等细节 | [Attention](../components/attention.md)、[KV Cache](../components/kv-cache.md)、[RMSNorm 与归一化边界](../components/normalization.md) |
| 目标 checkpoint 的配置字段与模块名 | 该 checkpoint 的 `config.json` 与源码（本库不提供） |

回答 AR 量化问题时必须说明：是在 prefill 还是 decode 上做、量化对象是权重 / 激活 / cache 中哪一类、序列长度与 batch，以及目标后端是否真的走低比特路径。

## 来源与证据

- [Attention Is All You Need](../../sources/papers/paper-transformer.md)：decoder self-attention 与因果 mask 的原始定义（encoder-decoder、Post-LN、正弦位置编码；不含 RoPE、KV cache 与现代实现）。
- [huggingface/transformers](../../sources/repos/repo-huggingface-transformers.md)：固定 commit 下 KV cache 的运行时契约（`cache_utils.py` 的 `DynamicCache`/`StaticCache`/量化缓存层）与各模型实现文件位置。

本页的执行路径描述以这两份来源的 `source-reported` 事实为限；具体实现对 cache 的处理与 fused kernel 都必须回到目标仓库核对。
