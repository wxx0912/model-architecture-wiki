---
{
  "id": "component-kv-cache",
  "title": "KV Cache",
  "type": "component",
  "tags": [
    "kv-cache",
    "attention",
    "gqa",
    "mla",
    "quantization"
  ],
  "related": [
    "component-attention",
    "component-normalization",
    "paradigm-ar",
    "paradigm-linear-attention"
  ],
  "sources": [
    "paper-transformer",
    "paper-deepseek-v2",
    "paper-kimi-linear",
    "repo-mamba"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "key value cache",
    "键值缓存"
  ],
  "updated": "2026-10-24"
}
---

# KV Cache

## 一句话理解

自回归 decode 每步只来一个新 token，若每步都在整段上下文上重算 attention，单步代价是 `O(n²d)`（每层，`n` 为当前序列长；§4 Table 1 给的是逐层 self-attention 复杂度 `O(n²·d)`，“单步已重算”是其直接推导）。KV cache 把历史 K/V 存下来、每步只算新 token 的 query 再读缓存，把解码时的重复计算换成了**随序列增长的运行时显存**。线性 attention / SSM 维护的是另一种固定大小的递推状态，不应称为普通 KV cache。

## 结构与数据流

```text
prefill：整段 prompt → 写出各层 K/V 到缓存
decode ：新 token 的 q → 读缓存 K/V → attention → 追加新 K/V
```

cache 的张量语义由 attention 变体决定（paper-deepseek-v2 §2.1.4，Table 1；按元素个数计，与存储精度无关）：

| 变体 | 缓存内容 | cache 元素 / token |
| --- | --- | --- |
| MHA | 显式 K、V | `2 n_h d_h l` |
| GQA | 显式 K、V（`n_g` 组） | `2 n_g d_h l` |
| MQA | 显式 K、V（1 组） | `2 d_h l` |
| MLA | latent `c^{KV}` 与跨头共享的 `k^R` | `(d_c + d_h^R) l ≈ (9/2) d_h l` |

MLA 的关键在于推理期把 `W^{UK}` 吸收进 query 投影、`W^{UV}` 吸收进输出投影，真正需要缓存的只有 `c^{KV}` 与解耦 key `k^R`（附录 C）。V2 取 `d_c = 4 d_h`、`d_h^R = d_h/2`；按 `d_c=512`、`d_h^R=64`、60 层算，每 token 缓存 `(512+64)×60 = 34560 ≈ 34.6K` 个元素。

## 变体对照

| 类别 | 运行时状态 | 是否随序列增长 | 备注 |
| --- | --- | --- | --- |
| MHA / GQA / MQA / MLA | 显式 K/V 或 latent | 是 | 每 token 新增一份，长度线性增长 |
| 线性 attention（KDA） | 每 head 一个 `d_k × d_v` 状态矩阵 | 否 | 固定大小；prefill 用 chunk kernel、decode 切 recurrent kernel |

## 实现差异

不同实现之间可能不同的地方：cache 的张量语义与 layout、dtype、page size、prefill 写入与 decode 读取是否走同一套 `update` 接口、是否跨 batch/请求复用（如 paged attention），以及是否启用 sliding window / chunked。仅凭"用了 KV cache"不能判断具体布局或精度，必须以目标实现与 config 为准。

## 量化关联

**结构事实**

- cache 是运行时状态，不是权重 checkpoint 的附属字段；它随 decode 增长（线性 attention 的递推状态除外）。
- MLA 的 latent cache 与 MHA/GQA 的 K/V cache 语义不同，必须分开分析。
- DeepSeek-V2 部署期把参数转 FP8，并对 KV cache 做量化（把每个元素平均压到 6 bit，引用外部方法）；KDA 的 KV cache 相对 full attention 结构上降低 up to 75%。

**量化推断**

- cache 量化涉及存储格式、scale 更新、page/layout、prefill 到 decode 的交接与 kernel 支持。
- 校准要覆盖 prefill/decode、上下文长度、batch 与 cache growth。
- 需要比较 cache 量化前后的 attention logits、长上下文任务与真实 decode latency；伪量化结果不能证明目标后端的 cache kernel 已生效。

## 结果边界

- Table 1 按**元素个数**度量 cache，明确与存储精度无关；element 数小不等于字节数按同比例小。
- DeepSeek-V2 的部署量化只有"参数 FP8 + cache 平均 6 bit"两句，**没有精度影响、校准方法或逐层误差**，不能据此判断 MLA 的 cache 更适合低比特。
- KDA 的 "KV cache 降低 75%" 是 3:1 混合结构的**结构性减少**，不是压缩或量化收益；其效率数字是自研 kernel 自报实测、未披露硬件。
- 三篇论文都不含量化精度实验；本页量化推断需在目标实现上另行测量，本库未运行任何量化或评测。

## Agent 如何使用

首轮读本页即可判断目标模型 decode 时到底缓存什么、是否随序列增长、以及是否属于普通 KV cache。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| attention 变体与 cache 内容、MLA 的矩阵吸收 | [Attention](./attention.md)、[`paper-deepseek-v2`](../../sources/papers/paper-deepseek-v2.md) |
| 线性 attention / KDA 的递推状态 | [`paper-kimi-linear`](../../sources/papers/paper-kimi-linear.md)、[Linear Attention](../paradigms/linear-attention.md) |
| 自回归推理的 prefill/decode 阶段 | [Autoregressive Transformer](../paradigms/ar.md) |
| 原始 Transformer 的 attention 复杂度 | [`paper-transformer`](../../sources/papers/paper-transformer.md) |
| 目标后端 cache kernel 是否支持该精度 / layout | 目标推理后端文档与源码（本库不提供） |

回答 cache 量化问题时必须说明：缓存的是显式 K/V 还是 latent、元素数与层数、dtype 与 layout/page size、prefill/decode 交接方式，以及目标后端 kernel 是否支持该精度。

## 来源与证据

- [Attention Is All You Need](../../sources/papers/paper-transformer.md)：self-attention 复杂度与 decoder 掩码（§3.1、§4）。
- [DeepSeek-V2](../../sources/papers/paper-deepseek-v2.md)：MLA 缓存内容与 Table 1 对比、矩阵吸收（附录 C）、部署期 FP8/6-bit 量化（§3.2.3）。
- [Kimi Linear](../../sources/papers/paper-kimi-linear.md)：KDA 固定大小递推状态（§6.3）与结构性 KV 减少（§1、结论）。
- [state-spaces/mamba](../../sources/repos/repo-mamba.md)：SSM 推理状态是固定大小的 `(conv_state, ssm_state)`，不随序列增长。

本页结构以上述来源为限；来源不含量化精度实验，量化内容均为结构推断。
