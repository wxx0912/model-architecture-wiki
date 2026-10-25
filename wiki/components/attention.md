---
{
  "id": "component-attention",
  "title": "Attention: MHA, GQA and MLA",
  "type": "component",
  "tags": [
    "attention",
    "mha",
    "gqa",
    "mla",
    "rope",
    "kv-cache"
  ],
  "related": [
    "paradigm-ar",
    "paradigm-moe",
    "component-kv-cache",
    "model-deepseek"
  ],
  "sources": [
    "paper-transformer",
    "paper-rope",
    "paper-deepseek-v2",
    "paper-deepseek-v3",
    "repo-huggingface-transformers"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "注意力",
    "self-attention",
    "multi-head attention",
    "多头注意力",
    "grouped-query attention",
    "分组查询注意力",
    "multi-head latent attention",
    "多头潜在注意力"
  ],
  "updated": "2026-10-24"
}
---

# Attention：MHA、GQA 与 MLA

## 一句话理解

Self-attention 让每个位置用 query 与序列中所有 key 求相似度、再对 value 加权求和，是 Transformer 主干的核心算子。MHA、GQA/MQA、MLA 在计算图里占同一个位置：MHA/GQA/MQA 只改 K/V head 的共享关系，MLA 则把 K/V 换成低秩压缩的 latent，并额外加一条解耦的 query 分支与一条跨头共享的解耦 key 分支（用于承载 RoPE）。本页描述这些结构差异与对应的量化边界，不代表某个后端已经支持低比特 attention kernel。

## 结构与数据流

核心算子是 scaled dot-product attention：

```text
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

输入 hidden state `h [B, T, d_model]`，位置信息按变体注入；随后 Q/K/V 投影 → `QKᵀ/√d_k` → mask → softmax → 对 V 加权 → 输出投影 `W^O`。三种变体在这段流程里位置相同，只在 K/V 的来源上分岔。下面记 `n_h` 为 query head 数、`d_h` 为每头维度（MHA 里即 `d_k`、`d_v`），`l` 为层数。

- **MHA**：每个 query head 配独立的 K/V head。`W^Q_i, W^K_i ∈ R^{d_model×d_k}`、`W^V_i ∈ R^{d_model×d_v}`、`W^O ∈ R^{h·d_v×d_model}`。
- **GQA / MQA**：只改 q_heads 与 kv_heads 的数量关系——若干（MQA 是全部）query head 共享同一个 K/V head，`W^K/W^V` 的 head 数相应减少。
- **MLA**：K/V 先做低秩联合压缩 `c^{KV} = W^{DKV} h`（`d_c ≪ d_h·n_h`，每个 token 只留 `c^{KV} ∈ R^{d_c}`），再上投影出 `k^C = W^{UK} c^{KV}`、`v^C = W^{UV} c^{KV}`。RoPE 若直接作用在 `k^C` 上会让上投影矩阵与位置矩阵耦合、无法被吸收，于是另设跨头共享的解耦 key `k^R = RoPE(W^{KR} h)`（每 token 一份，`∈ R^{d_h^R}`）与解耦 query `q^R = RoPE(W^{QR} c^Q)`；每头的 `q_i = [q^C_i; q^R_i]`、`k_i = [k^C_i; k^R]`，attention 分母相应变为 `√(d_h + d_h^R)`。query 侧的低秩压缩 `c^Q = W^{DQ} h` 论文明确只用于降低训练期激活显存。

位置信息注入方式随变体不同：正弦位置编码加在输入 embedding 上，RoPE 类模型作用在 Q/K 上，而 MLA 只作用在解耦的 `q^R`/`k^R` 上，不碰压缩分支的 `q^C`/`k^C`。推理分两个阶段（AR 通用，见 [Autoregressive Transformer](../paradigms/ar.md)）：prefill 一次算整段，decode 每步只算新 token 的 query、读历史缓存；MLA 把 `W^{UK}` 并入 query 侧投影、`W^{UV}` 并入 `W^O` 之后，decode 只需要读缓存的 `c^{KV}` 与 `k^R`（缓存布局与更新方式见 [KV Cache](./kv-cache.md)）。

## 变体对照

`l` 为层数，`n_h` 为 query head 数、`n_g` 为 KV head 组数。cache 一列按元素个数计（DeepSeek-V2 Table 1 的口径）。

| 变体 | query ↔ K/V head | 推理缓存的内容 | cache 元素 / token | RoPE 作用位置 |
| --- | --- | --- | --- | --- |
| MHA | 一一对应 | 显式 K、V | `2 n_h d_h l` | Q、K |
| GQA | `n_h / n_g` 个 query 共享 1 个 KV | 显式 K、V | `2 n_g d_h l` | Q、K |
| MQA | 全部 query 共享 1 个 KV | 显式 K、V | `2 d_h l` | Q、K |
| MLA | K/V 由 latent 上投影，K 另加共享解耦分量 | `c^{KV}` 与跨头共享的 `k^R` | `(d_c + d_h^R) l` | 只作用在 `q^R` / `k^R` |

DeepSeek-V2 取 `d_c = 4 d_h`、`d_h^R = d_h / 2`，论文据此说 MLA 的 cache 量约等于“2.25 个 group 的 GQA”；按 V2 自己的 `d_c=512`、`d_h^R=64`、60 层算，是 `(512+64)×60 = 34560 ≈ 34.6K` 元素 / token。

## 实现差异

同一个变体在不同实现之间可能不同的地方：QKV 投影是否融合、`head_dim` 取值、q_heads / kv_heads 的配置字段名与张量并行按 head 的切分方式；RoPE 是作用在投影后的 Q/K 还是解耦分支、旋转前还是旋转后写入 cache；KV cache 的 layout 与 page 大小；MLA 的 `d_c`、`d_h^R` 取值以及上投影矩阵是否在推理期被合并。这些要按目标实现逐项核对：仅凭模块名不能判断 Q/K/V 是否已融合，也不能假设厂商实现与 transformers 的独立重写逐位一致。

## 量化关联

**结构事实**

- Q/K/V/O 投影、RoPE、softmax、KV cache 与输出投影是彼此不同的数值路径。
- GQA/MQA/MLA 改变 KV cache 的形状与更新方式；MLA 推理只需缓存 `c^{KV}` 与跨头共享的 `k^R`，其语义与 MHA/GQA 的显式 K/V cache 不同，要分开分析。
- DeepSeek-V2 §3.2.3 给出部署期量化陈述：参数转 FP8、KV cache 每个元素平均压到 6 bit；论文没有给这次量化的误差分析、校准方法或逐层影响。

**量化推断**

- 低比特误差会改变 attention logits 与 softmax 分布，进而影响长上下文行为；Q、K、V、O、latent 投影与 RoPE 输入的取值范围各自不同，prefill 与 decode 的激活分布也不同，验证要分开做。
- 能否对 QKV 做统一量化、能否复用同一组 scale，取决于具体后端与 packing 规则，不是结构决定的通用约束。
- MLA 把 K/V 压成 latent 之后缓存内容变了，但论文没有讨论对 latent cache 做低比特量化的误差；“压缩后更省显存就更省量化代价”不成立。

## 结果边界

- 本页的 MHA 定义来自 Transformer 原论文；GQA、MQA 在本库没有登记论文一手来源页，只出现在 DeepSeek-V2 的对比里，不能读成“这些变体都由 DeepSeek 提出”。
- 具体模型用哪种 head 关系、`d_c` / `d_h^R` 取多少、cache 的真实 layout 与 kernel，以目标 checkpoint 的 `config.json`、源码和后端为准，本页不能替代。
- 变体对照表的 cache 公式按元素个数度量，不含字节数、访存与 kernel 开销，不能据此推断显存占用或速度。
- 本页只描述结构，没有在目标后端跑过低比特 attention。

## Agent 如何使用

首轮读本页即可判断目标 attention 属于 MHA / GQA / MLA 中的哪一种、推理时缓存什么。需要精确证据时按下面展开：

| 需要确认的内容 | 继续读取 |
| --- | --- |
| attention 原始定义、正弦位置编码 | [`paper-transformer`](../../sources/papers/paper-transformer.md) |
| MLA 的低秩压缩、解耦 RoPE、矩阵吸收与 cache 对比 | [`paper-deepseek-v2`](../../sources/papers/paper-deepseek-v2.md)、[`paper-deepseek-v3`](../../sources/papers/paper-deepseek-v3.md) |
| 各模型的注意力实现类与 config 字段 | [`repo-huggingface-transformers`](../../sources/repos/repo-huggingface-transformers.md) |
| 目标模型实际的 head 关系、RoPE 插入点、cache layout | 目标 checkpoint 的 `config.json`、源码与 runtime |

回答 attention 相关问题时，必须说明是 MHA/GQA/MLA 的哪一种（q_heads / kv_heads 各多少）、缓存的是显式 K/V 还是 latent、RoPE 作用在哪一步，以及目标后端是否真的会走对应的 kernel。

## 来源与证据

- [Attention Is All You Need](../../sources/papers/paper-transformer.md)：MHA 与 scaled dot-product attention 的原始定义（该篇为 Post-LN + 正弦位置编码，不含 GQA/MQA/MLA/RoPE）。
- [RoFormer / RoPE](../../sources/papers/paper-rope.md)：旋转位置编码作用在 query/key 上的构造（MLA 如何应用 RoPE 见 DeepSeek-V2/V3 来源）。
- [DeepSeek-V2](../../sources/papers/paper-deepseek-v2.md)、[DeepSeek-V3](../../sources/papers/paper-deepseek-v3.md)：MLA 的低秩联合压缩、解耦 RoPE、推理期矩阵吸收、cache 公式对比，以及 V2 §3.2.3 的部署期 FP8 / 6-bit 陈述。
- [huggingface/transformers](../../sources/repos/repo-huggingface-transformers.md)：固定 commit 下各模型的注意力实现类与 config 字段。

结构定义以论文的 `source-reported` 描述为主；head 数、latent 维度、cache 布局属于实现事实，必须回到目标 checkpoint 的 `config.json` 与源码核对。
