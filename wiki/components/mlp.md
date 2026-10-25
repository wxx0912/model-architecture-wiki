---
{
  "id": "component-mlp",
  "title": "MLP and Gated FFN",
  "type": "component",
  "tags": [
    "mlp",
    "swiglu",
    "geglu",
    "quantization"
  ],
  "related": [
    "paradigm-ar",
    "paradigm-moe",
    "component-attention",
    "component-normalization"
  ],
  "sources": [
    "paper-transformer",
    "paper-mixtral",
    "paper-llada",
    "paper-deepseek-v2",
    "paper-deepseek-v3"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-24"
}
---

# MLP 与 Gated FFN

## 一句话理解

Transformer 的 FFN 是 **position-wise 的两层 MLP**：先把 hidden 投到更宽的中间维，再投回 `d_model`。现代 LLM 多用 **gated FFN**（gate 与 up 两条投影，激活后逐元素相乘，再经 down 投影）；稀疏 MoE 里每个专家就是一个这样的 FFN。

## 结构与数据流

```text
标准 FFN ： x[B,T,d_model] → W1(d_ff) → act → W2 → [B,T,d_model]
gated FFN： x → [W_gate x, W_up x](d_ff) → act(W_gate x) ⊙ (W_up x) → W_down → [B,T,d_model]
```

- 原始 Transformer（§3.3，Eq.2）：`FFN(x) = max(0, xW_1 + b_1)W_2 + b_2`，`d_model=512`、`d_ff=2048`；同一层内所有位置共享同一组参数，层与层之间不共享。
- Mixtral（§2.1）：专家用 **SwiGLU** 作专家函数 `E_i(x)`（gated FFN 族的一个实例；§2.1 明确 “the same SwiGLU architecture as the expert function”，图注的 “standard feedforward block” 不代表两层 ReLU FFN），取 top-2，输出 `y = Σ_i Softmax(Top2(x·W_g))_i · SwiGLU_i(x)`。
- DeepSeekMoE（§2.2.1）：`h'_t = u_t + Σ_{i=1}^{N_s} FFN^{(s)}_i(u_t) + Σ_{i=1}^{N_r} g_{i,t} FFN^{(r)}_i(u_t)`；共享专家与路由专家各是一个 FFN。

## 变体对照

| FFN 形式 | 结构 | 来源例证 |
| --- | --- | --- |
| 标准 FFN | 两层 + ReLU | Transformer：`d_ff=2048`、`d_model=512` |
| gated FFN 族（例：SwiGLU） | gate/up 两条投影 + 逐元素乘 + down | Mixtral 专家、LLaDA backbone 均用 SwiGLU |
| MoE FFN | 多个独立 FFN + router | DeepSeek-V2：每个 MoE 层 2 共享 + 160 路由（首层为 dense FFN），专家中间维 1536（Lite 为 2 + 64、1408） |

## 实现差异

不同实现之间可能不同的地方：激活 / 门控变体（ReLU / SwiGLU / GeGLU / Squared-ReLU）、gate 与 up 是否 fused 成一个权重、是否有 bias、张量并行下按哪种维度切分、中间张量是否物化，以及 MoE 里是否用 grouped GEMM 与 3D 专家权重（激活变体与后几项是通用工程约定，不属本页来源的事实；本页来源只涉及 ReLU 与 SwiGLU）。实现名称相同不代表权重 shape 或融合方式相同，需按目标实现的模块前缀与权重布局确认。

## 量化关联

**结构事实**

- gate 与 up 共享同一输入，逐元素乘法会改变误差传播路径；down 投影把误差重新注入 residual stream。
- dense FFN 的中间维通常比 `d_model` 宽（`d_ff > d_model`）；MoE 的细粒度专家则相反（DeepSeek-V2 专家中间维 1536 ≪ hidden 5120）。
- MoE 专家的 MLP 还受 token 路由与专家负载影响，各专家分布不同。
- DeepSeek-V3 的 FP8 训练契约（§3.3）：激活按 `1x128` tile、权重按 `128x128` block 分组缩放；**MoE 的 SwiGLU 输入按 FP8 缓存**（反向重算），MoE 上投影前的**激活**与下投影前的**激活梯度**各自量化成 FP8 后再参与 dispatch 通信，而 forward/backward 的 `combine` 组件仍保持 BF16；MoE gating 算子保留高精度。

**量化推断**

- 应分别统计 gate、up、逐元素乘法前后与 down 的误差，而不是只看单条 Linear 的 MSE。
- gate/up 是否需要同精度取决于具体 fused kernel 与量化格式，不应默认强制统一。
- 只做单个 Linear 的量化误差不必然反映 gated FFN 的 block 输出质量；dense FFN 的宽中间维通常是激活异常值的主要落点，需单独统计。

## 结果边界

- `FFN(x)=max(0,xW_1+b_1)W_2+b_2`、`d_ff=2048` 是原始 Transformer（2017）的口径，不含 SwiGLU、GeGLU 或 MoE；这些更晚的结构要引各自的来源。
- gated FFN 在本库的一手例证是 Mixtral（SwiGLU 专家）与 LLaDA（LLaMA 类 backbone），不能把它们的具体激活/维度当作所有 gated FFN 的通用定义。
- DeepSeek-V3 的 FP8 细节属于其**训练契约**，不能直接当作其它模型或推理后端的量化方案。
- 本库未运行任何量化或评测；本页量化内容除已注明来源者外均为结构推断。

## Agent 如何使用

首轮读本页即可判断目标是标准 FFN、gated FFN 还是 MoE FFN，gate/up/down 的投影关系与中间维。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 标准 FFN 公式与维度 | [`paper-transformer`](../../sources/papers/paper-transformer.md) |
| SwiGLU 专家的门控形式 | [`paper-mixtral`](../../sources/papers/paper-mixtral.md) |
| MoE 专家布局与路由 | [`paper-deepseek-v2`](../../sources/papers/paper-deepseek-v2.md)、[Sparse MoE](../paradigms/moe.md) |
| FP8 训练下 MLP/MoE 的量化与通信契约 | [`paper-deepseek-v3`](../../sources/papers/paper-deepseek-v3.md) |
| 目标实现的 gate/up/down 命名、权重 shape 与后端 kernel | 目标模型源码 / 配置（本库不提供） |

回答量化问题时必须说明：FFN 是 dense 还是 MoE、激活类型、gate/up 是否 fused、`d_ff`/专家中间维、量化的是权重还是激活，以及目标后端与精度。

## 来源与证据

- [Attention Is All You Need](../../sources/papers/paper-transformer.md)：标准 FFN 公式与 `d_ff`（§3.3）。
- [Mixtral of Experts](../../sources/papers/paper-mixtral.md)：SwiGLU 专家的门控形式（§2.1）。
- [LLaDA](../../sources/papers/paper-llada.md)：LLaMA 类 backbone 的 SwiGLU 组件（附录 B.2 Table 5）。
- [DeepSeek-V2](../../sources/papers/paper-deepseek-v2.md)：MoE 专家结构、共享/路由专家与中间维（§2.2.1、§3.1.2）。
- [DeepSeek-V3](../../sources/papers/paper-deepseek-v3.md)：FP8 训练下激活/权重的分组量化与 SwiGLU 缓存（§3.3）。

本页结构以上述来源为限；来源均为论文口径，量化内容除注明者外为结构推断。
