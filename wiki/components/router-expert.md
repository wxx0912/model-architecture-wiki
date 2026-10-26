---
{
  "id": "component-router-expert",
  "title": "MoE Router, Dispatch and Expert MLP",
  "type": "component",
  "tags": [
    "moe",
    "routing",
    "mlp",
    "quantization"
  ],
  "related": [
    "paradigm-moe",
    "paradigm-latent-moe",
    "component-mlp"
  ],
  "sources": [
    "paper-switch-transformer",
    "paper-mixtral",
    "paper-deepseek-v2",
    "paper-deepseek-v3",
    "repo-deepseek-v3"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "mixture of experts",
    "混合专家"
  ],
  "updated": "2026-10-26"
}
---

# MoE Router、Dispatch 与 Expert MLP

## 一句话理解

MoE block 可拆成四段：**router** 给每个 token 打分并选 top-k、**dispatch** 按专家分组（分布式下是 all-to-all）、**专家 MLP** 各自计算、**combine** 按 routing weight 加权并回到原 token 序。把"连续数值误差"与"离散 top-k 路由变化"分开，是量化分析的关键。

## 结构与数据流

```text
hidden [N, d] → router scores [N, E] → top-k → dispatch（按专家分组）
             → 各专家 FFN → combine（按 routing weight 加权、回原序）
```

- Switch Transformer（§2.1）：top-1 路由，`p_i(x) = softmax(h(x))_i`，`y = Σ_{i∈TopK(p(x))} p_i(x)·E_i(x)`。
- Mixtral（§2.1）：`G(x) := Softmax(TopK(x·W_g))`（未选中的 logits 置 `−∞`，可整块跳过），取 top-2。
- DeepSeekMoE（§2.2.1）：`h'_t = u_t + Σ_{i=1}^{N_s} FFN^{(s)}_i(u_t) + Σ_{i=1}^{N_r} g_{i,t} FFN^{(r)}_i(u_t)`，`g_{i,t}` 取 affinity 的 top-`K_r`（共享专家 + 路由专家）。

**容量与丢弃**（Switch §2.2）：`expert capacity = (tokens per batch / number of experts) × capacity factor`；路由溢出的 token 跳过该层、经残差直接传给下一层；论文设定下丢弃通常 `<1%`。均衡用辅助损失 `loss = α·N·Σ_i f_i P_i`（`f_i` 不可微、`P_i` 可微），取 `α = 1e-2`。

**均衡与路由约束**：

- DeepSeek-V2（§2.2.2–2.2.4）：device-limited routing（每个 token 的目标专家最多分布在 `M` 台设备，`M ≥ 3` 时性能接近不限）；三类辅助损失 `L_ExpBal`/`L_DevBal`/`L_CommBal`；训练期按设备 token dropping（容量因子 1.0）。
- DeepSeek-V3（§2.1.2）：affinity 改用 **sigmoid**，给每个专家一个 bias `b_i`、**只加在 top-K 选择上**（`s_{i,t} + b_i ∈ TopK`），训练每 step 末按 batch 负载调整（过载 `−γ`、欠载 `+γ`）；仍保留序列级 `L_Bal` 但系数极小；node-limited routing；训练与推理**都不丢 token**。

## 变体对照

| 变体 | affinity | top-k | 均衡方式 | 丢弃 |
| --- | --- | --- | --- | --- |
| Switch Transformer | softmax | top-1 | 辅助损失 `α=1e-2` | 容量因子 + 丢弃（训练 <1%） |
| Mixtral | `Softmax(TopK(x·W_g))` | top-2 | 论文未描述容量/辅助损失机制 | 论文未描述容量/丢弃机制 |
| DeepSeek-V2 | softmax affinity | top-K + 共享专家 | 三类辅助损失 + device-limited | 训练期按设备丢 token |
| DeepSeek-V3 | sigmoid affinity | top-K + 共享 + 路由 bias | bias 动态调整为主 + 极小系数序列级 `L_Bal`，node-limited | 训练/推理都不丢 token |

## 实现差异

不同实现之间可能不同的地方：affinity 用 softmax 还是 sigmoid、top-k 取值、是否有共享专家、容量与丢弃策略、均衡用辅助损失还是路由 bias、dispatch/combine 的通信顺序与并行方式，以及专家权重的布局与 grouped GEMM 约定（后几项是通用工程约定，不属本页来源的事实）。DeepSeek-V3 官方推理实现里 `Gate`、`Expert`、`MoE` 是分开的模块（model.py:535/601/636），配置从 `configs/*.json` 读取。

## 量化关联

**结构事实**

- router 输出是**离散选择**（top-K），专家是否参与计算由它决定；`combine` 按 routing weight 加权。
- DeepSeek-V2 用 device-limited routing + 三类辅助损失 + 训练期 token dropping（容量因子 1.0）；DeepSeek-V3 用 sigmoid affinity、专家 bias 只参与 top-K 选择、node-limited routing，且不丢 token。

**实现事实**（来自固定版本源码，须按该 revision 核对）

- DeepSeek-V3 官方推理实现含 fp8 路径（`act_quant` 激活量化 + `fp8_gemm`）；671B 配置为 256 路由专家 + 1 共享、每 token 激活 8 个。

**量化推断**

- router 分数的微小变化会造成离散专家切换，不能只用 router 输出的 MSE 评价。
- dispatch/combine 的索引、权重与通信必须保持精确语义；专家权重与激活应按专家访问分布校准。
- fused grouped GEMM 的 group size、packing、scale layout 与专家维度是导出前硬约束。

**未知项**

- 五份来源都不含量化精度的完整对比；例外是 DeepSeek-V3 用 FP8 vs BF16 相对损失误差 <0.25% 验证其 FP8 训练框架。

## 结果边界

- Switch 的容量公式假定张量形状在编译期静态确定（其 TPU 语境），"丢弃 <1%" 是其数据与配置内的观测，不能外推。
- Mixtral 的路由分析显示**没有明显的按主题的专家专业化**，专家选择更对应句法（如 Python 的 `self` 与英文 `Question` 常走同一专家），这是其层 0/15/31 上的观测。
- DeepSeek-V3 的"不丢 token"与 V2 的"训练期丢 token"是各自配置，不能互推；官方代码里 `ModelArgs` 默认值（64 路由专家、2 共享）**不是模型配置**，671B 须用 `configs/config_671B.json` 的 256/1/8。
- DeepSeek-V3 论文的 FP8 是**训练精度框架**（附录 B.1 给出 FP8 vs BF16 相对损失误差 <0.25%）；官方**推理实现仓库**的 fp8 路径（`act_quant`/`fp8_gemm`）是另一回事，属推理契约，不能当作误差分析。本库未运行任何量化或评测，量化内容均为结构推断。

## Agent 如何使用

首轮读本页即可判断目标 MoE 的 affinity、top-k、是否有共享专家、均衡与丢弃策略、以及模块划分。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| top-1 路由、容量因子、丢弃与辅助损失 | [`paper-switch-transformer`](../../sources/papers/paper-switch-transformer.md) |
| 全层替换、top-2、路由分析 | [`paper-mixtral`](../../sources/papers/paper-mixtral.md) |
| DeepSeekMoE 的细粒度/共享专家、device-limited 与辅助损失 | [`paper-deepseek-v2`](../../sources/papers/paper-deepseek-v2.md) |
| 无辅助损失均衡、路由 bias、node-limited | [`paper-deepseek-v3`](../../sources/papers/paper-deepseek-v3.md) |
| 官方推理实现里的 Gate/Expert/MoE 与 fp8 路径 | [`repo-deepseek-v3`](../../sources/repos/repo-deepseek-v3.md) |
| 目标 MoE 实现的 affinity/top-k/均衡与后端 kernel | 目标模型源码 / 配置（本库不提供） |

回答量化问题时必须说明：affinity 与 top-k、专家总数与激活数、是否有共享专家、容量/丢弃策略、并行方式与通信顺序、以及目标后端 kernel 支持。

## 来源与证据

- [Switch Transformer](../../sources/papers/paper-switch-transformer.md)：top-1 路由、容量因子、丢弃与辅助损失。
- [Mixtral of Experts](../../sources/papers/paper-mixtral.md)：top-2 门控、全层替换与路由分析。
- [DeepSeek-V2](../../sources/papers/paper-deepseek-v2.md)：DeepSeekMoE、device-limited routing、三类辅助损失。
- [DeepSeek-V3](../../sources/papers/paper-deepseek-v3.md)：sigmoid affinity、路由 bias 动态调整（论文称 auxiliary-loss-free）与 node-limited routing。
- [DeepSeek-V3 官方实现](../../sources/repos/repo-deepseek-v3.md)：固定 commit 下 Gate/Expert/MoE 模块、fp8 推理路径与 671B 配置。

本页结构以上述来源为限；来源不含量化精度实验，量化内容均为结构推断。
