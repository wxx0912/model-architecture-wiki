---
{
  "id": "paradigm-multihead-latent-moe",
  "title": "Multi-Head LatentMoE",
  "type": "paradigm",
  "tags": [
    "latent-moe",
    "moe",
    "routing",
    "quantization"
  ],
  "related": [
    "paradigm-latent-moe",
    "paradigm-latent-moe-nvidia",
    "paradigm-moe",
    "component-router-expert",
    "check-quantization-preflight"
  ],
  "sources": [
    "paper-multihead-latentmoe"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-26"
}
---

# Multi-Head LatentMoE

## 一句话理解

Multi-Head LatentMoE 把 token 投影后切成 `N_h` 个子 token，每个子 token 由一个**独立的**小 MoE 实例处理，再拼回原维。它配套一条 **Head Parallel**：把 all-to-all 挪到路由**之前**，做到通信量与激活专家数 `k` 无关（`O(1)`）、流量均衡且通信模式确定。

## 结构与数据流

结构式（§3.1，Eq.5–6）：

```text
[x_{t,1}, …, x_{t,N_h}] = split(W_in x_t)          # W_in ∈ R^{d×d}
o_t = W_out · concat(f_1(x_{t,1}), …, f_{N_h}(x_{t,N_h}))
```

每个 `f_i` 是独立的 MoE 实例（自己的 router 与专家集，彼此不共享参数）；惯例取 `N_h · d_h = d`，但论文强调不必须相等。除线性投影外，总 FLOPs 与等价 MoE 相同。

**Head Parallel**（§3.2）：前提 `P ≤ N_h` 且 `N_h` 能被 `P` 整除。每 GPU 初始持有 `(B, T, N_h, d_h)` 的子 token 张量，一次 all-to-all 后变成 `(B, T, P, N_h/P, d_h)`（每个 GPU 拿到某个 head 子集的**全部**子 token），MoE 计算后再反向 all-to-all 送回。由此得到三个结果：

1. 每 token 只发一次 → **通信量与 `k` 无关**（`O(1)`）；
2. 各 GPU 收发等量 → 流量确定均衡；
3. 通信模式与路由无关 → **不需要预先交换 metadata**，也避开"等最慢节点 / 某 GPU 路由到过多 token 导致 OOM"这类边界。

**两个 IO-aware 算子**（§3.3–3.4）：

1. routing 把 token 与专家都分块，在 SRAM 内算分数并在线维护 top-k 累加器，只把 top-k 的 score/index 写回 HBM（bias 在 top-k 前加入、之后再减回），HBM 访问由 `O(N_e)` 降到 `O(k)`（§3.3 原句；§1 另处写作由 `O(T·N_e + N_e)` 降到 `O(T + N_e)`，两处口径不同）且**精确**；
2. 专家计算利用 FFN 与 attention 的对偶，把稀疏专家改写成 block-sparse attention 并复用 FlexAttention（取 `score_mod(s) = log(gelu(s) + 1)`，再乘 `ℓ` 并减掉每 token 对应专家的 `Σ V_j` 恢复精确结果），实现是纯 PyTorch、不需自写 CUDA kernel。

## 变体对照

EP 与 Head Parallel 的对照（§4.2，Figure 3）：EP 的 all-to-all 延迟与峰值显存随 `k` 和负载倾斜一起增长；HP 对所有 `k`/倾斜**保持不变**，因为它在路由前只发一次。

Head 配置消融（§4.3，Table 2，保持 `N_h·d_h = d`）：

| 配置 | val loss | 训练成本 | SRAM 压力 |
| --- | --- | --- | --- |
| `16×64` | 3.56 | 1.34× | 低 |
| `8×128` | 3.48 | 1.00× | 中 |
| `4×256` | 3.41（最低） | 1.07× | 高 |

主结果（§4.1，Table 1；参数按"激活-总参"记，训练成本以同档 MoE EP 为 1.00×）：2B 总参档 MoE EP 35.36h / ppl 15.56 / 平均 43.95，LatentMoE EP 32.71h（0.93×）/ 15.31 / 44.83，**MH LatentMoE HP 31.68h（0.90×）** / 15.61 / 43.93；4B 总参档 MoE EP 55.34h / 15.01 / 45.30，LatentMoE EP 53.76h（0.97×）/ 14.84 / 44.55，**MH LatentMoE HP 34.41h（0.62×）** / 15.02 / 45.17。训练提速因此是 **2B 档 1.11×（31.68 vs 35.36h）、4B 档 1.61×（34.41 vs 55.34h）**。

另一种"分离路由 token"设计（§4.4，把 token 投影成 `2N_h` 个子 token）只在训练早期略有改善，且使 HP 下 all-to-all 体积翻倍，未被采用。

## 实现差异

不同实现之间可能不同的地方：head 数 `N_h` 与子维 `d_h`、是否 `N_h·d_h = d`、router 是否用分离的 token、子 MoE 是否共享参数、是否启用 HP（以及 HP/EP 的拓扑拆分），以及 IO-aware 算子是否落到对应实现。`N_h` 与 `d_h` 都是设计旋钮，`8×128` 只是该论文的实验折中。

## 量化关联

**结构事实**

- token 被投影到 `N_h` 个 `d_h` 维子 token，每个子 token 有自己的 router 与专家。
- 论文用 **FP32** 计算 router 以保证数值稳定（§2.2、附录 A）。

**量化推断**

- 低维专家权重是量化对象：head 间共享 scale、跨设备通信、低维专家权重与恢复投影都需要独立评估，不能沿用单 MoE 的配置。
- 量化后需按 head / 专家 / 恢复路径分别做局部数值与端到端验证，并检查路由一致率（本库新增的检查项，source 无此依据）。
- 论文的 FP32 router 是数值稳定措施，**不是**量化机制，不能当作"router 必须高精度"的实验结论。

**未知项**

- 论文全文不含量化或低比特路径。

## 结果边界

- 所有实验都是小规模对照：12 层、`d=1024`、0.2B 激活参数、10B token、H100 + NVLink；"1.61× / 1.11× 训练提速"只在这个规模与该实现下成立（摘要与 §4.1 对 1.11× 档位的口径不一致，两处均为原文）。
- 论文 §6 明确方法在**超稀疏区间**（专家数大、专家小）最有效；HP 的上界是 head 数，大规模下建议 HP 走跨节点、EP 走节点内。这些是适用条件，不是通用承诺。
- Figure 3 的负载不均是 Zipf **模拟**分布，不是真实训练路由；HP 的"不随 `k`/倾斜变化"来自设计，端到端延迟仍受其它因素影响。
- Head 配置消融只在 3 个 `(N_h, d_h)` 组合、单一规模下做；`8×128` 的"最优"是工程折中，`4×256` 的 val loss 最低但成本与 SRAM 压力更高。
- 论文不涉及量化；本页量化内容全为结构推断，不能证明任何具体 checkpoint 或推理后端已支持该结构。
- 本库只做论文与来源页的静态阅读，**未运行任何训练或推理、未做数值验证**。

## Agent 如何使用

首轮读本页即可判断目标 MoE 是否把 token 切成多 head 子 token、是否用 Head Parallel、通信在路由前还是路由后。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 结构式、Head Parallel 契约、IO-aware 算子与全部实验 | [`paper-multihead-latentmoe`](../../sources/papers/paper-multihead-latentmoe.md) |
| MoE 路由、专家布局与负载均衡的通用机制 | [Sparse MoE](moe.md) |
| 术语分流与另一条 LatentMoE 工作 | [LatentMoE 术语与变体](latent-moe.md) |
| 目标模型源码、导出器与推理运行时的独立验证 | 目标仓库与目标后端文档（本库不提供） |

回答量化问题时必须说明：`N_h` 与 `d_h` 的实际取值、子 MoE 是否共享参数、通信阶段与并行切分、head 间是否共享 scale，以及目标后端。

## 来源与证据

- [Multi-Head LatentMoE and Head Parallel](../../sources/papers/paper-multihead-latentmoe.md)：结构式（Eq.5–6）、Head Parallel 通信契约、IO-aware routing/expert、Table 1 主结果与 head 配置消融、§6 适用区间。

本页结构以上述来源为限；论文不涉及量化，量化内容均为结构推断。
