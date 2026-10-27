---
{
  "id": "paradigm-latent-moe-nvidia",
  "title": "NVIDIA LatentMoE",
  "type": "paradigm",
  "tags": [
    "latent-moe",
    "moe",
    "routing",
    "quantization"
  ],
  "related": [
    "paradigm-latent-moe",
    "paradigm-moe",
    "component-router-expert",
    "component-mlp"
  ],
  "sources": [
    "paper-latentmoe"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-26"
}
---

# NVIDIA LatentMoE

## 一句话理解

LatentMoE 先把 token 从原始 hidden 维 `d` 投影到低维 latent 空间 `ℓ`，再在 latent 空间执行 routed expert 计算，最后投影回 `d`。路由与 shared expert 仍在原始 `d` 维；省下的预算用于放大专家容量——`LMoE-eff` 只放大 `N`，推荐的 `LMoE-acc` 同时放大 `K`。

## 机制与数据流

设计出发点（§2）是两条 roofline 结论：低延迟服务受**权重加载带宽**支配，吞吐场景受 **all-to-all 通信**支配（本示例 `t_comm/t_comp ≈ 9`）；而有效非线性预算 `U_eff ∝ K·m`、`d` 有任务相关下界 `r_eff`，所以 `K` 与 `m` 都不能压。可压的只剩 `d`。把 `d` 压 `α` 倍、同时把 `N`、`K` 放大 `α` 倍，通信与带宽成本不变而表达力提高（论文标明这是**假设**，由实验验证）。

两个结构变体（§3）：

```text
LMoE-eff(x) = W_↑ · ( Σ_{i∈T_{K,N'}} p'_i E_i(W_↓ x; ℓ) ) + Σ_{j=N'+1}^{N'+S} E_j(x; d)   # Eq.1
LMoE-acc    与 eff 的唯一区别：top-k 取 K' = αK                                          # Eq.2
N' = αN,  α = d/ℓ,  p' = softmax(W'_r x),  W'_r ∈ R^{N'×d}
```

routed 专家在 latent 空间计算（`W_FC1, W_gate ∈ R^{m×ℓ}`、`W_FC2 ∈ R^{ℓ×m}`），**只压输入维、中间维 `m` 不变**（故 `U_eff` 不变）；shared 专家与路由仍在原始 `d` 维；dispatch/aggregate 也在 latent 空间，因此通信量与专家权重加载带宽都降 `α` 倍。

成本对照（§3，Table 1）：

| 变体 | 通信 | 每专家权重加载 | 效果 |
| --- | --- | --- | --- |
| 标准 MoE | `(N/EP)·t_exp·d` | `d·m` | 基线 |
| LMoE-eff | `(N/EP)·t_exp·ℓ` | `ℓ·m` | 效率↑、精度持平 |
| LMoE-acc（推荐） | `(N/EP)·t_exp·d` | `d·m` | 精度↑、效率持平 |

## 变体对照

| 配置 | `d` | `N` | `K` | `S` | `m` | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| 16BT-2BA | 2048 | 64 | 6 | 2 | 1408 | 27 层；SwiGLU；消融主载体 |
| 95BT-8BA | 4096 | 128 | 6 | 2 | 2688 | 32 层；Squared-ReLU |
| Hybrid-73BT-8BA | 4096 | 128 | 6 | 2 | 2688 | 52 层（原文括注 “24 Mamba/MoE, 4 Attn.”，按 24+24+4 读才自洽、原文未写清）；Mamba-2 块 128 heads、head dim 64、state dim 128、8 groups |

消融（16BT-2BA，§4.1）：压缩比扫描显示 `α ≤ 4` 时质量保持，后续统一取 `α = 4`；只压 `d` 4× 而不扩专家会明显掉点，把 `N` 扩 `α` 倍可补回。

## 实现差异

不同实现之间可能不同的地方：latent 维 `ℓ` 与压缩比 `α`、`N`/`K` 的放大方式、`m` 是否保持、shared 专家的数量与位置、router 所在空间，以及 dispatch/aggregate 是在 latent 还是原始维进行。`ℓ` 与 `α` 都是设计旋钮，不能从论文的 `α = 4` 反推其它实现的取值。

## 量化关联

**结构事实**

- routed 专家权重在 latent 空间：`W_FC1, W_gate ∈ R^{m×ℓ}`、`W_FC2 ∈ R^{ℓ×m}`；router 与 shared 专家在原始 `d` 维。
- 论文 §4.3 的实测推理用 vLLM + **FP8 per-tensor**（vLLM 的现成推理配置）。

**量化推断**

- latent 空间的专家权重通道统计与原始 `d` 维不同，`W_↓`（down 投影）/`W_FC1`/`W_FC2` 的 scale 与 group size 不应套用原维 dense FFN 的默认值。
- router 在原始 `d` 维工作，路由误差与 latent reconstruction error 要分开记录。
- 低维投影、专家通信与导出 packing 要先过目标后端约束；量化后需分层统计路由一致率、latent 误差、专家输出误差与端到端质量。

**未知项**

- 论文没有量化敏感性实验，FP8 推理配置不代表对该结构做过量化评估。

## 结果边界

- `α ≤ 4` 不损质量是 16BT-2BA 上的经验结论（论文称在 95B 也经验验证），不是理论上限或通用最优。
- 所有精度数字来自三个具体配置与特定 token 预算，且沿用 baseline 超参、未为 LatentMoE 调参；不能外推到其它 `α`、`ℓ`、`N`、`K` 组合或其它模型家族。
- FP8 per-tensor 是 vLLM 的现成配置，论文没有量化敏感性实验；**不能推断 LatentMoE 对量化更友好或不友好**。
- 本页只是论文级证据，不证明任何具体 checkpoint 的行为，也不能替代在目标实现上的运行验证。

## Agent 如何使用

首轮读本页即可判断目标专家块是否把专家输入压到 latent、路由在哪个空间、`N`/`K` 是否配平放大。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 设计原则、`LMoE-eff`/`LMoE-acc` 公式、成本对照与三档配置 | [`paper-latentmoe`](../../sources/papers/paper-latentmoe.md) |
| MoE 路由、负载均衡与专家布局的通用机制 | [Sparse MoE](moe.md) |
| 术语分流与另一条 LatentMoE 工作 | [LatentMoE 术语与变体](latent-moe.md) |
| 目标实现的 latent 维与路由契约 | 该实现 / 论文附带的配置与源码（本库不提供） |

回答 NVIDIA LatentMoE 量化问题时必须说明：`d`、`ℓ`、`α`、`N`、`K`、`m` 的实际取值，量化的是 latent 专家、router 还是 down/up 投影，以及目标后端与精度。

## 来源与证据

- [LatentMoE](../../sources/papers/paper-latentmoe.md)：五条设计原则、LMoE-eff/LMoE-acc 结构式、Table 1 成本对照、三档模型规格与消融、FP8 per-tensor 实测。

本页结构以上述论文为限；论文不含量化实验，量化内容均为结构推断，本库未做独立数值或运行验证。
