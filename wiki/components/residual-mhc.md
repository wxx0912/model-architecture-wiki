---
{
  "id": "component-residual-mhc",
  "title": "Residual, Hyper-Connections and mHC",
  "type": "component",
  "tags": [
    "residual",
    "mhc",
    "quantization"
  ],
  "related": [
    "paradigm-ar",
    "paradigm-ssm",
    "component-attention-residuals"
  ],
  "sources": [
    "paper-mhc",
    "repo-tile-kernels"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "manifold-constrained hyper-connections",
    "hyper-connections"
  ],
  "updated": "2026-10-24"
}
---

# Residual、Hyper-Connections 与 mHC

## 一句话理解

普通 residual 是单条 stream 的加法。Hyper-Connections（HC）把 residual stream 拓宽成 `n` 条并行流、用 `H^res/H^pre/H^post` 混合它们，但复合转移矩阵不受约束、会破坏 identity mapping；mHC 把 residual 混合矩阵 `H^res` 约束到**双随机矩阵流形**（Birkhoff 多面体），恢复稳定性。

## 结构与数据流

HC 的形式（§1，Eq.3）：

```text
x_{l+1} = H^res_l · x_l + H^{post,T}_l · F(H^pre_l · x_l, W_l)
x_l ∈ R^{n×C}（n 条并行 stream），H^res_l ∈ R^{n×n}，H^pre_l, H^post_l ∈ R^{1×n}
```

`H^res` 在流内混合特征，`H^pre/H^post` 负责读出 / 写回；三者都由"输入相关动态项 + 静态偏置"组成（§3，Eq.5，动态项经 `tanh` 与初始化为小值的门控标量 `α`）。论文典型扩张率 `n = 4`。

HC 的失效机制（§1，Eq.4）：递归展开后前向信号由复合映射 `∏ H^res` 支配，而该乘积不保持全局均值 → 信号无界放大或衰减。§3.1 的实测（Figure 2–3）显示 27B 模型上约 12k step 出现 loss 突增，Amax Gain Magnitude（复合映射行/列和绝对值最大值）峰值达到 **3000**。

mHC 的核心约束（§4.1，Eq.6）：

```text
H^res ∈ { H ∈ R^{n×n} : H·1 = 1, 1ᵀ·H = 1ᵀ, H ≥ 0 }   # 双随机 / Birkhoff 多面体
```

`n = 1` 时双随机退化为标量 1，回到标准 identity mapping。三条性质：谱范数 `‖H^res‖₂ ≤ 1`（非扩张）、双随机对乘法封闭（任意深度复合仍是双随机）、是置换矩阵的凸包；`H^pre/H^post` 另加非负约束。参数化（§4.2，Eq.7–9）：`vec(x_l) ∈ R^{1×nC}` → 动态投影 `φ^pre, φ^post ∈ R^{nC×n}`、`φ^res ∈ R^{nC×n²}`，再加静态偏置与门控 `α`，最后 `H^res = Sinkhorn-Knopp(H̃^res)`（从 `exp(H̃^res)` 起交替行列归一化，实验取 `t_max = 20`）。

## 变体对照

| 变体 | residual 结构 | 混合约束 | 依据 |
| --- | --- | --- | --- |
| 标准 residual | `x + F(x)`（单 stream） | 无 | paper-mhc §1，Eq.1 |
| HC | `n` 条 stream + `H^res/H^pre/H^post` | 无 | paper-mhc §1，Eq.3 |
| mHC | 同上 | `H^res` 双随机，`H^pre/H^post` 非负 | paper-mhc §4.1，Eq.6 |
| AttnRes（对照） | 跨层 softmax 聚合历史层 | `Σα = 1` | 见 [Attention Residuals](./attention-residuals.md) |

## 实现差异

不同实现之间可能不同的地方：stream 数 `n`、`H^res` 的参数化与归一化方式（是否真的用 Sinkhorn、迭代次数 `t_max`）、`H^pre/H^post` 的非负参数化、混合精度策略，以及融合 kernel 与重算边界。`n` 是设计旋钮，`n=1` 才退化标准 residual，不能由 `n=4` 的结果反推其它 `n`。

## 量化关联

**结构事实**

- 多流 residual 引入 `H^res`（`n×n`）、`H^pre/H^post` 三类系数张量及其激活、Sinkhorn 迭代与反向重算路径。
- mHC 的精度策略（§4.3.1，Eq.10–19）：`φ` 用 **tfloat32**、`vec(x_l)` 用 **bfloat16**，偏置 / 门控标量 / 系数用 **float32**。
- `repo-tile-kernels` 在 `tile_kernels/mhc/` 提供 9 个 kernel 模块（含 `mhc_sinkhorn_fwd/bwd`，`repeat`/`eps` 为显式参数），分 NVIDIA 与 Ascend 两套后端——属官方 kernel 实现。

**量化推断**

- 应单独检查 mixing 权重、归一化统计、非负性与行列和约束，这不是普通 Linear 的完整替代。
- 低比特误差是否跨层放大需要运行验证，不能由"约束稳定"直接推导。

## 结果边界

- 论文不含量化或低比特误差实验，只规定了混合精度策略；不能据此推出 mHC 下"哪些张量该量化、误差如何传播"。
- 三条理论性质只对**精确**双随机矩阵成立；实现用 20 次 Sinkhorn 迭代得到近似解，单层反向增益已偏离 1、复合后最大偏离约 1.6，不能把"双随机"当成实现里严格满足的等式。
- `n = 4` 时额外时间开销 6.7% 是 DeepSeek 内部大规模训练（DualPipe + TileLang kernel）的测量值，依赖具体实现与硬件。
- mHC 并非全面超过 HC：27B 八项基准里七项更好，**MATH 一项是 HC 更好（26.4 vs 26.0）**；引用时不能写成"全面超过"。
- `repo-tile-kernels` 只证明官方有 kernel 实现，不等于某个 checkpoint 默认启用 mHC；`n`、`t_max` 等须回目标实现核验。
- 本库未运行任何量化或评测，本页量化内容均为结构推断。

## Agent 如何使用

首轮读本页即可判断目标是标准 residual、HC 还是 mHC，stream 数、混合矩阵是否受双随机约束。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| mHC 的约束、参数化、Sinkhorn 与主实验 | [`paper-mhc`](../../sources/papers/paper-mhc.md) |
| 官方 mHC kernel 算子与导出签名 | [`repo-tile-kernels`](../../sources/repos/repo-tile-kernels.md) |
| 另一种跨层残差方案（softmax 聚合） | [Attention Residuals](./attention-residuals.md) |
| 目标 checkpoint 是否启用 mHC、`n` 与 `t_max` | 目标模型源码 / 配置（本库不提供） |

回答量化问题时必须说明：是标准 residual / HC / mHC、`n` 取值、混合矩阵是否受约束、Sinkhorn 迭代次数与实现精度，以及目标后端与 kernel 支持。

## 来源与证据

- [mHC](../../sources/papers/paper-mhc.md)：HC 形式与失效机制（Eq.3–4）、mHC 双随机约束（Eq.6）、Sinkhorn 与精度策略（§4.2–4.3）、27B 主结果。
- [deepseek-ai/TileKernels](../../sources/repos/repo-tile-kernels.md)：固定 commit 下 `tile_kernels/mhc/` 的算子清单与签名。

本页结构以上述来源为限；来源不含量化实验，量化内容均为结构推断。
