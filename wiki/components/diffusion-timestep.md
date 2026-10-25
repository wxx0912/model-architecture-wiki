---
{
  "id": "component-diffusion-timestep",
  "title": "Diffusion Time/Noise and Discrete Mask State",
  "type": "component",
  "tags": [
    "timestep",
    "diffusion",
    "dit",
    "dllm",
    "mask-state",
    "calibration"
  ],
  "related": [
    "paradigm-dit",
    "paradigm-dllm",
    "component-normalization",
    "check-calibration-axes"
  ],
  "sources": [
    "paper-dit",
    "paper-llada",
    "repo-dit"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-24"
}
---

# Diffusion Timestep 与离散 Mask 状态

## 一句话理解

扩散主干需要一个**时间/噪声水平条件**来知道"现在去噪到哪一步"，它被编码成向量、注入归一化 / attention / MLP / 输出头；DiT 用 adaLN 调制做这件事。dLLM 的 mask ratio、生成 step、block 状态是**离散迭代状态**，语义上与连续 timestep/noise 不同，不能互相等同。

## 结构与数据流

**DiT 的连续 timestep**（paper-dit 附录 A）：timestep 用 **256 维频率 embedding**，后接维度等于 transformer hidden size 的**两层 MLP**（激活 SiLU）；每个 adaLN 层把 timestep embedding 与 class embedding 之**和**送入 SiLU + 一个线性层，输出维度是 hidden size 的 **4×（adaLN）或 6×（adaLN-Zero）**。adaLN-Zero 在 adaLN 基础上再回归逐维 `α`，作用于 block 内每个残差连接之前，并把 `α` 的 MLP 初始化为恒输出 0，使整个 DiT block 初始为恒等映射。

官方实现（repo-dit）：条件 `c = t + y`（`t` 为 timestep embedding、`y` 为类别 embedding），`DiTBlock` 里 `adaLN_modulation = Sequential(SiLU, Linear(hidden, 6*hidden))`，输出 `chunk(6)`（attention 与 MLP 各一组 shift/scale/gate）；`final_layer` 也带一套 adaLN 调制。`learn_sigma=True` 时输出通道翻倍，同时预测噪声与方差。

**dLLM 的离散 mask 状态**（paper-llada）：前向过程按线性概率独立掩码 token；反向从全掩码定长序列出发，按 `N` 个均匀时间步从 `t=1` 走到 `0`、步长 `1/N`（`s = t − 1/N`）。

反向采样循环（§2.4；算法在附录 A 的 Algorithm 4/5，Algorithms 小节为 A.3）：

1. 当前时刻 `t` 的序列里所有被掩码位置同时预测（非因果 backbone，一次前向出全部位置的 logits）；
2. 对本轮新预测（原本被掩码）的位置按 `s/t` 的概率重新掩码，期望保留 `1−s/t`；
3. 默认用**低置信度 remasking**：置信度取 argmax 预测的概率 `c^i = p_θ(r_0^i | p_0, r_t)`，**保留置信度最高的 `n_un = ⌊L(1−s)⌋` 个位置、其余重新掩码**（`L` 为生成长度）；
4. 令 `t ← s` 进入下一轮，直到 `t = 0`。

## 条件与状态对照

| 类别 | 状态变量 | 取值 | 注入 / 使用方式 |
| --- | --- | --- | --- |
| DiT timestep | 连续噪声水平 | `t_max=1000` 线性方差表 `1e-4 → 2e-2`（沿用 ADM） | 256 维频率 embedding + 2 层 MLP，与类别 embedding 求和后进 adaLN |
| DiT 类别条件 | 类别标签 | 离散 | 类别 embedding，与 `t` 相加后共用同一 adaLN 函数 |
| dLLM mask 状态 | mask ratio / step / block | 离散（`N` 步、`s = t − 1/N`） | 逐位置 mask，非因果同时预测；按 `s/t` 概率重新掩码，或低置信度 remasking |

## 实现差异

不同实现之间可能不同的地方：timestep 频率 embedding 的维度与频带、编码 MLP 的层数与激活、调制输出是 `4×` 还是 `6×`、条件是否相加或拼接、adaLN-Zero 的零初始化范围，以及是否把条件当额外 token（in-context）或走 cross-attention。dLLM 侧则可能不同：时间步数 `N`、remasking 策略（随机 / 低置信度）、block 长度与调度。这些都是设计选择，不能从 DiT 的 adaLN-Zero 直接外推。

## 量化关联

**结构事实**

- DiT 把 timestep 与类别条件之和送进 adaLN MLP（SiLU + Linear），输出 4×/6× hidden；adaLN-Zero 把 `adaLN_modulation` 的**末层 Linear**（以及 `final_layer` 的调制末层与线性层）零初始化。
- dLLM 的 mask ratio / step 是**离散**调度变量，反向过程分 `N` 步、逐位置重新掩码。
- 范围事实：三份来源都不含量化、低比特或激活范围统计。

**量化推断**

- timestep/noise 或 mask/step 都可能改变激活分布，但属于不同架构语义，校准轴不可互换。
- 校准集应按目标模型分别记录连续去噪轴或离散 mask/迭代轴；单个 timestep 或单个 mask ratio 的局部误差都不能代表完整采样轨迹。

## 结果边界

- DiT 的 timestep 规格（256 维、2 层 MLP、`4×`/`6×`）与 `adaLN-Zero` 结论来自 class-conditional ImageNet 256×256 的论文与官方实现，不能直接外推到带文本条件或不同归一化的扩散 Transformer。
- dLLM 的 mask 状态是离散迭代量，"mask ratio ≈ 连续 noise level" 只是类比，不能当作等价关系；LLaDA 的采样参数（`N`、低置信度 remasking）是论文的实验设置。
- 论文与仓库都不涉及量化；本页"校准要覆盖采样轨迹"的结论属结构推断，需在目标实现上另行测量。
- 本库只做论文与来源页的静态阅读，未运行任何采样或做数值验证。

## Agent 如何使用

首轮读本页即可判断目标模型的条件是连续 timestep/noise 还是离散 mask/step、注入在哪里、调制输出维度是几倍 hidden。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| DiT 的 adaLN/adaLN-Zero、timestep embedding 规格与四种条件注入 | [`paper-dit`](../../sources/papers/paper-dit.md) |
| 官方实现里 `TimestepEmbedder`/`DiTBlock`/`FinalLayer` 的代码形态 | [`repo-dit`](../../sources/repos/repo-dit.md) |
| dLLM 的掩码前向/反向与 remasking 规则 | [`paper-llada`](../../sources/papers/paper-llada.md) |
| 校准轴应覆盖哪些采样轨迹 | [Calibration Axes](../checks/calibration-axes.md) |
| 目标模型实际用的条件注入与采样调度 | 目标模型源码 / 配置（本库不提供） |

回答量化问题时必须说明：条件是连续 timestep 还是离散 mask、采样步数/调度、注入位置与调制维度、以及量化的是权重还是条件向量本身。

## 来源与证据

- [DiT](../../sources/papers/paper-dit.md)：timestep embedding 与 adaLN/adaLN-Zero 规格（附录 A）、四种条件注入与 block 对比（§5）。
- [facebookresearch/DiT](../../sources/repos/repo-dit.md)：固定 commit 下 `models.py` 的 `TimestepEmbedder`、`DiTBlock`、`FinalLayer` 与初始化。
- [LLaDA](../../sources/papers/paper-llada.md)：连续/离散对应的 mask 前向过程、反向采样与低置信度 remasking。

本页结构以上述来源为限；三份来源都不含量化实验，量化内容均为结构推断。
