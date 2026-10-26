---
{
  "id": "paradigm-multimodal",
  "title": "Multimodal Transformer and Omni Models",
  "type": "paradigm",
  "tags": [
    "multimodal",
    "ar",
    "diffusion",
    "quantization"
  ],
  "related": [
    "paradigm-ar",
    "paradigm-dit",
    "component-embedding-lm-head",
    "component-multimodal-connector",
    "check-quantization-preflight"
  ],
  "sources": [
    "paper-qwen2-vl",
    "repo-qwen2-vl",
    "repo-huggingface-transformers"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "vlm",
    "vision language model",
    "多模态"
  ],
  "updated": "2026-10-26"
}
---

# 多模态与 Omni 模型

## 一句话理解

多模态模型把非文本模态编码成 token 或 latent，再经 connector 接进语言或扩散主干一起计算。它不是一个单一结构：模态数量、token 化方式、时间轴建模和输出头都可以完全不同，所以 "Omni" 这类名字本身不构成结构描述，必须绑定到具体模型的配置与代码。

## 模态接入路径与数据流

通用路径（跨模型归纳，见「结果边界」的说明）是：

```text
raw modality → encoder → resampler / projector（connector） → modality tokens
             → 与文本 token 混排 → 主干 → 输出头
```

以 Qwen2-VL 为具体案例（本页唯一的结构样本）：

- **分辨率**：ViT（`patch_size=14`）抽出 patch，**去掉绝对位置编码、改用 2D-RoPE**，因而支持任意分辨率和可变数量的 visual token；推理时把不同分辨率的图像 pack 进同一条序列，用 pack 后的长度控制显存。
- **token 压缩**：ViT 之后接一个 MLP，把相邻 `2×2` 个 token 压成 1 个，再在首尾加 `<|vision_start|>` / `<|vision_end|>`。论文给的例子是 224×224 图像压缩后变成 66 个 token（224/14=16 → 16×16=256 个 patch，`2×2` 压缩后 256/4=64，再加首尾两个特殊 token 得 66）。
- **位置编码 M-RoPE**：rotary 拆成 temporal / height / width 三个分量。文本三者用同一 position ID（等价 1D-RoPE）；图像 temporal 固定、height/width 按 token 位置赋不同 ID；视频每帧 temporal 递增；多模态混排时每个模态从前一模态的最大 position ID +1 开始。三分量如何切分 head_dim 与频率，论文未披露。
- **图像与视频统一**：视频按 **2 fps** 采样，用 depth=2 的 3D 卷积处理 3D tube（I3D 思路），每张图像当作两帧相同帧，动态调整每帧分辨率，训练期每段视频总 token 上限 16384。

三档规模（2B / 7B / 72B）共用同一个约 675M 的 ViT，因此 ViT 的算力负载与 LLM 规模无关。

## 实现差异

不同实现之间可能不同的地方（工程归纳，非论文结论）：图像分辨率与 patch size、视频帧采样率与是否用 3D 卷积、connector 类型（MLP、重采样、cross-attention）、token 压缩方式、位置编码的拆分方式、特殊 token 与 processor、是否共享主干、以及 encoder 与主干是否同精度、同后端。**模型名相同也可能配不同 processor 与 checkpoint 变体**，需要按实际加载的配置确认。

## 量化关联

**结构事实**

- 模态 encoder、connector 与语言主干是分开的模块，输入分布不同；视觉 token 的数量随分辨率变化。

**实现事实**（来自固定版本源码，须按该 revision 核对）

- Qwen2-VL 的输入端预处理（resize、`min_pixels`/`max_pixels` 一类像素约束）在 `qwen-vl-utils` 里，建模代码在 transformers 的 `models/qwen2_vl/` 下——两者是不同来源。

**量化推断**

- 模态 encoder、connector 与主干的数据分布不同，统一校准集或统一 bit 容易造成某一模态退化，应分开评估。
- connector 的小误差会改变主干看到的输入语义；模态对齐误差要单独看。
- 视频/音频存在时间轴与长序列，校准要覆盖帧率、时长与缺失模态等边界。
- 需要分别报告模态理解质量、生成质量、首 token / 解码延迟和真实后端支持情况。

## 结果边界

- 本页的结构描述以 Qwen2-VL 为样本；"Omni" 不是单一结构，其它多模态模型的模态数与融合方式必须绑定各自的官方配置与代码。
- Qwen2-VL 的仓库 URL 现在 301 重定向到 Qwen3-VL，且**不含 Qwen2-VL 建模文件**；输入端预处理在 `qwen-vl-utils`，实现级事实要看 transformers 的 `models/qwen2_vl/`。
- M-RoPE 不是全面优于 1D-RoPE：论文消融显示它在视频基准更好，但 MMStar 持平、RealWorldQA 与 InfoVQA 略低。
- 上文「通用路径」与「实现差异」中列出的知识是跨模型的工程归纳，不是任何一份来源的论文结论；Qwen2-VL 的专有项均已单独标注。
- 三份来源都不含量化实验；`repo-qwen2-vl` 的 README 只记录上游发布过量化权重产物（Qwen2.5-VL 的 AWQ，以及 Qwen2-VL-72B 的 AWQ / GPTQ-Int4 / GPTQ-Int8），不含算法或误差证据。本页量化内容全为结构推断。

## Agent 如何使用

首轮读本页即可判断目标模型有没有独立 encoder 与 connector、视觉 token 怎么产生、位置编码怎么拆。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 分辨率处理、2×2 压缩、M-RoPE、视频采样与训练阶段 | [`paper-qwen2-vl`](../../sources/papers/paper-qwen2-vl.md) |
| 官方仓库现状与 `qwen-vl-utils` 的像素约束 | [`repo-qwen2-vl`](../../sources/repos/repo-qwen2-vl.md) |
| transformers 里 Qwen2-VL 的建模文件与 config 字段 | [`repo-huggingface-transformers`](../../sources/repos/repo-huggingface-transformers.md) |
| 多模态页面对应的组件细节 | [Token Embedding 与 LM Head](../components/embedding-lm-head.md)、[Multimodal Encoder 与 Connector](../components/multimodal-connector.md) |
| 目标 checkpoint 的预处理、视觉 token 数与后端支持 | 该 checkpoint 的 config / processor / 运行时（本库不提供） |

回答多模态量化问题时必须说明：量化的是 encoder、connector 还是主干（还是全部）、输入分辨率与视觉 token 数量、是否含视频/音频长序列，以及 processor 与 checkpoint 的具体版本。

## 来源与证据

- [Qwen2-VL](../../sources/papers/paper-qwen2-vl.md)：vision encoder → connector → LLM 结构、Naive Dynamic Resolution、2×2 token 压缩、M-RoPE 与视频处理。
- [QwenLM/Qwen2-VL](../../sources/repos/repo-qwen2-vl.md)：仓库重定向现状与 `qwen-vl-utils` 提供的输入端预处理。
- [huggingface/transformers](../../sources/repos/repo-huggingface-transformers.md)：固定 commit 下 `models/qwen2_vl/` 的建模与配置落点。

本页的结构描述以 Qwen2-VL 为准；量化结论均为结构推断。
