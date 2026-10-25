---
{
  "id": "model-qwen2-vl",
  "title": "Qwen2-VL",
  "type": "model",
  "tags": [
    "multimodal",
    "attention",
    "rope",
    "embedding",
    "quantization"
  ],
  "related": [
    "paradigm-multimodal",
    "paradigm-ar",
    "component-multimodal-connector",
    "component-position-encoding",
    "component-embedding-lm-head"
  ],
  "sources": [
    "paper-qwen2-vl",
    "repo-qwen2-vl",
    "repo-huggingface-transformers"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-24"
}
---

# Qwen2-VL

## 一句话理解

Qwen2-VL 是视觉语言模型（LVLM），沿用 `vision encoder → connector → LLM` 范式，LLM 用 Qwen2 系列。两个结构改动：**naive dynamic resolution**（ViT 去掉绝对位置编码、改用 2D-RoPE，支持任意分辨率→可变 visual token 数）与 **M-RoPE**（rotary 拆成 temporal/height/width 三分量）。

## 结构规格表

| 项 | Qwen2-VL |
| --- | --- |
| 整体 | `vision encoder → connector → LLM`；LLM 用 Qwen2 系列 |
| 规模 | 2B = 675M ViT + 1.5B LLM；7B = 675M + 7.6B；72B = 675M + 72B（**三档共用同一个约 675M 的 ViT**） |
| 视觉编码 | ViT（DFN 初始化）删掉 absolute position embedding，改 **2D-RoPE**，支持任意分辨率 |
| token 压缩 | ViT 后接 MLP 把相邻 `2×2` 个 token 压成 1 个；论文 §2.1 只称其为 “a simple MLP layer”，§2.3 的并行设置里把它叫作 vision merger，本库据此归入 connector（工程推断）；`patch_size`（论文示例为 14） |
| 位置编码 | **M-RoPE**：temporal / height / width 三分量 |
| 视频 | 2 fps 采样；depth=2 的 3D 卷积；每张图当作两帧；训练期每段视频总 token 上限 16384 |
| 训练 | 三阶段；累计 1.4T token（600B + 800B），只监督 text token；知识截止 2023-06 |
| 外推 | 训练每视频 ≤16K token，推理到 80K 仍稳定（仅 72B 在 Video-MME Medium 验证） |

## 结构与数据流

图像由 processor 按 `min_pixels`/`max_pixels` 约束切成可变数量的 patch，过 ViT（2D-RoPE，无绝对位置）→ 2×2 MLP 压缩 → 首尾加 `<|vision_start|>`/`<|vision_end|>` → 进入 LLM。论文给的例子：**224×224、`patch_size=14` → 压缩后 66 个 token**。推理阶段把不同分辨率图像 pack 进同一序列，用 pack 后长度控制显存。

M-RoPE（§2.1 Figure 3）把 rotary 拆成 temporal / height / width 三分量：文本三者用相同 position ID（等价 1D-RoPE）；图像 temporal ID 恒定、height/width 按 token 位置赋不同 ID；视频每帧 temporal ID 递增；多模态混排时每个模态从前一模态最大 position ID +1 开始。论文称它既改善位置建模，又降低 image/video 的 position ID 数值、便于推理期外推。

图像与视频混合训练：视频按 2 fps 采样，用 depth=2 的 3D 卷积处理 3D tube（I3D 思路）而非 2D patch，从而在不增加序列长度的情况下处理更多帧；每张图像当作两帧相同帧，训练期每段视频总 token 上限 16384（推理可外推到 80K，仅 72B 在 Video-MME Medium 验证）。

实现落点：`repo-qwen2-vl` 登记的 URL 现在 **301 重定向到 Qwen3-VL**，且当前仓库不含 Qwen2-VL 建模代码，其预处理工具为 `qwen-vl-utils`（resize / `min_pixels` / `max_pixels`）；Qwen2-VL 的建模实现在 transformers 的 `models/qwen2_vl/`（transformers 侧独立实现，见 `repo-huggingface-transformers`）。

## 量化关联

**结构事实**

- visual token 数随分辨率与动态分辨率设定变化，不是固定序列长度；三档只差 LLM 规模、共用同一 ViT。
- 位置编码是三分量 M-RoPE；视觉侧含一个 2×2 压缩 MLP（论文 §2.1 称 “a simple MLP layer”，§2.3 的并行设置称 vision merger），本库把该压缩 MLP 与 connector 一并归入 connector（工程推断）。
- 范围事实：论文不含量化/低比特/数值误差分析；`repo-qwen2-vl` 的 README 记录上游发布了 Qwen2-VL-72B 的 AWQ / GPTQ-Int4 / GPTQ-Int8 权重产物（发布事实，不含算法证据）。

**量化推断（本库推断）**

- 视觉 encoder、connector 与语言主干应分别校准；动态分辨率使视觉 token 数与激活范围随输入变化。
- 图像、视频与纯文本路径要分开评估，并把 M-RoPE 的位置分配与 attention mask 一起检查。
- connector 输出量化可能造成视觉语义对齐偏移，应同时记录视觉局部输出、语言 logits 与端到端任务指标。

## 结果边界

- "224×224、`patch=14` → 66 token"是论文给的单个示例，不是通用公式；任意分辨率的 token 数取决于 `min_pixels`/`max_pixels` 等预处理设定，须按目标实现核验。
- M-RoPE"改善位置建模、降低 position ID 数值"是论文的解释；16K→80K 外推只在一个模型（72B）与一个基准（Video-MME Medium）验证，不能当作任意长度/模态的保证。
- 三档共用 675M ViT 是结构事实，不等于三档视觉表征相同，不能互推精度或量化难度。
- §3 的数字是论文自报口径（自建 OCR 基准、自建 agent 数据、每视频 ≤768 帧等）；"72B 超过 GPT-4o/Claude 3.5"有正文点名例外（MMMU、Arabic），VLN 任务（R2R/REVERIE）正文只称与 GPT-4o 相当，不能笼统写成全面领先。
- README 里的 AWQ/GPTQ 只说明上游发布了量化后权重；量化算法本身不属本篇证据（且 Qwen2.5-VL 的 AWQ 属下一代，不能与 Qwen2-VL 混用）。
- 论文与仓库都不含量化实验；本页只做论文/源码静态阅读，本库未运行任何量化或评测。

## Agent 如何使用

首轮读本页即可判断目标是 Qwen2-VL 这一代（共用 ViT、动态分辨率、M-RoPE），还是被 301 重定向到的 Qwen3-VL/Qwen2.5-VL（不能与 Qwen2-VL 混用）。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 整体结构、动态分辨率、2×2 压缩、M-RoPE 与训练配方 | [`paper-qwen2-vl`](../../sources/papers/paper-qwen2-vl.md) |
| 输入端预处理（resize、`min_pixels`/`max_pixels`）与仓库漂移 | [`repo-qwen2-vl`](../../sources/repos/repo-qwen2-vl.md) |
| Qwen2-VL 的建模文件与配置字段落点 | [`repo-huggingface-transformers`](../../sources/repos/repo-huggingface-transformers.md) |
| 多模态连接器与位置编码的通用机制 | [Multimodal Transformer](../paradigms/multimodal.md)、[RoPE 与位置编码](../components/position-encoding.md) |
| 目标 checkpoint 的预处理默认值、视觉 token 数与后端支持 | 该 checkpoint 的预处理配置与 processor 文件、`config.json`、`modeling_*.py` 与运行时（本库不提供） |

回答 Qwen2-VL 量化问题时必须说明：具体档位与 checkpoint、ViT 是否冻结、动态分辨率与 `min_pixels`/`max_pixels` 设定、是否启用 2×2 压缩、M-RoPE 的模态 ID 分配，以及目标后端 kernel 支持。

## 来源与证据

- [Qwen2-VL 技术报告](../../sources/papers/paper-qwen2-vl.md)：整体结构、naive dynamic resolution、2×2 压缩、M-RoPE、训练与评测口径（§2–§3）。
- [QwenLM/Qwen2-VL](../../sources/repos/repo-qwen2-vl.md)：已 301 重定向到 Qwen3-VL，不含 Qwen2-VL 建模代码；含 `qwen-vl-utils` 预处理工具与微调/评测代码。
- [huggingface/transformers](../../sources/repos/repo-huggingface-transformers.md)：`models/qwen2_vl/` 的建模文件与 `configuration_qwen2_vl.py` 配置字段落点。

本页结构以上述来源为限；论文不含量化实验，仓库只记录量化接口与权重发布、均未运行验证；量化内容除注明者外均为本库推断。
