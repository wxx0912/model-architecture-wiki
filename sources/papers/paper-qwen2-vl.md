---
{
  "id": "paper-qwen2-vl",
  "title": "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution",
  "category": "paper",
  "url": "https://arxiv.org/abs/2409.12191",
  "version": "v2（2024-10-03）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1（Figure 1–2、Table 1）、§2.1（Figure 3）、§2.2、§2.3、§3 引言（Table 2）、§3.1、§3.2（Table 3–6）、§3.3.1（Table 7、Figure 4）、§3.3.2（Table 8、Figure 5）、§3.3.3（Figure 6）、§4、附录 A（Figure 7–31）",
  "topic": "architectures",
  "excerpt_sha256": null
}
---
## 这份来源是什么

Qwen2-VL 的官方技术报告（阿里 Qwen 团队；arXiv 2409.12191，v2 2024-10-03）。给出三档开源 LVLM（2B/7B/72B）的结构与训练配方，核心是两个结构改动：**naive dynamic resolution**（ViT 去掉绝对位置编码、改用 2D-RoPE，任意分辨率→可变 visual token 数，并用一个 MLP 做 2×2 压缩）与 **M-RoPE**（把 rotary embedding 拆成 temporal/height/width 三分量）。

是本库 `models/qwen2-vl`、`paradigm-multimodal`、`component-multimodal-connector`、`component-position-encoding`（M-RoPE）的论文依据。与相邻来源的关系：沿用 Qwen-VL 的 `vision encoder → connector → LLM` 框架，LLM 用 Qwen2 系列；vision encoder 用 DFN 的 ViT 初始化，但把其固定位置编码换成 RoPE-2D。

范围边界：这是**结构 + 训练**报告，全文不含量化、低比特或数值精度误差分析，也没有激活/权重数值分布统计；能提供的是结构路径与 token 计数类事实。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源 `template.tex`（单文件；附录定性例子图在 `cases.tex`，由 `template.tex:858` 的 `\include{cases}` 插入），版本号取自 abs 页 Submission history（v1 2024-09-18、v2 2024-10-03），本地源码即 v2；另抓 arXiv HTML 渲染版仅用于交叉核对编号。
- **编号口径**：按源码浮动体顺序独立数得 **Figure 1–31、Table 1–8**。正文 6 张图是 Figure 1–6；附录 25 张是 Figure 7–31（`template.tex` 9 张 + `cases.tex` 16 张，编号 Figure 8–23）。两处易错点：`template.tex:991–1037` 的 Function Calling 浮动体**没有 `\caption`**、不占图号；只数 `template.tex` 里带 `\caption` 的浮动体会漏掉三处 `\captionof{figure}` 产生的 Figure 29–31。arXiv HTML 渲染版编号与之一致，已交叉验证。

## 关键事实

- **整体结构**（§2 开篇与 §2.1；Table 1）：沿用 Qwen-VL 的 vision encoder → connector → LLM 范式，LLM 用 Qwen2 系列；三档规模**共用同一个约 675M 的 ViT**（2B = 675M ViT + 1.5B LLM、7B = 675M + 7.6B、72B = 675M + 72B），论文明确说因此 ViT 的算力负载与 LLM 规模无关。
- **Naive Dynamic Resolution**（§2.1）：思路来自 NaViT，直接删掉 ViT 原来的 absolute position embedding、改用 **2D-RoPE**，从而支持任意分辨率→可变数量的 visual token；推理阶段把不同分辨率的图像 pack 进同一条序列，用 pack 后的长度控制显存。
- **2×2 token 压缩**（§2.1）：ViT 之后接一个简单 MLP，把相邻 `2×2` 个 token 压缩成 1 个，压缩后的序列首尾加 `<|vision_start|>`/`<|vision_end|>`。论文给的例子：**224×224 图像、ViT patch_size=14 → 压缩后 66 个 token** 进入 LLM。
- **M-RoPE**（§2.1 Figure 3）：把 rotary embedding 拆成 temporal / height / width 三分量。文本：三者用相同 position ID，等价 1D-RoPE；图像：temporal ID 恒定，height/width 按 token 位置赋不同 ID；视频：每帧 temporal ID 递增；多模态混排时每个模态从前一模态的最大 position ID +1 开始。论文称它既改善位置建模，又**降低 image/video 的 position ID 数值**，便于推理期外推。
- **图像与视频统一**（§2.1）：图像与视频混合训练；视频按 **2 fps** 采样；引入 **depth=2 的 3D 卷积**（I3D 思路）处理 3D tube 而非 2D patch，从而在不增加序列长度的情况下处理更多帧；**每张图像当作两帧相同帧**；动态调整每帧分辨率，**每段视频总 token 上限 16384**。
- **三阶段训练**（§2.2）：① 只训 ViT（大量图文对）；② 解冻全部参数、用更广数据训练；③ 冻结 ViT、只用指令数据微调 LLM。
- **数据与规模**（§2.2）：数据含图文对、OCR、交错图文文章、VQA、视频对话、图像知识（清洗网页 + 开源数据集 + 合成数据），**知识截止 2023 年 6 月**。第一阶段约 **600B token**，第二阶段再加 **800B token** 图像相关数据，**累计 1.4T token**，但**只对 text token 提供监督**。
- **数据格式**（§2.2）：`<|vision_start|>`/`<|vision_end|>` 界定图像；对话用 ChatML；grounding 的 bbox 坐标归一化到 [0,1000) 并用 `<|box_start|>`/`<|object_ref_start|>` 等标注。
- **基础设施**（§2.3）：阿里云 PAI-Lingjun，3D 并行（DP+TP+PP）+ DeepSpeed ZeRO-1 + 序列并行（带选择性激活重算）；TP 时 vision encoder 与 LLM 一起切、vision merger 不切；72B 用 1F1B PP。为消除卷积算子非确定性导致的 TP 下共享权重不一致，用离线归约解决并省掉一次 all-reduce。
- **主结果**（§3.1 Table 2）：论文称 72B 在多数指标上达到或超过 GPT-4o / Claude 3.5 Sonnet，文档理解明显领先（DocVQA 96.5、InfoVQA 84.5、TextVQA 85.5、OCRBench 877、MTVQA 30.9）；正文**明确承认在 MMMU（64.5 vs GPT-4o 69.1）落后**，而 MathVision（25.9 vs 30.4）的差距只体现在表格（§3.2.4 的表述是 "set a new open-source benchmark on MathVision with 25.9"）。
- **多语言 OCR**（§3.2 Table 3）：在内部多语言 OCR 基准上，除 **Arabic（70.7 vs GPT-4o 75.9）** 外，其余 7 种语言（Korean/Japanese/French/German/Italian/Russian/Vietnamese）都超过 GPT-4o。
- **视频**（§3.2 Table 4）：MVBench 73.6、PerceptionTest 68.0、EgoSchema 77.9 均为表中最好；Video-MME（w/o subs）71.2 低于 Gemini 1.5-Pro 的 75.0，论文说明评测时**每个视频最多抽 768 帧**可能压低长视频成绩。
- **Visual Agent**（§3.2 Table 5）：FnCall 的 TM 93.1 / EM 53.2 均高于 GPT-4o；但 **R2R（51.7 vs 专用模型 79.0）与 REVERIE（31.0 vs 61.0）显著落后专用 VLN 模型**，论文归因于模型从多张图生成的地图信息不完整、无结构。
- **动态分辨率消融**（§3.3.1 Table 7、Figure 4）：固定 token 数（64/576/1600/3136）下没有任何一个固定分辨率在所有基准上最优；动态分辨率平均只用 **1924 token** 就达到或接近最好（InfoVQA 75.89、RealWorldQA 70.07、OCRBench 866、MMMU 53.44）。把过小的图放大到 min_pixels 阈值会提升若干基准，但 **OCRBench 在 min_pixels 过大时严重下降**（论文推测该基准含大量极小图、过度放大后成为 out-of-distribution 样本）。
- **M-RoPE 消融**（§3.3.2 Table 8、Figure 5）：用 Qwen2-1.5B + ViT-L 的预训练模型对比，M-RoPE 相对 1D-RoPE 在视频基准更好（PerceptionTest 47.4 vs 46.6、NextQA 46.0 vs 43.9、STAR 57.9 vs 55.5），多数图像基准略好，但 **MMStar 持平、RealWorldQA 反而 53.7 < 54.5、InfoVQA 50.3 < 50.8**。长度外推：训练每视频最多 16K token，推理到 **80K** 仍稳定（仅 Qwen2-VL-72B 在 Video-MME Medium 上验证）。
- **模型 scaling**（§3.3.3 Figure 6）：性能随规模单调提升，数学能力与参数量正相关最明显，OCR 类小模型也不弱；7B 第二阶段性能随 token 增加而提升，但 **VQA 类任务有波动**。

## 不能推出的结论

- 论文不含量化、低比特或数值精度误差分析，也没有激活/权重数值分布统计；不能据此推出哪些张量对量化敏感、2×2 token 压缩或 M-RoPE 对量化误差有何影响，本页只提供结构路径（ViT → 2×2 MLP 压缩 → connector → LLM）。
- 三档模型共用 675M ViT 是**结构事实**，不等于"三档的视觉表征相同"；它们只是 LLM 规模不同，不能互推精度或量化难度。训练配方（600B + 800B = 1.4T token、只监督 text token、知识截止 2023-06）与 §2.3 的并行/存储设置都是论文自述，不能推断某个 checkpoint 的实际训练量或部署配置。
- "224×224、patch=14 → 66 token" 是论文给的单个示例，不是通用公式；任意分辨率下的 token 数取决于 min_pixels/max_pixels 等预处理设定，必须按目标实现核验。
- M-RoPE 的三分量分解是论文的建模选择，"降低 position ID 数值从而便于外推"是论文的解释；16K 训练→80K 推理的长度外推只在一个模型（72B）与 Video-MME Medium 上验证，不能当作任意长度、任意模态的外推保证。
- §3 的对比数字取自论文自报口径（自建内部多语言 OCR 基准、自建 agent 数据集、每视频最多 768 帧、Video-MME 的 w/o subs 列等），"最高分"多为相对当时 SoTA；不能跨来源混用，也不构成独立复现。
- "72B 超过 GPT-4o / Claude 3.5" 是论文的整体表述：正文**明确点名**的例外是 MMMU、Arabic；VLN 任务（R2R/REVERIE）正文称与 GPT-4o 相当、都明显落后专用 VLN 模型，Video-MME 与 MathVision 的差距只在表格里，表内还有正文未点名的落后项（MMMU-Pro 46.2 vs 51.9、ChartQA 88.3 vs 90.8）。引用时不能笼统写成全面领先。
- 本页只是论文级证据，不能证明某个具体 checkpoint 的预处理（min_pixels/max_pixels、每视频帧数上限）、ViT 是否被冻结、token 压缩是否启用；这些必须回到目标实现核验。
