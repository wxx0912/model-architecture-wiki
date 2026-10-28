---
{
  "id": "paper-qwen35-omni",
  "title": "Qwen3.5-Omni Technical Report",
  "category": "paper",
  "url": "https://arxiv.org/abs/2604.15804",
  "version": "v2（2026-04-21）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1、§2.1 概览、§2.2 AuT、§2.3 Perceivation（文本/音频/图像/视频与音视频时间戳）、§2.4 语音生成、§2.5 流式与并发（Table 1 模块表 + 首包延迟、Table 2 延迟表）、§3 预训练三阶段",
  "topic": "architectures",
  "excerpt_sha256": null
}
---
## 这份来源是什么

Qwen3.5-Omni 技术报告（arXiv 2604.15804 v2，Qwen Team）。报告覆盖架构、预训练、后训练与评测；本库只取结构相关部分：§2.1 概览、§2.2 音频编码器 AuT、§2.3 多模态输入编码与时间戳方案、§2.4 语音生成、§2.5 流式与并发（含 Table 1 模块表）、§3 预训练阶段划分。它是 `model-qwen35-omni` 的唯一结构依据，也为 `paradigm-multimodal`、`paradigm-moe`、`paradigm-linear-attention`、`component-multimodal-connector` 提供"Thinker–Talker 双模块 + 混合注意力 MoE"的现实实例。

范围边界：只覆盖 Qwen3.5-Omni（Plus 与 Flash 两个变体）；报告不含逐层配置表，也不含量化实验；§2.5 给的是内部服务上的理论首包延迟与配置口径，不是可复用的部署配方。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2604.15804v2`），未读 LaTeX 源与 PDF。
- 定位：§2.1 的模块分工、§2.2 的 AuT 结构与训练数据、§2.3 的四种输入编码与位置编号规则、§2.4 的 Talker 与 ARIA、§2.5 的 Table 1（模块与首包延迟）与 Table 2（不同并发下的理论首包延迟）、§3 的三阶段预训练；§4 之后的评测数字未取。

## 关键事实

- **整体形态**（§2.1）：延续 Qwen2.5-Omni 引入的 **Thinker–Talker 架构**。Thinker 负责文本生成，Talker 接收 Thinker 的高层表示并自回归生成流式语音 token；每个解码步由 MTP 模块输出当前帧的残差码本，再由 Code2Wav 增量合成波形。
- **模块构成**（Table 1）：Audio Encoder = **AuT**；Vision Encoder = **SigLIP2**；Thinker = **Hybrid MoE Transformer**；Talker = **Hybrid MoE Transformer**；MTP = Dense Transformer；Code2wav = ConvNet。
- **骨干**（§2.5）：Thinker 与 Talker 都建立在 Qwen3.5 引入的 **Hybrid-Attention MoE** 架构上，其中包含 **Gated Delta Net（GDN）** 模块，用于加速长音视频序列建模、降低长上下文推理的 KV-cache I/O。
- **AuT**（§2.2）：transformer 结构的 attention-encoder-decoder 音频编码器，**从零训练**，训练数据为 Qwen3-ASR 生成的 4,000 万小时 audio-text 对；filter bank 特征经 4 个 Conv2D 块下采样 16 倍后进入 self-attention，输出音频 token 速率为 **6.25 Hz**（每个输出帧约对应 160 ms 原始信号）；多语言覆盖 20+ 语言，中英多语比例为 3.5 : 3.5 : 3；训练采用动态注意力窗口大小机制。
- **文本输入**（§2.3）：使用 Qwen3.5 tokenizer，byte-level BPE，词表 **250k**（相对此前的 150k 提升），多数语言编解码效率提升 10–60%。
- **音频输入**（§2.3）：波形重采样到 16 kHz，转成 **128 通道 mel 频谱**，窗长 25 ms、hop 10 ms；由 AuT 编码，音频部分每 160 ms 一个 temporal ID。
- **视觉输入**（§2.3）：图像与视频共用视觉编码器；视频按**动态帧率**采样。Table 1 给的架构名是 **SigLIP2**，§2.3 与 §3 正文则称"采用 Qwen3.5 的视觉编码器"——两处**称法不同**，报告没有说明二者的关系。
- **时间戳与位置编号**（§2.3）：沿用 TM-RoPE，但不再用 temporal position ID 直接编码绝对时间，而是给每个视频/音视频 temporal patch **前置一个显式时间戳**（以秒为单位的格式化文本字符串），音频序列另按随机间隔插入时间戳；视频帧的时间 ID 按实际时间戳动态调整到 160 ms 分辨率；跨模态位置编号连续——每个后续模态从上一个模态最大位置 ID 加 1 开始。
- **Talker 与 ARIA**（§2.4）：Talker 直接作用于 `Qwen3.5-Omni-Audio-Tokenizer` 产出的 RVQ token，用 MTP 模块建模残差码本，最后由 causal ConvNet 重建波形；为 Talker 引入**专用 system prompt** 描述目标音色，从而支持零样本声音克隆与可控语音生成。**ARIA**（Adaptive Rate Interleave Alignment）把 Qwen3-Omni 的双通道生成改为**单通道交错**形式，约束条件是"生成序列任意前缀的累计 speech-to-text token 比例不超过对应的 item 级全局比例"。
- **长度与量表**（§1）：支持 **256k token** 上下文、**10 小时以上音频**、**400 秒 720P（1 FPS）** 视频；模型规模为"数百 B 参数"量级（未给具体数）。
- **预训练**（§3）：三阶段——第一阶段冻结 LLM 参数，只训练视觉与音频编码器（先训 adapter）；第二阶段解冻全部参数，用 32,768 序列长度训练；第三阶段用 262,144 序列长度数据。LLM 由 Qwen3.5 初始化，视觉编码器取自 Qwen3.5，音频编码器由 AuT 初始化。
- **首包延迟**（Table 1、Table 2）：Table 1 给单点值——Qwen3.5-Omni-Plus 为 435 ms（音频输入）/ 651 ms（视频输入），Flash 为 235 ms / 426 ms。Table 2 是**理论**首包延迟，在 1/4/8 并发下给出 Thinker TTFT、Talker TTFC、Thinker/Talker TPOP、TPS 与 Overall Latency；评测环境为内部 vLLM，并对 MTP 模块与 codec decoder 启用 `torch.compile` 与 CUDA Graph。

## 不能推出的结论

- 报告**不含逐层配置表**：Thinker / Talker 的层数、hidden、head 数、专家数、MoE 中间维，以及 GDN 层与 attention 层的配比，论文都没有给出。
- 截至本页核对时，经 `hf-mirror.com` 查询本次未找到 Qwen3.5-Omni 的官方权重仓（`Qwen/Qwen3.5-Omni-Plus`、`-Flash` 均查不到）——这只是本次镜像查询未命中，不代表官方未发布；因此**没有 `config.json` 可以佐证**上述任何结构量；本页不登记 repo 来源。
- GDN 的状态更新公式、Hybrid-Attention MoE 中线性注意力与全注意力的交错方式，本文均未给出，只指向 Qwen3.5（Team, 2026），不能据本文推断。
- 报告不含量化实验：位宽、校准、导出格式与精度结论均未披露，不能推断量化后表现。
- Table 2 明确标注为 **theoretical** first-packet latency，且依赖内部 vLLM 与 `torch.compile` / CUDA Graph；Plus 与 Flash 的部署资源与并行策略不同，论文自己声明两者延迟与吞吐**不能严格横向比较**。Table 1 只写 end-to-end first-packet latency，但其数值与 Table 2 在 1 并发下的 Overall Latency 完全相同，本库因此按同一理论口径对待。
- Table 1 与 §2.3 / §3 对视觉编码器的称法不同（架构名 SigLIP2 vs "来自 Qwen3.5"），报告未说明二者是否为同一编码器，须读 Qwen3.5 报告确认。
- "数百 B 参数"是量级描述，不是具体参数量。
- 本页只读 HTML 渲染版，未核对 LaTeX 源；评测数字未逐表复核。
