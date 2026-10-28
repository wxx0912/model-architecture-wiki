---
{
  "id": "model-qwen35-omni",
  "title": "Qwen3.5-Omni",
  "type": "model",
  "tags": [
    "multimodal",
    "ar",
    "moe",
    "gdn",
    "linear-attention",
    "state-update",
    "quantization"
  ],
  "aliases": [
    "Qwen3.5-Omni",
    "Qwen3.5-Omni-Plus",
    "Qwen3.5-Omni-Flash",
    "全模态 omni 模型"
  ],
  "related": [
    "paradigm-multimodal",
    "paradigm-moe",
    "paradigm-linear-attention",
    "component-multimodal-connector",
    "component-router-expert",
    "component-kv-cache",
    "model-qwen2-vl"
  ],
  "sources": [
    "paper-qwen35-omni"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "updated": "2026-10-28"
}
---

# Qwen3.5-Omni

## 一句话理解

Qwen3.5-Omni 是端到端全模态模型（文本 / 音频 / 图像 / 视频输入，文本 + 流式语音输出），沿用 Thinker–Talker 架构：Thinker 做多模态理解与文本生成，Talker 生成流式语音 token，两者骨干都是含 **Gated Delta Net（GDN）** 的 **Hybrid-Attention MoE**。语音侧走 RVQ 多码本 + MTP + causal ConvNet，支持 256k 上下文、10 小时音频与 400 秒 720P 视频。非纯文本 decoder-only，量化要拆成"多编码器 + 双 MoE 骨干 + 语音解码链"处理。

## 结构规格表

| 模块 | 架构 | 说明 |
| --- | --- | --- |
| Audio Encoder | AuT | attention-encoder-decoder，从零训练；输出 6.25 Hz 音频 token（每帧约 160 ms） |
| Vision Encoder | SigLIP2 | Table 1 给的架构名；§2.3 与 §3 正文称"采用 Qwen3.5 的视觉编码器"，两处称法不同 |
| Thinker | Hybrid MoE Transformer | 多模态理解与文本生成；含 GDN |
| Talker | Hybrid MoE Transformer | 接收 Thinker 的高层表示，生成流式语音 token；含 GDN |
| MTP | Dense Transformer | 每个解码步输出当前帧的残差码本 |
| Code2wav | ConvNet | causal 流式波形合成 |

论文只在 Table 1 给出这份模块表，**没有逐层配置表**；模型规模只写"数百 B 参数"量级，未给具体参数量。本表不能替代 checkpoint 的 `config.json`——本次经镜像查询未找到官方的 `Qwen/Qwen3.5-Omni-*` 权重仓，因此本页没有实现层来源。

**量表与接口**（§1、§2.3）：上下文 256k token；支持 10 小时以上音频、400 秒 720P 视频（1 FPS）；文本词表 250k。

## 结构与数据流

- **总装**（§2.1）：文本 / 音频 / 图像 / 静音视频进入 Thinker；Thinker 输出文本，并把高层表示交给 Talker；Talker 自回归产出 RVQ 多码本 token，MTP 补残差码本，Code2wav 逐帧合成波形。
- **音频通路**（§2.2、§2.3）：波形重采样 16 kHz → 128 通道 mel（窗 25 ms、hop 10 ms）→ AuT（filter bank 经 4 个 Conv2D 下采样 16 倍后进 self-attention）→ 6.25 Hz 音频 token；音频每 160 ms 对应一个 temporal ID。
- **视觉通路**（§2.3）：图像与视频共用视觉编码器；视频按动态帧率采样，帧的时间 ID 按实际时间戳动态调整到 160 ms 分辨率。
- **时间与位置**（§2.3）：沿用 TM-RoPE，但每个视频/音视频 temporal patch 前置一个以秒为单位的显式时间戳文本串，音频序列另按随机间隔插入时间戳；跨模态位置编号连续，后续模态从上一个模态最大位置 ID + 1 开始。
- **语音生成**（§2.4）：Talker 直接作用于 `Qwen3.5-Omni-Audio-Tokenizer` 的 RVQ token；用专用 system prompt（而非说话人 embedding）描述目标音色，支持零样本声音克隆。ARIA 把双通道生成改为单通道交错，约束"任意前缀的累计 speech-to-text token 比例不超过 item 级全局比例"。
- **预训练**（§3）：第一阶段冻结 LLM，只训编码器与 adapter；第二阶段解冻全部参数、序列长度 32,768；第三阶段序列长度 262,144。LLM 由 Qwen3.5 初始化，音频编码器由 AuT 初始化。

## 量化关联

**结构事实**

- Thinker 与 Talker 都是 Hybrid-Attention MoE 骨干，且都含 GDN 模块（§2.5）。
- MTP 是 Dense Transformer，Code2wav 是 ConvNet（Table 1）。
- 音频编码器 AuT 与 LLM 是分开训练、分开初始化的模块（§2.2、§3）：AuT 输出 6.25 Hz 音频 token。
- Talker 的生成对象是 RVQ 多码本 token，而非单个码本（§2.4）。
- 时间信息以"显式文本时间戳"的形式进入序列（§2.3）。
- 视觉编码器在 Table 1 的架构名是 SigLIP2，§2.3 与 §3 正文称其来自 Qwen3.5（两处称法不同，报告未说明关系）。

**量化推断**

- Thinker 与 Talker 都是 MoE：专家权重占参数总量绝大部分而每 token 只激活少数专家，per-expert 覆盖与路由稳定是量化误差和部署格式的主要来源。
- GDN 属于线性注意力/状态更新路径：它的状态是递推量而不是 KV cache，长序列下会被反复读写，量化误差会沿时间累积；其敏感度与 KV cache 低比特化不是同一件事，需单独评测。
- 语音侧是一条独立的量化对象链（Talker 骨干 → MTP 残差码本 → ConvNet 解码器）：这条链至少要单独校准、单独统计。语音 token 与文本 token 对量化误差的容忍度孰高孰低、能否共用同一比特预算，都没有本页可用的证据，须由实测决定。
- AuT 与 LLM 是分开训练、分开初始化的模块：分开训练不能推出两者激活尺度不同，激活范围能否共用一套校准统计是待实测项。端到端量化时至少要把 AuT 与 LLM 当作两个待校准对象分别评估，不能只校准 LLM 侧。
- 显式时间戳是文本 token，位置/时间对齐信息与文本 token 混在同一序列里，文本分支的量化误差会同时影响时序对齐。

**未知项**

- 论文不给逐层配置：Thinker / Talker 的层数、hidden、head 数、专家数、专家中间维，以及 GDN 层与全注意力层的配比都是缺口。
- 本次经镜像查询未找到官方的 `Qwen/Qwen3.5-Omni-*` 权重仓（查询只命中第三方同名仓），因此没有 `config.json` 可核对，也没有权重命名、加载类或默认参数证据；这只说明本次查询未命中，不代表官方未发布。
- GDN 的状态更新公式与 Hybrid-Attention MoE 的交错方式只指向 Qwen3.5（Team, 2026），本页未读该来源。
- 论文不含任何量化实验：位宽、校准、导出格式与精度结论均未披露。
- Table 1 与 §2.3 / §3 对视觉编码器的称法不同（SigLIP2 vs "来自 Qwen3.5"），是否为同一编码器须读 Qwen3.5 报告确认。

## 结果边界

- Table 2 明确标注为 **theoretical**，并依赖内部 vLLM 与 `torch.compile`、CUDA Graph；Plus 与 Flash 的部署资源与并行策略不同，论文自述两者不能严格横向比较。Table 1 未标 theoretical，但其数值与 Table 2 在 1 并发下的 Overall Latency 相同。
- "数百 B 参数"是量级描述；10 小时音频 / 400 秒 720P 视频是支持口径，不是实测吞吐或显存。
- 论文评测（215 项音视频子任务）是论文训练与推理设定下的结果，不代表量化后表现。
- 本库未加载权重、未运行量化或评测，本页没有任何实现层或运行时证据。

## Agent 如何使用

首轮读本页即可判断：是否全模态（是）、骨干是否 MoE（Thinker 与 Talker 都是）、是否含线性注意力（含 GDN）、是否有独立音频编码器与语音解码链（AuT / Talker-MTP-Code2wav）。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| 多模态模型的输入编码与连接器共性结构 | [`component-multimodal-connector`](../components/multimodal-connector.md)、[`paradigm-multimodal`](../paradigms/multimodal.md) |
| MoE 路由与专家的共性结构 | [`component-router-expert`](../components/router-expert.md)、[`paradigm-moe`](../paradigms/moe.md) |
| 线性注意力与状态更新（GDN 所属范式） | [`paradigm-linear-attention`](../paradigms/linear-attention.md)、[`component-kv-cache`](../components/kv-cache.md) |
| 上一代多模态模型的视觉接口 | [`model-qwen2-vl`](qwen2-vl.md) |
| 模块表、AuT 结构、时间戳方案、ARIA 与延迟口径 | [`paper-qwen35-omni`](../../sources/papers/paper-qwen35-omni.md) |
| 逐层配置、加载类、权重命名 | 论文未给、本库未登记权重仓；须查目标 checkpoint 的 `config.json` 与官方源码 |

回答量化问题时必须说明：具体变体（Plus / Flash）、骨干是 Hybrid MoE 且含 GDN、语音侧是 RVQ 多码本 + MTP + ConvNet、音频编码器为单独训练的 AuT、显式时间戳进入文本序列，并且必须声明本页没有配置层与运行时证据。

## 来源与证据

- [Qwen3.5-Omni 技术报告](../../sources/papers/paper-qwen35-omni.md)：§2.1–§2.5 架构与 Table 1 / Table 2、§3 预训练阶段、§1 长度与规模口径。

本页结构完全以该报告为限；报告不含逐层配置与量化实验，本页的实现层内容为零，量化内容除注明者外均为本库推断。
