---
{
  "id": "paper-glm5",
  "title": "GLM-5: from Vibe Coding to Agentic Engineering",
  "category": "paper",
  "url": "https://arxiv.org/abs/2602.15763",
  "version": "v2（2026-02-24）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "html"
  ],
  "read_locator": "摘要、§1、§2.1（模型规模、Multi-latent Attention、MTP with Parameter Sharing）、§2.1.1（Continued Pre-Training with DSA，含 Table 3）、§2.1.2（高效注意力变体消融）、§2.4.3（INT4 Quantization-aware Training）、§5（Adapting GLM-5 to Chinese Chip Infrastructure）、附录 A（Table 10：架构超参）",
  "topic": "architectures",
  "excerpt_sha256": null
}
---
## 这份来源是什么

GLM-5 技术报告（arXiv 2602.15763 v2，Zhipu AI & Tsinghua University）。报告覆盖预训练、中训练、后训练、面向国产芯片的适配与评测；本库只取与结构相关的部分：§2.1（模型规模、MLA 与 Muon Split、MLA-256、MTP 参数共享）、§2.1.1（用持续预训练引入 DSA）、§2.1.2（高效注意力变体消融）、§2.4.3（SFT 阶段的 INT4 量化感知训练与位级一致 kernel）、§5（W4A8 混合精度与融合算子）、附录 A Table 10（架构超参）。它是 `model-glm5` 的结构依据，也为 `component-attention`（MLA 变体）、`component-kv-cache`（latent KV cache）、`component-router-expert`、`paradigm-moe` 提供"MHA 式 MLA + DSA 稀疏注意力 + MoE"的现实实例。

范围边界：只覆盖 GLM-5 一个模型；§2.1.2 的注意力变体消融（SWA / GDN / SimpleGDN）在 GLM-9B 上完成，同节的 DSA 小规模验证在 GLM-4.7-Flash 上完成，两者都不是 GLM-5 本体。

## 已读定位与阅读形式

- 阅读形式：arXiv HTML 渲染版（`arxiv.org/html/2602.15763v2`），未读 LaTeX 源与 PDF。
- 定位：§2.1 的 Model size scaling、Multi-latent Attention、Multi-token Prediction with Parameter Sharing（含 Table 1、Table 2）；§2.1.1 的 DSA 两阶段过渡（含 Table 3）；§2.1.2 的变体消融（GLM-9B）与 DSA 小规模验证（GLM-4.7-Flash，含 Table 6）；§2.4.3 的 INT4 QAT；§5 的 Mixed-Precision W4A8 quantization 与 High-Performance fusion kernels；附录 A 的 Table 10 架构超参；§6 评测数字未取。

## 关键事实

- **规模**（§2.1）：GLM-5 扩到 **256 个专家**，并把**层数降到 80** 以降低专家并行的通信开销，得到 **744B 总参 / 40B 激活**（GLM-4.5 为 355B / 32B，总参翻倍）。§1 另写训练 token 预算扩到 **28.5T**。
- **MLA**（§2.1）：用 Multi-latent Attention 压缩 KV。论文给出的对比是 **576 维 latent KV cache（MLA）对 2048 维 KV cache（GQA-8）**；作者发现原 Muon 配方下 MLA 追不上 GQA-8，于是提出 **Muon Split**——把 up-projection 矩阵 `W^{UQ}`、`W^{UK}`、`W^{UV}` 按 head 拆成小矩阵后分别做矩阵正交化，使不同 head 的投影权重以不同尺度更新；论文报告 Muon Split 下 attention logits 尺度稳定、无需裁剪（§2.1；Table 1 给出对应结果）。
- **MLA-256**（§2.1）：针对 MLA 解码时 576 维点积的高开销，把 head dimension 从 **192 提到 256**、attention head 数**减少 1/3**，保持训练计算量与参数量不变，降低解码计算；Table 1 显示 MLA-256 与 MLA 性能相当。
- **MTP 参数共享**（§2.1）：训练时**共享 3 层 MTP 的参数**（而不是训练 n 层 MTP），使 draft 模型的内存开销与"只训 1 层 MTP"的 DeepSeek-V3 一致，同时提高接受率；Table 2 给出 GLM-5 接受长度 **2.76**、DeepSeek-V3.2 为 **2.55**（私有 prompt 集、投机步数 4）。
- **DSA 的引入方式**（§2.1.1）：沿用 DeepSeek Sparse Attention，用基于内容的动态细粒度选择替换稠密 `O(L²)` 注意力，**作用于全部层**；通过**持续预训练**从稠密基座迁移，两阶段为"dense warm-up + sparse adaptation"——warm-up 训 1000 步（每步 14 条 202,752 token 的序列、最大学习率 5e-3），sparse adaptation 训 **20B tokens**；论文只说沿用标准 DSA 配方，未明确 GLM-5 的 warm-up 是否冻结基座（冻结基座、只训 indexer 的细节出自同节 GLM-4.7-Flash 实验）。论文报告 DSA 把长序列注意力计算降低约 **1.5–2×**，128K 上下文可用一半 GPU 成本；Table 3 在 128K 长文基准上 MLA / DSA 分别为 MQ-NIAH 100.0 / 100.0、MV-NIAH 95.5 / 97.0、SQuAD 79.7 / 86.0、HotpotQA 66.3 / 63.0。
- **变体消融**（§2.1.2，GLM-9B）：与线性注意力类方案对比，GDN 等线性注意力能提升质量但**增加参数**，SimpleGDN 因最大化复用预训练权重而平衡最好；这些方案即使一半层保留全注意力，在细粒度检索上仍有固有精度差（RULER@128K 最多 5.69、RepoQA@128K 最多 7.33）；论文称 DSA 因 lightning indexer 做 token 级稀疏而不丢弃长程依赖，可无损地用于所有层。同节另一组小规模实验在 GLM-4.7-Flash 上做 DSA：warmup 阶段只训 indexer 1000 步（batch 16、冻结基座），随后与模型联合训练 150B tokens；RULER 4K–128K 结果见 Table 6。
- **W4A8 混合精度**（§5）：为把 GLM-5 放进单台 Atlas 800T A3，论文采用混合精度——**标准 Attention 与 MLP 块用 W8A8（INT8），MoE 专家压到 W4A8（INT4）**；用 **QuaRot** 做 outlier 抑制、**Flex_AWQ_SSZ** 做 scaling 校准，工具为 **msModelSlim**。同一节把模型称为"750B parameter GLM-5"。
- **融合算子**（§5）：为 Ascend NPU 开发 Lightning Indexer（把打分、ReLU、TopK 融合成单算子）、Sparse Flash Attention（并行处理"从 KV cache 选 TopK token"与稀疏注意力计算）、MLAPO（把 13 个小预处理算子融合成一个，用 Vector / Cube 单元并行）。
- **推理引擎适配**（§5）：vLLM-Ascend 与 SGLang；含 D2H 采样拷贝与下一步 decode 的异步重叠、RadixCache 前缀共享与 Prefix Cache、Attention DP + MoE EP 混合并行、以及用 MTP 提高 NPU 计算密度。

## 不能推出的结论

- 报告**不含低比特量化精度对比实验**：§5 只给到 Ascend 部署的"组件级精度分配 + 所用算法名"，§2.4.3 只说 SFT 阶段做 INT4 QAT 且有训练/推理位级一致的量化 kernel，均未给出逐层敏感度、精度对比数字或校准数据口径；不能据此推断 GLM-5 在 int4 / int8 下的精度损失，也不能外推到其它后端。
- §5 说"750B parameter"，§2.1 说 744B——同一报告内数值不一致，引用参数量时须说明口径。
- 层数口径在同一报告内不一致：§2.1 说"层数降到 80"，但附录 A Table 10 为 3 层 dense + 75 层 MoE = 78，与权重仓 `config.json` 的 `num_hidden_layers=78` 一致；引用层数时须说明口径，具体以目标 checkpoint 为准。
- 576 维 latent KV cache 与权重仓 `kv_lora_rank=512` + `qk_rope_head_dim=64` 的对应关系（512+64=576）是**推断**，论文没有给出这两个字段的分解。
- §2.1.2 的注意力变体消融在 GLM-9B / GLM-4.7-Flash 上完成，**不能直接外推到 GLM-5**；DSA 的 indexer 结构与稀疏选择细节论文未给，指向 DeepSeek-V3.2。
- 附录 A Table 10 给出架构超参（hidden 6144、dense/MoE intermediate、QK/V head dim、Q/KV LoRA、head 数、indexer 参数、专家数、词表），但与权重仓 `config.json` 口径不完全一致：Table 10 的 QK Head Dim=192 对应 config `qk_nope_head_dim=192`，config `qk_head_dim=256` 另含 rope 部分；Table 10 的 "Routed Experts"=8 是每 token 激活数，config `n_routed_experts=256` 是专家总数。引用时须注明口径。
- 论文评测（ARC 基准、CC-Bench-V2 等）是论文设定下的结果，不代表量化版本的表现。
- 本页只读 HTML 渲染版，未核对 LaTeX 源；§6 评测数字未逐表复核。
