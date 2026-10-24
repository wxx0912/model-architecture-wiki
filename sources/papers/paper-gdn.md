---
{
  "id": "paper-gdn",
  "title": "Gated Delta Networks: Improving Mamba2 with Delta Rule",
  "category": "paper",
  "url": "https://arxiv.org/abs/2412.06464",
  "version": "v3（2025-03-06）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1、§2.1、§2.2、§3.1（Eq.10 与 Table 1）、§3.2（Table 2）、§3.3、§3.4（Figure 1）、§4（Setup、Table 3、Table 4、Figure 2、Table 5、Figure 3）、§5、§6、附录 A、附录 B.1、附录 B.2（Table S.1–S.2）",
  "topic": "architectures",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---
## 这份来源是什么

Gated DeltaNet 论文（arXiv 2412.06464，v3，ICLR 2025 风格）。它把 Mamba2 的**门控衰减**与 DeltaNet 的 **delta 更新**合成一条 gated delta rule，并给出配套的硬件高效 chunkwise 训练算法，再给出纯线性（Gated DeltaNet）与混合（+滑窗注意力 / +Mamba2）两种架构。

是本库 `paradigm-linear-attention` 与 `model-kimi-linear` 的一手材料：线性注意力族的"门控 + delta"状态更新、state 的矩阵形态、chunkwise 记法、以及"线性层与注意力层如何混排"的证据。与相邻来源的关系：`paper-mamba2` 提供被改进的 gated 更新与 SSD/chunkwise 记法，DeltaNet（§2.2 转述、本库未单独登记）提供 delta 规则。

范围边界：实验只有 1.3B/100B token 的等条件对照与 400M/500M、15B token 的消融；不涉及量化、低比特或 state 数值范围统计；只给算法级公式，没有 chunk size 取值与 kernel 实现细节。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源 `main.tex`（含 `\input` 的 `figures/model.tex`），另读 arXiv HTML 渲染版交叉核对编号。v1 2024-12-09、v2 2025-03-05、v3 2025-03-06，本地源码即 v3。
- 下文章节号、式号、表号、图号按 v3：§1 引言、§2 Preliminary、§3 Gated Delta Networks、§4 Experiments、§5 Related Work、§6 Conclusion；附录 A（Extended WY 表示的证明）、附录 B.1/B.2。正文表 **Table 1–5**、图 **Figure 1–3**；**附录表在源码里被重编号为 Table S.1/S.2**（不是 A.1/A.2）。

## 关键事实

- **gated delta rule**（§3.1 Eq.10）：`S_t = S_{t-1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ`；数据相关的标量门 `α_t ∈ (0,1)` 控制状态衰减——`α_t → 0` 迅速清空记忆，`α_t → 1` 退化为纯 delta 规则（§1）。
- **两个前身**（§2.1、§2.2）：Mamba2 为 `S_t = α_t S_{t-1} + v_t k_tᵀ`，用同一比例衰减**所有** KV 关联；DeltaNet 为 `S_t = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`，逐条（软）替换旧 KV，`β_t ∈ (0,1)` 为写入强度，脚注给 `β_t` 可取 `(0,2)` 以允许负特征值。
- **在线学习视角**（§3.1 Table 1）：把 LA、Mamba2、Longhorn、DeltaNet、Gated DeltaNet 的更新统一写成"在线目标 + 闭式更新"；`α_t` 放松了既往模型阻止状态偏离的正则项。另一种读法是 delta 规则等价于对回归损失 `½‖S_t k_t − v_t‖²` 做 test-time SGD（`β_t` 为学习率），门控相当于在这个 SGD 上再乘一个自适应权重衰减。
- **chunkwise 训练**（§3.3，附录 A 给证明）：把 DeltaNet 的 WY 表示推广到带门控形式，用 UT transform 写成 `Ũ_t = [I + strictLower(diag(β_t)(Γ_t ⊙ K_t K_tᵀ))]⁻¹ diag(β_t) V_t`；分块缩放沿用论文箭头记法——`←Q`、`←W` 乘 `γ^r`，`→K` 乘 `γ^C/γ^r`、`→S` 乘 `γ^C`。
- **block 设计**（§3.4 Figure 1 右）：宏观沿用 Llama（token mixer 层 + SwiGLU MLP 层），把 self-attention 换成 gated delta rule 的 token mixing。`q,k,v` 由线性投影 → 短卷积 → SiLU 产生，`q,k` 做 **L2 归一化**；`α,β` 只走线性投影，`α` 沿用 Mamba2 参数化但论文在脚注里**明确省略了具体参数化细节**；输出先归一化与输出门控，再做输出投影。
- **混合架构**（§3.4）：H1 把线性层与滑窗注意力（SWA）交替；H2 把 Mamba2、Gated DeltaNet、SWA 三者混排。动机是线性模型的固定状态难以处理局部位移/比较与检索。
- **state 是矩阵值状态**（§1、§2.1）：线性注意力被改写为"矩阵值状态的线性 RNN"，`S ∈ R^{d_v×d_k}`（每 head 一个矩阵，逐 token 递推 `o_t = S_t q_t`），**不随序列长度增长**；论文未给该状态的数值范围或分布统计。
- **S-NIAH 案例**（§3.2 Table 2，1.3B）：Gated DeltaNet 在 S-NIAH-1@8K 得 **91.8**、S-NIAH-2@4K **92.2**、S-NIAH-3@4K **27.6**；同期 Mamba2 的 S-NIAH-1@8K 掉到 **30.4**、S-NIAH-3@4K 仅 **4.6**，DeltaNet 的 S-NIAH-2@4K 仅 **18.6**。论文由此得三条观察：衰减损害长程保持、门控利于过滤无关信息、delta 规则利于复杂模式的记忆。
- **训练设置**（§4 Setup）：所有模型等条件对照——1.3B 参数、100B token（FineWeb-Edu 子集）、AdamW 峰值 lr 4e-4、weight decay 0.1、梯度裁剪 1.0、cosine 调度含 1B token warmup、batch 0.5M token、Llama2 tokenizer（词表 32000）、训练长度 4K（Samba 与混合模型滑窗为 2K）。
- **主结果**（§4 Table 3，1.3B）：Gated DeltaNet 的 WikiText ppl **16.42**、LAMBADA ppl **12.17**、常识平均 **55.32**，优于 Mamba2（16.56/12.56/54.89）与 DeltaNet（17.71/16.88/52.14）；LAMBADA 与常识平均也优于 Transformer++（18.32/52.25）与 Samba（13.29/54.00），但 **WikiText 16.42 略差于 Samba 16.13**（该列最低为 Gated DeltaNet-H2 的 15.91）；混合模型平均更高（H1 56.40、H2 56.18）。**口径提醒**：正文写该表覆盖 400M 与 1.3B 两个规模，v3 源码表体只有一组行。
- **真实检索**（§4 Table 4，输入截断 2K）：Gated DeltaNet 平均 **30.6**，高于 Mamba2（29.8）与 DeltaNet（26.2），但明显低于 Transformer++（37.0）与 Samba（37.3）；混合模型反超纯注意力（H1 39.0、H2 40.1）。论文自述纯线性模型相对注意力仍有明显差距，并把 Gated DeltaNet 相对前身提升幅度的收窄归因于未做指令对齐的小模型容易重复、与更新规则的选择基本无关（§4）。
- **长上下文**（§4 Figure 2、Table 5）：外推到 **20K** token，覆盖 GovReport、QMSum、NarrativeQA、Qasper、CodeParrot、PG19，Gated DeltaNet 在 RNN 类里困惑度最低；LongBench 14 个任务上平均 **16.6**（H1 17.8、H2 18.4），高于 Mamba2（13.5）、DeltaNet（13.6）、Mamba（14.6）。
- **吞吐**（§4 Figure 3，1.3B 模型、单张 H100）：gated delta rule 相对 delta rule 只有极小开销，Gated DeltaNet 与 DeltaNet 吞吐基本一致，约比 Mamba2 慢 **2–3K tokens/sec**；2K 窗口内 Transformer++ 最高（FlashAttention-2），故含 SWA 的模型更快。
- **消融**（附录 B.2 Table S.1，400M/15B）：基线（head dim 128）27.35 ppl / 47.26 acc；换朴素 delta 规则明显变差（30.87/45.12）；去短卷积（28.95/46.16）与去输出门（29.12/45.46）都掉点；去输出归一化只有边际变化（27.55/47.07）；L2 归一化优于 L1；SiLU 优于 ReLU 与 1+ELU；head dim 128 是效率与质量折中（64 更差、256 略好）。Table S.2（500M/15B）四种顺序里 `Mamba2 + Gated DeltaNet + SWA` 最好（23.54/48.73）。
- **与相关工作的位置**（§5）：串起门控/遗忘（源自 LSTM 一脉）与 delta 规则两条谱系；指出 DeltaNet 的 `I − β_t k_t kᵀ` 比 Mamba2 的 `α_t I` 有 "greater flexibility"（该比较转引 yang2024parallelizing，非本文自证）；列出可无缝套用到 Gated DeltaNet 的他人扩展（负特征值、多个 Householder 乘积，分别引 Grazzi2024 与 siems2025）与同期工作 RWKV-7（`S_t = S_{t-1}(diag(d_t) − a_t b_tᵀ) + v_t k_tᵀ`）。
- **代码**：`https://github.com/NVlabs/GatedDeltaNet`（摘要）；§5 脚注另指向 Flash Linear Attention 的 `generalized_delta_rule`。

## 不能推出的结论

- 实验只覆盖 1.3B/100B token 的等条件对照与 400M/500M、15B token 的消融；不能把小模型排名外推到更大规模、其他数据或 tokenizer。
- 论文不含量化、低比特或 state 数值范围统计；只能说明"state 是每 head 一个 `d_v×d_k` 矩阵、不随序列增长"这一结构事实，不能推出 state 量化/低比特存储的误差路径、校准需求或后端支持。
- 论文**未披露** 1.3B 模型的 hidden dim、head 数、层数以及 H1/H2 的层比例；结构识别不能把这些当成已知参数。
- 论文只给算法级公式（WY 表示、UT transform、分块缩放），**不给 chunk size 取值、CUDA 算子或实现级细节**；某个实现的 kernel 行为须回官方仓库与 Flash Linear Attention 核验。
- Table 2/3/4/5 是自研实现、训练长度 4K 下的结果，且评测口径不统一（Table 4 用 Cloze Completion Formatting 提示并截断到 2K，消融表用 lm-evaluation-harness）；不能跨实现直接比较，也不能当作不同算力预算下的绝对排名。
- "Gated DeltaNet 与 DeltaNet 吞吐相当、比 Mamba2 慢 2–3K tokens/sec"是 1.3B 模型在单张 H100 上的实测，随 kernel、后端与序列长度变化，不是算法固有开销。
- 正文称结果表覆盖 400M 与 1.3B 两个规模，但 v3 的 Table 3 只列一组行；引用具体数值须同时写明表格与设定。
- Table S.1/S.2 的结论（head dim 128 最优、输出归一化收益边际、`Mamba2+GDN+SWA` 顺序最好）来自 400M/500M、15B token 的单次消融，是经验折中，不是普适最优配置。
- 论文没有系统扫描"线性层 : 注意力层"的比例（H1 为 1:1 交替，H2 为三层一组）；`paper-mamba2` 里约 10% 注意力层的结论属另一篇论文的设定，不能移到本页。
- 摘要"在 in-context retrieval 等基准上超过 Mamba2/DeltaNet"不能反推真实长文档检索已解决：论文自己的 Table 4 显示纯线性模型在真实检索上仍落后注意力。
- 本页只是论文级证据，不证明任何具体 checkpoint 的实际行为，也不能替代在目标实现上的运行验证。
