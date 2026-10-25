---
{
  "id": "paper-deepseek-v2",
  "title": "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
  "category": "paper",
  "url": "https://arxiv.org/abs/2405.04434",
  "version": "v5（2024-06-19）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1、§2.1（Figure 3、Table 1）、§2.2、§3.1.2–3.1.4、§3.2.3、附录 B（DeepSeek-V2-Lite）、附录 C、附录 D（Table 8–9）",
  "topic": "architectures",
  "excerpt_sha256": null
}
---
## 这份来源是什么

DeepSeek-V2 技术报告（arXiv 2405.04434，v5）。236B 总参数 / 每 token 激活 21B、支持 128K 上下文的 MoE 语言模型，主要贡献两个结构：**MLA（Multi-head Latent Attention）**——用低秩联合压缩 KV，把推理期 KV cache 压成一个 latent 向量；**DeepSeekMoE**——细粒度专家切分 + 共享专家隔离（MoE 部分论文自述沿用 DeepSeekMoE 论文的架构）。论文另给 DeepSeek-V2-Lite（附录 B）与 V2 部署时的量化方案（§3.2.3）。

支撑本库 `paradigm-moe`（每层 2 共享 + 160 路由专家、每 token 激活 6 个、device-limited routing 与三类负载均衡损失）、`component-attention` 与 `component-kv-cache`（MLA 的压缩对象与推理期矩阵吸收、部署期 KV cache 6 bit 量化）、`models/deepseek`。

边界：论文的量化内容只有**部署期**一小段（约两句：参数转 FP8、KV cache 平均 6 bit），没有低比特量化的精度对比、校准方法或误差分析；附录中评测格式类表格本页未逐条读取。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源 `main.tex`；另读 arXiv abs 页 HTML（用其版本历史与摘要）。v1 2024-05-07 … v5 2024-06-19。
- 定位：§2 Architecture（2.1 MLA、2.2 DeepSeekMoE）、§3.1.2–3.1.4（超参、基础设施、长上下文）、§3.2.3（部署量化）、附录 B（Lite）、C（矩阵吸收）、D（Table 8–9 消融）。表号按浮动体顺序：Table 1 = KV cache 对比、Table 2 = 主评测表、Table 8 = 附录 D.1 的 MHA/GQA/MQA 消融、Table 9 = 附录 D.2 的 MLA vs MHA。

## 关键事实

- **总体规模与效率**（摘要、§3.2.3）：236B 总参数、每 token 激活 21B、上下文 128K；相对 DeepSeek 67B 声称省 42.5% 训练成本（每训练 1T token：67B 需 300.6K GPU 小时，V2 只需 172.8K）、KV cache 减少 93.3%、最大生成吞吐提升到 5.76×。
- **MLA 的低秩联合压缩**（§2.1.2）：`c^{KV}_t = W^{DKV} h_t`（`d_c ≪ d_h n_h`），再上投影出 `k^C_t = W^{UK} c^{KV}_t`、`v^C_t = W^{UV} c^{KV}_t`；推理时**只缓存 `c^{KV}_t`**，故这部分 cache 每 token 只有 `d_c·l` 个元素。query 侧的低秩压缩 `c^Q_t = W^{DQ} h_t`、`q^C_t = W^{UQ} c^Q_t` 论文明确**只用于降低训练期激活显存**。
- **推理期矩阵吸收与解耦 RoPE**（§2.1.2–2.1.3、附录 C）：`W^{UK}` 可吸收进 `W^Q`、`W^{UV}` 可吸收进 `W^O`，推理时甚至不需要把 keys/values 显式算出来；附录 C 用蓝框标出真正需要缓存的只有 `c^{KV}_t` 与 `k^R_t`。RoPE 若直接作用在 `k^C_t` 上会让 `W^{UK}` 与位置相关矩阵耦合、无法吸收，故另设**多头解耦 query `q^R_{t,i}`** 与**跨头共享的解耦 key `k^R_t = RoPE(W^{KR} h_t)`** 专门承载 RoPE（`q^R_t = RoPE(W^{QR} c^Q_t)`）；每头 `q_{t,i} = [q^C_{t,i}; q^R_{t,i}]`、`k_{t,i} = [k^C_{t,i}; k^R_t]`，attention 分母是 `sqrt(d_h + d_h^R)`。
- **KV cache 对比**（§2.1.4，Table 1）：MHA `2 n_h d_h l`、GQA `2 n_g d_h l`、MQA `2 d_h l`、MLA `(d_c + d_h^R) l ≈ (9/2) d_h l`；论文据此说 MLA 的 cache 相当于"只有 2.25 个 group 的 GQA"，但性能强于 MHA。表中 V2 取 `d_c = 4 d_h`、`d_h^R = d_h/2`；按 §3.1.2（`d_c=512`、`d_h^R=64`、60 层）算，V2 每 token 的 cache 是 `(512+64)×60 = 34560 ≈ 34.6K` 个元素（与附录 D.2 大 MoE 那一列的 34.6K 数值相同，但论文未给该大 MoE 的 `d_c`/`d_h^R`/层数，只是数值相同、不代表两者超参一致）。
- **DeepSeekMoE**（§2.2.1）：`h'_t = u_t + Σ_{i=1}^{N_s} FFN^{(s)}_i(u_t) + Σ_{i=1}^{N_r} g_{i,t} FFN^{(r)}_i(u_t)`，其中 `g_{i,t}` 取 affinity `s_{i,t} = Softmax_i(u_tᵀ e_i)` 的 top-`K_r`、其余置 0（`e_i` 是第 i 个路由专家的质心）；两个关键设计是**细粒度专家切分**与**共享专家隔离**。
- **路由与均衡**（§2.2.2–2.2.4）：**device-limited routing** 让每个 token 的目标专家最多分布在 `M` 台设备上（先选出亲和度最高的 `M` 台设备，再在其中做 top-K），论文称 `M ≥ 3` 时性能与不限制的 top-K 大致一致；三类负载均衡辅助损失 `L_ExpBal`/`L_DevBal`/`L_CommBal` 分别配 `α_1/α_2/α_3`（通信均衡损失与 device-limited routing 配合）；训练期按设备做 token dropping（容量因子 1.0，丢弃亲和度最低的 token，并保证约 10% 训练序列的 token 永不被丢），推理时按效率需求决定是否丢弃。
- **模型超参**（§3.1.2）：60 层、hidden 5120、可学习参数按 std 0.006 初始化；`n_h=128`、`d_h=128`、`d_c=512`、`d'_c=1536`、`d_h^R=64`；除第一层外全部 FFN 换成 MoE 层，每层 **2 个共享专家 + 160 个路由专家**、专家中间维 1536、**每 token 激活 6 个路由专家** → 236B 总参 / 21B 激活；另在压缩 latent 之后加 RMSNorm、在宽度瓶颈乘缩放因子以保证训练稳定。
- **训练超参**（§3.1.2）：8.1T token、BBPE 100K 词表；AdamW `β=(0.9,0.95)`、`weight_decay=0.1`；最大学习率 `2.4e-4`，前 2K 步线性 warmup，训练到约 60% 与 90% token 时各乘 0.316；梯度裁剪 1.0；batch size 在前 225B token 从 2304 升到 9216；最大序列 4K。`D=8`、`M=3`、`α_1=0.003`、`α_2=0.05`、`α_3=0.02`。
- **长上下文扩展**（§3.1.4）：用 YaRN 把默认 4K 扩到 128K，且**只作用在解耦共享 key `k^R_t`** 上；`s=40`、`α=1`、`β=32`、目标最大长度 160K；长度缩放因子取 `√t = 0.0707·ln s + 1`（与原始 YaRN 略有偏离）；额外在 32K 序列、batch 576 下训 1000 步，NIAH 在到 128K 的各长度上表现良好。
- **部署期量化**（§3.2.3）：为部署先把**参数转成 FP8**，再对 KV cache 做量化（引用 `kv_quant`、`atom`），把 cache 里每个元素**平均压到 6 bit**；配合 MLA，实测部署所需 KV cache 明显小于 DeepSeek 67B，因此能用大得多的 batch。单节点 8×H800 上生成吞吐超 50K tokens/s（为 67B 最大生成吞吐的 5.76 倍）、prompt 输入吞吐超 100K tokens/s。
- **消融**（附录 D）：MLA vs MHA（Table 9）——同架构、只换 attention 的 MoE 对比，小 MoE（约 16B 总参，训 1.33T token）KV cache 每 token 15.6K vs 110.6K（约 14%），大 MoE（约 250B 总参，**只训了 420B token**）34.6K vs 860.2K（约 4%），**8 个评测格里有 7 格 MLA ≥ MHA**（例外是小 MoE 的 C-Eval：MHA 51.6 vs MLA 50.9）。MHA/GQA/MQA（Table 8）——7B dense、1.33T token、对齐参数量后，MHA 在 BBH/MMLU/C-Eval/CMMLU 上显著优于 GQA 与 MQA。
- **DeepSeek-V2-Lite**（附录 B）：**15.7B 总参 / 2.4B 激活**（标题写 16B），27 层、hidden 2048、16 个 attention head（每头 128 维）、`d_c=512`、`d_h^R=64`，与完整版不同的是**不压缩 query**；除第一层外全部 FFN 换成 MoE，每层 2 个共享专家 + **64 个路由专家**、专家中间维 1408、每 token 激活 6 个；上下文 32K、训 5.7T token。
- **基础设施**（§3.1.3）：基于 HAI-LLM，16 路 zero-bubble 流水并行 + 8 路专家并行 + ZeRO-1 数据并行，**不使用张量并行**（激活参数少、部分算子重算）；共享专家计算与专家并行 all-to-all 通信重叠；为通信、路由与融合线性算子定制 CUDA kernel；MLA 基于改进版 FlashAttention-2 优化；实验在 H800 集群完成。

## 不能推出的结论

- 「21B 激活参数达到 top-tier」是**与其它开源模型比较**的相对结论，不能理解成它等价于一个 21B 稠密模型，也不代表单 token 算力只有 21B 稠密模型的水平（总参数 236B，路由与通信另有开销）。
- KV cache 表按**元素个数**度量、明确与存储精度无关；element 数小不等于字节数按同比例小。论文只在部署一节写了"参数转 FP8"且"KV cache 压到平均 6 bit"（后者引用外部方法），**没有给出该量化的精度影响、校准方法或逐层误差**，本页不能据此判断 MLA 的 cache 更适合低比特或量化更友好。
- 42.5% 训练成本、93.3% KV cache、5.76× 吞吐都是相对 DeepSeek 67B、在论文自己的部署设定（8×H800 单节点、特定 prompt/生成长度分布）下测得的；不能外推到其它硬件、batch 或实现。
- MLA 的推理期「吸收」依赖矩阵乘法结合律做算子合并，是**实现层面**的等价改写；不能由它推出训练/推理激活的数值分布相同。
- 长上下文扩展只作用于解耦共享 key `k^R_t`，这是 MLA 特有的处理；不能把它当作标准 RoPE 的 YaRN 用法。
- 附录 D.2 的两组对比不是等训练量（小 MoE 1.33T token、大 MoE 仅 420B token），也不是严格等参（15.7B/16B 与 ~250B 两档）；「MLA 不弱于 MHA」的结论要连同这些条件一起引用，且 C-Eval 上确实有一格 MHA 更高。
- 训练期使用 token dropping、评测期不丢 token；论文声称两者一致，但本页不能据此推断任意推理配置下丢弃策略都无害。
- 本库关于 MoE 路由/专家激活分布对量化影响的结论需要另行在目标实现上测：论文没有给激活范围统计，也没有给路由分布与量化误差的关系。
