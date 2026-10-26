---
{
  "id": "model-deepseek",
  "title": "DeepSeek-V2 / DeepSeek-V3",
  "type": "model",
  "tags": [
    "ar",
    "attention",
    "mla",
    "moe",
    "routing",
    "quantization"
  ],
  "related": [
    "paradigm-ar",
    "paradigm-moe",
    "component-attention",
    "component-router-expert",
    "component-mlp",
    "component-kv-cache"
  ],
  "sources": [
    "paper-deepseek-v2",
    "paper-deepseek-v3",
    "repo-deepseek-v3",
    "repo-huggingface-transformers"
  ],
  "confidence": "source-reported",
  "evidence_scope": "mixed",
  "aliases": [
    "multi-head latent attention",
    "多头潜在注意力"
  ],
  "updated": "2026-10-26"
}
---

# DeepSeek-V2 / DeepSeek-V3

## 一句话理解

DeepSeek-V2/V3 是 decoder-only 自回归模型，两个标志性结构是 **MLA**（低秩 latent 压缩 KV，推理只缓存压缩向量）与 **DeepSeekMoE**（细粒度路由专家 + 共享专家）。V3 在此之上加了以**路由 bias 动态调整**为主的负载均衡（论文称 auxiliary-loss-free，仍保留极小系数的序列级 `L_Bal`）、**MTP** 训练目标与 **FP8 混合精度训练**。系列内不同版本配置不同，不能从系列名推断单个 checkpoint。

## 结构规格表

| 项 | DeepSeek-V2 | DeepSeek-V3（671B） |
| --- | --- | --- |
| 总参 / 激活 | 236B / 21B | 671B / 37B |
| 层数 | 60（第 1 层 dense） | 61（`n_dense_layers=3`） |
| hidden | 5120 | 7168 |
| attention | MLA：`n_h=128`、`d_h=128`、`d_c=512`、`d_h^R=64` | MLA：`q_lora_rank=1536`、`kv_lora_rank=512`、`qk_nope=128`、`qk_rope=64`、`v_head_dim=128` |
| MoE | 每层 2 共享 + 160 路由，每 token 激活 6，专家中间维 1536 | 每层 256 路由 + 1 共享，每 token 激活 8；`n_expert_groups=8`、`n_limited_groups=4`、`route_scale=2.5` |
| 均衡 | device-limited routing + 三类辅助损失 | sigmoid affinity + 路由 bias 动态调整（论文称 auxiliary-loss-free，仍保留极小系数序列级 `L_Bal`）+ node-limited routing |
| 上下文 | 128K | 128K |

V2-Lite（附录 B）为 15.7B / 2.4B、27 层、hidden 2048、16 head、`d_c=512`、`d_h^R=64`、2 共享 + 64 路由（中间维 1408），与完整版的差别之一是**不压缩 query**。V2 列的论文符号与 V3 列的 config 字段一一对应：`d_c↔kv_lora_rank`、`d'_c↔q_lora_rank`、`d_h↔qk_nope_head_dim`、`d_h^R↔qk_rope_head_dim`（V2 的 query 压缩维为 `d'_c=1536`，只有 V2-Lite 不压缩 query）。按 `q_lora_rank` 是否大于 0 可区分是否压缩 query；按 `n_routed_experts`/`n_shared_experts`/`n_activated_experts` 可区分 16B-Lite / 236B / 671B。
表中两列分别取自论文与官方 `configs/*.json`；引用具体 checkpoint 时必须用目标 `config.json` 核对，本表不能替代。

## 结构与数据流

主干是 embedding → 多个 `Block`（MLA + MLP/MoE）→ final norm → LM head。

- **MLA**（V2 §2.1.2）：`c^{KV} = W^{DKV} h` → 上投影出 `k^C/v^C`；推理期把 `W^{UK}` 吸收进 query、`W^{UV}` 吸收进输出投影，真正缓存的只有 `c^{KV}` 与跨头共享的解耦 key `k^R`。
- **MoE**（V2 §2.2.1）：`h' = u + Σ_s FFN_s(u) + Σ_r g_r FFN_r(u)`；V3 的 affinity 改用 sigmoid，专家 bias 只参与 top-K 选择，每 step 末按 batch 负载动态调整。
- **MTP**（V3）：训练目标是 D 个模块在串行因果链上**依次预测 D 个额外 token**（偏移 2…D+1；论文采用串行，区别于 Meta 的并行多头。本 case `D=1`，即在 next token 之外再预测一个额外 token），与主模型共享 embedding 与输出头；推理时可丢弃，也可复用于投机解码，是否保留由 checkpoint 与推理图决定。
- 官方推理实现（repo-deepseek-v3）：`Gate`(model.py:535) + `Expert`(601) + `MoE`(636)，配置从 `configs/*.json` 读；`config_671B.json` 与 `config_16B.json`/`config_236B.json`/`config_v3.1.json` 并存。

## 量化关联

**结构事实**

- MLA 推理只缓存压缩向量（`kv_lora_rank` 维）与跨头共享的解耦 key（`qk_rope_head_dim` 维），不是完整 K/V。
- V2 部署期把参数转 FP8，并把 KV cache 平均压到 6 bit（引用外部方法）。
- V3 的 FP8 训练契约：激活 `1x128` tile、权重 `128x128` block 分组缩放；MoE SwiGLU 输入 FP8 缓存、上投影激活与下投影激活梯度各自量化成 FP8 通信、`combine` 保持 BF16，MoE gating 保高精度。

**实现事实**（来自固定版本源码，须按该 revision 核对）

- 官方推理实现含 fp8 路径（`act_quant` + `fp8_gemm`）。
- 官方推理实现用 `kv_cache` / `pe_cache` 两个 buffer 承载上述缓存，只有 `attn_impl="naive"` 才注册完整 `k_cache`/`v_cache`。

**量化推断**

- MLA latent cache 与 MHA/GQA 的 K/V cache 语义不同，量化对象与后端 kernel 需按实现分别定义。
- 专家访问频率不均，需要 per-expert 覆盖；router top-k 变化应与专家 MLP 输出误差分开测量。
- MTP 分支若保留在推理图中应独立检查其权重与输出；若仅用于训练，则不是推理量化对象。

**未知项**

- 论文未给 V2 部署量化的精度影响、校准方法与逐层结论。

## 结果边界

- 代码默认值（`ModelArgs` 的 64 路由 / 2 共享）**不是模型配置**；引用实际专家数必须用 `configs/` 下对应 JSON（671B 为 256/1/8）。
- V3 论文的 FP8 是**训练精度框架**（附录 B.1 给了 FP8 vs BF16 相对误差 <0.25%），仓库里的 fp8 路径是**推理实现**；两者都不是通用量化方案，论文未给 INT8/INT4 等更低比特与部署实验。
- V2 的"KV cache 减少 93.3%""吞吐 5.76×"是相对 DeepSeek 67B、在其部署设定下测得。
- 本库未运行任何量化或评测；未对具体 checkpoint 做数值验证。

## Agent 如何使用

首轮读本页即可判断目标 checkpoint 属于哪一版、attention 是否 MLA、MoE 专家数与均衡策略、是否含 fp8 路径。

| 需要确认的内容 | 继续读取 |
| --- | --- |
| MLA 与 DeepSeekMoE 的结构、辅助损失 | [`paper-deepseek-v2`](../../sources/papers/paper-deepseek-v2.md) |
| V3 的 sigmoid affinity、无辅助损失均衡、MTP 与 FP8 | [`paper-deepseek-v3`](../../sources/papers/paper-deepseek-v3.md) |
| Gate/Expert/MoE 模块、671B 配置与 fp8 路径 | [`repo-deepseek-v3`](../../sources/repos/repo-deepseek-v3.md) |
| transformers 里 DeepSeek-V2/V3 的建模文件 | [`repo-huggingface-transformers`](../../sources/repos/repo-huggingface-transformers.md) |
| 目标 checkpoint 的真实配置与 runtime 行为 | 该 checkpoint 的 `config.json` / 官方源码 / 运行时（本库不提供） |

回答量化问题时必须说明：具体版本与 config、attention 是否 MLA、专家数与 top-k、是否启用 fp8、cache 精度与后端支持。

## 来源与证据

- [DeepSeek-V2](../../sources/papers/paper-deepseek-v2.md)：MLA、DeepSeekMoE、device-limited routing 与部署量化（§3.2.3）。
- [DeepSeek-V3](../../sources/papers/paper-deepseek-v3.md)：无辅助损失均衡（§2.1.2）、MTP（§2.2）、FP8 混合精度训练（§3.3）。
- [DeepSeek-V3 官方实现](../../sources/repos/repo-deepseek-v3.md)：固定 commit 下模块划分、fp8 路径与 671B 配置。
- [huggingface/transformers](../../sources/repos/repo-huggingface-transformers.md)：`models/deepseek_v3/`、`models/deepseek_v2/` 的实现落点。

本页结构以上述来源为限；除 V3 的 FP8 vs BF16 消融外，来源不含更低比特的精度实验，量化内容除注明者外均为本库推断。
