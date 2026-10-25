---
{
  "id": "repo-deepseek-v3",
  "title": "deepseek-ai/DeepSeek-V3（官方推理实现、配置与权重转换）",
  "category": "repo",
  "url": "https://github.com/deepseek-ai/DeepSeek-V3",
  "version": "9b4e9788e4a3a731f7567338ed15d3ec549ce03b",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "inference/model.py（ModelArgs、ParallelEmbedding、Linear、ColumnParallelLinear、RowParallelLinear、RMSNorm、MLA、MLP、Gate、Expert、MoE、Block、Transformer）、inference/kernel.py（act_quant、weight_dequant、fp8_gemm）、inference/configs/config_671B.json、inference/{convert.py,fp8_cast_bf16.py,generate.py}",
  "topic": "implementations",
  "excerpt_sha256": null
}
---

## 这份来源是什么

DeepSeek-V3 的官方推理实现仓库。它给 `models/deepseek`、`component-attention`（MLA 的 cache 布局）、`component-router-expert` 提供实现级事实：MLA 的投影拆分与 KV cache 只存压缩向量、MoE 的 Gate/Expert 结构、以及 fp8 推理路径与官方 671B 配置。与 `paper-deepseek-v3` 的分工：论文给方法与训练口径，本页给"官方推理代码怎么组织、配置里是什么"。

## 已读定位与阅读形式

- 阅读形式：GitHub 仓库文件树 + `inference/model.py`、`inference/configs/config_671B.json`（`repo-tree`），按 commit `9b4e9788e4a3a731f7567338ed15d3ec549ce03b` 读取。
- 已读定位：`inference/model.py` 的类定义与 `ModelArgs`、`MLA.__init__/forward`、`inference/kernel.py` 的函数清单、`inference/configs/` 目录。

## 关键事实

- `ModelArgs`（inference/model.py:20）的代码默认值为 `n_routed_experts=64`、`n_shared_experts=2`、`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`、`dtype∈{"bf16","fp8"}`（model.py:57–79）——注意这只是默认值，实际配置见 `configs/`。
- MLA（model.py:396）：`q_lora_rank=0` 时走单条 `wq`，否则 `wq_a` → `q_norm` → `wq_b`；`wkv_a` 一次输出 `kv_lora_rank + qk_rope_head_dim`，`kv_norm` 后 `wkv_b` 出 `n_heads*(qk_nope_head_dim + v_head_dim)`，`wo` 为 RowParallel；**默认路径的 KV cache 只注册 `kv_cache`（`kv_lora_rank`）与 `pe_cache`（`qk_rope_head_dim`）两个缓冲，不存完整 K/V**；只有 `attn_impl="naive"` 才注册完整的 `k_cache`/`v_cache`（model.py:440–444）。
- MoE 的组成是 `Gate`（model.py:535）+ `Expert`（model.py:601）+ `MoE`（model.py:636）；`Block` 为 MLA 加上 MLP 或 MoE（model.py:696）；`Transformer` 为总装（model.py:738）。
- FP8 推理路径：模块级函数 `linear`（model.py:131 起，fp8 分支在 158–163 行）用 `act_quant` 激活量化 + `fp8_gemm`，`class Linear`（166 行）的 forward 调用它；`inference/kernel.py` 提供 `act_quant`、`weight_dequant`、`fp8_gemm`；`inference/fp8_cast_bf16.py` 是权重格式转换脚本，`inference/convert.py` 负责 checkpoint 转换。
- 官方 671B 配置（`inference/configs/config_671B.json`，逐字段）：`vocab_size=129280`、`dim=7168`、`inter_dim=18432`、`moe_inter_dim=2048`、`n_layers=61`、`n_dense_layers=3`、`n_heads=128`、`n_routed_experts=256`、`n_shared_experts=1`、`n_activated_experts=8`、`n_expert_groups=8`、`n_limited_groups=4`、`route_scale=2.5`、`score_func="sigmoid"`、`q_lora_rank=1536`、`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`、`dtype="fp8"`。同目录另有 `config_16B.json`、`config_236B.json`、`config_v3.1.json`。
- 该仓库是**推理侧实现**（`generate.py` + 权重转换），不含训练代码。

## 不能推出的结论

- 代码默认值（如 `n_routed_experts=64`、`n_shared_experts=2`）不是模型配置；引用 DeepSeek-V3 的实际专家数须用 `configs/config_671B.json` 的 256/1/8，两者混用会得出错误结构。
- 推理实现里的 fp8 路径是"官方推理契约"，不等于训练时的量化契约；`act_quant`/`fp8_gemm` 的 scale 粒度与累加精度本页未逐行核对，不能据此推算量化误差。
- 本页未在目标硬件上运行该实现，不能据此认定某个 checkpoint 的显存占用或吞吐。
- MLA 的 cache "只存压缩向量 + rope 分量"是官方推理实现的结构事实；不能据此推断可以低比特存储该 cache，也不能推断其它实现（如任意框架）采用相同 cache 布局。
