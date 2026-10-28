---
{
  "id": "repo-deepseek-v4",
  "title": "deepseek-ai/DeepSeek-V4 系列权重仓（config.json）",
  "category": "repo",
  "url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro",
  "version": "b5968e9190ef611bbf34a7229255be88a0e937c1",
  "version_kind": "commit",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "deepseek-ai/DeepSeek-V4-Pro@b5968e9 的 config.json；deepseek-ai/DeepSeek-V4-Flash@60d8d70 的 config.json",
  "topic": "implementations",
  "excerpt_sha256": null
}
---
## 这份来源是什么

DeepSeek-V4 系列的官方权重仓（HuggingFace `deepseek-ai/DeepSeek-V4-Pro`、`deepseek-ai/DeepSeek-V4-Flash`）。本库登记它的用途是给出**具体 checkpoint 的配置字段**：论文 §4.2.1 已给出层数、head 数、专家数、`m`、`m′`、attention top-k 等结构量，本页用于核对字段名、checkpoint 侧取值，以及论文未给出的那份数据——**逐层 `compress_ratios` 列表**。它为 `model-deepseek-v4` 的"实现事实"提供固定版本依据。

范围边界：只读了两个权重仓的 `config.json`，不含建模代码与推理实现，因此不能证明 forward 的数值行为与压缩分支的实际切分。

## 已读定位与阅读形式

- 阅读形式：仓库文件列表 + `config.json`（`repo-tree`）。主锚点 commit `b5968e9190ef611bbf34a7229255be88a0e937c1`（`DeepSeek-V4-Pro`），并读取 `60d8d70770c6776ff598c94bb586a859a38244f1`（`DeepSeek-V4-Flash`）。
- 直连 `huggingface.co` 在本环境不可达，以上内容经 `hf-mirror.com` 按上述同一 commit 读取。
- 定位：两个模型的 `config.json`（权重分片未下载）。

## 关键事实

- **两个 config 一致的字段**：`attention_bias=false`、`hidden_act=silu`、`rms_norm_eps=1e-06`、`head_dim=512`、`num_key_value_heads=1`、`qk_rope_head_dim=64`、`index_n_heads=64`、`index_head_dim=128`、`o_lora_rank=1024`、`num_nextn_predict_layers=1`、`hc_mult=4`、`hc_eps=1e-06`、`hc_sinkhorn_iters=20`、`n_shared_experts=1`、`num_experts_per_tok=6`、`num_hash_layers=3`、`sliding_window=128`、`compress_rope_theta=160000`、`scoring_func=sqrtsoftplus`、`topk_method=noaux_tc`、`norm_topk_prob=true`、`swiglu_limit=10.0`、`max_position_embeddings=1048576`、`rope_theta=10000`、`vocab_size=129280`、`torch_dtype=bfloat16`、`expert_dtype=fp4`。
- **`rope_scaling`**：两个 config 都是 `{"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1}`。
- **`quantization_config`**：两个 config 都声明 `{"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "scale_fmt": "ue8m0", "weight_block_size": [128, 128]}`；同时 `expert_dtype=fp4`。
- **DeepSeek-V4-Pro**（`DeepseekV4ForCausalLM` / `model_type=deepseek_v4`）：`num_hidden_layers=61`、`hidden_size=7168`、`num_attention_heads=128`、`q_lora_rank=1536`、`o_groups=16`、`index_topk=1024`、`n_routed_experts=384`、`moe_intermediate_size=3072`、`routed_scaling_factor=2.5`。
- **DeepSeek-V4-Flash**（`DeepseekV4ForCausalLM`）：`num_hidden_layers=43`、`hidden_size=4096`、`num_attention_heads=64`、`q_lora_rank=1024`、`o_groups=8`、`index_topk=512`、`n_routed_experts=256`、`moe_intermediate_size=2048`、`routed_scaling_factor=1.5`。
- **`compress_ratios`**：逐层压缩比列表，Pro 为 62 项、Flash 为 44 项（均比 `num_hidden_layers` 多 1 项），取值 `0` / `4` / `128`。Pro 以 `[128, 128, 4, 128, 4, …]` 开头、以 `0` 结尾；Flash 以 `[0, 0, 4, 128, 4, …]` 开头、以 `0` 结尾。首两项与 `paper-deepseek-v4` §4.2.1 互为佐证：Pro 前两层是 HCA（`128`）、Flash 前两层是纯滑动窗口（`0`）；其后 `4` 对应 CSA、`128` 对应 HCA。
- 两个 config 都**没有** `n_group` / `topk_group` 字段，与 `paper-deepseek-v4` §2.1"去掉路由目标节点数量约束"一致。

## 不能推出的结论

- 只读 `config.json`，**没有读建模代码与官方推理实现**：indexer 与核心注意力如何共享压缩隐向量、压缩窗口如何切分，都须按 `modeling_*.py` 与官方 `inference/` 目录核对；config 字段存在不等于运行时一定命中。
- `compress_ratios` 的长度是 `num_hidden_layers + 1`，多出的那一项对应 MTP 层还是别的位置**未核实**，引用单项时不能直接按层号索引（首两项已由 §4.2.1 确认为 Pro 的 HCA、Flash 的纯滑动窗口）。
- `quantization_config` 只是权重仓声明的推理精度契约（FP8 e4m3 + ue8m0 scale、128×128 block）与 `expert_dtype=fp4` 的存储类型，**不含量化误差数据**；不能据此推断精度损失，也不能推断其它后端是否支持这些格式。
- `max_position_embeddings=1048576` 是 config 声明的上限，配合 `rope_scaling(yarn, factor=16, original=65536)` 使用；这不等于训练序列长度，也不证明任何后端能在 1M 上下文下跑起来。
- 本页读取的是上述 commit 在读取时的取值；上游后续 commit 可能改动字段，引用字段前须按目标 revision 核对。
- 未下载或加载权重，未运行任何前向，不能据此断言显存占用、KV cache 大小或吞吐。
