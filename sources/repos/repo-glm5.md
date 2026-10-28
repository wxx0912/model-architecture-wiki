---
{
  "id": "repo-glm5",
  "title": "zai-org/GLM-5（官方权重与配置）",
  "category": "repo",
  "url": "https://huggingface.co/zai-org/GLM-5",
  "version": "c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2",
  "version_kind": "commit",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "config.json（zai-org/GLM-5@c183ef8）",
  "topic": "implementations",
  "excerpt_sha256": null
}
---
## 这份来源是什么

GLM-5 官方权重仓（HuggingFace `zai-org/GLM-5`）。本库登记它的用途是把 `paper-glm5` 的结构描述落到具体 checkpoint 的配置字段：论文 §2.1 与附录 A Table 10 给出架构超参，权重仓 `config.json` 提供目标 checkpoint 的精确字段口径（如 `qk_head_dim=256`、`kv_lora_rank=512`、`n_routed_experts=256`）。它为 `model-glm5` 的"实现事实"提供固定版本依据。

范围边界：只读了权重仓的 `config.json`，不含建模代码，因此不能证明 forward 的数值行为。

## 已读定位与阅读形式

- 阅读形式：仓库文件列表 + `config.json`（`repo-tree`），按 commit `c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2` 读取。
- 直连 `huggingface.co` 在本环境不可达，以上内容经 `hf-mirror.com` 按同一 commit 读取。
- 定位：`config.json`（main）。

## 关键事实

- 架构类：`architectures=GlmMoeDsaForCausalLM`、`model_type=glm_moe_dsa`、`num_hidden_layers=78`、`first_k_dense_replace=3`、`moe_layer_freq=1`、`num_nextn_predict_layers=1`、`ep_size=1`。
- attention（MLA 形态）：`num_attention_heads=64`、`num_key_value_heads=64`、`head_dim=64`、`qk_head_dim=256`、`qk_nope_head_dim=192`、`qk_rope_head_dim=64`、`v_head_dim=256`、`kv_lora_rank=512`、`q_lora_rank=2048`、`attention_bias=false`、`rope_interleave=true`、`rope_parameters={"rope_theta": 1000000, "rope_type": "default"}`。
- DSA indexer：`index_n_heads=32`、`index_head_dim=128`、`index_topk=2048`、`indexer_rope_interleave=true`。
- MoE：`n_routed_experts=256`、`n_shared_experts=1`、`num_experts_per_tok=8`、`moe_intermediate_size=2048`、`routed_scaling_factor=2.5`、`scoring_func=sigmoid`、`topk_method=noaux_tc`、`norm_topk_prob=true`、`n_group=1`、`topk_group=1`。
- 其它：`hidden_size=6144`、`intermediate_size=12288`、`hidden_act=silu`、`rms_norm_eps=1e-05`、`max_position_embeddings=202752`、`vocab_size=154880`、`tie_word_embeddings=false`、`dtype=bfloat16`、`use_cache=true`、`eos_token_id=[154820, 154827, 154829]`。

## 不能推出的结论

- 只读 `config.json`，**没有读建模代码**：MLA 的投影拆分与 `kv_lora_rank` / `qk_rope_head_dim` 如何组合成 cache、DSA indexer 如何与核心注意力共享 query、Muon Split 在实现里的落点，都须按 `modeling_*.py` 核对；config 字段存在不等于运行时一定命中。
- `num_hidden_layers=78` 与 `paper-glm5` §2.1 的"层数降到 80"不一致，但与论文附录 A Table 10（3 dense + 75 MoE = 78）一致，即不一致发生在论文内部（§2.1 对 Table 10）；引用层数时须说明口径。
- `kv_lora_rank=512` 与 `qk_rope_head_dim=64` 加总为 576，与论文所说"576 维 latent KV cache"数值吻合，但"两者相加就是 cache 维度"是**推断**，config 未直接给出该组合关系。
- `scoring_func=sigmoid` 与 `paper-glm5` §2.1 的 MLA / MTP 讨论无关，但它说明 GLM-5 的路由打分函数是 sigmoid（与 GLM-4.5 一致），不是 DeepSeek-V4 的 `sqrtsoftplus`。
- 以上字段按 commit `c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2` 读取；引用字段前须按目标 revision 核对。
- 权重仓不含任何量化信息，不能据此推断量化精度或后端支持；`dtype=bfloat16` 只是权重存储类型，不代表推理时的 activation 或 cache 精度。
