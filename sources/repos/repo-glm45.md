---
{
  "id": "repo-glm45",
  "title": "zai-org/GLM-4.5（官方权重与配置）",
  "category": "repo",
  "url": "https://huggingface.co/zai-org/GLM-4.5",
  "version": "cbb2c7cfb52fa128a9660cb1a7a78e017899e115",
  "version_kind": "commit",
  "retrieved_at": "2026-10-26",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "config.json（main 分支，commit cbb2c7c）",
  "topic": "implementations",
  "excerpt_sha256": null
}
---
## 这份来源是什么

GLM-4.5 官方权重仓库（HuggingFace `zai-org/GLM-4.5`）。本库登记它的用途是**把论文的结构描述落到具体 checkpoint 的配置字段**，为 `model-glm45` 的"实现事实"提供固定版本依据。与官方推理代码仓分工：本页只覆盖权重仓的 `config.json`。

范围边界：只读了一个文件（`config.json`），不含建模代码，因此不能证明 forward 的数值行为。

## 已读定位与阅读形式

- 阅读形式：仓库文件列表 + `config.json`（`repo-tree`），按 commit `cbb2c7cfb52fa128a9660cb1a7a78e017899e115` 读取。
- 定位：`config.json`（main）。

## 关键事实

- 架构类：`architectures=Glm4MoeForCausalLM`、`model_type=glm4_moe`、`num_hidden_layers=92`、`first_k_dense_replace=3`、`num_nextn_predict_layers=1`。
- attention：`num_attention_heads=96`、`num_key_value_heads=8`、`head_dim=128`、`partial_rotary_factor=0.5`、`use_qk_norm=true`、`attention_bias=true`、`attention_dropout=0.0`、`rope_theta=1000000`、`rope_scaling=null`。
- MoE：`n_routed_experts=160`、`n_shared_experts=1`、`num_experts_per_tok=8`、`moe_intermediate_size=1536`、`routed_scaling_factor=2.5`、`norm_topk_prob=true`、`n_group=1`、`topk_group=1`。
- 其它：`hidden_size=5120`、`intermediate_size=12288`、`hidden_act=silu`、`rms_norm_eps=1e-05`、`max_position_embeddings=131072`、`vocab_size=151552`、`tie_word_embeddings=false`、`torch_dtype=bfloat16`。

## 不能推出的结论

- 只读 `config.json`，**没有读建模代码**：`partial_rotary_factor=0.5` 给出旋转比例，但 QK-Norm 的实施位置、sigmoid gate 的具体形式、MoE 路由顺序都需按 `modeling_*.py` 核对；config 字段存在不等于运行时一定命中。
- 权重仓配置不证明某个推理后端支持这些结构：partial RoPE、QK-Norm、MoE 融合在各 runtime 的支持程度不同。
- `torch_dtype=bfloat16` 是权重存储类型，不代表推理时的 activation 或 cache 精度。
- 该文件是读取时 main 的 HEAD 内容，会随上游更新；引用字段前须按目标 revision 核对。
