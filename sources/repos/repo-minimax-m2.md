---
{
  "id": "repo-minimax-m2",
  "title": "MiniMaxAI/MiniMax-M2（官方权重与配置）",
  "category": "repo",
  "url": "https://huggingface.co/MiniMaxAI/MiniMax-M2",
  "version": "04312f0cfbdeefb912575595b7c045512a29293c",
  "version_kind": "commit",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "config.json（MiniMaxAI/MiniMax-M2@04312f0）；同 commit 的 configuration_minimax_m2.py 与 modeling_minimax_m2.py 中被引用到的字段",
  "topic": "implementations",
  "excerpt_sha256": null
}
---
## 这份来源是什么

MiniMax-M2 官方权重仓（HuggingFace `MiniMaxAI/MiniMax-M2`）。本库登记它的用途是把 `paper-minimax-m2` 的结构描述落到具体 checkpoint 的配置字段，并记录**官方发布的量化格式**（该 checkpoint 的 config 自带 `quantization_config`）。它为 `model-minimax-m2` 的"实现事实"提供固定版本依据。

范围边界：读了权重仓的 `config.json`，并查阅同 commit 的 `configuration_minimax_m2.py` 与 `modeling_minimax_m2.py` 以确认字段如何被引用；未下载或加载权重文件，因此不能证明 forward 的数值行为。

## 已读定位与阅读形式

- 阅读形式：仓库文件列表 + `config.json`（`repo-tree`），并按 commit 核对 `configuration_minimax_m2.py` 与 `modeling_minimax_m2.py` 的字段引用。
- 直连 `huggingface.co` 在本环境不可达，以上内容经 `hf-mirror.com` 按同一 commit `04312f0cfbdeefb912575595b7c045512a29293c` 读取。
- 定位：`config.json` 顶层字段，以及建模代码中 `MiniMaxM2MLP` / `MiniMaxM2Experts` / `MiniMaxM2SparseMoeBlock` / attention 类的字段引用。

## 关键事实

- **顶层**：`architectures=["MiniMaxM2ForCausalLM"]`、`model_type=minimax_m2`、`hidden_act=silu`、`rms_norm_eps=1e-06`、`tie_word_embeddings=false`、`use_cache=true`。
- **主干**：`num_hidden_layers=62`、`hidden_size=3072`、`vocab_size=200064`、`max_position_embeddings=196608`、`sliding_window=null`。
- **attention**：`num_attention_heads=48`、`num_key_value_heads=8`（GQA）、`head_dim=128`、`rotary_dim=64`、`rope_theta=5000000`、`use_qk_norm=true`、`qk_norm_type=per_layer`、`attention_dropout=0.0`。
- **MoE**：`num_local_experts=256`、`num_experts_per_tok=8`、`intermediate_size=1536`、`mlp_intermediate_size=8192`、`shared_intermediate_size=0`、`scoring_func=sigmoid`、`use_routing_bias=true`、`router_aux_loss_coef=0.001`、`router_jitter_noise=0.0`、`output_router_logits=false`、`shared_moe_mode=sigmoid`。
- **MTP**：`use_mtp=true`、`num_mtp_modules=3`、`mtp_transformer_layers=1`。
- **注意力类型表**：`attn_type_list` 为 62 个 `1`（与论文"全层 full attention"一致）。
- **quantization_config**：`quant_method=fp8`、`fmt=float8_e4m3fn`、`activation_scheme=dynamic`、`weight_block_size=[128,128]`、`modules_to_not_convert=["gate","e_score_correction_bias","lm_head"]`。
- **建模代码确认的字段语义**：
  - `MiniMaxM2MLP.ffn_dim = config.intermediate_size`，且每个专家是一个 `MiniMaxM2MLP`，因此 **`intermediate_size=1536` 是每个专家的 FFN 中间维**。
  - `MiniMaxM2SparseMoeBlock` 中 `gate = Linear(hidden_size, num_local_experts, bias=False)`，路由先 `sigmoid(logits)`，再 `+ e_score_correction_bias` 选 top-k，最后用 sigmoid 值归一化作为专家权重，对应 `scoring_func=sigmoid` 与 `use_routing_bias=true`。
  - attention 的 `q_norm` / `k_norm` 是作用在 `head_dim × num_heads` 与 `head_dim × num_kv_heads` 整块上的 RMSNorm，不是 per-head。

## 不能推出的结论

- 只读 `config.json` 与少量建模代码字段引用，**未加载权重、未运行前向**：MoE 路由数值行为、QK-Norm 的实际效果、`rope_theta` 在长上下文下的展开方式，都须按完整代码与运行核对。
- `mlp_intermediate_size=8192`、`shared_intermediate_size=0`、`attn_type_list`（62 个 `1`）、`use_mtp` / `num_mtp_modules` / `mtp_transformer_layers`、`qk_norm_type`、`shared_moe_mode` 等字段**未在 `modeling_minimax_m2.py` 中被引用**；同 commit 的 HF 建模文件不含 MTP 实现（`auto_map` 只映射 `MiniMaxM2ForCausalLM`），也不含 `qk_norm_type` 等字段的逻辑，其含义须按具体推理后端（vLLM / SGLang 等）核对，不能从本页推断。
- `quantization_config` 是**权重发布格式的声明**，不是精度结论：它说明该 checkpoint 按 FP8（`float8_e4m3fn`、block 128×128、动态激活）量化，且 `gate`、`e_score_correction_bias`、`lm_head` 在排除列表里；但**哪些张量真正落在量化集合内、精度损失多少**，都要读权重映射与实际评测，不能由 config 推断。
- `max_position_embeddings=196608` 是配置声明上限，不证明任何后端能在该长度下服务。
- `tie_word_embeddings=false` 等只描述结构声明，不代表推理时的 activation 或 cache 精度（该 config 没有 `dtype` / `torch_dtype` 字段）。
- 以上字段按 commit `04312f0cfbdeefb912575595b7c045512a29293c` 读取；引用字段前须按目标 revision 核对。
