---
{
  "id": "repo-kimik3",
  "title": "moonshotai/Kimi-K3（官方权重与配置）",
  "category": "repo",
  "url": "https://huggingface.co/moonshotai/Kimi-K3",
  "version": "f831ab66814297da540d832a5235f8e904f29d06",
  "version_kind": "commit",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "config.json（moonshotai/Kimi-K3@f831ab6，含 text_config、vision_config、quantization_config）",
  "topic": "implementations",
  "excerpt_sha256": null
}
---
## 这份来源是什么

Kimi K3 官方权重仓（HuggingFace `moonshotai/Kimi-K3`）。本库登记它的用途有两个：一是把 `paper-kimik3` 的结构描述落到具体 checkpoint 的字段（层数、head 数、latent rank、哪几层是 KDA / 哪几层是 MLA、MoE 维度），二是记录**官方发布的量化格式**——该 checkpoint 的 config 自带 `quantization_config`。它为 `model-kimi-k3` 的"实现事实"提供固定版本依据。

范围边界：只读了权重仓的 `config.json`（含嵌套 `text_config`、`vision_config`、`quantization_config`），不含建模代码与权重文件，因此不能证明 forward 的数值行为，也不能确认哪些张量真正按该格式存储。

## 已读定位与阅读形式

- 阅读形式：仓库文件列表 + `config.json`（`repo-tree`），按 commit `f831ab66814297da540d832a5235f8e904f29d06` 读取。
- 直连 `huggingface.co` 在本环境不可达，以上内容经 `hf-mirror.com` 按同一 commit 读取。
- 定位：`config.json` 的顶层字段与 `text_config` / `vision_config` / `quantization_config` 三块。

## 关键事实

- **顶层**：`architectures=["KimiK3ForConditionalGeneration"]`、`model_type=kimi_k3`、`dtype=bfloat16`、`tie_word_embeddings=false`、`vocab_size` 侧的 `bos/eos/pad_token_id` 分别为 163584 / 163586 / 163839。
- **text_config**：`architectures=["KimiLinearForCausalLM"]`、`model_type=kimi_linear`、`num_hidden_layers=93`、`hidden_size=7168`、`intermediate_size=33792`、`num_attention_heads=96`、`num_key_value_heads=96`、`first_k_dense_replace=1`、`moe_layer_freq=1`、`vocab_size=163840`、`max_position_embeddings=1048576`、`rms_norm_eps=1e-05`、`num_nextn_predict_layers=0`、`attn_res_block_size=12`。
- **MLA 字段**（text_config）：`kv_lora_rank=512`、`q_lora_rank=1536`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`、`mla_use_nope=true`、`mla_use_output_gate=true`。
- **KDA / 线性注意力字段**（`linear_attn_config`）：`head_dim=128`、`num_heads=96`、`gate_lower_bound=-5.0`、`use_full_rank_gate=true`、`short_conv_kernel_size=4`；并**逐层列出** `full_attn_layers=[4, 8, 12, …, 92, 93]`（共 24 层）与 `kda_layers=[1, 2, 3, 5, 6, 7, …, 89, 90, 91]`（共 69 层）。
- **Stable LatentMoE 字段**（text_config）：`num_experts=896`、`num_experts_per_token=16`、`num_shared_experts=2`、`moe_intermediate_size=3072`、`routed_expert_hidden_size=3584`、`routed_scaling_factor=1.0`、`moe_router_activation_func=sigmoid`、`moe_renormalize=true`、`latent_moe_use_norm=true`、`topk_method=noaux_tc`、`use_grouped_topk=true`、`num_expert_group=1`、`topk_group=1`。
- **激活函数**（text_config）：`hidden_act=situ`、`activation_situ_beta=4.0`、`activation_situ_linear_beta=25.0`。
- **vision_config**（MoonViT-V2）：`patch_size=14`、`vt_num_hidden_layers=27`、`vt_hidden_size=1024`、`vt_intermediate_size=4096`、`vt_num_attention_heads=12`、`qkv_hidden_size=1536`、`mlp_type=mlp2`、`norm_type=rmsnorm`、`attn_bias=false`、`linear_bias=false`、`activation_func=gelu_pytorch_tanh`、`merge_type=sd2_tpool`、`merge_kernel_size=[2, 2]`、`mm_projector_type=patchmergerv2`、`mm_hidden_size=1024`、`text_hidden_size=7168`、`init_pos_emb_height=64`、`init_pos_emb_width=64`、`init_pos_emb_time=4`。
- **quantization_config**：`quant_method=compressed-tensors`、`format=mxfp4-pack-quantized`、`quantization_status=compressed`；`config_groups.group_0` 的 `targets=["Linear"]`、权重 `num_bits=4`、`type=float`、`group_size=32`、`strategy=group`、`symmetric=true`、`observer=minmax`、`scale_dtype=torch.uint8`、`dynamic=false`；`ignore` 列表为 `re:.*self_attn.*`、`re:.*shared_experts.*`、`re:.*mlp\.(gate|up|gate_up|down)_proj.*`、`re:.*lm_head.*`、`re:.*vision_tower.*`、`re:.*mm_projector.*`；`kv_cache_scheme=null`、`input_activations=null`、`output_activations=null`。

## 不能推出的结论

- 只读 `config.json`，**没有读建模代码与权重文件**：`full_attn_layers` / `kda_layers` 的层号语义、`routed_expert_hidden_size=3584` 与 `moe_intermediate_size=3072` 的差别（哪个是专家内部维度、哪个是 latent 宽度）、SiTU-GLU 的两个 beta 如何组合，都须按 `modeling_*.py` 核对。
- `quantization_config` 是**权重发布格式的声明**，不是精度结论：它说明该 checkpoint 的 `Linear` 目标按 MXFP4（4-bit float、group size 32、对称、minmax）打包，且 attention、共享专家、若干 `mlp.*_proj`、`lm_head`、视觉塔与投影器在 `ignore` 列表里；但该权重声明**不带激活量化**（`input/output_activations=null`），而 `paper-kimik3` §4.1.4 的 QAT 用 MXFP8 激活——两者指不同对象（发布权重格式 vs 训练/推理的计算精度）。哪些张量真正落在量化集合内、精度损失多少，都要读权重映射与实际评测，不能由 config 推断。
- `num_nextn_predict_layers=0` 说明该 checkpoint 的 config **不含 MTP 模块**；而 `paper-kimik3` 明确说 K3 预训练带 1 层 MTP（Table 1、§4.1.4），两者口径不一致，不能由 config 反推训练期与推测解码是否使用 MTP。
- `mla_use_nope=true` 与论文 §2.1.2 的"MLA 层使用 NoPE"一致，但 config 字段名不等于运行时一定不做位置编码，仍须按建模代码核对。
- `max_position_embeddings=1048576` 是配置声明上限，不证明任何后端能在该长度下服务。
- 以上字段按 commit `f831ab66814297da540d832a5235f8e904f29d06` 读取；引用字段前须按目标 revision 核对。
- 未下载或加载权重，未运行任何前向，不能据此断言显存、吞吐或量化后的精度。
