---
{
  "id": "repo-qwen3",
  "title": "Qwen/Qwen3 系列官方权重仓（config.json）",
  "category": "repo",
  "url": "https://huggingface.co/Qwen/Qwen3-235B-A22B",
  "version": "8efa61729e24bd65b1d152b5ab5409052aa80e65",
  "version_kind": "commit",
  "retrieved_at": "2026-10-28",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "Qwen/Qwen3-235B-A22B@8efa617 的 config.json；同 org 的 Qwen3-30B-A3B@ad44e77、Qwen3-32B@9216db5、Qwen3-14B@40c0698、Qwen3-8B@b968826、Qwen3-4B@1cfa9a7、Qwen3-1.7B@70d244c、Qwen3-0.6B@c1899de 的 config.json",
  "topic": "implementations",
  "excerpt_sha256": null
}
---
## 这份来源是什么

Qwen3 系列的官方权重仓（HuggingFace `Qwen/Qwen3-*`）。本库登记它的用途是把论文的结构描述落到**具体 checkpoint 的配置字段**，为 `model-qwen3` 的"实现事实"提供固定版本依据。与 `paper-qwen3` 的分工：论文给系列构成、架构改动与训练口径，本页给"某个 checkpoint 的 `config.json` 里有什么"。

范围边界：只读了权重仓的 `config.json`，不含建模代码，因此不能证明 forward 的数值行为。

## 已读定位与阅读形式

- 阅读形式：仓库文件列表 + `config.json`（`repo-tree`）。主锚点 commit `8efa61729e24bd65b1d152b5ab5409052aa80e65`（`Qwen/Qwen3-235B-A22B`）。
- 同一份登记另覆盖 30B-A3B（`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`）、32B（`9216db5781bf21249d130ec9da846c4624c16137`）、14B（`40c069824f4251a91eefaf281ebe4c544efd3e18`）、8B（`b968826d9c46dd6066d109eabc6255188de91218`）、4B（`1cfa9a7208912126459214e8b04321603b3df60c`）、1.7B（`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`）、0.6B（`c1899de289a04d12100db370d81485cdf75e47ca`）的 `config.json`，各按上列 commit 读取。
- 直连 `huggingface.co` 在本环境不可达，以上内容经 `hf-mirror.com` 按上述同一 commit 读取。
- 定位：各模型的 `config.json`（权重分片未下载、未加载）。

## 关键事实

- **公共字段**（下列 8 个 config 全部一致）：`attention_bias=false`、`head_dim=128`、`hidden_act=silu`、`rms_norm_eps=1e-06`、`max_position_embeddings=40960`、`rope_theta=1000000`、`rope_scaling=null`、`vocab_size=151936`、`torch_dtype=bfloat16`、`use_cache=true`、`use_sliding_window=false`、`sliding_window=null`。
- **Qwen3-8B**（`Qwen3ForCausalLM` / `model_type=qwen3`）：`num_hidden_layers=36`、`hidden_size=4096`、`num_attention_heads=32`、`num_key_value_heads=8`、`intermediate_size=12288`、`tie_word_embeddings=false`。
- **Qwen3-32B**（`Qwen3ForCausalLM`）：`num_hidden_layers=64`、`hidden_size=5120`、`num_attention_heads=64`、`num_key_value_heads=8`、`intermediate_size=25600`、`tie_word_embeddings=false`。
- **Qwen3-14B**（`Qwen3ForCausalLM`）：`num_hidden_layers=40`、`hidden_size=5120`、`num_attention_heads=40`、`num_key_value_heads=8`、`intermediate_size=17408`、`tie_word_embeddings=false`。
- **Qwen3-4B**（`Qwen3ForCausalLM`）：`num_hidden_layers=36`、`hidden_size=2560`、`num_attention_heads=32`、`num_key_value_heads=8`、`intermediate_size=9728`、`tie_word_embeddings=true`。
- **Qwen3-1.7B**（`Qwen3ForCausalLM`）：`num_hidden_layers=28`、`hidden_size=2048`、`num_attention_heads=16`、`num_key_value_heads=8`、`intermediate_size=6144`、`tie_word_embeddings=true`。
- **Qwen3-0.6B**（`Qwen3ForCausalLM`）：`num_hidden_layers=28`、`hidden_size=1024`、`num_attention_heads=16`、`num_key_value_heads=8`、`intermediate_size=3072`、`tie_word_embeddings=true`。
- **Qwen3-30B-A3B**（`Qwen3MoeForCausalLM` / `model_type=qwen3_moe`）：`num_hidden_layers=48`、`hidden_size=2048`、`num_attention_heads=32`、`num_key_value_heads=4`、`intermediate_size=6144`、`moe_intermediate_size=768`、`num_experts=128`、`num_experts_per_tok=8`、`decoder_sparse_step=1`、`norm_topk_prob=true`、`tie_word_embeddings=false`。
- **Qwen3-235B-A22B**（`Qwen3MoeForCausalLM`）：`num_hidden_layers=94`、`hidden_size=4096`、`num_attention_heads=64`、`num_key_value_heads=4`、`intermediate_size=12288`、`moe_intermediate_size=1536`、`num_experts=128`、`num_experts_per_tok=8`、`decoder_sparse_step=1`、`norm_topk_prob=true`、`output_router_logits=false`、`router_aux_loss_coef=0.001`、`mlp_only_layers=[]`、`max_window_layers=94`、`tie_word_embeddings=false`。
- **这些 config 没有 QK-Norm 相关字段**，也没有 MTP / next-token-prediction 相关字段；QK-Norm 只出现在 `paper-qwen3` §2 的叙述里。
- `head_dim` 是显式字段，不等于 `hidden_size / num_attention_heads`：235B-A22B 的 `64 × 128 = 8192` 大于 `hidden_size=4096`。
- 词表：8 个 config 均为 `vocab_size=151936`，与 `paper-qwen3` §2 的 151,669 不同。

## 不能推出的结论

- 只读 `config.json`，**没有读建模代码**：QK-Norm 的实施位置与归一维度、MoE 的 gate 形式与 routing 顺序、`rope_scaling` 在长上下文下的实际展开方式，都须按 `modeling_*.py` 核对；config 字段存在不等于运行时一定命中。
- 配置里**没有 QK-Norm 字段**（与 GLM-4.5 权重仓的 `use_qk_norm=true` 不同），因此不能从权重仓字段确认 Qwen3 是否真的带 QK-Norm，也不能确定它归一的是 Q / K 的哪个维度。
- `max_position_embeddings=40960` 是 config 支持的序列上限，不等于论文 Table 1 / Table 2 的 128K；128K 依赖推理期扩展，引用时须区分。
- `vocab_size=151936` 与论文的 151,669 不一致，未定位差异来源，不能混用。
- 本页读取的内容是上述 commit 在读取时的取值；上游后续 commit 可能改动字段，引用字段前须按目标 revision 核对。
- `torch_dtype=bfloat16` 是权重存储类型，不代表推理时的 activation 或 cache 精度。
- 权重仓不含任何量化信息，不能据此推断量化精度、可量化模块或后端支持。
- 未下载或加载权重，未运行任何前向，不能据此断言显存占用或吞吐。
