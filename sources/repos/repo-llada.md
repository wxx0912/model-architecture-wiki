---
{
  "id": "repo-llada",
  "title": "ML-GSAI/LLaDA（LLaDA 官方仓库：推理与评测脚本）",
  "category": "repo",
  "url": "https://github.com/ML-GSAI/LLaDA",
  "version": "9182493720ed723ef8031210d85959364e51cbe0",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "README.md、generate.py（generate/var_generate/add_gumbel_noise/get_num_transfer_tokens）、get_log_likelihood.py、chat.py、app.py、eval_llada.py、eval_illada.py、eval_instruct/、opencompass/",
  "topic": "implementations",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

## 这份来源是什么

LLaDA 的官方 GitHub 仓库。它给 `models/llada`、`paradigm-dllm` 提供实现级事实：官方仓库**只放推理与评测脚本，模型定义代码不在这里**——脚本一律通过 HuggingFace Hub 以 `trust_remote_code=True` 加载权重仓库里的 `modeling_*`。与 `paper-llada` 的分工：论文给前向/反向过程与训练口径，本页给"官方代码放在哪、怎么跑"。

## 已读定位与阅读形式

- 阅读形式：GitHub 仓库文件树 + `README.md`、`generate.py`、`get_log_likelihood.py` 源码（`repo-tree`），按 commit `9182493720ed723ef8031210d85959364e51cbe0` 读取。
- 已读定位：根目录脚本与 `eval_instruct/`（改过的 lm-eval-harness）、`opencompass/`（含 `models/dllm.py`）两个评测框架的文件清单。

## 关键事实

- 仓库**不含模型定义**：脚本经 HuggingFace Hub 拉模型（多为 `AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base'|'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)`，如 generate.py:233、get_log_likelihood.py:83、chat.py:9；`eval_reverse.py` 用 `AutoModelForCausalLM`），并注明需要 `transformers==4.38.2`。也就是说建模代码在 HF 模型仓库里，不在本仓库。
- 生成入口是 `generate.py`：`generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0., ...)`（generate.py:95），配套 `add_gumbel_noise`、`get_num_transfer_tokens`（把步数换算成每步解出的 token 数）与变长版 `var_generate`（generate.py:204）。
- 条件似然评估在 `get_log_likelihood.py`；lm-eval-harness 适配在 `eval_llada.py`（`AutoModel.from_pretrained(..., torch_dtype=torch.bfloat16)`，eval_llada.py:78）；另有 `eval_illada.py`、`eval_reverse.py`、`chat.py`、`app.py`。
- README 记录该系列的后续版本线（论文之外）：LLaDA-V、LLaDA 1.5、LLaDA-MoE-7B-A1B、iLLaDA-8B（后者设 `mask_id=5` 即可复用 LLaDA 的推理代码）。

## 不能推出的结论

- 模型结构（层数、头数、KV heads、RoPE、无 causal mask 的实现细节）**不在本仓库**；要核实结构必须查 HuggingFace Hub 上对应 checkpoint 的 `config.json` 与 `modeling_*.py`（本库 `paper-llada` 的模型规格表是论文口径，不等于 Hub 上某个版本的实现）。
- 脚本里的默认超参（`steps=128`、`gen_length=128`、`block_length=128`）是示例默认值，不是论文的评测配置。
- 仓库没有量化/低比特实现；不能据此判断 LLaDA 的量化适配（例如 mask token 在量化推理中的处理）。
- 仓库存在不等于某个 checkpoint 的实际行为；`trust_remote_code=True` 意味着真正执行的建模代码随权重仓库版本变化，必须按目标版本核验。
