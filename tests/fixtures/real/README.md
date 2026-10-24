# Real configuration snapshots

These are small, weight-free `config.json` snapshots fetched on 2026-10-22 through the public Hugging Face resolve endpoint. Each snapshot is pinned by the source repository commit returned in the response header:

| File | Model | HF repository commit | Source |
|---|---|---|---|
| `qwen2_7b_instruct_config.json` | Qwen2-7B-Instruct | `f2826a00ceef68f0f2b946d945ecc0477ce4450c` | https://huggingface.co/Qwen/Qwen2-7B-Instruct/resolve/main/config.json |
| `mixtral_8x7b_instruct_config.json` | Mixtral-8x7B-Instruct-v0.1 | `eba92302a2861cdc0098cc54bc9f17cb2c47eb61` | https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/main/config.json |
| `qwen2_vl_7b_instruct_config.json` | Qwen2-VL-7B-Instruct | `eed13092ef92e448dd6875b2a00151bd3f7db0ac` | https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/resolve/main/config.json |
| `mamba2_130m_config.json` | state-spaces/mamba2-130m | `3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49` | https://huggingface.co/state-spaces/mamba2-130m/resolve/main/config.json |

The snapshots confirm configuration fields only. They do not prove state-dict layout, source execution paths, tensor shapes, kernel behavior, or quantization quality.
