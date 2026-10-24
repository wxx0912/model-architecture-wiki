---
{
  "id": "repo-mamba",
  "title": "state-spaces/mamba（Mamba / Mamba-2 / Mamba-3 官方实现）",
  "category": "repo",
  "url": "https://github.com/state-spaces/mamba",
  "version": "e9594ce1c732d97440f0332fdc43170a2294dbfa",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "mamba_ssm/modules/{mamba_simple.py,mamba2.py,mamba3.py,block.py,mha.py,mlp.py}、mamba_ssm/models/{config_mamba.py,mixer_seq_simple.py}、mamba_ssm/ops/{selective_scan_interface.py,triton/,tilelang/mamba3/,cute/mamba3/}、csrc/selective_scan/",
  "topic": "implementations",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

## 这份来源是什么

Mamba 系列的官方实现仓库（state-spaces/mamba）。它给 `paradigm-ssm`、`models/mamba2` 与 `component-kv-cache`（SSM 的固定状态）提供实现级事实：三代模块的参数化与默认值、状态是每层固定大小的张量而非随序列增长的 cache、以及算子层分别用 CUDA selective scan / Triton SSD / TileLang+Cute 实现。与 `paper-mamba`、`paper-mamba2` 的分工：论文给方法与结论，本页给"官方代码里有哪些模块、默认超参是什么"。

## 已读定位与阅读形式

- 阅读形式：GitHub 仓库文件树 + `mamba_ssm/modules/` 下三个模块源码（`repo-tree`），按 commit `e9594ce1c732d97440f0332fdc43170a2294dbfa` 读取。
- 已读定位：`mamba_simple.py`、`mamba2.py`、`mamba3.py` 的 `__init__`/`forward`/`step`；`mamba_ssm/ops/` 与 `csrc/selective_scan/` 的文件清单。

## 关键事实

- `Mamba`（mamba_simple.py:31，Mamba-1）：默认 `d_state=16`、`d_conv=4`、`expand=2`，`dt_rank=ceil(d_model/16)`；参数含 `in_proj`（出 `2·d_inner`）、`conv1d`、`x_proj`（出 `dt_rank + 2·d_state`）、`dt_proj`（出 `d_inner`）、`A_log`、`D`、`out_proj`；同时提供整段 `forward` 与单步 `step`（mamba_simple.py:208）。
- `Mamba2`（mamba2.py:37）：默认 `headdim=64`、`ngroups=1`、`chunk_size=256`；`d_inner = expand·d_model`、`nheads = d_ssm // headdim`、`d_in_proj = 2·d_inner + 2·ngroups·d_state + nheads`；带 `RMSNormGated`，并支持张量并行（`ColumnParallelLinear`/`RowParallelLinear`、`world_size` 切分）；前向里 `A = -exp(A_log)`（mamba2.py:182）。
- `Mamba3`（mamba3.py）：默认 `d_state=128`、`expand=2`、`headdim=64`；用 `heavy_tail_activation` 生成数据相关的 `A`（mamba3.py:27–34、:194）；带 `B_norm`/`C_norm`（`RMSNormGated(d_state)`，mamba3.py:126–127）与 MIMO rank、rotary 角度分量；kernel 走 Triton（`mamba_ssm/ops/triton/mamba3/`）以及 TileLang/CuTe 实现（`ops/tilelang/mamba3/`、`ops/cute/mamba3/`）。
- 算子层分工：Mamba-1 的 selective scan 在 `csrc/selective_scan/`（fp32/fp16/bf16 的 fwd/bwd CUDA kernel）；Mamba-2 的 SSD 走 Triton（`ops/triton/ssd_bmm.py`、`ssd_chunk_scan.py`、`ssd_chunk_state.py`、`ssd_state_passing.py`、`ssd_combined.py`）；单步更新在 `ops/triton/selective_state_update.py`。
- 语言模型装配入口是 `mamba_ssm/models/mixer_seq_simple.py`（配 `models/config_mamba.py`）；另有 `modules/block.py`、`modules/mha.py`、`modules/mlp.py` 供混合堆叠。

## 不能推出的结论

- 上面的默认值（`d_state=16/128`、`headdim=64`、`expand=2`、`chunk_size=256`）是**代码默认值**，不代表论文或某个 checkpoint 的实际配置；引用具体模型规模必须看对应 config 与权重。
- 仓库同时含多套 kernel 后端（CUDA / Triton / TileLang / CuTe），本页只读了文件清单与模块参数，**没有逐个核对 kernel 的数值实现、tile 尺寸与精度**；"有实现"不等于与目标硬件/精度对齐。
- 仓库没有量化/低比特路径（`rocm_patch/rocm6_0.patch` 只是 ROCm 6.0 适配补丁），不能据此推断 SSM 状态或投影权重对量化的敏感性；SSM 状态"大小固定"是结构事实，不是"可以低比特存储"的证据。
- 本页不构成任何 checkpoint 的行为证据，也不能替代在目标实现上的运行验证。
