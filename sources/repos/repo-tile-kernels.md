---
{
  "id": "repo-tile-kernels",
  "title": "deepseek-ai/TileKernels（DeepSeek 的 TileLang 算子库：mHC / 量化 / MoE 路由等）",
  "category": "repo",
  "url": "https://github.com/deepseek-ai/TileKernels",
  "version": "c1cf914a82e6901c679445f872051ce44ac2cc63",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "README.md、pyproject.toml、tile_kernels/mhc/{__init__.py,sinkhorn_kernel.py,...}、tile_kernels/quant/、tile_kernels/{moe,engram,transform,rand}/、tests/{mhc,quant,moe,engram,transform,rand}/",
  "topic": "implementations",
  "excerpt_sha256": null
}
---

## 这份来源是什么

DeepSeek 开源的高性能算子库 TileKernels（基于 TileLang）。它给 `component-residual-mhc`、`component-attention-residuals`（对照）与量化库的"算子/实现"层提供事实：mHC 的 Sinkhorn 归一化与 mix 拆分/应用等**逐个算子**在代码中确实存在，并且分成 NVIDIA 与 Ascend 两套后端。与 `paper-mhc` 的分工：论文给结构与实验口径，本页给"官方实现里有哪些算子、依赖什么栈"。

## 已读定位与阅读形式

- 阅读形式：GitHub 仓库文件树 + README、`tile_kernels/mhc/__init__.py`、`tile_kernels/mhc/sinkhorn_kernel.py` 源码（`repo-tree`），按 commit `c1cf914a82e6901c679445f872051ce44ac2cc63` 读取。
- 已读定位：根目录与 `tile_kernels/`、`tests/` 的文件清单；`mhc/` 与 `quant/` 的模块导出与函数签名。

## 关键事实

- README 的 Features 列出 7 条：MoE routing、Quantization（per-token / per-block / per-channel 的 FP8/FP4 cast 与反量化、fused SwiGLU+quant）、Engram、Manifold HyperConnection（mHC）、Transform（RoPE）、Rand，外加一条 `Modeling`（Engram gate 的高层 `torch.autograd.Function` 封装，只有该模块有）；README 声明这些算子已用于其内部训练与推理。
- mHC 目录 `tile_kernels/mhc/` 共 9 个算子实现文件（`*_kernel.py`），每个算子由 `*_kernel.py`（实现）+ `*_cuda.py`（NVIDIA 绑定）+ `*_asc.py`（Ascend 绑定）三份组成；`tile_kernels/mhc/__init__.py` 共导出 10 组入口（`norm_fn_kernel.py` 一个文件导出 2 组），导出名：`expand_to_mhc_fwd/bwd`、`mhc_head_compute_mix_fwd/bwd`、`mhc_multilayer_recompute`、`mhc_fn_normw_merge_fwd/bwd`、`mhc_reduce_partials_and_rmsnorm_fwd/bwd`、`mhc_post_fwd/bwd`、`mhc_pre_apply_mix_fwd/bwd`、`mhc_pre_big_fuse_fwd`、`mhc_pre_split_mixes_fwd/bwd`、`mhc_sinkhorn_fwd/bwd`。
- Sinkhorn 算子签名（sinkhorn_kernel.py:6/15）：`mhc_sinkhorn_fwd(comb_res_mix, comb_res_mix_out, repeat: int, eps: float)` 与对应的 `_bwd`——即把 Sinkhorn 迭代次数 `repeat` 与 `eps` 作为显式参数（与 `paper-mhc` 的 `t_max=20` 一致地说明迭代次数可配）。
- 双后端：README 记 2026-09-30 新增华为 Ascend 支持（Ascend 950 NPU / CANN ≥9.2.0），运行时自动选择后端，同一套 Python API 同时跑 NVIDIA GPU 与华为 NPU。
- 依赖（README）：Python ≥3.12、PyTorch ≥2.13、TileLang ≥0.1.15；NVIDIA 路径要求 CUDA Toolkit ≥13.1 且为 SM90 或 SM100。
- `tile_kernels/quant/` 给出量化相关算子（部分成对 fwd/bwd，部分为单向 cast）：`per_token_cast(_and_cast_back)`、`per_block_cast`、`per_block_cast_lossless`、`per_channel_cast`、`cast`/`cast_back`、`batched_transpose_weight_sf`、`norm_forward/backward` 等。

## 不能推出的结论

- **算子存在不等于目标模型启用了它**：仓库里有 mHC 算子只说明官方实现了 mHC 前反向，不证明某个 checkpoint/训练配置真的用了 mHC（须查模型配置与训练代码）。
- 本页只读了 README 与文件清单、以及 sinkhorn 的函数签名，**没有逐个核对 kernel 的数值正确性、tile 尺寸、精度与 scaling 语义**；引用量化算子的具体行为（FP8/FP4 的粒度、scale 布局）必须回到对应 `*_kernel.py` 逐行核对。
- 版本/后端绑定很紧（TileLang ≥0.1.15、CUDA ≥13.1、SM90/SM100、Ascend 950 + CANN ≥9.2.0）；不在该组合上不能假定可直接复用，也不能把某个后端的功能当作另一后端也有。
- 仓库里的 `tests/` 是单元测试，不代表端到端模型的正确性或性能。
