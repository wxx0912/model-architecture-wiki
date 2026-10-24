---
{
  "id": "repo-dit",
  "title": "facebookresearch/DiT（DiT 官方实现）",
  "category": "repo",
  "url": "https://github.com/facebookresearch/DiT",
  "version": "ed81ce2229091fd4ecc9a223645f95cf379d582b",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "models.py（TimestepEmbedder、LabelEmbedder、DiTBlock、FinalLayer、DiT、initialize_weights、DiT_models）、train.py、sample.py、sample_ddp.py、diffusion/{gaussian_diffusion,respace,timestep_sampler,diffusion_utils}.py",
  "topic": "implementations",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

## 这份来源是什么

DiT（Scalable Diffusion Models with Transformers）的官方 PyTorch 实现。它给出 `paradigm-dit` 与 `component-diffusion-timestep` 需要的实现级事实：Transformer 主干如何接入扩散时间步与类别条件（adaLN-Zero）、隐空间 patch 化与位置编码、以及输出如何拆成噪声与方差。与 `paper-dit` 的分工：论文给方法与实验口径，本页给"官方代码里长什么样"。

## 已读定位与阅读形式

- 阅读形式：GitHub 仓库文件树 + `models.py` 源码（`repo-tree`），按 commit `ed81ce2229091fd4ecc9a223645f95cf379d582b` 读取。
- 已读定位：`models.py` 的类定义与权重初始化、`DiT_models` 配置表、`DiT.forward`；`train.py`/`sample.py`/`sample_ddp.py` 入口；`diffusion/` 下的扩散调度模块。

## 关键事实

- `DiT`（models.py:145）由 `PatchEmbed` → 固定 sin-cos `pos_embed`（`nn.Parameter(..., requires_grad=False)`，models.py:174）→ `depth` 个 `DiTBlock` → `FinalLayer` 组成；前向顺序为 patch 化 → 逐 block `x = block(x, c)` → `final_layer` → `unpatchify`（models.py:233–248）。
- `DiTBlock`（models.py:101）用 `nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)` 作 `norm1/norm2`，条件 `c` 经 `adaLN_modulation = Sequential(SiLU, Linear(hidden, 6*hidden))` 后 `chunk(6)` 出 shift/scale/gate 三组（models.py:107–119）。
- **adaLN-Zero**：`initialize_weights`（models.py:182）把每个 block 的 `adaLN_modulation[-1].weight/bias` 与 `final_layer.adaLN_modulation[-1]`、`final_layer.linear.weight/bias` 全部零初始化（models.py:209–216）。
- 条件输入 `c = t + y`，其中 `t = t_embedder(t)`、`y = y_embedder(y, training)`（models.py:241–243）；`learn_sigma=True` 时 `out_channels = in_channels * 2`（models.py:163–165），即同时预测噪声与方差。
- 12 个规模配置由 `DiT_models` 字典给出（models.py:365）；例如 `DiT_XL_2` = `depth=28, hidden_size=1152, patch_size=2, num_heads=16`（models.py:328–330），另有 XL/L/B/S × patch 2/4/8。
- 结构基础件复用 `timm.models.vision_transformer` 的 `PatchEmbed`、`Attention`、`Mlp`（models.py:16），本仓库不含自研 attention/MLP 实现。
- 训练与采样入口是 `train.py`、`sample.py`、`sample_ddp.py`；扩散过程（噪声调度、重参数化）在 `diffusion/gaussian_diffusion.py` 与 `diffusion/respace.py`。

## 不能推出的结论

- 这是面向类条件 ImageNet 的**单分辨率隐空间 DiT**（条件只有类别标签 + 时间步），没有文本条件、多分辨率或多模态分支；不能把它的 adaLN-Zero 细节直接外推到其它 DiT 变体（如带文本条件或不同归一化的扩散 Transformer）。
- 仓库不含量化/低比特路径（无 fp8/int8/fp4 代码），不能据此推断 DiT 的量化敏感性。
- 本页只读了 `models.py` 的结构与初始化，未逐行核对扩散采样的数值实现（`diffusion/` 只确认文件存在）；引用调度细节应以目标实现为准。
- 仓库存在不等于某个具体 checkpoint 使用默认配置；实际配置须查 README 的下载链接与对应 config/命令。
