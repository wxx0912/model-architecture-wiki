---
{
  "id": "repo-kimi-linear",
  "title": "MoonshotAI/Kimi-Linear（Kimi Linear 官方技术报告仓库）",
  "category": "repo",
  "url": "https://github.com/MoonshotAI/Kimi-Linear",
  "version": "8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "仓库根目录（README.md、tech_report.pdf、figures/、LICENSE）",
  "topic": "implementations",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

## 这份来源是什么

Kimi Linear 的官方 GitHub 仓库。截至所读 commit，仓库**只有 README、技术报告 PDF、figures/ 图片与 LICENSE，没有训练或推理代码**。因此它在本库里是"技术报告的官方出处与配图来源"，不是 KDA 的实现依据：`models/kimi-linear`、`paradigm-linear-attention`、`component-attention` 里 KDA 的递推式、chunkwise 算法、3:1 混合排布等事实全部来自 `paper-kimi-linear`。

与相邻来源的分工：KDA 的官方 kernel 实现不在本仓库，`paper-kimi-linear` 指向的是 Flash Linear Attention（`fla-org/flash-linear-attention` 的 `fla/ops/kda`）与 HuggingFace checkpoint；本页不覆盖那两处。

## 已读定位与阅读形式

- 阅读形式：GitHub 仓库文件树（`repo-tree`），按 commit `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3` 读取。
- 已读定位：根目录文件清单，共 8 个 blob。

## 关键事实

- 根目录内容是 `LICENSE`、`README.md`、`tech_report.pdf`，以及 `figures/` 下的 `arch.png`、`arxiv.png`、`banner.png`、`logo.png`、`perf_speed.png`（commit `8c1d85e` 的文件树）。
- 该 commit 下不存在 Python/CUDA 源码目录，也没有 `kda` 或 `fla` 相关实现文件。

## 不能推出的结论

- 本仓库不含代码，不能提供任何实现级证据：不能据此判断 KDA kernel 的 chunk 大小、全精度二级 chunking 的具体实现、或推理时的状态布局。
- 仓库存在不等于权重公开，也不能证明某个 checkpoint 的 KDA/MLA 层比例或上下文长度。
- 需要实现层证据时必须转向 Flash Linear Attention 与 HuggingFace 上的 checkpoint 仓库（本库当前未把这两处登记为来源）。
