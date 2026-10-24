---
{
  "id": "repo-attention-residuals",
  "title": "MoonshotAI/Attention-Residuals（AttnRes 官方资料仓库）",
  "category": "repo",
  "url": "https://github.com/MoonshotAI/Attention-Residuals",
  "version": "85e22310fe5ee860b4a023de312d791de8a5a5e6",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "仓库根目录（README.md、Attention_Residuals.pdf、assets/）",
  "topic": "implementations",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

## 这份来源是什么

Attention Residuals（AttnRes）的官方 GitHub 仓库，由 `paper-attnres` 的标题页给出（MoonshotAI/Attention-Residuals）。截至所读 commit，仓库**只有 README、技术报告 PDF 与图片素材，没有训练或推理代码**。所以它在本书中的作用是"论文的官方出处与配图来源"，不是实现依据：`component-attention-residuals` 与 `models/kimi-linear` 里关于 AttnRes 的结构事实全部来自 `paper-attnres`。

与相邻来源的分工：若要在**实现层**核实 AttnRes 的每层伪 query、RMSNorm、跨层 softmax 与 block 划分，本仓库帮不上忙（该 commit 下没有可读的模型代码）；对照 `repo-tile-kernels`，后者才有可读的 mHC 相关算子。

## 已读定位与阅读形式

- 阅读形式：GitHub 仓库文件树（`repo-tree`），按 commit `85e22310fe5ee860b4a023de312d791de8a5a5e6` 读取。
- 已读定位：根目录文件清单，共 6 个 blob。

## 关键事实

- 仓库根目录只有 `README.md`、`Attention_Residuals.pdf`，以及 `assets/logo.png`、`assets/overview.png`、`assets/scaling_law.png`、`assets/training_dynamics.png`（commit `85e22310` 的文件树）。
- 该地址与 `paper-attnres` 标题页给出的代码入口一致（`https://github.com/MoonshotAI/Attention-Residuals`）。

## 不能推出的结论

- 本仓库不含代码，不能提供任何实现级证据：不能据此判断 AttnRes 的 block 数、伪 query 初始化方式、推理期需要缓存的深度方向状态，或后端 kernel 支持情况。
- `Attention_Residuals.pdf` 只是技术报告的 PDF 呈现，本页未逐字复核其内容；引用结构细节一律以 `paper-attnres` 的正文定位为准。
- 仓库公开不等于权重/checkpoint 公开，也不证明某个 checkpoint 启用了 AttnRes。
