---
{
  "id": "repo-qwen2-vl",
  "title": "QwenLM/Qwen2-VL（Qwen-VL 系列官方仓库与 qwen-vl-utils；现重定向到 Qwen3-VL）",
  "category": "repo",
  "url": "https://github.com/QwenLM/Qwen2-VL",
  "version": "96588727e44c78b25ba03ea03b8e12f7e64fd0da",
  "version_kind": "commit",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "repo-tree"
  ],
  "read_locator": "README.md、qwen-vl-utils/src/qwen_vl_utils/（`QwenLM/Qwen3-VL`@96588727，即原 `QwenLM/Qwen2-VL` 的 301 重定向目标）",
  "topic": "implementations",
  "requires_external_verification": true,
  "excerpt_sha256": null
}
---

## 这份来源是什么

Qwen-VL 系列的官方仓库。本库以 `repo-qwen2-vl` 登记它，原意是"Qwen2-VL 官方实现"。核实后需要更正口径：**该 URL 现在 301 重定向到 `QwenLM/Qwen3-VL`，且当前仓库不含 Qwen2-VL 的建模代码**——随仓库发布的是视觉输入预处理工具 `qwen-vl-utils` 与文档。Qwen2-VL 的官方建模实现在 HuggingFace transformers 里（见 `repo-huggingface-transformers` 的 `models/qwen2_vl/`）。所以本页在本书中的角色是"Qwen-VL 系列仓库与预处理工具的落点"，而不是结构依据；`models/qwen2-vl` 的结构事实来自 `paper-qwen2-vl`。

## 已读定位与阅读形式

- 阅读形式：GitHub 仓库文件树 + README（`repo-tree`）。因原地址重定向，读取的实际是重定向目标 `QwenLM/Qwen3-VL` 在 commit `96588727e44c78b25ba03ea03b8e12f7e64fd0da` 的内容；这一点必须与"当年 Qwen2-VL 名字下的提交"区分开。
- 已读定位：`qwen-vl-utils/src/qwen_vl_utils/`（预处理工具）、`README.md`；并实测 `https://github.com/QwenLM/Qwen2-VL` 的 HTTP 响应头。

## 关键事实

- 记录条目 `https://github.com/QwenLM/Qwen2-VL` 现在返回 **301**，`location: https://github.com/QwenLM/Qwen3-VL`（`curl -sI` 实测）——仓库已改名/迁移，历史 ID 与 URL 仍可访问但落到新一代上。
- 当前（重定向目标）仓库**不含 Qwen2-VL 的建模文件**：`qwen2_vl/modeling_qwen2_vl.py`、`qwen3_vl/modeling_qwen3_vl.py` 等路径均 404；实际存在的是 `qwen-vl-utils/src/qwen_vl_utils/__init__.py`（预处理工具）与 `README.md`。
- README 是 Qwen-VL 系列的版本线文档：记录 Qwen2.5-VL、Qwen2.5-VL-32B、`qwen-vl-finetune` 微调代码，以及 Qwen2.5-VL 的 AWQ 量化权重发布（3B/7B/72B）；并说明 `qwen-vl-utils`（示例固定为 `qwen-vl-utils==0.0.14`）向后兼容 Qwen2.5-VL，用于按视觉输入施加像素约束。
- 也就是说，本库 `models/qwen2-vl` 里"图像如何被切成 visual token"的实现级落点应在 transformers 的 `models/qwen2_vl/`，本仓库只提供**输入端预处理**（resize / min_pixels / max_pixels 等）。

## 不能推出的结论

- 不能用本仓库证明 Qwen2-VL 的结构：该 commit 下没有 Qwen2-VL 建模代码，结构事实只能来自 `paper-qwen2-vl` 与 `repo-huggingface-transformers`（`models/qwen2_vl/`）。
- 重定向说明仓库已跨代迁移；Qwen3-VL（以及 README 里的 Qwen2.5-VL）与 Qwen2-VL **不是同一代模型**，代码、预处理默认值与配置不能互推。
- README 里"Qwen2.5-VL 发布了 AWQ 权重"只能说明上游发布了某种量化后的权重产物；量化算法本身的证据在量化算法库的来源里，不能由这条 README 得出。
- 本页未运行 `qwen-vl-utils`，不能据此断言目标实现的实际 token 数或预处理行为（预处理默认值会随版本变化）。
- 登记 URL 会随时间继续漂移；引用时必须以 commit 为准，并核对当时的仓库名。
