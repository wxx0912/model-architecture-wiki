---
{
  "id": "check-structure-identification",
  "title": "Model Structure Identification",
  "type": "check",
  "tags": [
    "quantization",
    "calibration",
    "attention",
    "moe",
    "multimodal",
    "ssm"
  ],
  "related": [
    "check-quantization-preflight",
    "component-attention",
    "component-router-expert"
  ],
  "sources": [
    "repo-huggingface-transformers",
    "repo-kernelwiki"
  ],
  "confidence": "inferred",
  "evidence_scope": "mixed",
  "aliases": [],
  "updated": "2026-10-23"
}
---

# 模型结构识别

## 目的

把“模型名称”转换成可量化的实际计算图。结构识别必须以当前 checkpoint、配置、源码和运行图为证据，不能只匹配类名或论文架构图。

## 识别顺序

1. 锁定 checkpoint、配置、代码版本、权重索引和 tokenizer/processor。
2. 从 module tree、state dict key 和实际 forward hook 构建模块清单。
3. 识别 block 类型、attention 变体、FFN/gate、norm、residual、router、state/cache、connector 和输出头。
4. 记录每个边界的输入/输出 shape、dtype、布局、融合/共享关系和并行切分。
5. 用最小输入运行一次 reference forward，确认识别结果覆盖真实执行路径。

## 特别容易误判的结构

- fused QKV、fused gate/up 与拆分权重；
- GQA/MLA 的 cache 与普通 MHA cache；
- linear attention/SSM state 与 KV cache；
- MoE 的 shared/routed expert、专家布局和 token 重排；
- DiT 的 timestep/adaLN 与 dLLM 的 mask/step；
- tied embedding/LM head 与多模态 connector；
- mHC/Hyper-Connections 等跨层 residual mixing。

## 输出

输出带证据链接的结构图、模块表、未知项清单和量化边界。对于未能从源码或运行图确认的字段使用 `unknown`，不要用模型系列的默认值填充。

## Profile 辅助

先选最接近的结构 profile，再传入目标 checkpoint 的本地 `config.json`：

```bash
python3 scripts/preflight.py <profile-id> --config <checkpoint>/config.json --checkpoint <name-or-path> --source-revision <commit-or-tag>
```

工具只把配置文件直接给出的字段标记为 `config-confirmed`；由 head 数关系推导的 MHA/GQA 和由专家字段推导的 MoE 标记为 `inferred`。配置字段不能证明运行时代码路径，仍须核对加载类、权重 key 和 reference forward；推断出的量化风险都需要实验验证。

当 Agent 准备进入量化前检查时，应同时提供 state_dict key manifest、reference forward observation 和源码 revision，并运行 `scripts/preflight.py --strict-runtime`。严格模式通过只说明结构证据齐全，不说明量化精度或部署性能已经达标。
