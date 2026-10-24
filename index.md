# 模型结构导航

## 知识关系与 Agent 调用路径

范式页说明模型的主执行路径；组件页解释可复用模块；模型页说明具体模型如何组合这些结构；profile 提供机器可选的代表性结构族；check 页将待核实项转成 preflight。Agent 应先由模型配置选择 profile，再用配置、权重 key 和 reference forward 覆盖通用描述，最后读取关联页面和量化检查。只有目标源码和后端证据才能把待验证建议升级为硬约束。

## 模型范式

- [AR](wiki/paradigms/ar.md)：自回归 next-token 生成、prefill/decode 和 KV Cache。
- [DiT](wiki/paradigms/dit.md)：以 Transformer 去噪器处理扩散时间步和图像/视频 token。
- [dLLM](wiki/paradigms/dllm.md)：掩码或块扩散式语言生成，反复更新 token。
- [MoE](wiki/paradigms/moe.md)：Router、专家选择、dispatch 和稀疏计算。
- [Linear Attention](wiki/paradigms/linear-attention.md)：有限状态更新替代完整 token-token attention。
- [SSM](wiki/paradigms/ssm.md)：选择性状态空间递推。
- [Multimodal](wiki/paradigms/multimodal.md)：视觉/音频编码器与语言主干的连接。

## 通用组件

- [Attention](wiki/components/attention.md)
- [MLP / FFN](wiki/components/mlp.md)
- [Residual and mHC](wiki/components/residual-mhc.md)
- [KV Cache](wiki/components/kv-cache.md)
- [Diffusion timestep](wiki/components/diffusion-timestep.md)
- [Normalization](wiki/components/normalization.md)
- [RoPE / position encoding](wiki/components/position-encoding.md)
- [Embedding / LM head](wiki/components/embedding-lm-head.md)
- [MoE router and experts](wiki/components/router-expert.md)
- [Multimodal connector](wiki/components/multimodal-connector.md)

## 模型案例

- [DeepSeek-V2/V3 family](wiki/models/deepseek.md)
- [LLaDA](wiki/models/llada.md)
- [Kimi Linear](wiki/models/kimi-linear.md)
- [Mamba-2](wiki/models/mamba2.md)
- [Qwen2-VL](wiki/models/qwen2-vl.md)
- [Attention Residuals](wiki/components/attention-residuals.md)

LatentMoE 是歧义术语，检索时必须区分：

- [NVIDIA LatentMoE](wiki/paradigms/latent-moe-nvidia.md)
- [Multi-Head LatentMoE](wiki/paradigms/multihead-latent-moe.md)

## 结构核查

- [Quantization preflight](wiki/checks/quantization-preflight.md)
- [Structure identification](wiki/checks/structure-identification.md)
- [Calibration axes](wiki/checks/calibration-axes.md)
- [Deployment contract](wiki/checks/deployment-contract.md)

## 按量化问题检索

- [Calibration axes](wiki/checks/calibration-axes.md)：按 AR、DiT、dLLM、MoE、SSM 和多模态组织校准覆盖。
- [Quantization preflight](wiki/checks/quantization-preflight.md)：记录可量化模块、联合约束、候选排除和证据。
- [Deployment contract](wiki/checks/deployment-contract.md)：过滤不满足目标格式、packing 或 kernel 的候选。

## 机器可读入口

- `data/architecture-profiles.json`：代表性结构族、来源入口和未锁定字段。
- `python3 scripts/preflight.py <profile-id>`：生成“结构特征—潜在量化风险—检查方法—候选行动”清单；输出中的风险仍标记为待实验验证。
- `python3 scripts/preflight.py --config <checkpoint>/config.json`：按 `model_type` 自动选择唯一 profile，并将配置字段和 JSON 路径写入证据。

[完整稳定 ID 与标签索引](index.generated.md) · [来源层](sources/README.md) · [来源页规范](sources/SPEC.md)
