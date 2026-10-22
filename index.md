# 模型结构导航

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
