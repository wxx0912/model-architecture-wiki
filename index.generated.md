# 页面索引

> 由 `python3 scripts/build_index.py` 生成，请勿手改。

## 范式

- `paradigm-ar` — [Autoregressive Transformer](wiki/paradigms/ar.md)
- `paradigm-dit` — [Diffusion Transformer (DiT)](wiki/paradigms/dit.md)
- `paradigm-dllm` — [Diffusion Language Model](wiki/paradigms/dllm.md)
- `paradigm-latent-moe` — [LatentMoE Terminology and Variants](wiki/paradigms/latent-moe.md)
- `paradigm-latent-moe-nvidia` — [NVIDIA LatentMoE](wiki/paradigms/latent-moe-nvidia.md)
- `paradigm-linear-attention` — [Linear Attention and Gated State Updates](wiki/paradigms/linear-attention.md)
- `paradigm-moe` — [Sparse Mixture of Experts](wiki/paradigms/moe.md)
- `paradigm-multihead-latent-moe` — [Multi-Head LatentMoE](wiki/paradigms/multihead-latent-moe.md)
- `paradigm-multimodal` — [Multimodal Transformer and Omni Models](wiki/paradigms/multimodal.md)
- `paradigm-ssm` — [Selective State Space Model](wiki/paradigms/ssm.md)

## 组件

- `component-attention` — [Attention: MHA, GQA and MLA](wiki/components/attention.md)
- `component-attention-residuals` — [Attention Residuals (AttnRes)](wiki/components/attention-residuals.md)
- `component-diffusion-timestep` — [Diffusion Time/Noise and Discrete Mask State](wiki/components/diffusion-timestep.md)
- `component-embedding-lm-head` — [Token Embedding and LM Head](wiki/components/embedding-lm-head.md)
- `component-kv-cache` — [KV Cache](wiki/components/kv-cache.md)
- `component-mlp` — [MLP and Gated FFN](wiki/components/mlp.md)
- `component-multimodal-connector` — [Multimodal Encoder and Connector](wiki/components/multimodal-connector.md)
- `component-normalization` — [RMSNorm and Normalization Boundaries](wiki/components/normalization.md)
- `component-position-encoding` — [RoPE and Position Encoding](wiki/components/position-encoding.md)
- `component-residual-mhc` — [Residual, Hyper-Connections and mHC](wiki/components/residual-mhc.md)
- `component-router-expert` — [MoE Router, Dispatch and Expert MLP](wiki/components/router-expert.md)

## 模型

- `model-deepseek` — [DeepSeek-V2 / DeepSeek-V3](wiki/models/deepseek.md)
- `model-glm45` — [GLM-4.5 / GLM-4.5-Air](wiki/models/glm45.md)
- `model-kimi-linear` — [Kimi Linear](wiki/models/kimi-linear.md)
- `model-llada` — [LLaDA](wiki/models/llada.md)
- `model-mamba2` — [Mamba-2](wiki/models/mamba2.md)
- `model-qwen2-vl` — [Qwen2-VL](wiki/models/qwen2-vl.md)

## 核查

- `check-calibration-axes` — [Calibration Axes by Architecture](wiki/checks/calibration-axes.md)
- `check-deployment-contract` — [Quantization Deployment Contract](wiki/checks/deployment-contract.md)
- `check-quantization-preflight` — [Quantization Preflight](wiki/checks/quantization-preflight.md)
- `check-structure-identification` — [Model Structure Identification](wiki/checks/structure-identification.md)

## 按结构特征检索

> 按页面标签聚合，一个页面可出现在多组；完整标签索引见下节。

- **范式（AR / DiT / dLLM）**（ar、dit、dllm）：[Diffusion Time/Noise and Discrete Mask State](wiki/components/diffusion-timestep.md)；[Token Embedding and LM Head](wiki/components/embedding-lm-head.md)；[DeepSeek-V2 / DeepSeek-V3](wiki/models/deepseek.md)；[GLM-4.5 / GLM-4.5-Air](wiki/models/glm45.md)；[LLaDA](wiki/models/llada.md)；[Autoregressive Transformer](wiki/paradigms/ar.md)；[Diffusion Transformer (DiT)](wiki/paradigms/dit.md)；[Diffusion Language Model](wiki/paradigms/dllm.md)；[Multimodal Transformer and Omni Models](wiki/paradigms/multimodal.md)
- **注意力布局**（attention、mha、gqa、mla）：[Attention: MHA, GQA and MLA](wiki/components/attention.md)；[Attention Residuals (AttnRes)](wiki/components/attention-residuals.md)；[KV Cache](wiki/components/kv-cache.md)；[RoPE and Position Encoding](wiki/components/position-encoding.md)；[DeepSeek-V2 / DeepSeek-V3](wiki/models/deepseek.md)；[GLM-4.5 / GLM-4.5-Air](wiki/models/glm45.md)；[Kimi Linear](wiki/models/kimi-linear.md)；[LLaDA](wiki/models/llada.md)；[Qwen2-VL](wiki/models/qwen2-vl.md)；[Autoregressive Transformer](wiki/paradigms/ar.md)；[Diffusion Transformer (DiT)](wiki/paradigms/dit.md)
- **位置编码**（rope）：[Attention: MHA, GQA and MLA](wiki/components/attention.md)；[RoPE and Position Encoding](wiki/components/position-encoding.md)；[GLM-4.5 / GLM-4.5-Air](wiki/models/glm45.md)；[Qwen2-VL](wiki/models/qwen2-vl.md)；[Autoregressive Transformer](wiki/paradigms/ar.md)
- **归一化与嵌入**（normalization、rmsnorm、embedding、lm-head）：[Token Embedding and LM Head](wiki/components/embedding-lm-head.md)；[Multimodal Encoder and Connector](wiki/components/multimodal-connector.md)；[RMSNorm and Normalization Boundaries](wiki/components/normalization.md)；[GLM-4.5 / GLM-4.5-Air](wiki/models/glm45.md)；[Qwen2-VL](wiki/models/qwen2-vl.md)
- **MLP 与门控激活**（mlp、swiglu、geglu）：[MLP and Gated FFN](wiki/components/mlp.md)；[MoE Router, Dispatch and Expert MLP](wiki/components/router-expert.md)；[Diffusion Transformer (DiT)](wiki/paradigms/dit.md)；[Sparse Mixture of Experts](wiki/paradigms/moe.md)
- **MoE 路由与专家**（moe、latent-moe、routing）：[MoE Router, Dispatch and Expert MLP](wiki/components/router-expert.md)；[DeepSeek-V2 / DeepSeek-V3](wiki/models/deepseek.md)；[GLM-4.5 / GLM-4.5-Air](wiki/models/glm45.md)；[LatentMoE Terminology and Variants](wiki/paradigms/latent-moe.md)；[NVIDIA LatentMoE](wiki/paradigms/latent-moe-nvidia.md)；[Sparse Mixture of Experts](wiki/paradigms/moe.md)；[Multi-Head LatentMoE](wiki/paradigms/multihead-latent-moe.md)
- **线性注意力与 SSM**（linear-attention、ssm、mamba、gdn、state-update）：[Kimi Linear](wiki/models/kimi-linear.md)；[Mamba-2](wiki/models/mamba2.md)；[Linear Attention and Gated State Updates](wiki/paradigms/linear-attention.md)；[Selective State Space Model](wiki/paradigms/ssm.md)
- **残差与超连接**（residual、mhc、attnres）：[Attention Residuals (AttnRes)](wiki/components/attention-residuals.md)；[Residual, Hyper-Connections and mHC](wiki/components/residual-mhc.md)
- **KV cache 与部署接口**（kv-cache、deployment、packing）：[Attention: MHA, GQA and MLA](wiki/components/attention.md)；[KV Cache](wiki/components/kv-cache.md)；[Autoregressive Transformer](wiki/paradigms/ar.md)
- **扩散与掩码生成阶段**（diffusion、timestep、mask-state）：[Diffusion Time/Noise and Discrete Mask State](wiki/components/diffusion-timestep.md)；[LLaDA](wiki/models/llada.md)；[Diffusion Transformer (DiT)](wiki/paradigms/dit.md)；[Diffusion Language Model](wiki/paradigms/dllm.md)；[Multimodal Transformer and Omni Models](wiki/paradigms/multimodal.md)
- **多模态连接**（multimodal、multimodal-connector）：[Multimodal Encoder and Connector](wiki/components/multimodal-connector.md)；[Qwen2-VL](wiki/models/qwen2-vl.md)；[Multimodal Transformer and Omni Models](wiki/paradigms/multimodal.md)

## 按机制标签

- `ar`：[Token Embedding and LM Head](wiki/components/embedding-lm-head.md)；[DeepSeek-V2 / DeepSeek-V3](wiki/models/deepseek.md)；[GLM-4.5 / GLM-4.5-Air](wiki/models/glm45.md)；[Autoregressive Transformer](wiki/paradigms/ar.md)；[Multimodal Transformer and Omni Models](wiki/paradigms/multimodal.md)
- `attention`：[Model Structure Identification](wiki/checks/structure-identification.md)；[Attention Residuals (AttnRes)](wiki/components/attention-residuals.md)；[Attention: MHA, GQA and MLA](wiki/components/attention.md)；[KV Cache](wiki/components/kv-cache.md)；[RoPE and Position Encoding](wiki/components/position-encoding.md)；[DeepSeek-V2 / DeepSeek-V3](wiki/models/deepseek.md)；[LLaDA](wiki/models/llada.md)；[Qwen2-VL](wiki/models/qwen2-vl.md)；[Autoregressive Transformer](wiki/paradigms/ar.md)；[Diffusion Transformer (DiT)](wiki/paradigms/dit.md)
- `attnres`：[Attention Residuals (AttnRes)](wiki/components/attention-residuals.md)
- `calibration`：[Calibration Axes by Architecture](wiki/checks/calibration-axes.md)；[Quantization Deployment Contract](wiki/checks/deployment-contract.md)；[Quantization Preflight](wiki/checks/quantization-preflight.md)；[Model Structure Identification](wiki/checks/structure-identification.md)；[Diffusion Time/Noise and Discrete Mask State](wiki/components/diffusion-timestep.md)；[LLaDA](wiki/models/llada.md)；[Diffusion Language Model](wiki/paradigms/dllm.md)；[Linear Attention and Gated State Updates](wiki/paradigms/linear-attention.md)
- `deployment`：[Quantization Deployment Contract](wiki/checks/deployment-contract.md)；[Quantization Preflight](wiki/checks/quantization-preflight.md)
- `diffusion`：[Diffusion Time/Noise and Discrete Mask State](wiki/components/diffusion-timestep.md)；[LLaDA](wiki/models/llada.md)；[Diffusion Transformer (DiT)](wiki/paradigms/dit.md)；[Diffusion Language Model](wiki/paradigms/dllm.md)；[Multimodal Transformer and Omni Models](wiki/paradigms/multimodal.md)
- `dit`：[Diffusion Time/Noise and Discrete Mask State](wiki/components/diffusion-timestep.md)；[Diffusion Transformer (DiT)](wiki/paradigms/dit.md)
- `dllm`：[Diffusion Time/Noise and Discrete Mask State](wiki/components/diffusion-timestep.md)；[Token Embedding and LM Head](wiki/components/embedding-lm-head.md)；[LLaDA](wiki/models/llada.md)；[Diffusion Language Model](wiki/paradigms/dllm.md)
- `embedding`：[Token Embedding and LM Head](wiki/components/embedding-lm-head.md)；[Multimodal Encoder and Connector](wiki/components/multimodal-connector.md)；[Qwen2-VL](wiki/models/qwen2-vl.md)
- `gdn`：[Kimi Linear](wiki/models/kimi-linear.md)；[Linear Attention and Gated State Updates](wiki/paradigms/linear-attention.md)
- `geglu`：[MLP and Gated FFN](wiki/components/mlp.md)
- `gqa`：[Attention: MHA, GQA and MLA](wiki/components/attention.md)；[KV Cache](wiki/components/kv-cache.md)；[GLM-4.5 / GLM-4.5-Air](wiki/models/glm45.md)
- `kv-cache`：[Attention: MHA, GQA and MLA](wiki/components/attention.md)；[KV Cache](wiki/components/kv-cache.md)；[Autoregressive Transformer](wiki/paradigms/ar.md)
- `latent-moe`：[NVIDIA LatentMoE](wiki/paradigms/latent-moe-nvidia.md)；[LatentMoE Terminology and Variants](wiki/paradigms/latent-moe.md)；[Multi-Head LatentMoE](wiki/paradigms/multihead-latent-moe.md)
- `linear-attention`：[Kimi Linear](wiki/models/kimi-linear.md)；[Linear Attention and Gated State Updates](wiki/paradigms/linear-attention.md)
- `lm-head`：[Token Embedding and LM Head](wiki/components/embedding-lm-head.md)
- `mamba`：[Mamba-2](wiki/models/mamba2.md)；[Selective State Space Model](wiki/paradigms/ssm.md)
- `mask-state`：[Diffusion Time/Noise and Discrete Mask State](wiki/components/diffusion-timestep.md)
- `mha`：[Attention: MHA, GQA and MLA](wiki/components/attention.md)
- `mhc`：[Residual, Hyper-Connections and mHC](wiki/components/residual-mhc.md)
- `mla`：[Attention: MHA, GQA and MLA](wiki/components/attention.md)；[KV Cache](wiki/components/kv-cache.md)；[DeepSeek-V2 / DeepSeek-V3](wiki/models/deepseek.md)；[Kimi Linear](wiki/models/kimi-linear.md)
- `mlp`：[MLP and Gated FFN](wiki/components/mlp.md)；[MoE Router, Dispatch and Expert MLP](wiki/components/router-expert.md)；[Diffusion Transformer (DiT)](wiki/paradigms/dit.md)；[Sparse Mixture of Experts](wiki/paradigms/moe.md)
- `moe`：[Model Structure Identification](wiki/checks/structure-identification.md)；[MoE Router, Dispatch and Expert MLP](wiki/components/router-expert.md)；[DeepSeek-V2 / DeepSeek-V3](wiki/models/deepseek.md)；[GLM-4.5 / GLM-4.5-Air](wiki/models/glm45.md)；[NVIDIA LatentMoE](wiki/paradigms/latent-moe-nvidia.md)；[LatentMoE Terminology and Variants](wiki/paradigms/latent-moe.md)；[Sparse Mixture of Experts](wiki/paradigms/moe.md)；[Multi-Head LatentMoE](wiki/paradigms/multihead-latent-moe.md)
- `multimodal`：[Model Structure Identification](wiki/checks/structure-identification.md)；[Multimodal Encoder and Connector](wiki/components/multimodal-connector.md)；[Qwen2-VL](wiki/models/qwen2-vl.md)；[Multimodal Transformer and Omni Models](wiki/paradigms/multimodal.md)
- `multimodal-connector`：[Multimodal Encoder and Connector](wiki/components/multimodal-connector.md)
- `normalization`：[RMSNorm and Normalization Boundaries](wiki/components/normalization.md)
- `packing`：[Quantization Deployment Contract](wiki/checks/deployment-contract.md)
- `quantization`：[Calibration Axes by Architecture](wiki/checks/calibration-axes.md)；[Quantization Deployment Contract](wiki/checks/deployment-contract.md)；[Quantization Preflight](wiki/checks/quantization-preflight.md)；[Model Structure Identification](wiki/checks/structure-identification.md)；[Attention Residuals (AttnRes)](wiki/components/attention-residuals.md)；[Token Embedding and LM Head](wiki/components/embedding-lm-head.md)；[KV Cache](wiki/components/kv-cache.md)；[MLP and Gated FFN](wiki/components/mlp.md)；[Multimodal Encoder and Connector](wiki/components/multimodal-connector.md)；[RMSNorm and Normalization Boundaries](wiki/components/normalization.md)；[RoPE and Position Encoding](wiki/components/position-encoding.md)；[Residual, Hyper-Connections and mHC](wiki/components/residual-mhc.md)；[MoE Router, Dispatch and Expert MLP](wiki/components/router-expert.md)；[DeepSeek-V2 / DeepSeek-V3](wiki/models/deepseek.md)；[GLM-4.5 / GLM-4.5-Air](wiki/models/glm45.md)；[Kimi Linear](wiki/models/kimi-linear.md)；[LLaDA](wiki/models/llada.md)；[Mamba-2](wiki/models/mamba2.md)；[Qwen2-VL](wiki/models/qwen2-vl.md)；[Autoregressive Transformer](wiki/paradigms/ar.md)；[Diffusion Transformer (DiT)](wiki/paradigms/dit.md)；[Diffusion Language Model](wiki/paradigms/dllm.md)；[NVIDIA LatentMoE](wiki/paradigms/latent-moe-nvidia.md)；[LatentMoE Terminology and Variants](wiki/paradigms/latent-moe.md)；[Linear Attention and Gated State Updates](wiki/paradigms/linear-attention.md)；[Sparse Mixture of Experts](wiki/paradigms/moe.md)；[Multi-Head LatentMoE](wiki/paradigms/multihead-latent-moe.md)；[Multimodal Transformer and Omni Models](wiki/paradigms/multimodal.md)；[Selective State Space Model](wiki/paradigms/ssm.md)
- `quantization-preflight`：[Quantization Preflight](wiki/checks/quantization-preflight.md)
- `residual`：[Attention Residuals (AttnRes)](wiki/components/attention-residuals.md)；[Residual, Hyper-Connections and mHC](wiki/components/residual-mhc.md)
- `rmsnorm`：[RMSNorm and Normalization Boundaries](wiki/components/normalization.md)；[GLM-4.5 / GLM-4.5-Air](wiki/models/glm45.md)
- `rope`：[Attention: MHA, GQA and MLA](wiki/components/attention.md)；[RoPE and Position Encoding](wiki/components/position-encoding.md)；[GLM-4.5 / GLM-4.5-Air](wiki/models/glm45.md)；[Qwen2-VL](wiki/models/qwen2-vl.md)；[Autoregressive Transformer](wiki/paradigms/ar.md)
- `routing`：[Calibration Axes by Architecture](wiki/checks/calibration-axes.md)；[MoE Router, Dispatch and Expert MLP](wiki/components/router-expert.md)；[DeepSeek-V2 / DeepSeek-V3](wiki/models/deepseek.md)；[GLM-4.5 / GLM-4.5-Air](wiki/models/glm45.md)；[NVIDIA LatentMoE](wiki/paradigms/latent-moe-nvidia.md)；[LatentMoE Terminology and Variants](wiki/paradigms/latent-moe.md)；[Sparse Mixture of Experts](wiki/paradigms/moe.md)；[Multi-Head LatentMoE](wiki/paradigms/multihead-latent-moe.md)
- `ssm`：[Model Structure Identification](wiki/checks/structure-identification.md)；[Mamba-2](wiki/models/mamba2.md)；[Selective State Space Model](wiki/paradigms/ssm.md)
- `state-update`：[Calibration Axes by Architecture](wiki/checks/calibration-axes.md)；[Kimi Linear](wiki/models/kimi-linear.md)；[Mamba-2](wiki/models/mamba2.md)；[Linear Attention and Gated State Updates](wiki/paradigms/linear-attention.md)；[Selective State Space Model](wiki/paradigms/ssm.md)
- `swiglu`：[MLP and Gated FFN](wiki/components/mlp.md)
- `timestep`：[Calibration Axes by Architecture](wiki/checks/calibration-axes.md)；[Diffusion Time/Noise and Discrete Mask State](wiki/components/diffusion-timestep.md)；[Diffusion Transformer (DiT)](wiki/paradigms/dit.md)；[Diffusion Language Model](wiki/paradigms/dllm.md)
