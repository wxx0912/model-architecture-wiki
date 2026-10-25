---
{
  "id": "paper-dit",
  "title": "Scalable Diffusion Models with Transformers",
  "category": "paper",
  "url": "https://arxiv.org/abs/2212.09748",
  "version": "v2（2023-03-02）",
  "version_kind": "paper-version",
  "retrieved_at": "2026-10-23",
  "read_form": [
    "latex",
    "html"
  ],
  "read_locator": "摘要、§1 与 Figure 2、§3.1、§3.2 与 Figure 3–6、Table 1、§4、§5 与 Figure 7–10、§5.1 Table 2–3、§5.2、§6、附录开头（Table 4）、附录 A（timestep 与 CFG 细节）、附录 D 与 Table 5",
  "topic": "architectures",
  "excerpt_sha256": null
}
---
## 这份来源是什么

Diffusion Transformer（DiT）论文（arXiv 2212.09748，v2；Peebles & Xie）。它把扩散模型的 U-Net 去噪主干换成**在 VAE latent 的 patch 上运行的 Transformer**，并以**前向 Gflops** 为尺度系统研究其可扩展性：更深的层/更宽的宽度/更多的输入 token 都能降低 FID，最大模型 DiT-XL/2 在 class-conditional ImageNet 256×256 上把 FID-50K 降到 2.27。

支撑本库 `paradigm-dit`（主干 = patchify + DiT block + 线性解码器；条件经 adaLN-Zero 调制注入）、`component-diffusion-timestep`（timestep 与类别条件一起进 adaLN 调制，并给出具体 embedding 维度）、`check-calibration-axes`（校准要覆盖采样轨迹）。

边界：论文只在 class-conditional ImageNet 图像生成上验证；VAE 用现成的 Stable Diffusion VAE（论文不训练编码器，但**专门做了一节解码器消融**，见附录 D）；全文不涉及量化、低比特或激活范围统计。

## 已读定位与阅读形式

- 阅读形式：arXiv e-print 的 LaTeX 源 `main.tex`（文件时间 2023-03-02，与 v2 一致）；另读 arXiv abs 页 HTML（用其版本历史与摘要）。v1 2022-12-19、v2 2023-03-02（未再更新）；abs 页 Comments 栏没有会议信息（源里保留 `\def\confName{CVPR}` 等模板残留），本页无法确认实际发表版本。
- 下文 §、Table、Figure 号按 v2 LaTeX 源的浮动体出现顺序。

## 关键事实

- **主干与数据流**（§3.2，Figure 4）：输入是 VAE latent（256×256×3 经下采样 8 得 32×32×4 的 `z`）；patchify 把空间输入线性嵌入成 `T=(I/p)²` 个 `d` 维 token，并加**标准 ViT 的正弦/余弦位置编码**。patch size `p` 减半会让 `T` 变为 4 倍、总 Gflops **至少**变为 4 倍，但对参数量几乎没有影响。
- **解码器**（§3.2）：最后一个 DiT block 之后用最终（adaLN 时为自适应）LayerNorm，把每个 token 线性解码成 `p×p×2C` 张量（`C` 为 latent 通道数），再重排回原始空间布局，同时得到预测噪声与预测的对角协方差。
- **设计空间只有三项**（§3.2 末）：patch size `p∈{2,4,8}`、block 架构、模型规模 S/B/L/XL。
- **四种条件注入 block**（§3.2，Figure 3/5）：① in-context——把 `t`、`c` 的 embedding 当两个额外 token 拼进序列，用未改动的标准 ViT block；② cross-attention——`t`/`c` 拼成长度 2 的独立序列，self-attention 之后加一层 cross-attention，Gflops 开销最大（约 +15%）；③ adaLN——用 `t`、`c` 的 embedding 之和回归每个 block 的 `γ`、`β` 以取代直接学习的 scale/shift，Gflops 增量最小，也是唯一对**所有 token 施加同一函数**的机制；④ adaLN-Zero——在 adaLN 基础上再回归逐维 `α`，作用于 block 内每个残差连接之前，并把 `α` 的 MLP 初始化成恒输出 0，从而把整个 DiT block 初始化为恒等映射。
- **timestep 条件的实现规格**（附录 A）：timestep 用 **256 维频率 embedding**，后接维度等于 transformer hidden size 的**两层 MLP**、激活为 **SiLU**；每个 adaLN 层把 timestep embedding 与 class embedding 之**和**送入 SiLU + 一个线性层，输出维度是 hidden size 的 **4×（adaLN）或 6×（adaLN-Zero）**；核心 transformer 的非线性是 **GELU（tanh 近似）**。
- **block 设计对比**（§5，Figure 5）：同为 XL/2，in-context 119.4、cross-attention 137.6、adaLN 118.6、adaLN-Zero 118.6 Gflops；adaLN-Zero 训练全程优于 cross-attention 与 in-context（400K 步时约为 in-context 的一半），论文另称它显著优于 vanilla adaLN。
- **模型配置**（§3.2，Table 1；Gflops 按 `I=32`、`p=4`）：DiT-S 12 层 / `d=384` / 6 头 / 1.4 Gflops；DiT-B 12 / 768 / 12 / 5.6；DiT-L 24 / 1024 / 16 / 19.7；DiT-XL 28 / 1152 / 16 / 29.1（覆盖 0.3–118.6 Gflops）。参数量（附录 A Table 4）：DiT-S 33M、DiT-B 130M、DiT-L 458M、DiT-XL/2 675M。
- **训练设置**（§4）：class-conditional、ImageNet、latent 空间；AdamW、恒定 lr `1e-4`、无 weight decay、batch 256、只用水平翻转增强；论文称不需要 warmup 与正则化即可训到高性能、未出现常见 loss spike；最后线性层零初始化；全程维护 EMA（decay 0.9999），**所有结果都用 EMA 权重**；所有规模与 patch size 共用同一套超参（论文明确说没调过 lr、decay/warmup 表、Adam β、weight decay）。
- **扩散超参与 VAE**（§4、附录 D）：VAE 是现成的 Stable Diffusion VAE（下采样 8）；扩散超参沿用 ADM（`t_max=1000` 的线性方差表 `1e-4 → 2e-2`、ADM 的协方差参数化与 timestep/label embedding）。附录 D 用 original LDM / ft-MSE / ft-EMA 三种现成解码器消融（编码器相同、只换解码器，无需重训扩散模型），FID 分别为 2.46 / 2.30 / 2.27；**scaling 分析用 ft-MSE，Table 2–3 的最终指标用 ft-EMA**。
- **分类器无关引导**（§3.1）：训练时随机丢掉 `c` 换成学习到的 null embedding；推理 `ε̂ = ε(x_t,∅) + s·(ε(x_t,c) − ε(x_t,∅))`。**CFG 只作用在 latent 的前 3 个通道**上：3 通道 scale=1.5 得 FID-50K 2.27，等效的 4 通道 scale=1.375 得 2.20（附录 A）。
- **评测口径**（§4）：主指标 FID-50K、250 步 DDPM 采样；所有数字导出样本后用 **ADM 的 TensorFlow 评测套件**算出；本节数字除特别说明外不用 CFG；另报 IS/sFID/Precision/Recall。算力用 JAX + TPU-v3 pod，XL/2 在 TPU v3-256、全局 batch 256 下约 5.7 iterations/s。
- **结果**（§5.1，Table 2–3）：256×256 上 DiT-XL/2-G（cfg=1.50）FID **2.27**（此前最好的扩散模型 LDM-4-G 为 3.60；**非扩散的 StyleGAN-XL 为 2.30，故 2.27 只是略优**）；不用引导时 XL/2 为 9.62（附录 A Table 4：这是训到 7M 步的结果，2.352M 步时 10.67，论文称从未观察到 FID 饱和），只训 2.35M 步的 XL/2 仍以 FID 2.55 超过所有先前扩散模型。论文称 XL/2 的 118.6 Gflops 在**同等算力下质量更好**于 latent 空间的 LDM-4（103.6 Gflops），相对像素空间的 ADM（1120 Gflops）与 ADM-U（742 Gflops）省算力得多。512×512 上 XL/2 处理 `64×64×4` latent → 1024 token、524.6 Gflops，训 3M 步、cfg=1.50 时 FID **3.04**（此前最好的扩散模型 ADM 为 3.85，ADM 用 1983 Gflops、ADM-U 2813 Gflops）；同表 StyleGAN-XL 的 2.41 仍优于 DiT，论文结论限定在扩散模型之间。
- **Gflops 而非参数量决定质量**（§5，Figure 6/8）：共训 12 个 DiT（S/B/L/XL × `p∈{8,4,2}`）；固定规模、减小 patch size 时参数量几乎不变（还略降）而质量提升；Gflops 相近的不同配置（如 DiT-S/2 与 DiT-B/4）FID 相近；FID-50K 与模型 Gflops 强负相关。**模型算力不能被采样算力替代**（§5.2，Figure 10）：DiT-L/2 用 1000 步采样每图耗 80.7 Tflops、FID-10K 25.9，仍差于 DiT-XL/2 用 128 步的 15.2 Tflops、FID-10K 23.7。训练算力按 `Gflops × batch × steps × 3` 估计（因子 3 粗略认为反向约为前向的 2 倍）；参数量与 Gflops 统计都不含 VAE（VAE 编解码合 84M 参数）。

## 不能推出的结论

- 这是 **class-conditional ImageNet 图像生成**论文，不能把它的条件注入方式与质量结论直接搬到文本到图像 DiT/MMDiT 或视频 DiT：论文只把 DALL·E 2、Stable Diffusion 的文本条件列为未来工作（§6），并未做实验。
- "超过此前最好"的口径按分辨率不同：256 上论文明确称达到**所有此前生成模型**的最低 FID（含非扩散的 StyleGAN-XL 2.30，DiT 2.27 只以极小差距领先）；512 上论文只限定在扩散模型之间（StyleGAN-XL 2.41 仍优于 DiT 3.04）。引用时不能把 512 的限定套到 256，也不能用 256 的口径宣称 512 也刷新了全部生成模型。
- Gflops 是**给定输入分辨率、patch size 与 token 数**下的前向复杂度，不是模型固有属性；表里 Gflops 都不含 VAE。绝对 FID 不能跨评测实现比较（本文特意统一到 ADM 的 TF 套件），更不能跨 VAE 解码器比较（scaling 用 ft-MSE、最终指标用 ft-EMA）。
- 「adaLN-Zero 最好」只在 4 个 block 变体、XL/2、256×256 图像 class-conditional 这一设定下测得；论文只写"训练四个不同 block 设计的 XL/2 模型"、**未说明是否重复实验**；§5 正文也没写分辨率，该对比的分辨率须由附录 Table 4 的 Image Resolution 列（四行均 `256×256`）确定。
- 论文只用固定的正弦/余弦位置编码与固定 patchify，未探索可学习位置编码、RoPE 或相对位置编码；官方实现用的是 2D sin-cos（`models.py` 的 `get_2d_sincos_pos_embed`），所以不能把它当作"未探索 2D 位置编码"的例证，本库的位置编码页面也不能引用它来区分这些方案。
- 「无需 warmup、未见 loss spike」是这些配置在 TPU、batch 256、恒定学习率下的观察，不能当作任意 DiT 规模的稳定性保证。
- 论文不涉及量化、低比特或激活范围统计；本库关于「timestep 会改变激活范围、校准要覆盖采样轨迹」的结论属推断，需在目标实现上另行测量。
