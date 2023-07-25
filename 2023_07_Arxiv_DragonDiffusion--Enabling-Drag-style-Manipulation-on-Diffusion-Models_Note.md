# DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models

> "DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models" Arxiv, 2023 Jul
> [paper](https://arxiv.org/abs/2307.02421) [website](https://mc-e.github.io/project/DragonDiffusion/) [Author's blog explanation](https://www.zhihu.com/question/612852389/answer/3125192805?s_r=0&utm_campaign=shareopn&utm_medium=social&utm_oi=973429178926792704&utm_psn=1665158692570902528&utm_source=wechat_session) :+1:  [AK Twitter Blog](https://twitter.com/_akhaliq/status/1676808539317182464)
> [paper local pdf](./2023_07_Arxiv_DragonDiffusion--Enabling-Drag-style-Manipulation-on-Diffusion-Models.pdf)

## Key-point

- motivation

  `DragGAN` 基于 GAN，泛化能力弱，生成质量不高编辑方式单一。DragonDiffusion 着眼于如何将 Diffusion 模型强大的生成能力迁移到细粒度的图像编辑上，同时，设计更加一般化的图像编辑方式。

DragonDiffusion enables various editing modes for the generated or real images, including **object moving, object resizing, object appearance replacement, and content dragging.** 支持多种编辑方式

- classifier guidance based on the strong correspondence of intermediate features in the diffusion model.

  transform the editing signals into gradients via **feature correspondence loss**

- consistency between the original image and the editing result

  - a cross-branch self-attention
  - build a multi-scale guidance to consider both semantic and geometric alignment

![](https://mc-e.github.io/project/DragonDiffusion/static/assets/teaser.png)



**Contributions**



## Related Work

- DragGAN

  其底层模型是基于生成对抗网络（GAN），这导致有两方面不足。一是泛化能力弱，通用性会受到预训练 GAN 模型容量的限制，无法直接对任意图像进行编辑；二是较难实现高质量的生成内容。

  单一的点对点的拖拽式编辑难以满足用户复杂的编辑需求。

- DragDiffusion

- "Emergent Correspondence from Image Diffusion" Arxiv, 2023, DIFT
  [paper](https://arxiv.org/abs/2306.03881)

  根据扩散模型中间特征具有强对应关系这一观察

- Score-based diffusion

  > "Score-Based Generative Modeling through Stochastic Differential Equations" Arxiv, 2020 Nov, Score-based
  > [paper](https://arxiv.org/abs/2011.13456)
  >
  > - Langevin dynamics :question:
  
- Classifier guidance :question:
  "Diffusion Models Beat GANs on Image Synthesis" NeurIPS, 2021 May :statue_of_liberty:
  [paper](https://arxiv.org/abs/2105.05233) [code](https://github.com/openai/guided-diffusion?utm_source=catalyzex.com)
  $$
  ∇x_tlogq(x_t, y) = ∇x_tlogq(x_t) + ∇x_tlogq(y|x_t)
  $$
  $y$ 是外部 condition 信息，可以是 sketches, mask
  
  
  
- Stability Al 于上周发布了涂鸦生图工具 Stable Doodle 正使用了 Zhang jian 团队开发的T2I-Adapter作为核心控制技术。

- Consistency between Original image and generation result

  - DDIM inversion

    "Null-text Inversion for Editing Real Images using Guided Diffusion Models" CVPR, 2022 Nov
    [paper](https://arxiv.org/abs/2211.09794) [website](https://null-text-inversion.github.io/) [code ](https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images)Google github repo for null-text inversion :+1:

  - "Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation" Arxiv, 2022 Dec,
    [paper](https://arxiv.org/abs/2212.11565)

  - "Zero-Shot Video Editing Using Off-The-Shelf Image Diffusion Models" Arxiv, 2023 Mar, vid2vid-zero :star:
    [paper](https://arxiv.org/abs/2303.17599) [code](https://github.com/baaivision/vid2vid-zero?utm_source=catalyzex.com)

  - "MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing" Arxiv, 2023 Apr
    [paper](https://arxiv.org/abs/2304.08465)



## methods

DragonDiffusion 的技术方法，主要包含两个核心，即如何将**编辑信号注入到模型的扩散过程**当中，以及如何保持编辑之后的图像与**原图像具有内容一致性**。

![](https://mc-e.github.io/project/DragonDiffusion/static/assets/arch.png)

方法设计包含两个扩散分支：[引导分支](https://www.zhihu.com/search?q=引导分支&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3125192805})和生成分支。首先，待编辑图像通过 Stable Diffusion 的逆过程找到该图像在[扩散隐空间](https://www.zhihu.com/search?q=扩散隐空间&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3125192805})中的表示，作为两个分支的输入。其中，引导分支会对原图像进行重建，重建过程中将原图像中的信息注入下方的生成分支。生成分支的作用是根据引导信息对原图像进行编辑，同时保持主要内容与原图一致。

> Our DragonDiffusion is built based on Stable Diffusion [27], without model fine-tuning or training. **直接用 Stable-diffusion 不需要训练！**

### Classifier-guidance-based Editing Design

在每一个[扩散迭代步](https://www.zhihu.com/search?q=扩散迭代步&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3125192805})中，将两个分支的隐变量通过相同的UNet去噪器转换到特征域

Loss 约束引导区域 + 生成区域一致性（余弦相似度）
- 保持其他未编辑区域与原图的一致性

- 将编辑信号通过基于特征强对应关系的 score function 转化为梯度 :question:

  利用下面地 loss 约束：编辑区域和原图区域，其余区域的相似度，在 loss 里面的 mask 体现编辑

$$
∇z_t^{gen}logq(z_t^{gen},m^{gen},m^{share})=∇z_t^{gen}logq(z_t^{gen})+∇z_t^{gen}logq(m^{gen},m^{share}|z_t^{gen}).
$$

 score function 的 classifier guidance 部分用 loss 来表示
$$
∇z_t^{gen}logq(m^{gen},m^{share}|z_t^{gen}) = η * \frac{dL}{dz_t^{gen}}
$$

### Cross-branch Self-attention

- 进一步约束编辑结果与原图的一致性，我们设计了一种跨分支的自注意力机制

  利用引导分支自注意力模块中的Key和Value替换生成分支自注意力模块中的Key和Value，以此来实现特征层面的参考信息注入

  论文 DIFT 中发现 encoder 中的特征关联性很弱，所以这个 **KV 替换之对于 U-net decoder 进行**



### multi-scale Guidance

The **decoder** of the Unet denoiser contains four blocks of different scales

讨论前面 loss 内部计算相似度的 Decoder 特征 F，用指定层作为 F 的效果，最后发现 layer2,3 的信息分别是 high/low level。
因此 $F^{gen}_t ,F^{gud}_t$ 用 two sets of features from layer2, 3



## Experiment

针对不同编辑方式：object moving, object resizing, object appearance replacement, and content dragging 展示了一下图像生成效果

## **Limitations**

- object moving 物体细节略微不一致，例如咖啡的奶花移动后不一致



## Summary :star2:

> learn what & how to apply to our task
> understanding status: 代码没 release & 粗看论文借鉴一下思路

- 利用 DIFT 论文中发现 Denoiser Decoder 中间特征的相关性
  1. 通过 guidance U-net decoder 的 KV，替换到 generator 的对应 KV **实现原图和生成结果的 consistency**
  2. 2 个 branch 的 **U-net decoder** 使用不同 loss 实现 guidance 和 generation
  3. Classifier guidance 用 Loss 代替， 用 U-net decoder 第 2&3 层的特征作为图像特征，计算 loss