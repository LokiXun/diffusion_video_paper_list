# Stable Diffusion :moyai: (High-Resolution Image Synthesis with Latent Diffusion Models )



> [2022_CVPR_High-Resolution Image Synthesis with Latent Diffusion Models.pdf](./2022_CVPR_High-Resolution Image Synthesis with Latent Diffusion Models.pdf)
> [github](https://github.com/CompVis/stable-diffusion)
> [Youtube 博主解读](https://www.youtube.com/watch?v=f6PtJKdey8E) [知乎博客](https://zhuanlan.zhihu.com/p/582693939) [towardScience 博客](https://towardsdatascience.com/paper-explained-high-resolution-image-synthesis-with-latent-diffusion-models-f372f7636d42) :+1:
> [What can Stable Diffusion do?](https://stable-diffusion-art.com/how-stable-diffusion-work/)

- checklist
  - [ ] 感知压缩 Perceptual Image Compression、扩散模型
  - [ ] 条件机制 Conditioning Mechanisms，通过cross-attention的方式来实现多模态训练，使得条件图片生成任务也可以实现
  - [ ] time-conditional UNe
  - basics deficit
    - [ ] VAE



**Background**

- Bacis Knowledge

  image generation has been tackled mainly through **four families of models**: Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), Autoregressive Models (ARMs), Diffusion Probabilistic Models(DMs)

  - GANs: show promising results for **data with limited variability**, bearing mode-collapse, **unstable training**

    > **Mode collapse**: *this phenomenon occurs when the generator can alternately generate a limited number of outputs that fool the discriminator. In general, GANs **struggle capturing the full data distribution.***

  - VAE(Variational Autoencoders)

    not suffer from mode-collapse and can efficiently generate high-resolution images. **sample quality** is **not** always **comparable to** that of **GANs**.

  - DPM

    operation in **pixel space** by adding or removing noise to a tensor of the same size as the original image results in **slow inference speed** and **high computational cost**

    



**Contributions**

- 大幅降低训练和采样阶段的计算复杂度，让文图生成等任务能够在消费级GPU上，在10秒级别时间生成图片，大大降低了落地门槛

**Related Work**

## **methods**

> - :question:  VAE?
> - KL-reg, VQ-reg

`High-Resolution Image Synthesis with Latent Diffusion Models`, we can break it down into **four main steps** :star:

### Perceptual Image Compression

**Latent Diffusion explicitly separates the image compression** phase to **remove high frequency details (perceptual compression**), and learns a semantic and conceptual composition of the data (**semantic compression**).

> 原有的非感知压缩的扩散模型有一个很大的问题在于，由于在像素空间上训练模型，如果我们希望生成一张分辨率很高的图片，这就意味着我们训练的空间也是一个很高维的空间。
> 引入感知压缩就是说**通过 VAE 这类自编码模型对原图片进行处理，忽略掉图片中的高频信息**，只保留重要、基础的一些特征。
>
> 感知压缩主要利用一个预训练的自编码模型，该模型能够学习到一个在**感知上等同于图像空间的潜在表示空间 latent space**。优势是只需要训练一个通用的自编码模型，就可以用于不同的扩散模型的训练，在不同的任务上（image2image，text2image）使用。
>
> **基于感知压缩的扩散模型的训练**本质上是一个两阶段训练的过程，第一阶段需要训练一个自编码器，第二阶段才需要训练扩散模型本身。
>
> $\mathcal{x} \in R^{H\times W \times 3} \rightarrow R^{h\times w \times 3} $，下采样因子 $f = H/h = W/w = 2^m$ 
>
> - 压缩比的选取，比较 FID，Inception Scores vs. training progress



#### Optimization

![VQGAN_approach_overview.jpg](C:\Users\Loki\workspace\LearningJourney_Notes\Tongji_CV_group\docs\VQGAN_approach_overview.jpg)

- Regularization

  - KL(Kullback-Leibler) 散度 [博客参考](https://zhuanlan.zhihu.com/p/100676922)

    **计算两个概率分布之间差异**，当用一个去近似另一个分布，计算会有多少损失。ps: KL 散度很像距离，但不是，因为 :Loss_KL(p|q) != Loss_KL(q|p) 不对称！





### Latent Diffusion Models

LDM 在训练时就可以利用编码器得到 $z_t$
$$
L_{DM} := E_{\epsilon(x),\varepsilon\sim\mathcal{N}(0,1),t}{[\abs{\abs{ \epsilon - \epsilon_\theta(x_t,t)}}^2_2]}
\\
L_{LDM} := E_{\epsilon(x),\varepsilon\sim\mathcal{N}(0,1),t}{[\abs{\abs{ \epsilon - \epsilon_\theta(z_t,t)}}^2_2]}
$$


### Conditioning Mechanisms





**Unsolved Limitations**



**Summary**

> learn what & how to apply to our task

