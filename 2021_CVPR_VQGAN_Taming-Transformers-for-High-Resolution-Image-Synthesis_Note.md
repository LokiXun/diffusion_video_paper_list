# VQ-GAN

> [2021_CVPR_VQGAN_Taming-Transformers-for-High-Resolution-Image-Synthesis.pdf](./2021_CVPR_VQGAN_Taming-Transformers-for-High-Resolution-Image-Synthesis.pdf)
> [CompVis/taming-transformers: Taming Transformers for High-Resolution Image Synthesis (github.com)](https://github.com/CompVis/taming-transformers)
> https://compvis.github.io/taming-transformers/
>
> [Youtuber The AI Epiphany VQ-GAN Paper explained](https://www.youtube.com/watch?v=j2PXES-liuc) :+1:
> [WeChat blog & code](https://mp.weixin.qq.com/s/iyfUDU93GUNqKtOJDUKjKA) >> 简单补一下 VQ-VAE

motivation: Stable Diffusion 的 Encoder 使用 VQ-GAN 中的方法训练

**Background**

VQ-GAN uses modified VQ-VAEs and a powerful transformer (GPT-2) to synthesize high-res images.

**Contributions**

- The first methods using Transformer to generate images
- An important modification of VQ-VAE 
  1. changing MSE for perceptual loss
  2. adding adversarial loss which makes the images way more crispy

**Related Work TODO**

- [ ] [VQ-VAEs](https://www.youtube.com/watch?v=VZFVUrYcig0)



## methods

VQGAN使用了**两阶段的图像生成方法**

- 训练时，先训练一个**图像压缩模型（VQGAN**，包括编码器和解码器两个子模型），再训练一个生成压缩图像的模型。
- 生成时，先用第二个模型生成出一个压缩图像，再用第一个模型（**基于Transformer的模型）复原成真实图像**。

> 从 Transformer 计算开销只能生成小图的限制 & VQ-VAE 生成的模糊，两个角度来展开 VQ-GAN 的改进点

相比擅长捕捉局部特征的CNN，Transformer的优势在于它能更好地融合图像的全局信息。可是，Transformer的自注意力操作开销太大，只能生成一些分辨率较低的图像。因此，作者认为，可以综合CNN和Transformer的优势，**先用基于CNN的VQGAN把图像压缩成一个尺寸更小、信息更丰富的小图像，再用Transformer来生成小图像。**

为提升 VQVAE 的生成效果，作者提出了两项改进策略：1) 图像压缩模型VQVAE仅使用了均方误差，压缩图像的复原结果较为模糊，可以把**图像压缩模型换成GAN**；2) 在**生成压缩图片这个任务上，基于CNN的图像生成模型比不过Transformer**，可以用Transformer代替原来的CNN。



### VQ-VAE

> [WeChat blog & code](https://mp.weixin.qq.com/s/iyfUDU93GUNqKtOJDUKjKA) >> 简单补一下 VQ-VAE





## Experiment

- Transformer and other auto-regressive Methods



**Unsolved Limitations**



**Summary(learn what & how to apply to our task) :star2:**



