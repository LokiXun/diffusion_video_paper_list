# LDMVFI: Video Frame Interpolation with Latent Diffusion Models

> "LDMVFI: Video Frame Interpolation with Latent Diffusion Models" Arxiv, 2023 Mar :+1:
> [paper](https://arxiv.org/abs/2303.09508) [code](https://github.com/danier97/LDMVFI)
> [paper local pdf](./2023_03_Arxiv_LDMVFI--Video-Frame-Interpolation-with-Latent-Diffusion-Models.pdf)

## **Key-point**

first generative modeling approach that addresses video frame interpolation，**contains two major components**

- VQ-FIGAN

  replace LDM encoder, decoder with VQ-FIGAN

  1. LDM encoder 没针对 VFI 任务设计，用 MaxViT atten & deformable conv 增强 VFI 性能
  2. **在 decoder 中加入 GT code  >> 增强 reconstruction consistency**

- denoising U-net

  针对 VFI 任务，inference 时候能够使用前后帧信息，用前后帧的 code 作为 condition



**Contributions**

- LDM used in VFI，用前后帧的 code 作为 condition
- replace LDM encoder, decoder with VQ-FIGAN
  1. LDM encoder 没针对 VFI 任务设计，用 MaxViT atten & deformable conv 增强 VFI 性能
  2. **在 decoder 中加入 GT code  >> 增强 reconstruction consistency**
- replace vanilla attention with MaxViT attention >> improve efficiency



## **Related Work**

- baseline VFI 分为两个类别

  - 光流预测
  - predict locally **adaptive convolution kernels** to synthesize output pixels

- 之前 DM 用于 VFI(video frame interpolation)

  "MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation" NeurIPS, 2022 May
  [paper](https://arxiv.org/abs/2205.09853) [website](https://mask-cond-video-diffusion.github.io/?utm_source=catalyzex.com)

  只能对于低分辨率生成。

  发现 LDM 还没人用于 VFI



## **methods**

> - 写作 trick
>   - 简述一下 LDM

- Task Definition
  $$
  I^0, I^1 >> I^n, ~n=1/up\_sample\_rate(if~up=2,n=0.5)
  $$
  

 The proposed LDMVFI contains **two main components**: 

1. VQ-FIGAN: a VFI-specific autoencoding model
   projects frames into a latent space, and reconstructs the target frame
2. denoising U-Net that performs reverse diffusion process in the latent space for conditional image generation



> 整体框架

![](https://camo.githubusercontent.com/b35bcef303836a082f561e0722a24cbd4447ac80b683a350e47cc56138972535/68747470733a2f2f64616e69657239372e6769746875622e696f2f4c444d5646492f6f766572616c6c2e737667)



### VQ-FIGAN

![](./docs/LDMVFI_VQ-FIGAN_structure.png)

**Training VQ-FIGAN**

作者发现 LDM 自带的 encoder 压缩效果不行，就自己设计了一个 Encoder 映射到 latent code

模仿 VQ-GAN

**如何在 decoder 融合 I0, I1 信息**

MaxViT attention: Q=latent code, KV=i0,i1 的 code. >> reversion Diffusion 预测两帧之间的差异，和 i0 关联得到一个权值矩阵，加到 i1 上

**locally adaptive deformable convolutions** 

> TODO: equ 4-6 decoder 最后在送入这个 deformable conv

### Conditional generate LDM

- 把 I0,i1 作为条件，reverse diffusion 如何实现? :question:

reverse diffusion, y 为前后帧 GT
$$
pθ(x_{t−1}|x_t, y)
$$

> Experiment Section 最后一小节有介绍
>
> conditioning the denoising U-Net on the latents z 0 , z1 （前后帧 GT 的 code）is concatenation





## **Experiment**

3090 GPUs, VQ-FIGAN 70 epochs, U-net 60 epochs

### Dataset

> Appendix J 提供了各个数据集的 url

- Training
  Vimeo90K + BVI-DVC quintuplets

  The final training set consists of 64612 frame septuplets from Vimeo90k and 17600 frame quintuplets from BVI-DVC provided by [11]

- evaluation on commonly used VFI benchmarks

  UCF101, DAVIS, SNU-FILM

- Full HD evaluation

  BVI-HFR [43] dataset

### Quantitative Result

- 10 baselines

  including BMBC [50], AdaCoF [37], CDFI [15], XVFI [60], ABME [51], IFRNet [35], VFIformer [41], ST-MFNet [11], FLAVR [30], and MCVD [70].

  **All these models were re-trained on our training dataset for fair comparison**



### Ablation Study

- Effectiveness of VQ-FIGAN

  对比 VQ-GAN 类似方式，Decoder 中的 MaxCABlocks 替换为 ResBlock && 去掉使用 $\phi^0, \phi^1$

- Downsampling Factor f

  f increases from 8 to 32, there is generally an increasing trend in model performance;
  f=64 效果降低很多

- U-net 参数量

  Table 3 reflects a decreasing trend in model performance as c is decreased



## **Limitations**

See Appendix H.

1. much slower inference speed 对比其他 SOTA 方法

2. model parameters in LDMVFI is also larger

   knowledge distillation [21] and model compression [4] can be used

3.  two-stage training strategy, large model size and slow inference speed of LDMVFI mean that large-scale training and evaluation processes

   训练推理慢。。。



**Summary :star2:**

> learn what & how to apply to our task

可以模仿这个把 LDM 用到 VFI 的思路

- 如果 LDM encoder 效果不好，尝试 replace LDM encoder, decoder with VQ-FIGAN && MaxViT 考虑换一下

  LDM encoder 没针对 VFI 任务设计，用 MaxViT atten & deformable conv 增强 VFI 性能

- **在 decoder 中加入 GT code  >> 增强 reconstruction consistency**

- 针对任务设计自己的 condition