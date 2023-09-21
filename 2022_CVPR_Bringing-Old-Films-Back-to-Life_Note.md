# Bringing-Old-Films-Back-to-Life

> "Bringing Old Films Back to Life" CVPR, 2022 Mar :star:
> [paper](https://arxiv.org/abs/2203.17276) [code](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life) [website](http://raywzy.com/Old_Film/)
> [lcoal paper pdf](./2022_CVPR_Bringing-Old-Films-Back-to-Life.pdf)

## **Keypoint**

老电影分辨率过低，存在退化问题（灰尘遮挡、划痕噪声，颜色缺失等问题)。人工修复成本太大，利用自动化方式对老电影进行修复，提升老电影观感质量。



**Contributions**

- unify the entire film restoration tasks with a single framework in which we conduct spatio-temporal **restoration and coloarization**.

  1. **the memorization nature of recurrent modules** benefits the temporal coherency whereas the **long-range modeling capability of transformers** helps the spatial restoration
  2. 上色：模仿人工方式，对一帧上色，然后 propagated 到其他帧

- bi-directional RNN 累计前后相邻帧的信息，减少 flickering。recurrent module 输出的 hidden_state embedding 表示场景内容

  1. 保持 temporal consistency 前后帧一致，避免 flickering
  2. 实现 occluded 区域只要在别的帧出现过，能够复原出来
  3. structured defects (划痕) can be localized 刚出现的地方前后帧差别很大

  - :grey_question: 为啥用 bidirectional：由于 flickering 问题，需要结合长时间的序列分析，因此用 bidirectional 信息

- `Swin Transformer ` 实现不同位置像素的信息交换，提升 restore mixed degradation 的效果，缓解 frame alignment 的问题





## **Related Work**

- Image restoration

  only focus on a single type of degradation. 无法应对真实场景的多种 degradation 同时出现的情况

- Video Restoration

  > [EF2GVI Video Inpainting](https://github.com/MCG-NKU/E2FGVI)

  1. denoising, deblurring, super-resolution 生成效果有限
  2. video inpainting 需要指定 mask 的区域，在 old-films 中没有


**Old-film restoration**

- 传统方法去除 **structed artifacts**( scratches, cracks, etc.)：先目标检测，再加 inpainting pipline

  无法处理 photometric degradations (e.g., blurriness and noises)；以来手工特征检测 structed artifacts，没有理解内容

- 修复 baseline

  - Deepremaster

  - Bring-Old-photos & temporal smoothing

    > "Learning blind video temporal consistency" ECCV, 2018 Aug
    > [paper](https://arxiv.org/abs/1808.00449)

  - BasicVSR

    > "BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond" CVPR, 2020 Dec
    >
    > [paper](https://arxiv.org/abs/2012.02181) [code](https://github.com/ckkelvinchan/BasicVSR-IconVSR) [website](https://ckkelvinchan.github.io/projects/BasicVSR/) [blog explanation](https://zhuanlan.zhihu.com/p/364872992)

  - Video Swim

    > "Video Swin Transformer" CVPR, 2021 Jun
    > [paper](https://arxiv.org/abs/2106.13230) [code](https://github.com/SwinTransformer/Video-Swin-Transformer?utm_source=catalyzex.com)

- 上色 Baseline

  - DeepExemplar

    > "Deep Exemplar-based Video Colorization" CVPR, 2019 Jun
    > [paper](https://arxiv.org/abs/1906.09909) [code](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization)
    > [local pdf](./2019_07_CVPR_Deep-Exemplar-based-Video-Colorization.pdf) [note](./2019_07_CVPR_Deep-Exemplar-based-Video-Colorization_Note.md)

- Flow estimation

  - RAFT



## **methods**

> RTN 框加，data-simulation, colorization, optimization

![RTN_pipline.jpg](.\docs\RTN_pipline.jpg)

**Spatial-temporal Module**

![RTN_temporal-spatial-restoration_module.jpg](./docs\RTN_temporal-spatial-restoration_module.jpg)

- Temporal Recurrent Network

  > :question: convolutional encoder E, W
  >
  > 1. F aggregates the states between history and current feature
  > 2. R further recovers and boosts the hidden representation
  > 3. 光流怎么得到
  > 4. 如何重建图像？

  $$
  st = R↑ ◦ F↑(E(x_t), W(s_{t−1}, f_{t−1→t}))\\
  yt = D(\hat{st} \frown st) ∈ R^{H×W}  \\\frown means~concatenate, \text{D is a convolution decoder}
  $$

- Spatial Transformer Restoration

  CNN 的 locality & pattern restoration 无法适用于很多 artifact 的情况。此时提取的光流（前后真对应像素的关系）会存在错误。

- Learnable Guided Mask

  > M 为 shallow network

  1. 检测 artifact 的方式，训练数据难搞，检测效果不好
  2. 有些 dirt 是透明的，还存在一些信息可以利用

  用一种动态加权方式进性结合
  $$
  M = \mathcal{M}(E(x_t)
  \frown W)
  $$

- 生成 old-film 数据

  1. Contaminant Blending
  2. Quality Degradation

  随机对视频帧中使用上面 2 种方式

- Colorization :question:

- Loss

  1. perceptual loss
  2. Temporal-PatchGAN
  3. hinge loss





## **Experiment**

> 4 x 2080Ti >> 44G 显存





### **Ablation Study**

用别的去替换 Temporal Bi-directional RNN， Learnable Guided Mask，Spatial Transformer 看效果

- 合成数据集上的效果，比较 PSNR，SSIM
  DAVIS dataset & degradation model(See [Dataset](#Dataset))
  blending random degradations with clean frames of DAVIS [34] dataset

- 真实老电影上的效果（作者收集的 63个老电影）比较 NIQE, BRISQUE && 看图

- 上色效果

  gray version of the subset of REDS 

  predict the colors of the first 50 frames by taking the 100th frame as the colorization reference 因为对比的是 example-based 上色方法



### setting

**During optimization**, we randomly crop 256 patches from REDS [33] dataset and apply the proposed video degradation model on the fly
batchsiz=4, 4*2080Ti

- learning rate is set to 2e-4 for both generators and disc
- 微调 RAFT



### Dataset

> :question: 4 张卡做 DDP batchsize=4, 每张卡 batchsize = 1?

- training set

  crop 256 patches from REDS [33] dataset and apply the proposed video degradation model on the fly

  **batchsize = 4, 4 卡 DDP, 训  2 天**

- Quantitative Comparison

  - DAVIS 视频 + 合成: 计算 PSNR, SSIM 等客观指标
  - 收集的 63 个真实老电影: 计算 NIQE, BRISQUE 指标





#### Degradation

- Contaminant Blending

  **collect 1k+ texture templates** from the Internet, which are further augmented with random rotation, local cropping, contrast change, and morphological operations. Then we **use addition, subtract and multiply blending modes** with various levels of opacity∈ [0.6, 1.0] to combine the scratch textures with the natural frames.

- blurring, noise and unsharpness

  - Gaussian noise and speckle noise with σ ∈ [5, 50]
  -  Isotropic and anisotropic Gaussian blur kernels
  - Random *downsampling and upsampling* with different interpolation methods
  - *JPEG compression* whose level is in the range of [40, 100].
  - *Random color jitter* through adjusting the brightness∈ [0.8, 1.2] and contrast∈ [0.9, 1.0].

**inject these defects randomly,** one observation of old films. **first define a set of template parameters for each video.** Then we further apply predefined parameters with slight perturbations on consecutive temporal frames to achieve more realistic rendering
随机将上述 DA 映射到随机的帧上，相邻帧之间用相同的 DA 组合 & 扰动

> Code implementation

- `def degradation_video_list_4`

  先做 UMS 锐化：随机取一个 gaussianBlur，然后原图 - blur 得到一个残差，这个残差和原图加权平均一下

  


### Loss

- L1 loss 多帧平均

- perceptual loss

  loss is accumulated over all frames in the generated video

- Spatial-Temporal Adversarial Loss

  Temporal-PatchGAN  enhance both perceptual quality and spatial-temporal coherence

对于上色 task，转换到 LAB 空间来处理





**Limitations**

- fail to distinguish the contaminant from the frame content

  存在退化被错误识别为场景内容，从而没有得到去除。例如黑色划痕被误识别为场景内的烟雾

- GAN may synthesize inadequate high-frequency details and artifacts

  生成的细节不足，存在噪声

- 对于 barely recognizable 的内容很难重建



## **Summary :star2:**

> learn what & how to apply to our task

- GAN generate pixel
- SwimTransformer





