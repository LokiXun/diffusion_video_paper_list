# Bringing-Old-Films-Back-to-Life

> [2022_CVPR_Bringing-Old-Films-Back-to-Life.pdf](./2022_CVPR_Bringing-Old-Films-Back-to-Life.pdf)
> [github repo](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life.git)
>
> - 4 x 2080Ti >> 44G 显存

**Background**

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





**Related Work**

- Image restoration

  only focus on a single type of degradation. 无法应对真实场景的多种 degradation 同时出现的情况

- Video Restoration

  > [EF2GVI Video Inpainting](https://github.com/MCG-NKU/E2FGVI)

  1. denoising, deblurring, super-resolution 生成效果有限
  2. video inpainting 需要指定 mask 的区域，在 old-films 中没有

- Old-film restoration

  - 传统方法去除 **structed artifacts**( scratches, cracks, etc.)：先目标检测，再加 inpainting pipline

    无法处理 photometric degradations (e.g., blurriness and noises)；以来手工特征检测 structed artifacts，没有理解内容

  - DeepRemaster



## **methods**

> RTN 框加，data-simulation, colorization, optimization

![RTN_pipline.jpg](C:\Users\Loki\workspace\LearningJourney_Notes\Tongji_CV_group\docs\RTN_pipline.jpg)

**Spatial-temporal Module**

![RTN_temporal-spatial-restoration_module.jpg](C:\Users\Loki\workspace\LearningJourney_Notes\Tongji_CV_group\docs\RTN_temporal-spatial-restoration_module.jpg)

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



**Unsolved Limitations**

- fail to distinguish the contaminant from the frame content

  存在退化被错误识别为场景内容，从而没有得到去除。例如黑色划痕被误识别为场景内的烟雾

- GAN may synthesize inadequate high-frequency details and artifacts

  生成的细节不足，存在噪声

- 对于 barely recognizable 的内容很难重建



**Summary(learn what & how to apply to our task) :star2:**

- GAN generate pixel
- SwimTransformer





